import math
import random
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import config
from .ml_ranking_model import (
    FINAL_PUNCTUATION_MARKS,
    GROUP_TO_ID,
    SPECIAL_EOS,
    SPECIAL_FEATURE_ID,
    SPECIAL_PAD,
    SUFFIX_OFFSET,
    resolve_torch_device,
)

MAX_POSITION_IN_WORD = 64
MAX_SENTENCE_WORDS = 512
CHAIN_ENCODER_LAYERS = 2
NO_LOSS = -100
MASKED_SCORE = -1e9


def end_marker_chain(final_token: Optional[int]) -> List[tuple]:
    token = SPECIAL_EOS if final_token is None else final_token
    return [(token, SPECIAL_FEATURE_ID, 1)]


def build_batch(sentences: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    chain_tokens = []
    chain_groups = []
    chain_positions = []
    sentence_index = []
    word_index = []
    candidate_index = []
    gold_rows = []
    most_words = 0
    most_candidates = 1

    for s, sentence in enumerate(sentences):
        words = list(sentence["candidates"]) + [[end_marker_chain(sentence["final_token"])]]
        gold = sentence.get("gold") or [None] * len(sentence["candidates"])
        most_words = max(most_words, len(words))
        gold_row = []
        for w, candidates in enumerate(words):
            most_candidates = max(most_candidates, len(candidates))
            if w < len(gold) and gold[w] is not None and len(candidates) > 1:
                gold_row.append(gold[w])
            else:
                gold_row.append(NO_LOSS)
            for k, chain in enumerate(candidates):
                chain_tokens.append([token[0] for token in chain])
                chain_groups.append([token[1] for token in chain])
                chain_positions.append([min(token[2], MAX_POSITION_IN_WORD - 1) for token in chain])
                sentence_index.append(s)
                word_index.append(w)
                candidate_index.append(k)
        gold_rows.append(gold_row)

    longest_chain = max(len(chain) for chain in chain_tokens)

    def padded(rows, value):
        return [row + [value] * (longest_chain - len(row)) for row in rows]

    chain_padding = [[False] * len(row) + [True] * (longest_chain - len(row)) for row in chain_tokens]
    word_padding = [[False] * len(row) + [True] * (most_words - len(row)) for row in gold_rows]
    gold = [row + [NO_LOSS] * (most_words - len(row)) for row in gold_rows]

    return {
        "chain_tokens": torch.tensor(padded(chain_tokens, SPECIAL_PAD), dtype=torch.long, device=device),
        "chain_groups": torch.tensor(padded(chain_groups, SPECIAL_FEATURE_ID), dtype=torch.long, device=device),
        "chain_positions": torch.tensor(padded(chain_positions, SPECIAL_FEATURE_ID), dtype=torch.long, device=device),
        "chain_padding": torch.tensor(chain_padding, dtype=torch.bool, device=device),
        "sentence_index": torch.tensor(sentence_index, dtype=torch.long, device=device),
        "word_index": torch.tensor(word_index, dtype=torch.long, device=device),
        "candidate_index": torch.tensor(candidate_index, dtype=torch.long, device=device),
        "word_padding": torch.tensor(word_padding, dtype=torch.bool, device=device),
        "gold": torch.tensor(gold, dtype=torch.long, device=device),
        "shape": (len(sentences), most_words, most_candidates),
    }


class CandidateDisambiguator(nn.Module):
    def __init__(self, suffix_vocab_size: int, closed_class_vocab_size: int, device: Optional[Any] = None):
        super().__init__()
        vocab_size = SUFFIX_OFFSET + suffix_vocab_size + closed_class_vocab_size + len(FINAL_PUNCTUATION_MARKS)
        width = config.embed_dim
        self.token_embed = nn.Embedding(vocab_size, width, padding_idx=SPECIAL_PAD)
        self.group_embed = nn.Embedding(len(GROUP_TO_ID), width)
        self.position_in_word_embed = nn.Embedding(MAX_POSITION_IN_WORD, width)
        self.chain_encoder = nn.TransformerEncoder(
            self._encoder_layer(), num_layers=CHAIN_ENCODER_LAYERS, enable_nested_tensor=False,
        )
        self.chain_norm = nn.LayerNorm(width)
        self.word_projection = nn.Linear(2 * width, width)
        self.word_position_embed = nn.Embedding(MAX_SENTENCE_WORDS, width)
        self.sentence_encoder = nn.TransformerEncoder(
            self._encoder_layer(), num_layers=config.num_layers, enable_nested_tensor=False,
        )
        self.score_head = nn.Sequential(
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.to(resolve_torch_device(device))

    @staticmethod
    def _encoder_layer() -> nn.TransformerEncoderLayer:
        return nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

    def encode_chains(self, batch: Dict[str, Any]) -> torch.Tensor:
        x = (
            self.token_embed(batch["chain_tokens"])
            + self.group_embed(batch["chain_groups"])
            + self.position_in_word_embed(batch["chain_positions"])
        )
        x = self.chain_encoder(x, src_key_padding_mask=batch["chain_padding"])
        present = (~batch["chain_padding"]).unsqueeze(-1).to(x.dtype)
        pooled = (x * present).sum(dim=1) / present.sum(dim=1).clamp_min(1.0)
        return self.chain_norm(pooled)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        chains = self.encode_chains(batch)
        sentences, words, candidates_per_word = batch["shape"]
        s = batch["sentence_index"]
        w = batch["word_index"]
        k = batch["candidate_index"]

        grid = chains.new_zeros(sentences, words, candidates_per_word, chains.size(-1))
        grid[s, w, k] = chains
        present = torch.zeros(sentences, words, candidates_per_word, dtype=torch.bool, device=chains.device)
        present[s, w, k] = True

        candidate_counts = present.sum(dim=2, keepdim=True).clamp_min(1).to(grid.dtype)
        mean = grid.sum(dim=2) / candidate_counts
        strongest = grid.masked_fill(~present.unsqueeze(-1), float("-inf")).max(dim=2).values
        strongest = strongest.masked_fill(batch["word_padding"].unsqueeze(-1), 0.0)

        positions = torch.arange(words, device=chains.device).clamp(max=MAX_SENTENCE_WORDS - 1)
        word_vectors = self.word_projection(torch.cat([mean, strongest], dim=-1)) + self.word_position_embed(positions)
        context = self.sentence_encoder(word_vectors, src_key_padding_mask=batch["word_padding"])

        word_context = context[s, w]
        pair = torch.cat([word_context, chains, word_context * chains], dim=-1)
        chain_scores = self.score_head(pair).squeeze(-1).float()

        scores = torch.full((sentences, words, candidates_per_word), MASKED_SCORE, device=chains.device)
        scores[s, w, k] = chain_scores
        return scores


class CandidateTrainer:
    def __init__(self, model: CandidateDisambiguator, device: Optional[Any] = None):
        self.device = resolve_torch_device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    @staticmethod
    def _learning_rate_factor(step: int, warmup: int, total_steps: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
        floor = float(config.lr_eta_min_ratio)
        return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))

    def _train_step(self, sentences: List[Dict[str, Any]]) -> Optional[float]:
        batch = build_batch(sentences, self.device)
        scored = batch["gold"] != NO_LOSS
        if not scored.any():
            return None
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            scores = self.model(batch)
        loss = F.cross_entropy(scores[scored], batch["gold"][scored])
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss.item())

    def train(
        self,
        train_sentences: List[Dict[str, Any]],
        dev_sentences: List[Dict[str, Any]],
        epochs: int,
        batch_size: int = 32,
        fit_sample_size: int = 2000,
    ) -> List[Dict[str, Any]]:
        total_steps = epochs * math.ceil(len(train_sentences) / batch_size)
        warmup = min(int(config.warmup_steps), max(1, total_steps // 10))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: self._learning_rate_factor(step, warmup, total_steps),
        )
        fit_sample = random.sample(train_sentences, min(fit_sample_size, len(train_sentences)))
        order = list(train_sentences)
        history = []
        best_dev_accuracy = -1.0
        best_state = None

        for epoch in range(1, epochs + 1):
            started = time.monotonic()
            random.shuffle(order)
            self.model.train()
            losses = []
            for start in range(0, len(order), batch_size):
                loss = self._train_step(order[start:start + batch_size])
                scheduler.step()
                if loss is not None:
                    losses.append(loss)

            dev_accuracy = self.accuracy(dev_sentences)
            train_accuracy = self.accuracy(fit_sample)
            mean_loss = sum(losses) / max(len(losses), 1)
            history.append({
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_ambiguous_accuracy": train_accuracy,
                "dev_ambiguous_accuracy": dev_accuracy,
            })
            print(
                f"   Candidate epoch {epoch}/{epochs}: loss={mean_loss:.4f} "
                f"train_ambiguous_acc={train_accuracy:.4f} dev_ambiguous_acc={dev_accuracy:.4f} "
                f"({time.monotonic() - started:.1f}s)",
                flush=True,
            )
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                best_state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}

        self.model.load_state_dict(best_state)
        return history

    def predict(self, sentences: List[Dict[str, Any]], batch_size: int = 64) -> List[List[int]]:
        self.model.eval()
        picks = []
        with torch.no_grad():
            for start in range(0, len(sentences), batch_size):
                chunk = sentences[start:start + batch_size]
                batch = build_batch(chunk, self.device)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    best = self.model(batch).argmax(dim=-1).tolist()
                for sentence, row in zip(chunk, best):
                    picks.append(row[:len(sentence["candidates"])])
        return picks

    def accuracy(self, sentences: List[Dict[str, Any]]) -> float:
        picks = self.predict(sentences)
        correct = 0
        total = 0
        for sentence, sentence_picks in zip(sentences, picks):
            for candidates, gold, pick in zip(sentence["candidates"], sentence["gold"], sentence_picks):
                if gold is None or len(candidates) < 2:
                    continue
                total += 1
                correct += pick == gold
        return correct / total if total else 0.0
