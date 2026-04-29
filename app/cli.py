import os
from typing import List, Optional, Dict, Any

from app.workflows import WorkflowEngine
from app.sequence_matcher import get_top_sentence_predictions
from app.input import sanitize_word, sanitize_sentence


class AppCLI:
    """Thin I/O layer for the interactive flow.

    Each command follows the same shape:
        raw text -> sanitize -> engine.analyze_* -> display -> user input
                 -> engine.commit_*

    Sanitation happens here at the boundary; the engine and analyzer only
    ever see canonical (tr_lower'd, apostrophe-stripped) input.
    """

    def __init__(self):
        self.engine = WorkflowEngine()

    def welcome(self):
        print("\n Commands:")
        print("  - Enter a word to analyze and train")
        print("  - 'sentence <text>' - Train on a full sentence")
        print("  - 'eval sentence <text>' - Evaluate model's top 10 guesses on a sentence")
        print("  - 'auto' - Start auto mode (random words from dictionary)")
        print("  - 'eval <word>' - Evaluate model on a word")
        print("  - 'relearn' - Train on all logged decompositions")
        print("  - '10FKV' - 10-fold cross-validation with 95% CI")
        print("  - 'stats' - Show training statistics")
        print("  - 'sample' - Analyze a text word by word.")
        print("  - 'sample sentence' - Analyze a text sentence by sentence.")
        print("  - 'save' - Save model")
        print("  - 'quit' - Exit")

    def show_message(self, msg: str):
        print(msg)

    def get_input(self, prompt: str) -> str:
        return input(prompt)

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def _show_single_option(self, display_idx: int, vm: Dict[str, Any]):
        print(f"\n[Option {display_idx}]")
        if vm.get('score') is not None:
            print(f"ML Score: {vm['score']:.4f}")
        print(f"Root:      {vm['root_str']}")
        if vm.get('has_chain'):
            print(f"Suffixes:  {vm['suffixes_str']}")
            print(f"Names:     {vm['names_str']}")
            print(f"Formation: {vm['formation_str']}")
        else:
            print(f"Formation: {vm['root_str']} (no suffixes)")
        print(f"Final POS: {vm['final_pos']}")
        print("-" * 70)

    def show_decompositions(self, word: str, view_models: List[Dict[str, Any]]):
        print(f"\n{word}")
        for i, vm in enumerate(view_models, 1):
            self._show_single_option(i, vm)

    def show_sentence_prediction(self, display_idx: int, score: float, words: List[str], view_models: List[Dict[str, Any]], aligned_str: str):
        print(f"\n[Option {display_idx}] Score: {score:.4f}")
        print(f"    {aligned_str}\n")
        for w, vm in zip(words, view_models):
            print(f"  {w}:")
            print(f"    Root:      {vm['root_str']}")
            if vm.get('has_chain'):
                print(f"    Suffixes:  {vm['suffixes_str']}")
                print(f"    Names:     {vm['names_str']}")
                print(f"    Formation: {vm['formation_str']}")
            else:
                print(f"    Formation: {vm['root_str']} (no suffixes)")
            print(f"    Final POS: {vm['final_pos']}")
        print("-" * 70)

    def show_stats(self, stats: Dict[str, Any]):
        print("\n Training Statistics:")
        print(f"  Total examples: {stats['total']}")
        if stats['recent_avg'] > 0:
            print(f"  Recent avg loss: {stats['recent_avg']:.4f}")
            print(f"  Latest loss: {stats['latest']:.4f}")
        if stats['best_val'] < float('inf'):
            print(f"  Best validation: {stats['best_val']:.4f}")

    def show_auto_summary(self, processed: int, skipped: int):
        print(f"\n{'='*70}\nAUTO MODE SUMMARY\n{'='*70}")
        print(f" Processed: {processed}\n Skipped: {skipped}\n{'='*70}\n")

    def get_user_choices(self, num_options: int) -> Optional[List[int]]:
        while True:
            choice = input(f"\nSelect correct (1-{num_options}, 's'=skip, 'q'=quit): ").strip().lower()
            if choice == 'q':
                return None
            if choice == 's':
                return [-1]

            normalized = choice.replace(',', ' ').replace('-', ' ').replace('/', ' ')
            parts = normalized.split()
            if not parts:
                print("No input provided.")
                continue
            try:
                choices = [int(p) for p in parts]
            except ValueError:
                print("Invalid number in input.")
                continue
            invalid = [c for c in choices if c < 1 or c > num_options]
            if invalid:
                print(f"Invalid choices: {invalid}. Use 1-{num_options}")
                continue
            return list(dict.fromkeys([c - 1 for c in choices]))

    def confirm_save(self) -> bool:
        return input("Save model before quitting? (y/n): ").strip().lower() == 'y'

    def _format_aligned_sentence(self, words: List[str], parts: List[str]) -> str:
        top_line = ""
        bot_line = ""
        for w, p in zip(words, parts):
            width = max(len(w), len(p))
            top_line += w.ljust(width) + "   "
            bot_line += p.ljust(width) + "   "
        return f"{top_line.strip()}\n    {bot_line.strip()}"

    def handle_word(self, raw_word: str) -> Optional[bool]:
        word = sanitize_word(raw_word)
        if not word:
            return False

        analysis = self.engine.analyze_word(word)
        if analysis is None:
            self.show_message(f"\n  No decompositions found for '{word}'")
            return False

        if len(analysis['decomps']) == 1:
            self.show_message(f"\n Only one decomposition for '{word}' - skipping")
            return False

        if analysis['vms'] and analysis['vms'][0].get('score') is not None:
            self.show_message("\n ML Model predictions shown")

        self.show_decompositions(word, analysis['vms'])

        choices = self.get_user_choices(len(analysis['decomps']))
        if choices is None:
            return None
        if choices == [-1]:
            return False

        loss, deleted_msgs = self.engine.commit_word(analysis, choices)

        for msg in deleted_msgs:
            print(f"  {msg}")

        self.show_message(f"\n Training complete. Loss: {loss:.4f}")
        self.show_message(f"Total examples: {self.engine.training_count}")
        return True

    def handle_sentence(self, raw_sentence: str, analyses: Optional[List[Dict]] = None) -> Optional[bool]:
        if analyses is None:
            words = sanitize_sentence(raw_sentence)
            if not words:
                self.show_message(f"\n Could not parse: {raw_sentence}")
                return False
            analyses = self.engine.analyze_sentence(words)
            if analyses is None:
                self.show_message(f"\n Could not parse all words in: {raw_sentence}")
                return False
            self.clear_screen()
            self.show_message(f"Sentence: {raw_sentence}\n")

        words = [a['word'] for a in analyses]

        while True:
            target_str = self.get_input("Enter correct decomposition string (or prefix) ['q' to cancel]: ").strip()
            if target_str.lower() == 'q':
                return False

            self.show_message("\nSearching for legal combinations...")
            all_sentences, furthest_text, furthest_idx = self.engine.evaluate_sentence_target(analyses, target_str)

            if not all_sentences:
                self.show_message(f"\n[!] Invalid Decomposition Provided: '{target_str}'")
                if furthest_idx < len(words):
                    failed_word = words[furthest_idx]
                    prefix_msg = f"'{furthest_text}'" if furthest_text else "(None)"
                    self.show_message(f" -> Matched successfully up to: {prefix_msg}")
                    self.show_message(f" -> Mismatch detected at word {furthest_idx + 1}: '{failed_word}'")
                    self.show_message(f" -> Valid morphological variations for '{failed_word}':")
                    unique_options = list(dict.fromkeys(analyses[furthest_idx]['typing_strings']))
                    for opt in unique_options:
                        self.show_message(f"      - {opt}")
                self.show_message("\nPlease try again.\n")
                continue
            break

        display_list = all_sentences

        if len(display_list) == 1:
            self.show_message("\nAuto-selected the only legal match:")
            c = display_list[0]
            aligned_display = self._format_aligned_sentence(words, c['parts'])
            vms = [analyses[w_idx]['vms'][c['combo_indices'][w_idx]] for w_idx in range(len(words))]
            self.show_sentence_prediction(0, c['score'], words, vms, aligned_display)
            correct_combo = c['combo_indices']
        else:
            varying_indices = []
            for w_idx in range(len(words)):
                unique_indices = set(c['combo_indices'][w_idx] for c in display_list)
                if len(unique_indices) > 1:
                    varying_indices.append(w_idx)
            if not varying_indices:
                varying_indices = list(range(len(words)))

            self.show_message("\nPredictions (showing only ambiguous words):")
            for i, c in enumerate(display_list):
                filt_words = [words[idx] for idx in varying_indices]
                filt_parts = [c['parts'][idx] for idx in varying_indices]
                aligned_display = self._format_aligned_sentence(filt_words, filt_parts)
                filt_vms = [analyses[w_idx]['vms'][c['combo_indices'][w_idx]] for w_idx in varying_indices]
                self.show_sentence_prediction(i, c['score'], filt_words, filt_vms, aligned_display)

            while True:
                choice = self.get_input(f"\nSelect by number (0-{len(display_list)-1}) or 'q' to cancel: ").strip()
                if choice.lower() == 'q':
                    return False
                if choice.isdigit() and int(choice) < len(display_list):
                    correct_combo = display_list[int(choice)]['combo_indices']
                    break
                self.show_message("Invalid selection.")

        self.show_message("\nTraining on full sentence context...")
        loss = self.engine.commit_sentence_training(raw_sentence, words, analyses, correct_combo)
        self.show_message(f"Sentence loss: {loss:.4f}")
        return True

    def handle_eval_sentence(self, raw_sentence: str) -> Optional[bool]:
        words = sanitize_sentence(raw_sentence)
        if not words:
            self.show_message("\n Could not parse sentence.")
            return False

        analyses = self.engine.analyze_sentence(words)
        if analyses is None:
            self.show_message("\n Could not parse sentence.")
            return False

        self.clear_screen()
        self.show_message(f"Sentence: {raw_sentence}\n")
        self.show_message("Evaluating top predictions (Beam Search)...")

        top_predictions = get_top_sentence_predictions(analyses, self.engine.trainer, top_k=10)

        for i, c in enumerate(top_predictions):
            aligned_display = self._format_aligned_sentence(words, c['parts'])
            vms = [analyses[w_idx]['vms'][c['combo_indices'][w_idx]] for w_idx in range(len(words))]
            self.show_sentence_prediction(i, c['score'], words, vms, aligned_display)

        while True:
            choice = self.get_input(f"\nSelect correct decomposition by number (0-{len(top_predictions)-1}), 'm' to manually enter target string, or 'q' to cancel: ").strip().lower()

            if choice == 'q':
                return False

            if choice == 'm':
                return self.handle_sentence(raw_sentence, analyses=analyses)

            if choice.isdigit() and int(choice) < len(top_predictions):
                correct_combo = top_predictions[int(choice)]['combo_indices']
                break

            self.show_message("Invalid selection.")

        self.show_message("\nTraining on full sentence context...")
        loss = self.engine.commit_sentence_training(raw_sentence, words, analyses, correct_combo)
        self.show_message(f"Sentence loss: {loss:.4f}")
        return True

    def run_auto_mode(self):
        self.show_message("   Words deleted if root exists in dictionary\n   Press 'q' to exit\n")
        processed, skipped = 0, 0
        while True:
            word = self.engine.data_manager.random_word()
            if not word:
                self.show_message("\n No more words!")
                break
            processed += 1
            result = self.handle_word(word)
            if result is None:
                processed -= 1
                self.show_message("\n Exiting auto mode...")
                break
            if result is False:
                skipped += 1
        self.show_auto_summary(processed, skipped)

    def run(self):
        self.welcome()
        while True:
            try:
                raw = self.get_input("\n Enter word or command: ").strip()
                cmd = raw.lower()
                if not cmd:
                    continue

                if cmd == 'quit':
                    if self.engine.training_count > 0 and self.confirm_save():
                        self.engine.save()
                    break
                elif cmd == 'save':
                    self.engine.save()
                    self.show_message("Model saved.")
                elif cmd == 'stats':
                    stats = self.engine.get_stats()
                    self.show_stats(stats)
                elif cmd == 'auto':
                    self.run_auto_mode()
                elif cmd == 'sample':
                    filename = self.get_input("Enter filename (default: sample.txt): ").strip()
                    if not filename:
                        filename = "sample.txt"
                    self.show_message("Decomposing and ranking words...")
                    success = self.engine.sample_text(filename)
                    if success:
                        self.show_message("Sample text processing complete.")
                    else:
                        self.show_message("Failed to process sample text.")
                elif cmd == 'sample sentence':
                    self.show_message("Processing sentences from sample/sample_sentence.txt...")
                    success = self.engine.sample_sentences()
                    if success:
                        self.show_message("Sentence sampling complete.")
                    else:
                        self.show_message("Failed to process sentence samples.")
                elif cmd == 'relearn':
                    trained, skipped = self.engine.relearn_all()
                    self.show_message(f"\n  Trained on {trained} examples, skipped {skipped}.")
                elif cmd == '10fkv':
                    self.engine.run_kfold_cv(k=10)
                elif cmd.startswith('eval sentence '):
                    result = self.handle_eval_sentence(raw[14:].strip())
                    if result is None and self.confirm_save():
                        self.engine.save()
                        break
                elif cmd.startswith('eval '):
                    word = sanitize_word(raw[5:])
                    vm = self.engine.evaluate_word(word)
                    if vm:
                        self.show_message("\n ML Model's top prediction:")
                        self.show_decompositions(word, [vm])
                    else:
                        self.show_message("\n  No decompositions found")
                elif cmd.startswith('sentence '):
                    result = self.handle_sentence(raw[9:].strip())
                    if result is None and self.confirm_save():
                        self.engine.save()
                        break
                else:
                    result = self.handle_word(raw)
                    if result is None and self.confirm_save():
                        self.engine.save()
                        break

            except KeyboardInterrupt:
                if self.confirm_save():
                    self.engine.save()
                break
            except Exception as e:
                print(f"\n Error: {e}")
