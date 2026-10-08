CAUSATIVE_IR_ROOTS = {
    "art", "aş", "bat", "bit", "doğ", "doy", "duy", "düş", "geç", "göç",
    "iç", "kaç", "piş", "şaş", "şiş", "taş", "uç", "vazgeç", "yat", "yit",
}

CAUSATIVE_AR_ROOTS = {
    "çık", "çök", "git", "kız", "kop", "uy", "yet",
}

AORIST_IR_MONOSYLLABLES = {
    "al", "bil", "bul", "dur", "gel", "gör", "kal", "ol", "öl", "san",
    "var", "ver", "vur",
}

VOWEL_DROPPING_NOUNS = {
    "aciz", "ağız", "ahit", "akıl", "akis", "akit", "alın", "asıl", "asır",
    "avuç", "azil", "azim", "bağır", "bahis", "beniz", "beyin", "beyit",
    "boyun", "böğür", "burun", "cebir", "cehil", "cisim", "cürüm", "çığır",
    "defin", "devir", "ehil", "emir", "fasıl", "fecir", "fesih", "fetih",
    "fikir", "fuhuş", "geniz", "göğüs", "gönül", "hacim", "hapis", "hasım",
    "hatır", "hayır", "hazım", "hışım", "hüküm", "hüsün", "hüzün", "ilim",
    "isim", "izin", "kabir", "kadir", "kahır", "karın", "kasır", "kasıt",
    "kavim", "kayıp", "kayıt", "kesir", "keşif", "keyif", "kibir", "kısım",
    "koyun", "kutup", "küfür", "lafız", "lütuf", "metin", "meyil", "misil",
    "mühür", "nabız", "nakil", "nakış", "nakit", "nefis", "nehir", "nesil",
    "nesir", "nutuk", "oğul", "omuz", "ömür", "özür", "rahim", "resim",
    "ritim", "rükün", "sabır", "satıh", "seyir", "sihir", "şahıs", "şehir",
    "şekil", "şükür", "tavır", "ufuk", "uğur", "vahiy", "vakıf", "vakit",
    "vasıf", "vehim", "zehir", "zihin", "zikir", "zulüm", "zülüf",
}

VOWEL_DROPPING_VERBS = {
    "ayır", "bağır", "buyur", "çağır", "çevir", "devir", "eğir", "evir", "kavuş",
    "kavur", "kayır", "kıvır", "savur", "sıyır", "süpür", "yoğur",
}

VOWEL_DROPPING_ROOTS = VOWEL_DROPPING_NOUNS | VOWEL_DROPPING_VERBS

FINAL_VOWEL_ELIDING_ROOTS = {
    "bura", "nere", "ora", "şura", "yumurta",
}
