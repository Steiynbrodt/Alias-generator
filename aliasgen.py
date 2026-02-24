from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict

import numpy as np
import yaml

# -----------------------------
# Embedding backend (fastText via gensim downloader)
# -----------------------------
class FastTextEmbedder:
    def __init__(self, model_name: str = "fasttext-wiki-news-subwords-300"):
        import gensim.downloader as api
        self.model = api.load(model_name)

    def vec(self, text: str) -> np.ndarray:
        # fastText liefert auch für OOV-Wörter Vektoren (Subwords)
        v = self.model.get_vector(text)
        # normalisieren für Cosine-Similarity
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def most_similar_from_vocab(self, theme_words: Sequence[str], vocab: Sequence[str], top_k: int) -> List[str]:
        # Theme-Vektor als Mittelwert
        tv = np.mean([self.vec(w) for w in theme_words if w], axis=0)
        tvn = np.linalg.norm(tv)
        tv = tv / tvn if tvn > 0 else tv

        scored = []
        for w in dict.fromkeys(vocab):  # unique, preserve order
            vw = self.vec(w)
            score = float(np.dot(vw, tv))  # cosine, weil normalisiert
            scored.append((w, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in scored[:top_k]]


# -----------------------------
# Name composition
# -----------------------------
VOWELS = "aeiouäöüAEIOUÄÖÜ"

def mash(a: str, b: str) -> str:
    """Kofferwort: schneidet an Vokalpositionen und verbindet."""
    a, b = a.strip(), b.strip()
    if not a or not b:
        return (a + b).lower()

    a_l, b_l = a.lower(), b.lower()

    def cut(word: str) -> int:
        for i in range(len(word) - 2, 1, -1):
            if word[i] in VOWELS:
                return i
        return max(2, len(word)//2)

    ca = cut(a_l)
    cb = cut(b_l)

    out = a_l[:ca] + b_l[cb:]
    out = re.sub(r"(.)\1\1+", r"\1\1", out)  # aaa -> aa
    return out

def stylize(s: str, leet_prob: float, camel_prob: float) -> str:
    if random.random() < camel_prob:
        parts = re.split(r"[_\-.]", s)
        s = "".join(p[:1].upper() + p[1:] for p in parts if p)
    if random.random() < leet_prob:
        s = (s.replace("o", "0")
              .replace("e", "3")
              .replace("i", "1")
              .replace("a", "4"))
    return s

def clean(s: str) -> str:
    return re.sub(r"[^0-9A-Za-zäöüÄÖÜ_\-\.]", "", s)


# -----------------------------
# Config + Generator
# -----------------------------
@dataclass
class AliasConfig:
    vocab: List[str]
    prefixes: List[str]
    suffixes: List[str]
    separators: List[str]
    germanish_suffixes: List[str]
    mash_prob: float = 0.65
    prefix_prob: float = 0.35
    suffix_prob: float = 0.35
    germanish_prob: float = 0.25
    leet_prob: float = 0.10
    camel_prob: float = 0.10
    use_embeddings: bool = True
    embedding_model: str = "fasttext-wiki-news-subwords-300"
    top_k: int = 40
    mix_with_random_prob: float = 0.35


class AliasGenerator:
    def __init__(self, cfg: AliasConfig):
        self.cfg = cfg
        self.embedder: Optional[FastTextEmbedder] = None
        if cfg.use_embeddings:
            try:
                self.embedder = FastTextEmbedder(cfg.embedding_model)
            except Exception:
                # Fallback: läuft ohne Embeddings weiter
                self.embedder = None

    def _theme_candidates(self, theme_words: Sequence[str]) -> Optional[List[str]]:
        if not self.embedder:
            return None
        # Kandidaten aus dem Vokabular semantisch ranken
        return self.embedder.most_similar_from_vocab(theme_words, self.cfg.vocab, self.cfg.top_k)

    def generate(self, n: int, themes: Sequence[str], seed: Optional[int] = None) -> List[str]:
        if seed is not None:
            random.seed(seed)

        themes = [t.strip() for t in themes if t and t.strip()]
        candidates = self._theme_candidates(themes) if themes else None

        out = []
        for _ in range(n):
            sep = random.choice(self.cfg.separators) if self.cfg.separators else ""

            # Wortauswahl: entweder embedding-guided oder random aus vocab
            def pick_word() -> str:
                if candidates and random.random() > self.cfg.mix_with_random_prob:
                    return random.choice(candidates)
                return random.choice(self.cfg.vocab)

            a = pick_word()
            b = pick_word()

            # Komposition
            if random.random() < self.cfg.mash_prob:
                base = mash(a, b)
            else:
                base = f"{a}{sep}{b}".lower()

            # Prefix/Suffix
            if self.cfg.prefixes and random.random() < self.cfg.prefix_prob:
                base = f"{random.choice(self.cfg.prefixes)}{sep}{base}"
            if self.cfg.germanish_suffixes and random.random() < self.cfg.germanish_prob:
                base = f"{base}{random.choice(self.cfg.germanish_suffixes)}"
            if self.cfg.suffixes and random.random() < self.cfg.suffix_prob:
                base = f"{base}{random.choice(self.cfg.suffixes)}"

            base = stylize(base, self.cfg.leet_prob, self.cfg.camel_prob)
            out.append(clean(base))

        return out


def load_config(path: str) -> AliasConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict = yaml.safe_load(f)
    return AliasConfig(**raw)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Theme-based alias generator (no LLM).")
    p.add_argument("--config", default="aliasgen.yaml")
    p.add_argument("--themes", nargs="+", required=True, help="Theme words, e.g. hacking baking bäckerei")
    p.add_argument("-n", type=int, default=25)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    gen = AliasGenerator(cfg)

    for a in gen.generate(args.n, args.themes, seed=args.seed):
        print(a)
