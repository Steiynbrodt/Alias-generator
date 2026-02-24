from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


# -----------------------------
# Embedding backends (no LLM)
# -----------------------------
class EmbedderBase:
    def most_similar_from_vocab(
        self,
        theme_words: Sequence[str],
        vocab: Sequence[str],
        top_k: int,
    ) -> Optional[List[str]]:
        raise NotImplementedError


class LocalKeyedVectorsEmbedder(EmbedderBase):
    """
    Loads local word vectors from gensim KeyedVectors file (e.g. vectors.kv).
    Works best if your vocab/theme words exist in the trained vectors.
    """
    def __init__(self, vectors_path: str):
        from gensim.models import KeyedVectors
        self.kv = KeyedVectors.load(vectors_path)

    def _vec(self, word: str) -> Optional[np.ndarray]:
        if word in self.kv:
            v = self.kv[word]
            n = np.linalg.norm(v)
            return (v / n) if n > 0 else v
        return None

    def most_similar_from_vocab(self, theme_words: Sequence[str], vocab: Sequence[str], top_k: int) -> Optional[List[str]]:
        vecs = []
        for w in theme_words:
            v = self._vec(w)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return None

        tv = np.mean(vecs, axis=0)
        n = np.linalg.norm(tv)
        tv = (tv / n) if n > 0 else tv

        scored: List[Tuple[str, float]] = []
        for w in dict.fromkeys(vocab):  # unique preserve order
            v = self._vec(w)
            if v is None:
                continue
            scored.append((w, float(np.dot(v, tv))))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in scored[:top_k]]


class GensimDownloaderFastTextEmbedder(EmbedderBase):
    """
    Uses gensim.downloader to load a pretrained model by name.
    Example: fasttext-wiki-news-subwords-300 (EN).
    Note: downloads can be large.
    """
    def __init__(self, model_name: str):
        import gensim.downloader as api
        self.model = api.load(model_name)

    def _vec(self, word: str) -> np.ndarray:
        # fastText supports OOV through subwords (for this particular model)
        v = self.model.get_vector(word)
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else v

    def most_similar_from_vocab(self, theme_words: Sequence[str], vocab: Sequence[str], top_k: int) -> Optional[List[str]]:
        if not theme_words:
            return None

        vecs = [self._vec(w) for w in theme_words if w]
        if not vecs:
            return None

        tv = np.mean(vecs, axis=0)
        n = np.linalg.norm(tv)
        tv = (tv / n) if n > 0 else tv

        scored: List[Tuple[str, float]] = []
        for w in dict.fromkeys(vocab):
            v = self._vec(w)
            scored.append((w, float(np.dot(v, tv))))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in scored[:top_k]]


def build_embedder(cfg: "AliasConfig") -> Optional[EmbedderBase]:
    """
    cfg.embedding_model can be:
      - "none" / ""  -> no embeddings
      - "local"      -> load cfg.vectors_path
      - any other    -> treated as gensim.downloader model name
    """
    if not cfg.use_embeddings:
        return None

    model = (cfg.embedding_model or "").strip().lower()
    if model in ("", "none", "off", "false", "0"):
        return None

    if model == "local":
        if not cfg.vectors_path:
            return None
        try:
            return LocalKeyedVectorsEmbedder(cfg.vectors_path)
        except Exception:
            return None

    # downloader model
    try:
        return GensimDownloaderFastTextEmbedder(cfg.embedding_model)
    except Exception:
        return None


# -----------------------------
# Name composition
# -----------------------------
VOWELS = "aeiouäöüAEIOUÄÖÜ"

def mash(a: str, b: str) -> str:
    a, b = a.strip(), b.strip()
    if not a or not b:
        return (a + b).lower()

    a_l, b_l = a.lower(), b.lower()

    def cut(word: str) -> int:
        for i in range(len(word) - 2, 1, -1):
            if word[i] in VOWELS:
                return i
        return max(2, len(word) // 2)

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
               .replace("a", "4")
               .replace("t", "7"))
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

    # embeddings
    use_embeddings: bool = True
    embedding_model: str = "local"  # "local" OR gensim downloader model name OR "none"
    vectors_path: str = "vectors.kv"
    top_k: int = 40
    mix_with_random_prob: float = 0.35


def load_config(path: str) -> AliasConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict = yaml.safe_load(f)

    # Provide safe defaults if fields missing
    raw.setdefault("vocab", [])
    raw.setdefault("prefixes", [])
    raw.setdefault("suffixes", [""])
    raw.setdefault("separators", [""])
    raw.setdefault("germanish_suffixes", [""])

    return AliasConfig(**raw)


class AliasGenerator:
    def __init__(self, cfg: AliasConfig):
        self.cfg = cfg
        self.embedder = build_embedder(cfg)

    def _theme_candidates(self, theme_words: Sequence[str]) -> Optional[List[str]]:
        if not self.embedder:
            return None
        return self.embedder.most_similar_from_vocab(theme_words, self.cfg.vocab, self.cfg.top_k)

    def generate(self, n: int, themes: Sequence[str], seed: Optional[int] = None) -> List[str]:
        if seed is not None:
            random.seed(seed)

        themes = [t.strip() for t in themes if t and t.strip()]
        candidates = self._theme_candidates(themes) if themes else None

        out: List[str] = []
        for _ in range(n):
            sep = random.choice(self.cfg.separators) if self.cfg.separators else ""

            def pick_word() -> str:
                if candidates and random.random() > self.cfg.mix_with_random_prob:
                    return random.choice(candidates)
                return random.choice(self.cfg.vocab) if self.cfg.vocab else ""

            a = pick_word()
            b = pick_word()

            if random.random() < self.cfg.mash_prob:
                base = mash(a, b)
            else:
                base = f"{a}{sep}{b}".lower()

            if self.cfg.prefixes and random.random() < self.cfg.prefix_prob:
                base = f"{random.choice(self.cfg.prefixes)}{sep}{base}"
            if self.cfg.germanish_suffixes and random.random() < self.cfg.germanish_prob:
                base = f"{base}{random.choice(self.cfg.germanish_suffixes)}"
            if self.cfg.suffixes and random.random() < self.cfg.suffix_prob:
                base = f"{base}{random.choice(self.cfg.suffixes)}"

            base = stylize(base, self.cfg.leet_prob, self.cfg.camel_prob)
            out.append(clean(base))

        return out


# -----------------------------
# Tiny GUI (Tkinter)
# -----------------------------
def run_gui(cfg_path: str):
    import tkinter as tk
    from tkinter import ttk, messagebox

    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        raise SystemExit(f"Failed to load config: {e}")

    gen = AliasGenerator(cfg)

    root = tk.Tk()
    root.title("Alias Generator (themes → aliases)")

    # Layout
    frm = ttk.Frame(root, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)

    # Config info
    emb_status = "OFF"
    if cfg.use_embeddings:
        if (cfg.embedding_model or "").strip().lower() == "local":
            emb_status = f"LOCAL ({cfg.vectors_path})" if gen.embedder else f"LOCAL (failed: {cfg.vectors_path})"
        elif (cfg.embedding_model or "").strip().lower() in ("none", "off", ""):
            emb_status = "OFF"
        else:
            emb_status = f"DOWNLOAD ({cfg.embedding_model})" if gen.embedder else f"DOWNLOAD (failed: {cfg.embedding_model})"

    ttk.Label(frm, text=f"Config: {cfg_path}").grid(row=0, column=0, columnspan=2, sticky="w")
    ttk.Label(frm, text=f"Embeddings: {emb_status}").grid(row=1, column=0, columnspan=2, sticky="w")

    # Inputs
    ttk.Label(frm, text="Themes (space-separated):").grid(row=2, column=0, sticky="w", pady=(10, 2))
    themes_var = tk.StringVar()
    themes_entry = ttk.Entry(frm, textvariable=themes_var)
    themes_entry.grid(row=2, column=1, sticky="ew", pady=(10, 2))

    ttk.Label(frm, text="Count:").grid(row=3, column=0, sticky="w", pady=(6, 2))
    count_var = tk.StringVar(value="25")
    count_entry = ttk.Entry(frm, textvariable=count_var, width=8)
    count_entry.grid(row=3, column=1, sticky="w", pady=(6, 2))

    ttk.Label(frm, text="Seed (optional):").grid(row=4, column=0, sticky="w", pady=(6, 2))
    seed_var = tk.StringVar(value="")
    seed_entry = ttk.Entry(frm, textvariable=seed_var, width=12)
    seed_entry.grid(row=4, column=1, sticky="w", pady=(6, 2))

    # Output
    ttk.Label(frm, text="Aliases:").grid(row=5, column=0, sticky="w", pady=(10, 2))
    out_txt = tk.Text(frm, height=14, width=50)
    out_txt.grid(row=5, column=1, sticky="nsew", pady=(10, 2))
    frm.rowconfigure(5, weight=1)

    # Buttons
    btns = ttk.Frame(frm)
    btns.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    btns.columnconfigure(0, weight=1)

    def do_generate():
        themes = [t for t in themes_var.get().split() if t.strip()]
        if not themes:
            messagebox.showinfo("Missing themes", "Please enter at least one theme word.")
            return

        try:
            n = int(count_var.get().strip())
            if n < 1 or n > 5000:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid count", "Count must be an integer between 1 and 5000.")
            return

        seed_s = seed_var.get().strip()
        seed = None
        if seed_s:
            try:
                seed = int(seed_s)
            except Exception:
                messagebox.showerror("Invalid seed", "Seed must be an integer (or empty).")
                return

        aliases = gen.generate(n, themes, seed=seed)
        out_txt.delete("1.0", tk.END)
        out_txt.insert(tk.END, "\n".join(aliases))

    def do_copy():
        text = out_txt.get("1.0", tk.END).strip()
        if not text:
            return
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        messagebox.showinfo("Copied", "Aliases copied to clipboard.")

    def do_clear():
        out_txt.delete("1.0", tk.END)

    ttk.Button(btns, text="Generate", command=do_generate).grid(row=0, column=0, sticky="w")
    ttk.Button(btns, text="Copy", command=do_copy).grid(row=0, column=1, sticky="w", padx=(8, 0))
    ttk.Button(btns, text="Clear", command=do_clear).grid(row=0, column=2, sticky="w", padx=(8, 0))

    themes_entry.focus()
    root.minsize(650, 420)
    root.mainloop()


# -----------------------------
# CLI entry
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Theme-based alias generator (local vectors or gensim download) + tiny GUI.")
    p.add_argument("--config", default="aliasgen.yaml", help="Path to YAML config")
    p.add_argument("--themes", nargs="*", help="Theme words (space-separated); omit when using --gui")
    p.add_argument("-n", type=int, default=25, help="Number of aliases")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--gui", action="store_true", help="Launch Tkinter GUI")

    args = p.parse_args()

    if args.gui:
        run_gui(args.config)
        return

    cfg = load_config(args.config)
    gen = AliasGenerator(cfg)

    if not args.themes:
        raise SystemExit("Provide --themes ... or run with --gui")

    for a in gen.generate(args.n, args.themes, seed=args.seed):
        print(a)


if __name__ == "__main__":
    main()
