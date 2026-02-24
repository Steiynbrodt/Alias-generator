#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_vectors.py
- Scans text files (or directories) to build a corpus
- Optionally extracts a top-N vocabulary (and writes/updates YAML)
- Trains FastText word vectors locally (no internet needed)
- Saves vectors as gensim KeyedVectors (.kv) usable by aliasgen.py
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import yaml
from gensim.models import FastText
from gensim.models.keyedvectors import KeyedVectors


# -----------------------------
# Helpers: file scanning + tokenization
# -----------------------------
TEXT_EXTS = {
    ".txt", ".md", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".rs", ".go",
    ".html", ".css", ".xml",
}


def iter_text_files(inputs: Sequence[str], exts: Optional[set[str]] = None) -> Iterable[Path]:
    exts = exts or TEXT_EXTS
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            yield p
        elif p.is_dir():
            for fp in p.rglob("*"):
                if fp.is_file() and (fp.suffix.lower() in exts):
                    yield fp


_WORD_RE = re.compile(r"[0-9A-Za-zÄÖÜäöüß]+", re.UNICODE)

def tokenize_line(line: str, lowercase: bool = True) -> List[str]:
    toks = _WORD_RE.findall(line)
    if lowercase:
        toks = [t.lower() for t in toks]
    return toks


def read_corpus(files: Iterable[Path], lowercase: bool = True, max_lines: Optional[int] = None) -> Iterable[List[str]]:
    line_count = 0
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if max_lines is not None and line_count >= max_lines:
                        return
                    line_count += 1
                    toks = tokenize_line(line, lowercase=lowercase)
                    if toks:
                        yield toks
        except Exception:
            # ignore unreadable files silently
            continue


# -----------------------------
# Vocab extraction
# -----------------------------
@dataclass
class VocabOptions:
    top_n: int = 2000
    min_len: int = 3
    min_count: int = 2
    drop_numeric: bool = True


def build_vocab(corpus: Iterable[List[str]], opts: VocabOptions) -> List[str]:
    cnt = Counter()
    for sent in corpus:
        cnt.update(sent)

    vocab = []
    for w, c in cnt.most_common():
        if len(vocab) >= opts.top_n:
            break
        if c < opts.min_count:
            break  # because most_common() is descending
        if len(w) < opts.min_len:
            continue
        if opts.drop_numeric and w.isdigit():
            continue
        vocab.append(w)

    return vocab


def write_vocab_yaml(vocab: List[str], out_path: str):
    data = {"vocab": vocab}
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def update_alias_yaml(alias_yaml_path: str, vocab: List[str], backup: bool = True):
    p = Path(alias_yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {alias_yaml_path}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("alias yaml must be a mapping (dict)")

    if backup:
        p.with_suffix(p.suffix + ".bak").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    raw["vocab"] = vocab
    p.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


# -----------------------------
# Training
# -----------------------------
def train_fasttext(
    corpus: List[List[str]],
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    workers: int,
    sg: int,
) -> FastText:
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,  # 1=skipgram, 0=cbow
    )
    model.build_vocab(corpus)
    model.train(corpus, total_examples=len(corpus), epochs=epochs)
    return model


def save_vectors_kv(model: FastText, out_path: str):
    # Save only keyed vectors (smaller, exactly what aliasgen.py needs)
    model.wv.save(out_path)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Train local FastText word vectors + extract vocab (no internet)."
    )

    ap.add_argument("inputs", nargs="+", help="Text file(s) and/or directory(ies) to scan.")
    ap.add_argument("--max-lines", type=int, default=None, help="Stop after reading N lines (for quick tests).")
    ap.add_argument("--lowercase", action="store_true", default=True, help="Lowercase tokens (default on).")
    ap.add_argument("--no-lowercase", dest="lowercase", action="store_false", help="Do not lowercase tokens.")

    # Vocab extraction
    ap.add_argument("--extract-vocab", action="store_true", help="Extract top-N vocab from inputs.")
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--min-count-vocab", type=int, default=2)
    ap.add_argument("--keep-numeric", action="store_true", help="Keep numeric-only tokens in vocab.")
    ap.add_argument("--vocab-out", default="vocab.yaml", help="Where to write extracted vocab YAML (default: vocab.yaml).")
    ap.add_argument("--update-alias-yaml", default=None, help="Path to aliasgen YAML to update its 'vocab' field.")
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak when updating alias yaml.")

    # Training options
    ap.add_argument("--train", action="store_true", help="Train vectors from inputs.")
    ap.add_argument("--vectors-out", default="vectors.kv", help="Output vectors file (KeyedVectors), default: vectors.kv")
    ap.add_argument("--dim", type=int, default=100, help="Vector size, default: 100")
    ap.add_argument("--window", type=int, default=5, help="Context window, default: 5")
    ap.add_argument("--min-count-train", type=int, default=2, help="Min token count for training, default: 2")
    ap.add_argument("--epochs", type=int, default=20, help="Training epochs, default: 20")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2), help="Workers, default: CPU count")
    ap.add_argument("--sg", type=int, choices=[0, 1], default=1, help="0=CBOW, 1=Skip-gram (default: 1)")

    return ap.parse_args()


def main():
    args = parse_args()

    files = list(iter_text_files(args.inputs))
    if not files:
        raise SystemExit("No input files found (check paths/extensions).")

    # Read corpus once into memory (simpler + allows vocab + training without re-scan)
    # For very large corpora you could stream; for alias usage, this is usually fine.
    corpus_iter = read_corpus(files, lowercase=args.lowercase, max_lines=args.max_lines)
    corpus = list(corpus_iter)

    if not corpus:
        raise SystemExit("Corpus is empty (no tokens found).")

    print(f"Files: {len(files)} | Sentences: {len(corpus)}")

    extracted_vocab: Optional[List[str]] = None

    if args.extract_vocab:
        vopts = VocabOptions(
            top_n=args.top_n,
            min_len=args.min_len,
            min_count=args.min_count_vocab,
            drop_numeric=not args.keep_numeric,
        )
        extracted_vocab = build_vocab(corpus, vopts)
        print(f"Extracted vocab: {len(extracted_vocab)} tokens")

        write_vocab_yaml(extracted_vocab, args.vocab_out)
        print(f"Wrote vocab YAML: {args.vocab_out}")

        if args.update_alias_yaml:
            update_alias_yaml(args.update_alias_yaml, extracted_vocab, backup=not args.no_backup)
            print(f"Updated alias YAML vocab: {args.update_alias_yaml}")

    if args.train:
        model = train_fasttext(
            corpus=corpus,
            vector_size=args.dim,
            window=args.window,
            min_count=args.min_count_train,
            epochs=args.epochs,
            workers=args.workers,
            sg=args.sg,
        )
        save_vectors_kv(model, args.vectors_out)
        print(f"Saved vectors: {args.vectors_out}")

        # sanity check
        kv: KeyedVectors = model.wv
        print(f"Vector dim: {kv.vector_size} | Vocab size: {len(kv.key_to_index)}")

        # quick sample similarity if possible
        if extracted_vocab and len(extracted_vocab) >= 2:
            w = extracted_vocab[0]
            if w in kv:
                sims = kv.most_similar(w, topn=min(5, len(kv)))
                print(f"Sample most_similar('{w}'): {sims}")

    if not args.extract_vocab and not args.train:
        print("Nothing to do: pass --extract-vocab and/or --train")


if __name__ == "__main__":
    main()