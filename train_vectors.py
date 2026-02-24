import argparse
import os
from gensim.models import FastText
from gensim.utils import simple_preprocess

def read_corpus(paths):
    for path in paths:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    tokens = simple_preprocess(line)
                    if tokens:
                        yield tokens
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fn in files:
                    fp = os.path.join(root, fn)
                    try:
                        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                tokens = simple_preprocess(line)
                                if tokens:
                                    yield tokens
                    except:
                        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Textdateien oder Ordner")
    ap.add_argument("--out", default="vectors.kv")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    corpus = list(read_corpus(args.inputs))

    print(f"Loaded {len(corpus)} sentences")

    model = FastText(
        vector_size=args.dim,
        window=5,
        min_count=1,
        workers=4
    )

    model.build_vocab(corpus)
    model.train(corpus, total_examples=len(corpus), epochs=args.epochs)

    model.wv.save(args.out)
    print(f"Saved vectors to {args.out}")

if __name__ == "__main__":
    main()
