import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from src.utils import load_datasets, PretrainedEmbeddings
from src.tokenization import tokenize_df, build_reduced_embeddings, to_ids
from src.data import PairDataset
from src.training import cross_validate

HERE = Path(__file__).parent

EMBEDDINGS = {  
    "glove": ("glove.2024.wikigiga.50d.txt", 50, False),
    "fasttext": ("wiki-news-300d-1M.vec.zip", 300, True),
}

BASE = dict(
    pooling="mean", freeze=True, combine="full",
    hidden_size=128, num_layers=1, dropout=0.0,
    lr=1e-3, weight_decay=0.0, batch_size=64, epochs=10, seed=64,
)

# each variant changes one thing compared to BASE
VARIANTS = {
    "base": {},
    "pool_sum": {"pooling": "sum"},
    "pool_max": {"pooling": "max"},
    "combine_concat": {"combine": "concat"},
    "finetune": {"freeze": False},
    "hidden256_2layers": {"hidden_size": 256, "num_layers": 2},
    "dropout0.3": {"dropout": 0.3},
    "lr3e-3": {"lr": 3e-3},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", choices=list(EMBEDDINGS), default="glove")
    parser.add_argument("--configs", nargs="*", default=list(VARIANTS), help="which variants to run")
    parser.add_argument("--quick", action="store_true", help="2,000 pairs, 2 folds, 2 epochs, to test that everything runs")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # same data and folds as Project 1
    train_df, dev_df, _ = load_datasets()
    df = pd.concat([train_df, dev_df]).reset_index(drop=True)
    if args.quick:
        df = df.sample(2000, random_state=0).reset_index(drop=True)
    n_splits = 2 if args.quick else 5
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=64).split(df, df["label"]))

    file_name, dims, skip_first = EMBEDDINGS[args.embedding]
    print(f"Loading {file_name} ...")
    emb = PretrainedEmbeddings(str(HERE / file_name), dims, skip_first_line=skip_first)

    print("Tokenizing ...")
    tokens1, tokens2 = tokenize_df(df, emb)
    matrix, remap = build_reduced_embeddings(tokens1 + tokens2, emb)
    dataset = PairDataset(
        [to_ids(t, emb, remap) for t in tokens1],
        [to_ids(t, emb, remap) for t in tokens2],
        df["label"].tolist(),
    )
    del emb  
    print(f"{len(dataset)} pairs, reduced embedding matrix: {tuple(matrix.shape)}")

    out = HERE / "results" / f"q3_{args.embedding}{'_quick' if args.quick else ''}.json"
    out.parent.mkdir(exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}

    for name in args.configs:
        if name in results:
            print(f"Skipping {name} (already in {out.name})")
            continue
        cfg = {**BASE, **VARIANTS[name]}
        if args.quick:
            cfg["epochs"] = 2
        print(f"\n=== {name}: {cfg}")
        start = time.time()
        results[name] = cross_validate(dataset, folds, matrix, cfg, device)
        results[name]["minutes"] = (time.time() - start) / 60
        out.write_text(json.dumps(results))  
        print(f"=== {name}: mean accuracy {results[name]['mean_accuracy']:.4f} "
              f"({results[name]['minutes']:.1f} min)")

    print("\nSummary:")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        print(f"  {name:20s} {r['mean_accuracy']:.4f}")


if __name__ == "__main__":
    main()
