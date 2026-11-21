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
    encoder="lstm", joint=False, rnn_hidden=128, bidirectional=True, rnn_layers=1, rnn_pooling="max",
    freeze=True, combine="full", hidden_size=128, num_layers=1, dropout=0.0,
    lr=1e-3, weight_decay=0.0, batch_size=64, epochs=15, seed=64,
    early_stopping=True, patience=3, train_fraction=1.0,
)

# Each variant changes one thing compared to BASE
VARIANTS = {
    "base": {},                                          
    "joint": {"joint": True},                           
    "gru": {"encoder": "gru"},
    "unidirectional": {"bidirectional": False},
    "rnn_pool_last": {"rnn_pooling": "last"},
    "hidden64": {"rnn_hidden": 64},
    "hidden256": {"rnn_hidden": 256},
    "finetune": {"freeze": False},
    "random_init": {"random_init": True, "freeze": False},
    "dropout0.3": {"dropout": 0.3},
    "data10": {"train_fraction": 0.1},
    "data25": {"train_fraction": 0.25},
    "data50": {"train_fraction": 0.5},
    # combining variants
    "finetune_hidden256": {"freeze": False, "rnn_hidden": 256},
    "finetune_dropout0.3": {"freeze": False, "dropout": 0.3},
    "combined": {"freeze": False, "rnn_hidden": 256, "dropout": 0.3, "epochs": 25},
    # seed variance
    "base_seed2": {"seed": 1064},
    "base_seed3": {"seed": 2064},
    "finetune_seed2": {"freeze": False, "seed": 1064},
    "finetune_seed3": {"freeze": False, "seed": 2064},
    "combined_seed2": {"freeze": False, "rnn_hidden": 256, "dropout": 0.3, "epochs": 25, "seed": 1064},
    "combined_seed3": {"freeze": False, "rnn_hidden": 256, "dropout": 0.3, "epochs": 25, "seed": 2064},
     # learning rate and batch size
    "combined_lr3e-4": {"freeze": False, "rnn_hidden": 256, "dropout": 0.3, "epochs": 25, "lr": 3e-4},
    "combined_lr3e-3": {"freeze": False, "rnn_hidden": 256, "dropout": 0.3, "epochs": 25, "lr": 3e-3},
    "combined_batch32": {"freeze": False, "rnn_hidden": 256, "dropout": 0.3, "epochs": 25, "batch_size": 32},
    "combined_batch256": {"freeze": False, "rnn_hidden": 256, "dropout": 0.3, "epochs": 25, "batch_size": 256},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", choices=list(EMBEDDINGS), default="glove")
    parser.add_argument("--configs", nargs="*", default=list(VARIANTS), help="which variants to run")
    parser.add_argument("--quick", action="store_true", help="2,000 pairs, 2 folds, 2 epochs, to test that everything runs")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Same data and folds as Project 1
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
    ids1 = [to_ids(t, emb, remap) for t in tokens1]
    ids2 = [to_ids(t, emb, remap) for t in tokens2]
    labels = df["label"].tolist()
    del emb

    pair_dataset = PairDataset(ids1, ids2, labels)
    sep_id = matrix.shape[0]
    torch.manual_seed(0)
    joint_matrix = torch.cat([matrix, torch.randn(1, matrix.shape[1]) * matrix[2:].std()])
    joint_dataset = PairDataset([a + [sep_id] + b for a, b in zip(ids1, ids2)], [[1]] * len(ids1), labels)
    print(f"{len(pair_dataset)} pairs, reduced embedding matrix: {tuple(matrix.shape)}")

    out = HERE / "results" / f"q4_{args.embedding}{'_quick' if args.quick else ''}.json"
    out.parent.mkdir(exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}

    for name in args.configs:
        if name in results:
            print(f"Skipping {name} (already in {out.name})")
            continue
        cfg = {**BASE, **VARIANTS[name]}
        if args.quick:
            cfg["epochs"] = 2
        dataset, emb_matrix = (joint_dataset, joint_matrix) if cfg["joint"] else (pair_dataset, matrix)
        print(f"\n=== {name}: {cfg}")
        start = time.time()
        results[name] = cross_validate(dataset, folds, emb_matrix, cfg, device)
        results[name]["minutes"] = (time.time() - start) / 60
        out.write_text(json.dumps(results))  # save after every configuration
        print(f"=== {name}: mean accuracy {results[name]['mean_accuracy']:.4f} "
              f"({results[name]['minutes']:.1f} min)")

    print("\nSummary:")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        print(f"  {name:20s} {r['mean_accuracy']:.4f}")


if __name__ == "__main__":
    main()
