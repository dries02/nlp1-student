from itertools import product
from os.path import exists

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold, StratifiedKFold
from src.paraphrasepipeline import ParaphrasePipeline
from src.trainconfig import TrainConfig
from src.utils import load_datasets, make_encoder, make_model
from src.tuning import get_hyperparameters
from cacheembeddings import main as generate_embeddings

ENCODERS = ["indicator", "bow", "sentencetransformer"]
MODELS = ["logreg", "shallowmlp", "deepmlp"]

def run_experiment(train_df, config, folds, params, X_all=None, y_all=None):
    accuracies = []

    for train_idx, val_idx in folds:
        if config.encoder == "sentencetransformer":
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]
            model = make_model(config, params)
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
        else:
            pipeline = ParaphrasePipeline(make_encoder(config), make_model(config, params))

            train_fold = train_df.iloc[train_idx]
            val_fold = train_df.iloc[val_idx]

            pipeline.fit(train_fold)
            val_preds = pipeline.predict(val_fold)
            y_val = val_fold["label"]

        accuracy = (val_preds == y_val).mean()
        accuracies.append(float(accuracy))

    return accuracies

def main() -> None:
    train_df, dev_df, test_df = load_datasets()
    train_df = pd.concat([train_df, dev_df]).reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=64)
    folds = list(cv.split(train_df, train_df["label"]))

    results = {}

    for encoder_name, model_name in product(ENCODERS, MODELS):
        print("running:", encoder_name, model_name)

        config = TrainConfig(encoder=encoder_name, model=model_name)

        best_accuracy = -1.0
        best_params = None
        best_scores = None

        if encoder_name == "sentencetransformer":
            if not exists("embeddings/train_dev_s1.npy") or not exists("embeddings/train_dev_s2.npy"):
                print("Embeddings not found on disk. Generating them now...")
                generate_embeddings()
            s1 = np.load("embeddings/train_dev_s1.npy")
            s2 = np.load("embeddings/train_dev_s2.npy")
            X_all = np.hstack([s1, s2])
            y_all = train_df["label"].values
        else:
            X_all = None
            y_all = None

        for params in get_hyperparameters(model_name):
            accuracies = run_experiment(train_df, config, folds, params, X_all=X_all, y_all=y_all)
            mean_accuracy = sum(accuracies) / len(accuracies)

            print(params, "accuracy:", mean_accuracy)

            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_params = params
                best_scores = accuracies

        results[encoder_name, model_name] = {
            "best_params": best_params,
            "accuracies": best_scores,
            "mean_accuracy": best_accuracy
        }

    formatted_results = {
        f"{encoder}_{model}": data
        for (encoder, model), data in results.items()
    }
    with open("tuning_results.json", "w") as f:
        json.dump(formatted_results, f, indent=4)
    print(results)

if __name__ == "__main__":
    main()
