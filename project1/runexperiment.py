from itertools import product

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from src.paraphrasepipeline import ParaphrasePipeline
from src.trainconfig import TrainConfig
from src.utils import load_datasets, make_encoder, make_model

ENCODERS = ["indicator", "bow"]
MODELS = ["logreg", "shallowmlp", "deepmlp"]


def run_experiment(train_df: pd.DataFrame, config: TrainConfig, folds) -> list[float]:
    accuracies = []

    for train_idx, val_idx in folds:
        pipeline = ParaphrasePipeline(make_encoder(config), make_model(config))

        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]

        pipeline.fit(train_fold)
        val_preds = pipeline.predict(val_fold)

        accuracy = (val_preds == val_fold["label"]).mean()
        accuracies.append(float(accuracy))

    return accuracies

def main() -> None:
    train_df, dev_df, test_df = load_datasets()
    train_df = pd.concat([train_df, dev_df])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=64)
    folds = list(cv.split(train_df, train_df["label"]))

    results = {}

    for encoder_name, model_name in product(ENCODERS, MODELS):
        print("running:", encoder_name, model_name)
        config = TrainConfig(encoder=encoder_name, model=model_name)

        results[encoder_name, model_name] = run_experiment(train_df, config, folds)

    print(results)

if __name__ == "__main__":
    main()
