from argparse import ArgumentParser, Namespace

import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.encoder import (
    BagOfWordsEncoder,
    Encoder,
    SentenceTransformerEncoder,
    TokenIndicatorEncoder,
)
from src.utils import load_datasets


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--encoder", type=str, choices=["indicator", "bow", "sentencetransformer"], required=True)
    parser.add_argument("--model", type=str, choices=["loglin", "shallowmlp", "deepmlp"], required=True)
    return parser.parse_args()


def make_encoder(config: Namespace) -> Encoder:
    """Build an encoder."""
    match config.encoder:
        case "indicator":
            return TokenIndicatorEncoder()
        case "bow":
            return BagOfWordsEncoder()
        case "sentencetransformer":
            model_name = "all-MiniLM-L6-v2"             # might be changed
            return SentenceTransformerEncoder(model_name)
        case _:
            msg = f"Unknown encoder: {config.encoder}"
            raise ValueError(msg)


def make_model(config: Namespace):
    """Build a model."""
    match config.model:
        case "loglin":
            return LogisticRegression()
        case "shallowmlp":
            raise NotImplementedError
        case "deepmlp":
            raise NotImplementedError
        case _:
            msg = f"Unknown model: {config.model}"
            raise ValueError(msg)

def train_model(model, encoder: Encoder, train_df: pd.DataFrame) -> None:
    encoder.fit(train_df["sentence1"], train_df["sentence2"])
    s1 = encoder.transform(train_df["sentence1"])
    s2 = encoder.transform(train_df["sentence2"])
    # model.fit()


def main() -> None:
    config = parse_args()
    train_df, dev_df, test_df = load_datasets()
    encoder = make_encoder(config)
    model = make_model(config)
    train_model(model, encoder, train_df)


if __name__ == "__main__":
    main()
