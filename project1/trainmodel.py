import pandas as pd
from src.encoder import (
    BagOfWordsEncoder,
    Encoder,
    SentenceTransformerEncoder,
    TokenIndicatorEncoder,
)
from src.paraphraseclassifier import (
    DeepMLP,
    LogLinear,
    ParaphraseClassifier,
    ShallowMLP,
)
from src.paraphrasepipeline import ParaphrasePipeline
from src.trainconfig import TrainConfig, parse_config
from src.utils import load_datasets


def make_encoder(config: TrainConfig) -> Encoder:
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


def make_model(config: TrainConfig) -> ParaphraseClassifier:
    """Build a model."""
    match config.model:
        case "loglinear":
            return LogLinear(config.loglinear_solver)
        case "shallowmlp":
            return ShallowMLP()
        case "deepmlp":
            return DeepMLP()
        case _:
            msg = f"Unknown model: {config.model}"
            raise ValueError(msg)


def train_model(pipeline: ParaphrasePipeline, train_df: pd.DataFrame, dev_df: pd.DataFrame) -> None:
    pipeline.fit(train_df)

    dev_pred = pipeline.predict(dev_df)

    print((dev_pred == dev_df["label"]).mean())

def main() -> None:
    config = parse_config()
    train_df, dev_df, test_df = load_datasets()

    train_df = train_df.sample(400)
    dev_df = dev_df.sample(400)

    encoder = make_encoder(config)
    model = make_model(config)
    pipeline = ParaphrasePipeline(encoder, model)

    train_model(pipeline, train_df, dev_df)


if __name__ == "__main__":
    main()
