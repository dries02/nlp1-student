import pandas as pd
from src.encoder import (
    BagOfWordsEncoder,
    Encoder,
    SentenceTransformerEncoder,
    TokenIndicatorEncoder,
)
from src.paraphraseclassifier import (
    DeepMLP,
    LogReg,
    ParaphraseClassifier,
    ShallowMLP,
)
from src.trainconfig import TrainConfig


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all splits from Huggingface."""
    splits = {
        'train': 'labeled_final/train-00000-of-00001.parquet',
        'test': 'labeled_final/test-00000-of-00001.parquet',
        'validation': 'labeled_final/validation-00000-of-00001.parquet',
    }
    
    train_df = pd.read_parquet("hf://datasets/google-research-datasets/paws/" + splits["train"])
    dev_df = pd.read_parquet("hf://datasets/google-research-datasets/paws/" + splits["validation"])
    test_df = pd.read_parquet("hf://datasets/google-research-datasets/paws/" + splits["test"])

    return train_df, dev_df, test_df


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
        case "logreg":
            return LogReg(config.loglinear_solver)
        case "shallowmlp":
            return ShallowMLP()
        case "deepmlp":
            return DeepMLP()
        case _:
            msg = f"Unknown model: {config.model}"
            raise ValueError(msg)
