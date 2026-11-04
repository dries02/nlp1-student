import pandas as pd


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
