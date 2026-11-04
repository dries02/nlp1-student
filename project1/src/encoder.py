from abc import ABC, abstractmethod

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


class Encoder(ABC):
    """Base class for encoders."""
    @abstractmethod
    def fit(self, s1: pd.Series, s2: pd.Series) -> None:
        """Fit the tokenizer on the training set."""

    @abstractmethod
    def transform(self, sentence: pd.Series):
        """Transform a single sentence."""


class _CountEncoder(Encoder):
    """Implements encoders that use counting."""
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.encoder = CountVectorizer(**kwargs)

    def fit(self, s1: pd.Series, s2: pd.Series) -> None:
        sentences = pd.concat([s1, s2])
        self.encoder.fit(sentences)

    def transform(self, sentences: pd.Series):
        return self.encoder.transform(sentences)


class TokenIndicatorEncoder(_CountEncoder):
    """Implements a token indicator tokenizer."""
    def __init__(self) -> None:
        super().__init__(binary=True)


class BagOfWordsEncoder(_CountEncoder):
    """Implements a bag of words tokenizer."""
    def __init__(self) -> None:
        super().__init__(binary=False)


class SentenceTransformerEncoder(Encoder):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.encoder = SentenceTransformer(model_name)

    def fit(self, s1: pd.Series, s2: pd.Series) -> None:
        """The pretrained model is not fitted on the training set."""

    def transform(self, sentence: pd.Series):
        """Use pretrained model to transform sentence."""
        return self.encoder.encode(sentence)
