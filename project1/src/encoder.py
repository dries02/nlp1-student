from abc import ABC, abstractmethod
from typing import override

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


class Encoder(ABC):
    """Base class for encoders."""
    @abstractmethod
    def fit(self, s1: pd.Series, s2: pd.Series) -> None:
        """Fit the encoder on the training set."""

    @abstractmethod
    def transform(self, sentences: pd.Series):
        """Transform sentences."""


class _CountEncoder(Encoder):
    """Implements encoders that use counting."""
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.encoder = CountVectorizer(min_df=2, max_features=5000, **kwargs)

    @override
    def fit(self, s1: pd.Series, s2: pd.Series) -> None:
        sentences = pd.concat([s1, s2])
        self.encoder.fit(sentences)

    @override
    def transform(self, sentences: pd.Series):
        return self.encoder.transform(sentences)


class TokenIndicatorEncoder(_CountEncoder):
    """Implements a token indicator encoder."""
    def __init__(self) -> None:
        super().__init__(binary=True)


class BagOfWordsEncoder(_CountEncoder):
    """Implements a bag of words encoder."""
    def __init__(self) -> None:
        super().__init__(binary=False)


class SentenceTransformerEncoder(Encoder):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.encoder = SentenceTransformer(model_name)

    @override
    def fit(self, s1: pd.Series, s2: pd.Series) -> None:
        """The pretrained model is not fitted on the training set."""

    @override
    def transform(self, sentences: pd.Series):
        """Use pretrained model to transform sentences."""
        return self.encoder.encode(sentences.to_list())
