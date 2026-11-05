import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from src.encoder import Encoder
from src.paraphraseclassifier import ParaphraseClassifier


def make_features(s1: csr_matrix, s2: csr_matrix):
    """Simple baseline: just stack the feature vectors"""
    if type(s1) != type(s2):
        msg = f"Type mismatch: {type(s1)}, {type(s2)}"
        raise ValueError(msg)

    match s1:
        case csr_matrix():
            return hstack([s1, s2])
        case np.ndarray():
            return np.hstack([s1, s2])
        case _:
            msg = f"Unexpected type: {type(s1)}"
            raise NotImplementedError(msg)

class ParaphrasePipeline:
    def __init__(self, encoder: Encoder, classifier: ParaphraseClassifier) -> None:
        self.encoder = encoder
        self.classifier = classifier

    def _features(self, df: pd.DataFrame):
        u = self.encoder.transform(df["sentence1"])
        v = self.encoder.transform(df["sentence2"])
        return make_features(u, v)

    def fit(self, train_df: pd.DataFrame) -> None:
        self.encoder.fit(train_df["sentence1"], train_df["sentence2"])
        self.classifier.fit(self._features(train_df), train_df["label"])

    def predict(self, df: pd.DataFrame):
        return self.classifier.predict(self._features(df))
