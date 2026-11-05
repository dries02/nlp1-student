from abc import ABC, abstractmethod
from typing import override

import numpy.typing as npt
from keras import Sequential
from keras.layers import Dense
from sklearn.linear_model import LogisticRegression


class ParaphraseClassifier(ABC):
    @abstractmethod
    def fit(self, X_train, y_train) -> None:
        """Fit the model on the training set."""

    @abstractmethod
    def predict(self, X) -> npt.NDArray:
        """Make predictions."""


class LogLinear(ParaphraseClassifier):
    def __init__(self, solver: str) -> None:
        super().__init__()
        self.model = LogisticRegression(solver=solver)

    @override
    def fit(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    @override
    def predict(self, X) -> npt.NDArray:
        return self.model.predict(X)


class MLPClassifier(ParaphraseClassifier):
    def __init__(self, model: Sequential, n_epochs: int) -> None:
        super().__init__()
        self.model = model
        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.n_epochs = n_epochs

    @override
    def fit(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train, epochs=self.n_epochs)

    @override
    def predict(self, X) -> npt.NDArray:
        probs = self.model.predict(X)
        return (probs >= 0.5).ravel()               # to flatten to (n_samples,)


class ShallowMLP(MLPClassifier):
    """Implements a multi-layer perceptron with a single hidden layer."""
    def __init__(self, hidden_units: int = 64, n_epochs: int = 20) -> None:
        model = Sequential([
            Dense(hidden_units, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        super().__init__(model, n_epochs)


class DeepMLP(MLPClassifier):
    def __init__(self, hidden_units: int = 64, n_epochs: int = 20) -> None:
        model = Sequential([
            Dense(hidden_units, activation="relu"),
            Dense(hidden_units, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        super().__init__(model, n_epochs)
