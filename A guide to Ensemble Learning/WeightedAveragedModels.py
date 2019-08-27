# based on https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# and https://www.kaggle.com/eikedehling/trying-out-stacking-approaches
from sklearn.base import BaseEstimator, TransformerMixin, clone, RegressorMixin
import numpy as np


class WeightedAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
        assert sum(self.weights) == 1

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.sum(predictions * self.weights, axis=1)
