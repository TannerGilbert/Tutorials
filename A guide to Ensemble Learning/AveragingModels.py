# based on https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# and https://www.kaggle.com/eikedehling/trying-out-stacking-approaches
from sklearn.base import BaseEstimator, TransformerMixin, clone, RegressorMixin
import numpy as np


class AveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

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
        return np.mean(predictions, axis=1)
