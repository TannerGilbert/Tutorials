from sklearn.base import BaseEstimator, TransformerMixin, clone, ClassifierMixin
import numpy as np


class MajorityVote(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions_array = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.array([np.argmax(np.bincount(predictions)) for predictions in predictions_array])
