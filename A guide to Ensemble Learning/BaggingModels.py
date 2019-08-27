from sklearn.base import BaseEstimator, TransformerMixin, clone, RegressorMixin
import numpy as np


class BaggingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, task_type='classification'):
        self.models = models
        self.task_type = task_type

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            X_tmp, y_tmp = self.subsample(X, y)
            model.fit(X_tmp, y_tmp)

        return self

    # Create a random subsample from the dataset with replacement
    @staticmethod
    def subsample(X, y, ratio=1.0):
        X_new, y_new = list(), list()
        n_sample = round(len(X) * ratio)
        while len(X_new) < n_sample:
            index = np.random.randint(len(X))
            X_new.append(X[index])
            y_new.append(y[index])
        return X_new, y_new

    def predict(self, X):
        predictions_array = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        if self.task_type == 'classification':
            return np.array([np.argmax(np.bincount(predictions)) for predictions in predictions_array])
        else:
            return np.mean(predictions_array, axis=1)

    def predict_proba(self, X):
        if self.task_type == 'classification':
            predictions = []
            for x in X:
                prediction = np.row_stack([
                    model.predict_proba([x]) for model in self.models_
                ])
                predictions.append(np.mean(prediction, axis=0))
            return np.array(predictions)
        return None
