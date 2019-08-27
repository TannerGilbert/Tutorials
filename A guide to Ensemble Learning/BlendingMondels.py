from sklearn.base import BaseEstimator, TransformerMixin, clone, RegressorMixin
from sklearn.model_selection import train_test_split
import numpy as np


class BlendingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, holdout_pct=0.2, use_features_in_secondary=False):
        self.base_models = base_models
        self.meta_model = meta_model
        self.holdout_pct = holdout_pct
        self.use_features_in_secondary = use_features_in_secondary

    def fit(self, X, y):
        """Fit all the models on the given dataset"""
        self.base_models_ = [clone(x) for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)

        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=self.holdout_pct)

        holdout_predictions = np.zeros((X_holdout.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models_):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_holdout)
            holdout_predictions[:, i] = y_pred
        if self.use_features_in_secondary:
            self.meta_model_.fit(np.hstack((X_holdout, holdout_predictions)), y_holdout)
        else:
            self.meta_model_.fit(holdout_predictions, y_holdout)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models_
        ])
        if self.use_features_in_secondary:
            return self.meta_model_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_model_.predict(meta_features)
