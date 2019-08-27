# based on https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# and https://www.kaggle.com/eikedehling/trying-out-stacking-approaches
from sklearn.base import BaseEstimator, TransformerMixin, clone, RegressorMixin
from sklearn.model_selection import KFold
import numpy as np


class StackingModelsRetrained(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5, use_features_in_secondary=False):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_features_in_secondary = use_features_in_secondary

    def fit(self, X, y):
        """Fit all the models on the given dataset"""
        self.base_models_ = [clone(x) for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Train cloned base models and create out-of-fold predictions
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        if self.use_features_in_secondary:
            self.meta_model_.fit(np.hstack((X, out_of_fold_predictions)), y)
        else:
            self.meta_model_.fit(out_of_fold_predictions, y)

        for model in self.base_models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            base_model.predict(X) for base_model in self.base_models_])
        if self.use_features_in_secondary:
            return self.meta_model_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_model_.predict(meta_features)

    def predict_proba(self, X):
        meta_features = np.column_stack([
            base_model.predict(X) for base_model in self.base_models_])
        if self.use_features_in_secondary:
            return self.meta_model_.predict_proba(np.hstack((X, meta_features)))
        else:
            return self.meta_model_.predict_proba(meta_features)
