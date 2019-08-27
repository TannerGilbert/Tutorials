# based on https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# and https://www.kaggle.com/eikedehling/trying-out-stacking-approaches
from sklearn.base import BaseEstimator, TransformerMixin, clone, RegressorMixin
from sklearn.model_selection import KFold
import numpy as np


class StackingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5, task_type='classification', use_features_in_secondary=False):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.task_type = task_type
        self.use_features_in_secondary = use_features_in_secondary

    def fit(self, X, y):
        """Fit all the models on the given dataset"""
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Train cloned base models and create out-of-fold predictions
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        if self.use_features_in_secondary:
            self.meta_model_.fit(np.hstack((X, out_of_fold_predictions)), y)
        else:
            self.meta_model_.fit(out_of_fold_predictions, y)

        return self

    def predict(self, X):
        if self.task_type == 'classification':
            meta_features = np.column_stack([[np.argmax(np.bincount(predictions)) for predictions in
                                              np.column_stack([model.predict(X) for model in base_models])]
                                             for base_models in self.base_models_])
        else:
            meta_features = np.column_stack([
                np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                for base_models in self.base_models_])
        if self.use_features_in_secondary:
            return self.meta_model_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_model_.predict(meta_features)

    def predict_proba(self, X):
        if self.task_type == 'classification':
            meta_features = np.column_stack([[np.argmax(np.bincount(predictions)) for predictions in
                                              np.column_stack([model.predict(X) for model in base_models])]
                                             for base_models in self.base_models_])
            if self.use_features_in_secondary:
                return self.meta_model_.predict_proba(np.hstack((X, meta_features)))
            else:
                return self.meta_model_.predict_proba(meta_features)
