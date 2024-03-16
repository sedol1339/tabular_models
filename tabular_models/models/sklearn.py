from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import (
    QuantileTransformer,
    StandardScaler,
)

from tabular_models.utils import (
    identity,
    sparse_to_dense_with_copy,
    symexp,
    symlog,
)

from ..task_type import TaskType
from .base import BaseModel


class SklearnModel(BaseModel):
    def __init__(
        self,
        task_type: TaskType,
        model: BaseEstimator,
        random_seed: int = 0,
        sparse_to_dense: bool = True,
    ):
        """
        Wrapper for sklearn models to match BaseModel interface
        """
        super().__init__(task_type=task_type, random_seed=random_seed)
        self._model = clone(model)
        self._sparse_to_dense = sparse_to_dense
        self._fitted = False

    @property
    def sklearn_model(self) -> BaseEstimator:
        return self._model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        if y.dtype == 'category':
            y = y.cat.codes
        if self._sparse_to_dense:
            X = sparse_to_dense_with_copy(X)
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._sparse_to_dense:
            X = sparse_to_dense_with_copy(X)
        if self._task_type == TaskType.REGRESSION:
            return self._model.predict(X)
        elif self._task_type == TaskType.BINARY:
            probas = self._model.predict_log_proba(X)
            return probas[:, 1] - probas[:, 0]
        elif self._task_type == TaskType.MULTICLASS:
            return self._model.predict_log_proba(X)

    def cross_val_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        n_jobs: int = -1,
    ) -> np.ndarray:
        # same processing as in .fit() method
        if y.dtype == 'category':
            y = y.cat.codes
        if self._sparse_to_dense:
            X = sparse_to_dense_with_copy(X)

        preds = cross_val_predict(
            self._model,
            X,
            y,
            cv=n_folds,
            n_jobs=n_jobs,
            method=(
                'predict'
                if self._task_type == TaskType.REGRESSION
                else 'predict_log_proba'
            ),
        )
        if self._task_type == TaskType.BINARY:
            preds = preds[:, 1] - preds[:, 0]

        return preds


TargetTransformType = Literal[
    'none', 'symlog', 'standard', 'quantile_normal', 'quantile_uniform'
]


def make_transformed_target_regressor(
    model: RegressorMixin,
    transform: TargetTransformType,
    random_seed: int = 0,
) -> TransformedTargetRegressor:
    if transform == 'symlog':
        transform_args = {'func': symlog, 'inverse_func': symexp}
    elif transform == 'none':
        transform_args = {
            'func': identity,
            'inverse_func': identity,
        }
    elif transform == 'standard':
        transform_args = {'transformer': StandardScaler()}
    elif transform == 'quantile_normal':
        transform_args = {
            'transformer': QuantileTransformer(
                output_distribution='normal',
                random_state=random_seed,
            )
        }
    elif transform == 'quantile_uniform':
        transform_args = {
            'transformer': QuantileTransformer(
                output_distribution='uniform',
                random_state=random_seed,
            )
        }
    else:
        raise AssertionError(f'unknown target transform {transform}')
    return TransformedTargetRegressor(model, **transform_args)
