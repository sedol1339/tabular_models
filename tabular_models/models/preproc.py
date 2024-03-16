from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from sklearn.utils import Bunch

from tabular_models.dataset import Dataset
from tabular_models.task_type import TaskType


class PreprocType(Enum):
    IDENTITY = 'identity'
    STANDARD = 'standard'
    ROBUST = 'robust'
    QUANTILE_UNIFORM = 'quantile_uniform'
    QUANTILE_NORMAL = 'quantile_normal'
    ONE_HOT_ENCODING = 'one_hot_encoding'
    TARGET_ENCODING = 'target_encoding'


def make_preproc(
    preproc: PreprocType,
    task_type: TaskType | None = None,
    random_seed: int = 0,
    one_hot_drop: str = 'if_binary',
    one_hot_sparse_output: bool = True,
    one_hot_min_frequency: int = 10,
    impute_num: bool = True,
    impute_num_add_indicator: bool = True,
) -> BaseEstimator:
    maybe_num_imputer = (
        SimpleImputer(
            strategy='mean',
            add_indicator=impute_num_add_indicator,
            keep_empty_features=False,
        )
        if impute_num
        else FunctionTransformer()
    )

    if preproc == PreprocType.IDENTITY:
        return clone(maybe_num_imputer)
    elif preproc == PreprocType.STANDARD:
        return make_pipeline(
            StandardScaler(),
            clone(maybe_num_imputer),
        )
    elif preproc == PreprocType.ROBUST:
        return make_pipeline(
            RobustScaler(),
            clone(maybe_num_imputer),
        )
    elif preproc == PreprocType.QUANTILE_UNIFORM:
        return make_pipeline(
            QuantileTransformer(
                output_distribution='uniform', random_state=random_seed
            ),
            clone(maybe_num_imputer),
        )
    elif preproc == PreprocType.QUANTILE_NORMAL:
        return make_pipeline(
            QuantileTransformer(
                output_distribution='normal', random_state=random_seed
            ),
            clone(maybe_num_imputer),
        )
    elif preproc == PreprocType.ONE_HOT_ENCODING:
        return OneHotEncoder(
            drop=one_hot_drop,
            min_frequency=one_hot_min_frequency,
            handle_unknown='infrequent_if_exist',
            sparse_output=one_hot_sparse_output,
            dtype=np.float32,
        )
    elif preproc == PreprocType.TARGET_ENCODING:
        assert task_type is not None, 'Specify task_type for target encoding'
        return TargetEncoder(
            cv=25,
            smooth='auto',
            categories='auto',
            target_type={
                TaskType.REGRESSION: 'continuous',
                TaskType.BINARY: 'binary',
                TaskType.MULTICLASS: 'multiclass',
            }[task_type],
        )


def combine_preprocs(
    preprocs: dict[str, BaseEstimator],
    add_dummy_col: bool = False,
    to_dense: bool = False,
    n_jobs: int = 1,
) -> BaseEstimator:
    """
    May be slow, since each transformer transforms a single column,
    and each transformer may try to run it's own parallelization,
    since it expects to accept >1 features, however accepts only one.
    Consider to set n_jobs=1 except for very large datasets.
    """
    transformer = ColumnTransformer(
        [
            (col_name, preproc, [col_name])
            for col_name, preproc in preprocs.items()
        ],
        remainder='drop',
        sparse_threshold=0 if to_dense else 0.3,
        n_jobs=n_jobs,
    )

    if add_dummy_col:
        transformer = make_pipeline(
            transformer, DummyAppender(new_col='__dummy__')
        )

    return transformer


class DummyAppender(TransformerMixin, BaseEstimator):
    def __init__(self, new_col='dummy'):
        self.new_col = new_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=None):
        if copy:
            X = X.copy()
        if isinstance(X, pd.DataFrame):
            return X.assign(**{self.new_col: 0})
        else:
            return np.concatenate([X, np.zeros((len(X), 1))], axis=1)


def get_one_hot_encoder_output_size(
    column: pd.Series,
    drop: Literal[None, 'first', 'if_binary'] = 'if_binary',
    one_hot_min_frequency: int = 10,
) -> int:
    value_counts = column.value_counts(dropna=False).to_numpy()
    is_frequent = value_counts >= one_hot_min_frequency
    # value_counts() may have zeros for categorical columns
    has_infrequent = ((value_counts >= 1) & ~is_frequent).any()

    total = is_frequent.sum() + has_infrequent

    if drop == 'first' or (total == 2 and drop == 'if_binary'):
        total -= 1

    return total


class AbstractPreproc:
    def get_num_transform(self) -> PreprocType | Literal['skip']:
        # override if using num features
        raise NotImplementedError()

    def get_cat_transform(self) -> PreprocType | Literal['skip', 'auto']:
        # override if using cat features
        raise NotImplementedError()

    def get_preproc_params(self) -> Bunch:
        return Bunch(
            one_hot_drop='if_binary',
            one_hot_min_frequency=10,
            impute_num=True,
            impute_num_add_indicator=True,
        )

    def get_preproc_types(self, dataset: Dataset) -> dict[str, PreprocType]:
        preproc_params = self.get_preproc_params()
        cat_transform = self.get_cat_transform()
        num_transform = self.get_num_transform()
        preproc_types: dict[str, PreprocType] = {}

        for feature_name in dataset.X_train.columns:
            if dataset.X_train[feature_name].dtype != 'category':
                # numerical feature
                if num_transform != 'skip':
                    preproc_types[feature_name] = num_transform
            else:
                # categorical feature
                if cat_transform == 'auto':
                    one_hot_dim = get_one_hot_encoder_output_size(
                        dataset.X_train[feature_name],
                        drop=preproc_params.one_hot_drop,
                        one_hot_min_frequency=(
                            preproc_params.one_hot_min_frequency
                        ),
                    )
                    target_dim = (
                        len(dataset.y_train.cat.categories)
                        if dataset.task_type == TaskType.MULTICLASS
                        else 1
                    )
                    if target_dim < one_hot_dim:
                        preproc = PreprocType.TARGET_ENCODING
                    else:
                        preproc = PreprocType.ONE_HOT_ENCODING
                    preproc_types[feature_name] = preproc
                elif cat_transform != 'skip':
                    preproc_types[feature_name] = cat_transform

        return preproc_types

    def get_preproc(self, dataset: Dataset, n_cpus: int = -1) -> BaseEstimator:
        preproc_params = self.get_preproc_params()
        preprocs = {
            feature_name: make_preproc(
                preproc_type, dataset.task_type, **preproc_params
            )
            for feature_name, preproc_type in self.get_preproc_types(
                dataset
            ).items()
        }
        return combine_preprocs(
            preprocs,
            add_dummy_col=True,
            to_dense=True,
            n_jobs=n_cpus,
        )
