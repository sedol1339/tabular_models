from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import ignore_warnings

from tabular_models.dataset import Dataset
from tabular_models.models.linear import make_linear_model
from tabular_models.models.preproc import (
    PreprocType,
    combine_preprocs,
    make_preproc,
)
from tabular_models.models.sklearn import SklearnModel
from tabular_models.pipeline_result import PipelineResult
from tabular_models.pipelines.base import Pipeline, PrevResults, Req
from tabular_models.pipelines.scoring import default_scoring
from tabular_models.predictions import Predictions, Scores
from tabular_models.task_type import TaskType


def _pearson(
    x: pd.Series, y: pd.Series, bootstrap_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - statistics: array of shape (n_classes,)
    - pvalues: array of shape (n_classes,)

    For regression or binary, n_classes == 1.
    """
    isna = x.isna()
    x = x[~isna].reset_index(drop=True)
    y = y[~isna].reset_index(drop=True)

    if bootstrap_seed is not None:
        rng = np.random.default_rng(bootstrap_seed)
        iloc = rng.choice(len(x), size=len(x), replace=True)
        x = x[iloc]
        y = y[iloc]

    if y.dtype != 'category':
        result = pearsonr(x, y)
        return np.array([result.statistic]), np.array([result.pvalue])
    elif len(y.cat.categories) <= 2:
        result = pearsonr(x, y.cat.codes)
        return np.array([result.statistic]), np.array([result.pvalue])
    else:
        results = [
            pearsonr(x, (y == label).astype(float))
            for label in y.cat.categories
        ]
        return (
            np.array([r.statistic for r in results]),
            np.array([r.pvalue for r in results]),
        )


class SingleFeature_Num_Correlation(Pipeline):
    def splits_and_requirements(self) -> dict[str, list[Req]]:
        return {'none': []}

    @ignore_warnings(category=ConstantInputWarning)
    def run(self, dataset: Dataset, reqs: PrevResults) -> PipelineResult:
        correlations: dict[str, dict] = {}
        for feature_name in dataset.num_features:
            results = [
                _pearson(
                    dataset.X_train[feature_name],
                    dataset.y_train,
                    bootstrap_seed=i,
                )
                for i in range(50)
            ]
            correlations[feature_name] = {
                # arrays of shape (n_bootstraps, (n_classes or 1))
                'statistics': np.array([stat for stat, _ in results]),
                'pvalues': np.array([pvalue for _, pvalue in results]),
            }

        return PipelineResult(correlations=correlations)


class Abstract_SingleFeature_Num_Sklearn(Pipeline, metaclass=ABCMeta):
    @abstractmethod
    def get_num_transform(self) -> PreprocType:
        ...

    def get_sklearn_model(self, task_type: TaskType) -> BaseEstimator:
        return make_linear_model(
            task_type=task_type, num_cpus=self.CPUS, logreg_solver='lbfgs'
        )

    def splits_and_requirements(self) -> dict[str, list[Req]]:
        return {'none': []}

    @ignore_warnings(category=UserWarning)
    @ignore_warnings(category=RuntimeWarning)
    @ignore_warnings(category=ConvergenceWarning)
    def run(self, dataset: Dataset, reqs: PrevResults) -> PipelineResult:
        cross_val_scores_per_feature: dict[str, Scores] = {}
        for feature_name in dataset.num_features:
            preproc = combine_preprocs(
                {feature_name: make_preproc(self.get_num_transform())},
                to_dense=True,
                n_jobs=1,  # this is faster
            )
            model = SklearnModel(
                dataset.task_type,
                make_pipeline(
                    preproc, self.get_sklearn_model(dataset.task_type)
                ),
            )
            cross_val_preds = Predictions(
                model.cross_val_predict(
                    dataset.X_trainval,
                    dataset.y_trainval,
                    n_folds=5,
                    n_jobs=self.CPUS,
                ),
                dataset.y_trainval,
                dataset.task_type,
                iloc={'cross_val': np.arange(len(dataset.y_trainval))},
            )
            cross_val_scores_per_feature[feature_name] = default_scoring(
                cross_val_preds, subsets=['cross_val']
            )

        return PipelineResult(scores=cross_val_scores_per_feature)


class SingleFeature_Num_Linear_Standard(Abstract_SingleFeature_Num_Sklearn):
    def get_num_transform(self) -> PreprocType:
        return PreprocType.STANDARD


class SingleFeature_Num_Linear_QuantileUniform(
    Abstract_SingleFeature_Num_Sklearn
):
    def get_num_transform(self) -> PreprocType:
        return PreprocType.QUANTILE_UNIFORM


class SingleFeature_Num_Linear_QuantileNormal(
    Abstract_SingleFeature_Num_Sklearn
):
    def get_num_transform(self) -> PreprocType:
        return PreprocType.QUANTILE_NORMAL


class SingleFeature_Num_Tree2Leaves(Abstract_SingleFeature_Num_Sklearn):
    def get_num_transform(self) -> PreprocType:
        return PreprocType.IDENTITY

    def tree_kwargs(self) -> dict[str, Any]:
        return {'max_depth': 1}

    def get_sklearn_model(self, task_type: TaskType) -> BaseEstimator:
        return (
            DecisionTreeRegressor(**self.tree_kwargs(), random_state=0)
            if task_type == TaskType.REGRESSION
            else DecisionTreeClassifier(**self.tree_kwargs(), random_state=0)
        )


class SingleFeature_Num_Tree3Leaves(SingleFeature_Num_Tree2Leaves):
    def tree_kwargs(self) -> dict[str, Any]:
        return {'max_leaf_nodes': 3}
