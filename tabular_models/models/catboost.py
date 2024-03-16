from __future__ import annotations

from collections.abc import Iterable

import catboost
import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from typing_extensions import Self

from tabular_models.predictions import HIGHER_IS_BETTER
from tabular_models.utils import get_best_iter

from ..task_type import TaskType
from .base import BaseModel


def get_default_catboost_params(task_type: TaskType, cpus: int = -1) -> Bunch:
    """Reasonable (for experiments) catboost_kwargs for CatBoostModel constructor"""
    kwargs = Bunch()
    kwargs.task_type = 'CPU'
    kwargs.thread_count = cpus
    kwargs.depth = 6
    kwargs.learning_rate = 0.1
    kwargs.metric_period = 1
    if task_type == TaskType.REGRESSION:
        kwargs.loss_function = 'RMSEWithUncertainty'
        kwargs.eval_metric = 'RMSE'
    elif task_type == TaskType.BINARY:
        kwargs.loss_function = 'Logloss'
        kwargs.eval_metric = 'AUC'
    elif task_type == TaskType.MULTICLASS:
        kwargs.loss_function = 'MultiClass'
        kwargs.eval_metric = 'MultiClass'
    return kwargs


class CatBoostModel(BaseModel):
    """
    Catboost wrapper with BaseModel interface.

    CatBoost random_seed goes to __init__, CatBoost parameters {max_iterations,
    early_stopping_rounds, use_best_model, verbose} go to .fit(), other
    CatBoost parameters go to catboost_kwargs.

    Defaults for catboost_kwargs can be obtained by
    get_default_catboost_params(task_type).
    """

    def __init__(
        self,
        task_type: TaskType,
        random_seed: int = 0,
        init_model: BaseModel | None = None,
        catboost_kwargs: dict | None = None,
    ):
        super().__init__(task_type=task_type, random_seed=random_seed)

        self._init_model = init_model
        self._catboost_kwargs = catboost_kwargs or {}

        assert (
            'eval_fraction' not in self._catboost_kwargs
        ), 'Set eval_set in .fit()'
        assert (
            'iterations' not in self._catboost_kwargs
            and 'num_boost_round' not in self._catboost_kwargs
            and 'n_estimators' not in self._catboost_kwargs
            and 'num_trees' not in self._catboost_kwargs
        ), 'Set max_iterations in .fit()'
        assert (
            'random_seed' not in self._catboost_kwargs
        ), 'Use random_seed param in __init__'

        self._fitted_using_init_predictions: bool | None = None
        self._catboost_model = None  # not fitted yet

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        early_stopping_rounds: int = 50,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        use_best_model: bool = True,
        max_iterations: int = 10000,
        verbose: int | bool = 0,
        fit_init_model: bool = True,
        init_predictions: np.ndarray | None = None,
        init_val_predictions: np.ndarray | None = None,
    ) -> Self:
        """
        If init_model is not None and fit_init_model, this method firstly
        does self.init_model.fit(X, y), then fits Catboost with
        init predictions obtained by self.init_model.predict(X)
        """

        catboost_kwargs = {
            'random_seed': self._random_seed,
            'iterations': max_iterations,
            'allow_writing_files': False,
            **self._catboost_kwargs,
        }

        if self._task_type == TaskType.REGRESSION:
            self._catboost_model = catboost.CatBoostRegressor(
                **catboost_kwargs
            )
        elif self._task_type in (TaskType.BINARY, TaskType.MULTICLASS):
            self._catboost_model = catboost.CatBoostClassifier(
                **catboost_kwargs, class_names=list(y.cat.categories)
            )

        cat_features = [col for col in X.columns if X[col].dtype == 'category']

        if self._init_model is not None:
            assert (
                init_predictions is None
            ), 'use either init model or init predictions'
            if fit_init_model:
                self._init_model.fit(X, y)
            init_predictions = self._init_model.predict(X)
            if eval_set is not None:
                init_val_predictions = self._init_model.predict(eval_set[0])
        elif init_predictions is not None:
            assert len(X) == len(init_predictions)
            if eval_set is not None:
                assert init_val_predictions is not None, (
                    'when passing init_predictions and eval_set, you should'
                    ' also pass init_val_predictions'
                )
                assert len(eval_set[0]) == len(init_val_predictions)
        else:
            init_predictions = None

        # RMSEWithUncertainty seems not to support baseline argument in catboost.Pool
        uncertainty = (
            self._catboost_kwargs.get('loss_function', None)
            == 'RMSEWithUncertainty'
        )

        if init_predictions is not None:
            if uncertainty:
                pool = catboost.Pool(
                    X, y - init_predictions, cat_features=cat_features
                )
            else:
                pool = catboost.Pool(
                    X,
                    y,
                    baseline=init_predictions,
                    cat_features=cat_features,
                )
            self._fitted_using_init_predictions = True
        else:
            pool = catboost.Pool(X, y, cat_features=cat_features)
            self._fitted_using_init_predictions = False

        if eval_set is not None:
            X_val, y_val = eval_set
            if init_predictions is not None:
                if uncertainty:
                    eval_pool = catboost.Pool(
                        X_val,
                        y_val - init_val_predictions,
                        cat_features=cat_features,
                    )
                else:
                    eval_pool = catboost.Pool(
                        X_val,
                        y_val,
                        baseline=init_val_predictions,
                        cat_features=cat_features,
                    )
            else:
                eval_pool = catboost.Pool(
                    X_val, y_val, cat_features=cat_features
                )
        else:
            eval_pool = None

        self._catboost_model.fit(  # type: ignore[attr-defined]
            pool,
            eval_set=eval_pool,
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=use_best_model and eval_set is not None,
            verbose=verbose,
        )

        return self

    def _process_predictions(
        self,
        predictions: np.ndarray,
        init_predictions: np.ndarray | None = None,
    ) -> np.ndarray:
        if (
            self._catboost_kwargs.get('loss_function', None)
            == 'RMSEWithUncertainty'
        ):
            predictions = predictions[:, 0]  # drop uncertainty
        if init_predictions is not None:
            predictions += init_predictions
        return predictions

    def predict(
        self,
        X: pd.DataFrame,
        ntree_start: int | None = 0,
        ntree_end: int | None = 0,  # if 0, use _catboost_model.tree_count_
        thread_count: int | None = -1,
        init_predictions: np.ndarray | None = None,
    ) -> np.ndarray:
        assert self._catboost_model is not None, 'fit model before predict'

        if self._init_model is not None:
            assert (
                init_predictions is None
            ), 'use either init model or init predictions'
            init_predictions = self._init_model.predict(X)
        elif self._fitted_using_init_predictions:
            assert init_predictions is not None, (
                'After fitting with init_predictions,'
                ' predict with init_predictions'
            )

        predictions: np.ndarray = self._catboost_model.predict(
            X,
            prediction_type='RawFormulaVal',
            ntree_start=ntree_start,
            ntree_end=ntree_end,
            thread_count=thread_count,
        )
        return self._process_predictions(predictions, init_predictions)

    def staged_predict(
        self,
        X: pd.DataFrame,
        eval_period: int = 1,
        ntree_start: int | None = 0,
        ntree_end: int | None = 0,  # if 0, use _catboost_model.tree_count_
        thread_count: int | None = -1,
        init_predictions: np.ndarray | None = None,
    ) -> Iterable[np.ndarray]:
        assert self._catboost_model is not None, 'fit model before predict'

        if self._init_model is not None:
            assert (
                init_predictions is None
            ), 'use either init model or init predictions'
            init_predictions = self._init_model.predict(X)
        elif self._fitted_using_init_predictions:
            assert init_predictions is not None, (
                'After fitting with init_predictions,'
                ' predict with init_predictions'
            )

        predictions: Iterable[
            np.ndarray
        ] = self._catboost_model.staged_predict(
            X,
            prediction_type='RawFormulaVal',
            ntree_start=ntree_start,
            ntree_end=ntree_end,
            thread_count=thread_count,
            eval_period=eval_period,
        )
        return (  # lazy map
            self._process_predictions(p, init_predictions) for p in predictions
        )

    @property
    def init_model(self) -> BaseModel | None:
        return self._init_model

    @property
    def catboost_model(self) -> catboost.CatBoost | None:
        return self._catboost_model

    @property
    def tree_count(self) -> int:
        assert self._catboost_model is not None, 'model is not fitted'
        return self._catboost_model.tree_count_

    @property
    def best_iteration(self) -> int:
        assert self._catboost_model is not None, 'model is not fitted'
        return self._catboost_model.best_iteration_


def recalc_catboost_best_iter(
    model: CatBoostModel,
    es_rounds: int,
    strict: bool = True,
) -> int:
    """Re-calculate best iteration with different value of early stopping rounds.

    If strict, will assert that early stopping condition was reached.
    """
    assert model.catboost_model is not None
    eval_metric = model.catboost_model.get_all_params()['eval_metric']
    val_scores = np.array(
        model.catboost_model.get_evals_result()['validation'][eval_metric]
    )
    if not HIGHER_IS_BETTER['catboost_' + eval_metric]:
        val_scores = -val_scores
    best_iter, es_reached = get_best_iter(val_scores, es_rounds)
    if strict:
        assert es_reached
    return best_iter
