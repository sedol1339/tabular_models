from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
from scipy.special import expit, softmax
from scipy.stats import percentileofscore
from sklearn.metrics import (
    balanced_accuracy_score,
    d2_absolute_error_score,
    log_loss,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.utils import Bunch
from sklearn.utils._testing import ignore_warnings

if TYPE_CHECKING:
    from tabular_models.dataset import Dataset

from tabular_models.utils import symlog

from .scores import Scores
from .task_type import TaskType

HIGHER_IS_BETTER = {
    'r2': True,
    'log_loss': False,
    'neg_log_loss': True,
    'roc_auc': True,
    'balanced_accuracy': True,
    'median_absolute_error': False,
    'neg_median_absolute_error': True,
    'quantile_mse_d2': True,
    'd2_absolute_error': True,
    'catboost_RMSE': False,
    'catboost_RMSEWithUncertainty': False,
    'catboost_Logloss': False,
    'catboost_MultiClass': False,
    'catboost_AUC': True,
}


@dataclass
class Predictions:
    """
    Stores predictions and ground truth. Also may store field .t which means time spent
    for inference. In all cases, .iloc but not .loc is used to match index in
    predictions and in truth.

    - TaskType.REGRESSION
        - requires target Series be numerical
        - predictions should be in shape (n_samples,)
    - TaskType.BINARY
        - requires target Series be categorical when len(truth.cat.categories) == 2
        - predictions should be logits for class#1 in shape (n_samples,), where class#1
          is truth.cat.categories[1]
    - TaskType.MULTICLASS
        - requires target Series be categorical
        - predictions should be logits for class#0, ..., class#(N-1) in shape
          (n_samples, len(truth.cat.categories)).

    Predictions may store predictions for different subsets, in this case .iloc field
    should contain mapping from subset name to list of indices. Typical subset names
    are 'train', 'val', 'trainval', 'test' (same as in Dataset, see Dataset docstring).
    """

    predictions: np.ndarray
    truth: pd.Series
    task_type: TaskType
    iloc: Bunch = field(default_factory=Bunch)
    t: float | None = None

    def __init__(
        self,
        predictions: np.ndarray,
        truth: pd.Series | None = None,
        task_type: TaskType | None = None,
        iloc: Bunch | dict | None = None,
        t: float | None = None,
        dataset: Dataset | None = None,
    ):
        # convert fields
        if iloc.__class__ == dict:
            iloc = Bunch(**iloc)

        # if dataset passed, extract information from it
        if dataset is not None:
            assert truth is None
            assert task_type is None
            assert iloc is None
            truth = dataset.y
            task_type = dataset.task_type
            iloc = dataset.iloc.copy() if dataset.iloc is not None else None
        else:
            assert truth is not None
            assert task_type is not None
            iloc = iloc or Bunch()

        # validate fields
        if task_type == TaskType.REGRESSION:
            assert predictions.shape == (len(truth),)
            assert truth.dtype != 'category'
        elif task_type == TaskType.BINARY:
            n_classes = len(truth.cat.categories)
            assert n_classes == 2
            assert predictions.shape == (len(truth),)
            assert truth.dtype == 'category'
        elif task_type == TaskType.MULTICLASS:
            n_classes = len(truth.cat.categories)
            assert n_classes > 2
            assert predictions.shape == (len(truth), n_classes)
            assert truth.dtype == 'category'

        # write fields
        self.predictions = predictions
        self.truth = truth
        self.task_type = task_type
        self.iloc = iloc
        self.t = t

    def get_subset(self, subset: str | np.ndarray | list) -> Predictions:
        if isinstance(subset, str):
            iloc = self.iloc[subset]
        else:
            iloc = subset
        # drop iloc and t
        return Predictions(
            self.predictions[iloc], self.truth.iloc[iloc], self.task_type
        )

    def __str__(self) -> str:
        t_str = 'None' if self.t is None else f'{self.t:g}'
        return (
            f'Predictions({self.task_type}, size={len(self.truth)}, t={t_str})'
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def trainval(self) -> Predictions:
        return self.get_subset('trainval')

    @property
    def train(self) -> Predictions:
        return self.get_subset('train')

    @property
    def val(self) -> Predictions:
        return self.get_subset('val')

    @property
    def test(self) -> Predictions:
        return self.get_subset('test')

    @property
    def logits(self) -> np.ndarray:
        assert self.task_type in (TaskType.BINARY, TaskType.MULTICLASS)
        return self.predictions

    @staticmethod
    def logits_to_probabilities(
        logits: np.ndarray,
        task_type: TaskType,
    ) -> np.ndarray:
        if task_type == TaskType.BINARY:
            return expit(logits)
        elif task_type == TaskType.MULTICLASS:
            return softmax(logits, axis=-1)
        else:
            raise AssertionError(
                'probabilities accessible for binary or multiclass'
            )

    @property
    def probabilities(self) -> np.ndarray:
        return self.logits_to_probabilities(self.predictions, self.task_type)

    def score_all(
        self,
        metrics: Sequence[str],
        subsets: Sequence[str | None],
        bootstrap_seeds: Sequence[int | None] | None = None,
    ) -> Scores:
        scores = []
        for metric in metrics:
            for subset in subsets:
                for bootstrap_seed in bootstrap_seeds or [None]:
                    scores.append(
                        {
                            'subset': subset,
                            'metric': metric,
                            'bootstrap_seed': bootstrap_seed,
                            'score': self.score(
                                metric=metric,
                                subset=subset,
                                bootstrap_seed=bootstrap_seed,
                            ),
                        }
                    )

        return Scores(scores_df=pl.DataFrame(scores), task_type=self.task_type)

    @ignore_warnings(category=UserWarning)
    def score(
        self,
        metric: str,
        subset: str | None = None,
        mask: np.ndarray | None = None,
        iloc: np.ndarray | None = None,
        bootstrap_seed: int | None = None,
    ) -> float:
        """
        Calculates score.

        Optionally, may calculate score for a specified subset.

        Optionally, may calculate score for a specified mask of iloc (list of indices),
        where mask of iloc are applied after obtaining a subset, so are relative to
        the specified subset. This may help to calculate bootstrap scores.

        If bootstrap_seed is not None, "iloc" param will be set to a sample of
        N samples from N with repeat, where N is size of the specified subset.
        """
        # getting preds and truth
        preds = self.predictions
        if self.task_type == TaskType.REGRESSION:
            truth = self.truth.to_numpy()
        elif self.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
            truth = self.truth.cat.codes.to_numpy()
        if subset is not None:
            preds = preds[self.iloc[subset]]
            truth = truth[self.iloc[subset]]

        if bootstrap_seed is not None:
            # bootstrapping
            assert mask is None, 'use either mask or bootstrap_seed'
            assert iloc is None, 'use either iloc or bootstrap_seed'
            rng = np.random.default_rng(bootstrap_seed)
            iloc = rng.choice(len(preds), size=len(preds), replace=True)
            preds = preds[iloc]
            truth = truth[iloc]
        elif mask is not None:
            # masking
            assert iloc is None, 'use either iloc or mask'
            preds = preds[mask]
            truth = truth[mask]
        elif iloc is not None:
            # indexing
            preds = preds[iloc]
            truth = truth[iloc]

        # calculating
        if metric == 'r2':
            assert self.task_type == TaskType.REGRESSION
            try:
                score = r2_score(truth, preds)
            except ValueError:   # contains inf
                score = np.nan
        elif metric == 'd2_absolute_error':
            assert self.task_type == TaskType.REGRESSION
            try:
                score = d2_absolute_error_score(truth, preds)
            except ValueError:   # contains inf
                score = np.nan
        elif metric == 'r2_symlog':
            assert self.task_type == TaskType.REGRESSION
            try:
                score = r2_score(symlog(truth), symlog(preds))
            except ValueError:   # contains inf
                score = np.nan
        elif metric == 'roc_auc':
            assert self.task_type == TaskType.BINARY
            try:
                score = roc_auc_score(truth, np.clip(preds, -1e10, 1e10))
            except ValueError:
                # Only one class present in y_true.
                # ROC AUC score is not defined in that case.
                score = np.nan
        elif metric == 'balanced_accuracy':
            assert self.task_type in (TaskType.BINARY, TaskType.MULTICLASS)
            if len(np.unique(truth)) == len(self.truth.cat.categories):
                if self.task_type == TaskType.MULTICLASS:
                    score = balanced_accuracy_score(
                        truth, preds.argmax(axis=1)
                    )
                else:
                    score = balanced_accuracy_score(
                        truth, (preds > 0).astype(int)
                    )
            else:
                score = np.nan
        elif metric in {'log_loss', 'neg_log_loss'}:
            assert self.task_type in (TaskType.BINARY, TaskType.MULTICLASS)
            score = log_loss(
                truth,
                Predictions.logits_to_probabilities(preds, self.task_type),
                labels=list(range(len(self.truth.cat.categories))),
            )
            return score if metric == 'log_loss' else -score
        elif metric in {'median_absolute_error', 'neg_median_absolute_error'}:
            assert self.task_type == TaskType.REGRESSION
            score = median_absolute_error(truth, preds)
            return score if metric == 'median_absolute_error' else -score
        elif metric == 'quantile_mse_d2':
            pred_quantiles = percentileofscore(truth, preds, 'mean') / 100
            true_quantiles = percentileofscore(truth, truth, 'mean') / 100
            score = r2_score(true_quantiles, pred_quantiles)
        else:
            raise AssertionError(f'unknown metric "{metric}"')

        return float(score)

    @classmethod
    def average(self, predictions: list[Predictions]) -> Predictions:
        """
        Constructs new Predictions object by taking mean of the specified predictions.

        In all predictions, truth, task_type and iloc are assumed to be the equal (this
        is not checked).
        """
        truth = predictions[0].truth
        task_type = predictions[0].task_type
        iloc = predictions[0].iloc
        values = np.mean([x.predictions for x in predictions], axis=0)
        t_sum: float | None = 0
        for x in predictions:
            if x.t is None:
                t_sum = None
                break
            t_sum += x.t  # type: ignore[operator]
        return Predictions(values, truth, task_type, iloc, t_sum)

    @classmethod
    def cumulative_average(
        self, predictions: list[Predictions]
    ) -> list[Predictions]:
        """
        Constructs list new Predictions objects by taking cumulative mean of the
        specified predictions. This means that given a sequence [p1, p2, p3], the
        results will be [p1, (p1+p2)/2, (p1+p2+p3)/3].

        In all predictions, truth, task_type and iloc are assumed to be the equal (this
        is not checked).
        """
        truth = predictions[0].truth
        task_type = predictions[0].task_type
        iloc = predictions[0].iloc
        partial_sums = np.cumsum([x.predictions for x in predictions], axis=0)
        t_sum = np.cumsum(
            [(x.t if x.t is not None else np.nan) for x in predictions]
        )
        t_sum = [(x if not np.isnan(x) else None) for x in t_sum]
        return [
            Predictions(
                partial_sums[i] / (i + 1), truth, task_type, iloc, t_sum[i]
            )
            for i in range(len(predictions))
        ]
