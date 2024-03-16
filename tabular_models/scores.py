from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt

from tabular_models.task_type import TaskType


@dataclass
class Scores:
    scores_df: pl.DataFrame
    task_type: TaskType | None = None

    def _resolve_metric(self, metric: str | dict[TaskType, str]) -> str:
        if isinstance(metric, dict):
            assert self.task_type is not None
            return metric[self.task_type]
        else:
            return metric

    def get(
        self,
        subset: str,
        metric: str | dict[TaskType, str],
        bootstrap_seed: int | None = None,
    ) -> float:
        filtered = self.scores_df.filter(
            subset=subset,
            metric=self._resolve_metric(metric),
            bootstrap_seed=bootstrap_seed,
        )
        assert len(filtered) == 1, 'Requested metric not found or duplicate'
        return filtered['score'][0]

    def get_all_metrics_as_dict(self, subset: str) -> dict[str, float]:
        filtered = self.scores_df.filter(subset=subset, bootstrap_seed=None)
        return dict(
            zip(
                filtered['metric'].to_list(),
                filtered['score'].to_list(),
            )
        )

    def bootstraps(
        self,
        subset: str,
        metric: str | dict[TaskType, str],
    ) -> np.ndarray:
        filtered = self.scores_df.filter(
            pl.col('bootstrap_seed').is_not_null(),
            subset=subset,
            metric=self._resolve_metric(metric),
        )
        assert len(filtered) > 0
        scores = filtered['score'].to_numpy()
        return scores

    def bootstrap_quantiles(
        self,
        subset: str,
        metric: str | dict[TaskType, str],
        quantiles: float | Sequence[float] = (0.1, 0.9),
    ) -> np.ndarray | float:
        bootstraps = self.bootstraps(subset, metric)
        return np.quantile(bootstraps, quantiles)

    @classmethod
    def concat(cls, scores: Iterable[Scores]) -> Scores:
        scores = list(scores)
        assert len({s.task_type for s in scores}) == 1
        score_dfs = [s.scores_df for s in scores]
        return Scores(pl.concat(score_dfs), scores[0].task_type)

    def __sub__(self, other: Scores) -> ScoreDeltas:
        return ScoreDeltas(self, other, strict=True)

    def __rsub__(self, other) -> float | None:
        if other is np.nan:
            return np.nan
        elif other is None:
            return None
        else:
            raise AssertionError()

    def __str__(self) -> str:
        return f'Scores(size={len(self.scores_df)})'


class ScoreDeltas(Scores):
    def __init__(self, scores1: Scores, scores2: Scores, strict: bool = True):
        df1 = scores1.scores_df
        df2 = scores2.scores_df
        df_deltas = (
            df1.join(
                df2, on=('subset', 'metric', 'bootstrap_seed'), how='outer'
            )
            .with_columns(score=pl.col('score') - pl.col('score_right'))
            .drop('score_right')
        )

        if strict:
            assert len(df_deltas) == len(df1) == len(df2)
            assert scores1.task_type == scores2.task_type

        self.scores_df = df_deltas

        if scores1.task_type != scores2.task_type:
            if strict:
                raise AssertionError()
        else:
            self.task_type = scores1.task_type


np_nan = float


def bootstrap_matrix(
    scores: Sequence[Scores | None | np_nan],
    subset: str,
    metric: str | dict[TaskType, str],
) -> np.ndarray | None:
    notnull_scores: Sequence[Scores] = [
        s for s in scores if isinstance(s, Scores)
    ]
    if len(notnull_scores) == 0:
        return None
    metric = notnull_scores[0]._resolve_metric(metric)

    filtered = [
        s.scores_df.filter(
            pl.col('bootstrap_seed').is_not_null(),
            subset=subset,
            metric=metric,
        )
        if isinstance(s, Scores)
        else np.nan
        for s in scores
    ]

    bootstrap_seeds = filtered[0]['bootstrap_seed'].to_numpy()
    for f in filtered[1:]:
        if f is not np.nan:
            assert (f['bootstrap_seed'].to_numpy() == bootstrap_seeds).all()

    matrix = np.array(
        [
            (
                f['score'].to_numpy()
                if f is not np.nan
                else np.full(len(bootstrap_seeds), np.nan)
            )
            for f in filtered
        ]
    )

    return matrix


def bootstrap_gain(
    matrix: np.ndarray | None,
    folds: Sequence[int] | None = None,
) -> float:
    if matrix is None:
        return np.nan

    if folds is not None:
        try:
            matrix = matrix[np.array(folds)]
        except IndexError:
            return np.nan

    if np.isnan(matrix).any():
        return np.nan

    positive_ratio = (matrix > 0).mean()
    negative_ratio = (matrix < 0).mean()

    return positive_ratio - negative_ratio


def _shorten_str(s: str, max_size: int = 30) -> str:
    if len(s) < max_size:
        return s
    else:
        left = max_size // 2
        right = max_size - left - 3
        return s[:left] + '...' + s[-right:]


def gains_summary_plot(
    matrices: pd.Series,
    ax: plt.Axes | None = None,
    fontsize: float = 8,
):
    """
    Matrices: dataset_name -> bootstrap matrix (n_folds, b_bootstraps)
    """
    ax = ax or plt.gca()

    ax = ax or plt.gca()
    matrices = matrices.sort_values(ascending=True)
    matrices = matrices[~matrices.isna()]

    ax.barh(
        [_shorten_str(x) for x in matrices.index.to_list()],
        matrices.to_numpy(),
    )
    ax.axvline(0, ls='dashed')
    ax.axvline(-1, ls='dashed')
    ax.axvline(1, ls='dashed')
    ax.tick_params(axis='y', which='major', labelsize=fontsize)


def plot_bootstrap_matrix(
    matrix: np.ndarray,
    ax: plt.Axes | None = None,
    color: str = 'C0',
    s: float = 10,
    y_range: tuple[float, float] = (0, 1),
):
    ax = ax or plt.gca()
    n_folds, n_bootstraps = matrix.shape

    def linear_projection(
        x: np.ndarray,
        from_range: tuple[float, float],
        to_range: tuple[float, float],
    ) -> np.ndarray:
        """
        linear transform such that it maps `from_range` interval to `to_range` interval
        """
        w = (to_range[1] - to_range[0]) / (from_range[1] - from_range[0])
        b = to_range[0] - w * from_range[0]
        return w * x + b

    for fold in range(n_folds):
        plt.scatter(
            matrix[fold],
            linear_projection(
                fold + np.random.uniform(-0.2, 0.2, size=n_bootstraps),
                from_range=(-1, n_folds),
                to_range=y_range,
            ),
            color=color,
            s=s,
        )
