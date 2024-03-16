import itertools
import re
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def identity(x):
    return x


def symlog(x):
    """https://arxiv.org/abs/2301.04104"""
    return np.sign(x) * np.log(np.abs(x) + 1)


def symexp(x):
    """https://arxiv.org/abs/2301.04104"""
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


def product_as_dict(**kwargs):
    """
    Example input:
    list(product_dict(
        target_scaling=['symlog', 'standard'],
        numerical_scaling=['any', 'robust'],
    ))

    Example output:
    [
        {'target_scaling': 'symlog', 'numerical_scaling': 'any'},
        {'target_scaling': 'symlog', 'numerical_scaling': 'robust'},
        {'target_scaling': 'standard', 'numerical_scaling': 'any'},
        {'target_scaling': 'standard', 'numerical_scaling': 'robust'}
    ]

    https://stackoverflow.com/questions/5228158
    /cartesian-product-of-a-dictionary-of-lists
    """
    for values in itertools.product(*kwargs.values()):
        yield dict(zip(kwargs.keys(), values))


def get_best_iter(
    scores: np.ndarray,
    early_stopping_rounds: int,
) -> tuple[int, bool]:
    """
    Drops np.nan from scores tail.
    Treat scores as higher is better.
    Returns best iter and flag if early stopping condition was reached
    (true if best_iter < len(scores) - early_stopping_rounds)
    """
    assert scores.ndim == 1
    arr_len = max(np.where(~np.isnan(scores))[0]) + 1
    scores = scores[:arr_len]
    best_iter = 0
    early_stopping_rounds_reached = False
    for step, score in enumerate(scores):
        if step > best_iter + early_stopping_rounds:
            early_stopping_rounds_reached = True
            break
        if score > scores[best_iter]:
            best_iter = step
    return best_iter, early_stopping_rounds_reached


def _validate_filename(
    filename: str,
    allow_slash: bool = False,
    allow_single_star: bool = False,
) -> bool:
    r"""
    Return False if disallowed symbols are found: ? \ : * < > |
    If allow_slash == False, also return False if slash "/" is found.
    If allow_single_star, allows single * but not **.

    Useful to check filename before constructing Path.
    """
    has_slash = '/' in filename
    has_star = '*' in filename
    has_double_star = '**' in filename
    has_other_bad_symbols = len(re.findall(r'[\?\\:<>|"]', filename)) > 0
    is_point = filename in {'.', '..'}

    if is_point or has_other_bad_symbols or has_double_star:
        return False
    if has_slash and not allow_slash:
        return False
    if has_star and not allow_single_star:
        return False

    return True


def display_full_dataframe(df: pd.DataFrame) -> None:
    with pd.option_context(
        'display.max_rows',
        None,
        'display.max_columns',
        None,
        'display.float_format',
        '{:g}'.format,
    ):
        from IPython.display import display

        display(df)


def sparse_to_dense_with_copy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.SparseDtype):
            df[col] = df[col].sparse.to_dense()
    return df


# def query_df(
#     df: pd.DataFrame | pl.DataFrame, **cols_and_values: dict[str, Any]
# ) -> pd.DataFrame:
#     if isinstance(df, pd.DataFrame):
#         for col, value in cols_and_values.items():
#             df = df[df[col] == value]
#         return df
#     elif isinstance(df, pl.DataFrame):
#         return df.filter(**cols_and_values)


# def query_df_row(
#     df: pd.DataFrame | pl.DataFrame, **cols_and_values: dict[str, Any]
# ) -> pd.Series:
#     df = query_df(df, **cols_and_values)
#     if len(df) == 0:
#         return None
#     elif len(df) == 1:
#         if isinstance(df, pd.DataFrame):
#             return df.iloc[0]
#         elif isinstance(df, pl.DataFrame):
#             return df.to_pandas().iloc[0]
#     else:
#         raise AssertionError(f'{len(df)} rows found')


class Timer:
    """
    Example:
        >>> from tabular_models import Timer
        >>> with Timer() as timer:
        >>>     do_something()
        >>> print(timer.duration)

    Timer is reusable:
        >>> from tabular_models import Timer
        >>> import time
        >>> timer = Timer()
        >>> with timer:
        >>>     time.sleep(1)
        >>> print(timer.duration)
        1.0010457038879395
        >>> with timer:
        >>>     time.sleep(1)
        >>> print(timer.duration)
        2.002077579498291
        >>> print(timer.last_duration)
        1.0010318756103516
    """

    def __init__(self):
        self.duration = 0
        self.last_duration = 0
        self.last_start = -1
        self.last_stop = -1
        self.started = False

    def __enter__(self):
        assert not self.started
        self.started = True
        self.last_start = time.time()
        return self

    def __exit__(self, *args):
        self.last_stop = time.time()
        self.last_duration = self.last_stop - self.last_start
        self.duration += self.last_duration
        assert self.started
        self.started = False


def visualize_feature_matrix(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    separate_last: bool = True,
):
    ax = ax or plt.gca()
    n_samples, n_features = df.shape
    ticks, labels = [], []
    for i in range(n_features):
        column = df.iloc[:, i]
        extent = (i - 0.5, i + 0.5, n_samples - 0.5, -0.5)
        if column.dtype != 'category':
            values = column.to_numpy()
            # numerical
            vmin, vmax = np.nanquantile(values, (0.05, 0.95))
            if vmax == vmin:
                vmin, vmax = np.nanmin(values), np.nanmax(values)
            if vmax == vmin:
                vmin, vmax = 0, 1
            values = np.clip((values - vmin) / (vmax - vmin), 0, 1)
            cmap = 'viridis'
        else:
            # categorical
            values = column.cat.codes.to_numpy()
            values = values / values.max()
            cmap = 'cool'
            ticks.append(i)
            labels.append(len(column.cat.categories))
        ax.imshow(
            values[:, None],
            aspect='auto',
            interpolation='none',
            vmin=0,
            vmax=1,
            extent=extent,
            cmap=cmap,
        )
    ax.set_xlim(-0.5, n_features - 0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)   # , rotation=90
    if separate_last:
        ax.axvline(x=n_features - 1.5, color='r')
