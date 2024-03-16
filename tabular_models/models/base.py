from abc import ABC, abstractmethod

import pandas as pd
from typing_extensions import Self

from ..task_type import TaskType


class BaseModel(ABC):
    """
    An abstract fit-predict model that works with pandas dataframe.
    """

    def __init__(
        self,
        task_type: TaskType,
        random_seed: int = 0,
    ):
        self._task_type = task_type
        self._random_seed = random_seed

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Self:
        """
        X: DataFrame (categorical columns for categorical features)
        y: Series (numerical or categorical, depending on task_type in constructor)

        If model may be fitted iteratively, param conventions are the following:
            early_stopping_rounds: int = 50,
            eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
            max_iterations: int | None = None,
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """
        X: DataFrame (categorical columns for categorical features)

        Should return the following:
        For regression: predictions of size (n_samples,)
        For binary: logits of size (n_samples,)
        For multiclass: logits of size (n_samples, n_classes)
        """
        pass
