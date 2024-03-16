from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from .task_type import TaskType


@dataclass
class Dataset:
    """
    Stores the dataset with the following agreements:

    - Numerical columns have numerical dtype.
    - Categorical columns have categorical type and .cat.categories contains a full
      list of categories. If ordered, column should have "ordered" flag.
    - Binary columns may be stored as numerical or categorical.
    - Target (y) is categorical column for TaskType.BINARY and TaskType.MULTICLASS,
      numerical column for TaskType.REGRESSION. For TaskType.BINARY,
     len(y.cat.categories) should equal 2.

    This class does not specify the desired metric or loss function.

    Metadata may contain name (str) and fold (int).

    The .iloc field is responsible for the division into subsets. Its keys are subset
    names, and values are lists of row indices in X and y (in Pandas, they represent
    "iloc" but not "loc" indices over X and y).

    Typically, a dataset contains 'trainval' and 'test' subsets, which do not
    intersect and their union is the whole dataset. Further, DatasetSplitter may be
    used to create new subsets 'train' and 'val' which do not intersect and their union
    is 'trainval'.
        >>> data_store = DataStore('/data/tabular_data')
        >>> dataset = data_store.load_dataset('eucalyptus', 0)
        >>> dataset = DatasetSplitter(method='holdout', val_ratio=0.1).split(dataset)
        >>> assert len(dataset.X_trainval) == len(dataset.X_train) + len(dataset.X_val)

    For the typical subsets {'train', 'val', 'trainval', 'test'} Dataset class has
    built-in accessors like .X_val, y_train etc. In general case, Dataset may contain
    any subsets. These operations are equal:
        >>> dataset.X_train
        >>> dataset.X.iloc[dataset.iloc.train]
        >>> dataset.X.iloc[dataset.iloc['train']]

    Dataset class also have accessors .y_numerical, .y_train_numerical etc. They
    return y.cat.codes for the given subset if the target is categorical, and
    plain y for the given subset otherwise. This may be useful for classification
    models that cannot be trained on string categories as target.

    The subsets (the .iloc field) may be used when constructing Predictions, this
    allows to perform model.predict() once over the whole dataset.X and then calculate
    metrics over each subset:
        >>> model = CatBoostModel(task_type=dataset.task_type)
        >>> model.fit(
        >>>     dataset.X_train,
        >>>     dataset.y_train,
        >>>     eval_set=(dataset.X_val, dataset.y_val),
        >>>     early_stopping_rounds=10
        >>> )
        >>> preds = Predictions(
        >>>     model.predict(dataset.X),
        >>>     dataset.y,
        >>>     dataset.task_type,
        >>>     iloc=dataset.iloc
        >>> )
        >>> preds.train.score(), preds.val.score(), preds.test.score()
    """

    X: pd.DataFrame
    y: pd.Series
    task_type: TaskType
    iloc: Bunch = field(default_factory=Bunch)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def Xy(self) -> pd.DataFrame:
        return pd.concat([self.X, self.y.rename('TARGET')], axis='columns')

    def __post_init__(self):
        if self.iloc.__class__ == dict:
            self.iloc = Bunch(**self.iloc)

    def __str__(self) -> str:
        result = 'Dataset:'
        result += '\n  Properties:'
        result += f'\n    n_features: {self.X.shape[1]}'
        result += f'\n    task_type: {self.task_type}'
        result += '\n  Subsets:'
        for k, v in self.iloc.items():
            result += f'\n    {k}: {len(v)} samples'
        result += '\n  Metadata:'
        for k, v in self.metadata.items():
            result += f'\n    {k}: {v}'
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def _to_numerical(self, series: pd.Series) -> pd.Series:
        if series.dtype == 'category':
            return series.cat.codes
        return series

    def get_subset(self, name: str) -> Dataset:
        return self.get_subset_by_iloc(list(self.iloc[name]))

    # def exclude_subset(
    #     self, name: str, copy: bool = False
    # ) -> tuple[Dataset, list[int]]:
    #     retained_iloc = list(set(range(len(self.X))) - set(self.iloc[name]))
    #     return self.get_subset_by_iloc(retained_iloc, copy=copy), retained_iloc

    def get_subset_by_iloc(self, subset_iloc: list[int]) -> Dataset:
        return Dataset(
            X=self.X.iloc[subset_iloc],
            y=self.y.iloc[subset_iloc],
            task_type=self.task_type,
            iloc=Bunch(),
            metadata={},
        )

    def set_fold(self, fold: int) -> None:
        is_test = self.metadata['fold_indices'] == fold
        train_indices = np.where(~is_test)[0]
        test_indices = np.where(is_test)[0]
        self.iloc = Bunch(
            train=train_indices,
            trainval=train_indices,
            test=test_indices,
        )

        if self.metadata.get('reversed', False):
            self.iloc = Bunch(
                train=self.iloc.test,
                trainval=self.iloc.test,
                test=self.iloc.trainval,
            )

    # bug: when converting to mask, the index order is lost. So,
    # the results will be different:
    # 1) dataset.y.iloc[dataset.iloc.train]
    # 2) mask_to_retain = np.full(len(dataset.X), False)
    #    mask_to_retain[dataset.iloc.train] = True
    #    dataset.y[mask_to_retain]
    # def get_subset_by_iloc(
    #     self, subset_iloc: list[int], copy: bool = False
    # ) -> Dataset:
    #     """
    #     Will de-duplicate if some subsets contain duplicate indices, since
    #     internally converts ilocs into masks (no warning raised)
    #     """

    #     existing_subset_masks = {}
    #     for subset_name, iloc in dict(self.iloc).items():
    #         mask = np.full(len(self.X), False)
    #         mask[iloc] = True
    #         existing_subset_masks[subset_name] = mask
    #     mask_to_retain = np.full(len(self.X), False)
    #     mask_to_retain[subset_iloc] = True

    #     new_X = self.X[mask_to_retain]
    #     new_y = self.y[mask_to_retain]
    #     if copy:
    #         new_X = new_X.copy()
    #         new_y = new_y.copy()

    #     new_ilocs = Bunch()
    #     for subset_name, original_subset_mask in existing_subset_masks.items():
    #         tmp = original_subset_mask.copy()
    #         tmp[mask_to_retain] = False
    #         if tmp.any():
    #             continue  # this subset is not fully present in new dataset
    #         new_mask = original_subset_mask[subset_iloc]
    #         new_iloc = np.where(new_mask)[0]
    #         new_ilocs[subset_name] = new_iloc

    #     return Dataset(
    #         X=new_X,
    #         y=new_y,
    #         task_type=self.task_type,
    #         iloc=new_ilocs,
    #         metadata={},
    #     )

    @property
    def train(self) -> Dataset:
        return self.get_subset('train')

    @property
    def val(self) -> Dataset:
        return self.get_subset('val')

    @property
    def trainval(self) -> Dataset:
        return self.get_subset('trainval')

    @property
    def test(self) -> Dataset:
        return self.get_subset('test')

    @property
    def y_numerical(self) -> pd.Series:
        return self._to_numerical(self.y)

    # TRAINVAL
    @property
    def X_trainval(self) -> pd.DataFrame:
        return self.X.iloc[self.iloc.trainval]

    @property
    def y_trainval(self) -> pd.Series:
        return self.y.iloc[self.iloc.trainval]

    @property
    def y_trainval_numerical(self) -> pd.Series:
        return self._to_numerical(self.y_trainval)

    # TEST
    @property
    def X_test(self) -> pd.DataFrame:
        return self.X.iloc[self.iloc.test]

    @property
    def y_test(self) -> pd.Series:
        return self.y.iloc[self.iloc.test]

    @property
    def y_test_numerical(self) -> pd.Series:
        return self._to_numerical(self.y_test)

    # TRAIN
    @property
    def X_train(self) -> pd.DataFrame:
        return self.X.iloc[self.iloc.train]

    @property
    def y_train(self) -> pd.Series:
        return self.y.iloc[self.iloc.train]

    @property
    def y_train_numerical(self) -> pd.Series:
        return self._to_numerical(self.y_train)

    # VAL
    @property
    def X_val(self) -> pd.DataFrame:
        return self.X.iloc[self.iloc.val]

    @property
    def y_val(self) -> pd.Series:
        return self.y.iloc[self.iloc.val]

    @property
    def y_val_numerical(self) -> pd.Series:
        return self._to_numerical(self.y_val)

    @staticmethod
    def _fill_missing_categorical(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            if X[col].dtype == 'category':
                existing_categories = X[col].cat.categories
                category_to_fill: str | None = None
                for possible_name in [
                    'missing',
                    'unknown',
                    'none',
                ]:   # , 'other'
                    for cat in existing_categories:
                        if possible_name.lower() == cat.lower():
                            category_to_fill = cat
                            break
                    if category_to_fill is not None:
                        break
                if category_to_fill is None:
                    X[col] = X[col].cat.add_categories('missing')
                    category_to_fill = 'missing'
                X[col] = X[col].fillna(category_to_fill)
        return X

    def fill_missing_categorical(self) -> Dataset:
        return Dataset(
            X=self._fill_missing_categorical(self.X),
            y=self.y.copy(),
            task_type=self.task_type,
            iloc=self.iloc.copy(),
            metadata=self.metadata.copy(),
        )

    @property
    def num_features(self) -> list[str]:
        return [
            col for col in self.X.columns if self.X[col].dtype != 'category'
        ]

    @property
    def cat_features(self) -> list[str]:
        return [
            col for col in self.X.columns if self.X[col].dtype == 'category'
        ]
