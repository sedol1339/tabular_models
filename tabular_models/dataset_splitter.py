from __future__ import annotations

import re
from typing import Literal

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils._testing import ignore_warnings

from .dataset import Dataset
from .task_type import TaskType


class DatasetSplitter:
    """
     A class for splitting datasets. Splits from_subset to mutually exclusive subsets
     to_subsets[0] and to_subsets[1]. Example:
         >>> data_store = DataStore('/data/tabular_data')
         >>> dataset = data_store.load_dataset('eucalyptus', 0)
         >>> dataset = DatasetSplitter(method='holdout', val_ratio=0.1).split(dataset)
         >>> assert len(dataset.X_trainval) == len(dataset.X_train) + len(dataset.X_val)

    When using holdout, val_ratio is responsible for the relative size of to_subsets[1],
    which is typically named "val". Note that increasing val_ratio while keeping
    random_seed fixed will result in val set monotonically grow.

    When specifying method "cv" (cross-validation), k-fold is used. n_folds means number
    of folds, fold_idx means index of fold.

    When specifying method "none", does nothing.

    When stratified=True, for classification both holdout and cv methods generate
    stratified splits.

    Parameter ensure_all_classes in classification works the following way. When using
    holdout method, and at least one sample for each class is added to train split.
    If there is only one sample for some class, it is added to train, not to val. This
    may lead to val set being smaller than than specified in val_ratio. When using cv
    method, train set is guaranteed to contain samples for all classes, this requires
    >=2 samples for each class. If this condition is not met, an exception is thrown.
    """

    def __init__(
        self,
        method: Literal['holdout', 'cv', 'none'],
        from_subset: str = 'trainval',
        to_subsets: tuple[str | None, str | None] = ('train', 'val'),
        val_ratio: float | None = None,
        n_folds: int | None = None,
        fold_idx: int | None = None,
        random_seed: int = 0,
        stratified: bool = True,
        ensure_all_classes: bool = True,
    ):
        self._from_subset = from_subset
        self._to_subsets = to_subsets
        self._method = method
        if method == 'holdout':
            assert val_ratio is not None
            assert n_folds is None
            assert fold_idx is None
            self._val_ratio = val_ratio
        elif method == 'cv':
            assert val_ratio is None
            assert n_folds is not None
            assert fold_idx is not None
            self._n_folds = n_folds
            self._fold_idx = fold_idx
        elif method == 'none':
            assert val_ratio is None
            assert n_folds is None
            assert fold_idx is None
        else:
            raise AssertionError(f'Unknown method {method}')
        self._random_seed = random_seed
        self._stratified = stratified
        self._ensure_all_classes = ensure_all_classes

    def __str__(self) -> str:
        if self._method == 'holdout':
            a, b = self._val_ratio, self._random_seed
            return f'holdout{a}_seed{b}'
        elif self._method == 'cv':
            a, b, c = self._n_folds, self._random_seed, self._fold_idx
            return f'cv{a}_seed{b}_fold{c}'
        if self._method == 'none':
            return 'none'

    def __repr__(self) -> str:
        return f'DatasetSplitter[{self.__str__()}]'

    @classmethod
    def from_string(cls, string: str) -> DatasetSplitter:
        _float = r'-?([0-9\.]+)'
        _int = r'-?([0-9]+)'
        if match := re.fullmatch(rf'holdout{_float}_seed{_int}', string):
            val_ratio, random_seed = match.groups()
            return DatasetSplitter(
                method='holdout',
                val_ratio=float(val_ratio),
                random_seed=int(random_seed),
            )
        elif match := re.fullmatch(rf'cv{_int}_seed{_int}_fold{_int}', string):
            n_folds, random_seed, fold_idx = match.groups()
            return DatasetSplitter(
                method='cv',
                n_folds=int(n_folds),
                random_seed=int(random_seed),
                fold_idx=int(fold_idx),
            )
        elif string == 'none':
            return DatasetSplitter(method='none')
        else:
            raise AssertionError(
                f'cannot instantiate DatasetSplitter from {string}'
            )

    @ignore_warnings(category=UserWarning)
    def split(self, dataset: Dataset) -> Dataset:
        if self._method == 'none':
            return Dataset(
                X=dataset.X,
                y=dataset.y,
                iloc={**dataset.iloc},
                task_type=dataset.task_type,
                metadata=dataset.metadata,
            )

        assert (
            self._from_subset in dataset.iloc
        ), f'{self._from_subset} should exist in dataset'

        if dataset.task_type == TaskType.REGRESSION:
            stratify = False
        elif dataset.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
            stratify = self._stratified

        if self._method == 'holdout':
            if not stratify:
                iloc0, iloc1 = train_test_split(
                    dataset.iloc[self._from_subset],
                    test_size=self._val_ratio,
                    random_state=self._random_seed,
                    shuffle=True,
                )
            else:
                iloc0, iloc1, _, _ = train_test_split(
                    dataset.iloc[self._from_subset],
                    dataset.y.iloc[dataset.iloc[self._from_subset]].values,
                    test_size=self._val_ratio,
                    random_state=self._random_seed,
                    shuffle=True,
                )
            iloc0 = list(iloc0)
            iloc1 = list(iloc1)
            if self._ensure_all_classes and dataset.task_type in (
                TaskType.BINARY,
                TaskType.MULTICLASS,
            ):
                y0, y1 = dataset.y.iloc[iloc0], dataset.y.iloc[iloc1]
                indices_from_1_to_0 = []
                for label in set(y1.unique()).difference(set(y0.unique())):
                    # label found in iloc1 but not in iloc0
                    indices_from_1_to_0.append(
                        iloc1[np.where(y1 == label)[0][0]]
                    )
                iloc0 = iloc0 + indices_from_1_to_0
                for index in indices_from_1_to_0:
                    iloc1.remove(index)

        elif self._method == 'cv':
            if not stratify:
                kfold_iterator = KFold(
                    n_splits=self._n_folds,
                    random_state=self._random_seed,
                    shuffle=True,
                ).split(dataset.iloc[self._from_subset])
            else:
                kfold_iterator = StratifiedKFold(
                    n_splits=self._n_folds,
                    random_state=self._random_seed,
                    shuffle=True,
                ).split(
                    dataset.iloc[self._from_subset],
                    dataset.y.iloc[dataset.iloc[self._from_subset]].values,
                )

            iloc_for_all_folds = np.array(dataset.iloc[self._from_subset])
            iloc_for_each_fold = []
            for _, indices_in_iloc in kfold_iterator:
                iloc = iloc_for_all_folds[indices_in_iloc]
                iloc_for_each_fold.append(list(iloc))
            assert len(iloc_for_each_fold) == self._n_folds

            if self._ensure_all_classes and dataset.task_type in (
                TaskType.BINARY,
                TaskType.MULTICLASS,
            ):
                all_y = dataset.y.iloc[dataset.iloc[self._from_subset]]
                all_y_counts = all_y.value_counts().to_dict()
                for label, _total_count in all_y_counts.items():
                    counts_in_folds = [
                        dataset.y.iloc[iloc].value_counts().to_dict()[label]
                        for iloc in iloc_for_each_fold
                    ]
                    if sum([c > 0 for c in counts_in_folds]) > 1:
                        # each class is present in at least 2 folds
                        continue
                    elif sum(counts_in_folds) < 2:
                        # not enough samples for the class {label}
                        raise AssertionError(
                            f'Class "{label}" has only one sample while'
                            ' ensure_all_classes==True: for any possible split there'
                            ' will be a fold with training part not containing this'
                            ' class'
                        )
                    else:
                        # enough samples
                        from_fold = np.argmax(counts_in_folds)
                        to_fold = 1 if from_fold == 0 else 0
                        # need to move one sample of class {label} between folds
                        from_iloc: list[int] = iloc_for_each_fold[from_fold]
                        to_iloc: list[int] = iloc_for_each_fold[to_fold]
                        from_y_values = dataset.y.iloc[from_iloc]
                        index = from_iloc[
                            np.where(from_y_values == label)[0][0]
                        ]
                        from_iloc.remove(index)
                        to_iloc.append(index)

            # val set
            iloc1 = iloc_for_each_fold[self._fold_idx]
            # train set
            iloc_01 = list(dataset.iloc[self._from_subset])
            iloc0 = list(set(iloc_01).difference(set(iloc1)))
            np.random.default_rng(seed=self._random_seed).shuffle(iloc0)

            assert set(iloc1).difference(set(iloc_01)) == set()

        new_iloc = dataset.iloc.copy()
        if self._to_subsets[0] is not None:
            new_iloc[self._to_subsets[0]] = iloc0
        if self._to_subsets[1] is not None:
            new_iloc[self._to_subsets[1]] = iloc1

        result = Dataset(
            X=dataset.X,
            y=dataset.y,
            iloc=new_iloc,
            task_type=dataset.task_type,
            metadata=dataset.metadata.copy(),
        )

        if 'split' not in result.metadata:
            result.metadata['split'] = self.__str__()
        else:
            result.metadata['split'] += ';' + self.__str__()

        return result
