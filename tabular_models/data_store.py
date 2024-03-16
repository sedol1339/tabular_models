import fnmatch
import pickle
import shutil
import time
from collections.abc import Sequence
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from sklearn.utils import Bunch
from sklearn.utils._testing import ignore_warnings
from tqdm.auto import tqdm
from yaml import CLoader

from .dataset import Dataset
from .pipeline_result import PipelineResult
from .task_type import TaskType
from .utils import _validate_filename


class DataStore:
    """
    Represents structure of a folder containing datasets and results
    - Dataset lists are stored in {root} in  .yaml format
    - Datasets are stored in {root}/data/{dataset}/{fold}
    - Pipeline results are stored in {root}/results/{dataset}/{fold}/{name}/{split},
    when {name} is a unique name of pipeline and hyperparameters, and {split} is a
    string representation of DatasetSplitter used to generate 'train' and 'val' subsets
    from 'trainval'. If DatasetSplitter was not used, which means no 'val' subset,
    {split} is 'none'.

    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def list_datasets(
        self,
        exclude: list[str] | None = None,
        min_trainval_samples: int | None = None,
        max_trainval_samples: int | None = None,
        min_features: int | None = None,
        max_features: int | None = None,
        task_types: list[TaskType] | None = None,
        rev: bool = True,
    ) -> dict[str, Bunch]:
        # loading names and metadata
        with open(self.root / 'datasets.yaml') as h:
            datasets_metadata = {
                name: Bunch(**metadata)
                for name, metadata in yaml.load(h, Loader=CLoader).items()
            }
        for metadata in datasets_metadata.values():
            metadata.task_type = TaskType.from_string(metadata.task_type)

        # add reversed versions
        if rev:
            for name, metadata in list(datasets_metadata.items()):
                new_name, new_metadata = name + '[rev]', deepcopy(metadata)
                datasets_metadata[new_name] = new_metadata
                (
                    new_metadata.n_samples_trainval,
                    new_metadata.n_samples_test,
                ) = (
                    new_metadata.n_samples_test,
                    new_metadata.n_samples_trainval,
                )

        # filtering rules
        def filter(name: str, metadata: Bunch) -> Literal['keep', 'remove']:
            if exclude is not None:
                for exclude_pattern in exclude:
                    if (
                        fnmatch.fnmatch(name, exclude_pattern)
                        or name == exclude_pattern
                    ):
                        return 'remove'
            if task_types is not None and metadata.task_type not in task_types:
                return 'remove'
            if (
                min_trainval_samples is not None
                and metadata.n_samples_trainval < min_trainval_samples
            ):
                return 'remove'
            if (
                max_trainval_samples is not None
                and metadata.n_samples_trainval >= max_trainval_samples
            ):
                return 'remove'
            if min_features is not None and metadata.n_features < min_features:
                return 'remove'
            if (
                max_features is not None
                and metadata.n_features >= max_features
            ):
                return 'remove'
            return 'keep'

        # filtering
        datasets_metadata = {
            name: metadata
            for name, metadata in datasets_metadata.items()
            if filter(name, metadata) == 'keep'
        }

        return datasets_metadata

    def _get_data_folder(
        self,
        name: str,
    ) -> Path:
        assert _validate_filename(name), f'bad filename {name}'
        return self.root / 'data' / name

    def _get_results_folder(
        self,
        name: str,
        fold: int,
    ) -> Path:
        assert _validate_filename(name), f'bad filename {name}'
        assert _validate_filename(str(fold)), f'bad filename {str(fold)}'
        return self.root / 'results' / name / str(fold)

    def load_dataset(
        self,
        name: str,
        fold: int | None = None,
    ) -> Dataset:
        """
        Loads the dataset.

        If name ends with "[rev]", for example name=="cmc[rev]", then loads the
        dataset ("cmc" in the example) with swapped train and test parts. This
        can be useful to enlarge test part, while shortening train part, for more
        precise test metrics estimation.

        """
        if name.endswith('[rev]'):
            name = name[:-5]
            rev = True
        else:
            rev = False

        filepath = self._get_data_folder(name) / 'dataset.pkl'

        with open(filepath, 'rb') as h:
            dataset = pickle.load(h)

        dataset.metadata['reversed'] = rev

        if fold is not None:
            dataset.set_fold(fold)

        # if rev:
        #     assert set(dataset.iloc.keys()) == {'train', 'trainval', 'test'}, (
        #         'To reverse dataset, it should include'
        #         ' only {train, trainval, test} splits'
        #     )
        #     dataset.iloc = Bunch(
        #         train=dataset.iloc.test,
        #         trainval=dataset.iloc.test,
        #         test=dataset.iloc.trainval,
        #     )

        if (
            dataset.task_type in (TaskType.BINARY, TaskType.MULTICLASS)
            and 'trainval' in dataset.iloc
            and 'test' in dataset.iloc
        ):
            trainval_categories = set(dataset.y_trainval.cat.categories)
            test_categories = set(dataset.y_test.cat.categories)
            assert trainval_categories >= test_categories, (
                'Test contains categories not present in trainval:'
                f' {test_categories - trainval_categories}'
            )

        return dataset

    @ignore_warnings(category=FutureWarning)
    def load_dataset_from_openml(
        self,
        task_id: int,
        fold: int | None = None,
        openml_cache_dir: str | Path | None = '{root}/openml',
    ) -> Dataset:

        import openml

        openml_cache_dir = str(openml_cache_dir)

        if openml_cache_dir is not None:
            if '{root}' in openml_cache_dir:
                openml_cache_dir = openml_cache_dir.replace(
                    '{root}', str(self.root)
                )
            openml.config.set_root_cache_directory(str(openml_cache_dir))

        task = openml.tasks.get_task(
            task_id,
            download_splits=True,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )

        n_repeats, n_folds, n_samples = task.get_split_dimensions()
        assert n_repeats == 1
        assert n_folds == 10
        assert n_samples == 1

        X, y = task.get_X_and_y(dataset_format='dataframe')

        for col in X.columns:
            values = X[col]
            if (
                values.dtype == 'category'
                or values.dtype == object
                or values.dtype == bool
                or values.nunique(dropna=True) <= 2
            ):
                # all category columns have string categories
                X[col] = values.astype(str).astype('category')
            else:
                X[col] = values.astype(np.float32)

        if y.dtype == object or y.dtype == bool:
            y = y.astype(str).astype('category')

        if y.dtype == 'category':
            y = y.cat.rename_categories(str)

        if y.dtype == 'category':
            if len(y.cat.categories) == 2:
                task_type = TaskType.BINARY
            else:
                task_type = TaskType.MULTICLASS
        else:
            task_type = TaskType.REGRESSION

        # get indices for all folds to write into metadata
        fold_indices = np.full(len(X), np.nan)
        for loop_fold in range(10):
            _, loop_test_indices = task.get_train_test_split_indices(
                fold=loop_fold
            )
            assert np.isnan(fold_indices[loop_test_indices]).all()
            fold_indices[loop_test_indices] = loop_fold
        assert not np.isnan(fold_indices).any()

        dataset = Dataset(
            X=X,
            y=y,
            task_type=task_type,
            iloc=Bunch(),
            metadata={'fold_indices': fold_indices},
        )

        # get indices for current fold to write into iloc
        if fold is not None:
            dataset.set_fold(fold)

        return dataset

    def save_dataset(
        self,
        dataset: Dataset,
        name: str,
    ) -> None:
        assert not name.endswith(
            '[rev]'
        ), 'Reversed datasets are not meant to be saved'
        if 'name' in dataset.metadata:
            assert (
                dataset.metadata['name'] == name
            ), 'inconsistency between specified name and metadata'

        path = self._get_data_folder(name)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'dataset.pkl', 'wb') as h:
            pickle.dump(dataset, h)

    def save_pipeline_result(
        self,
        result: PipelineResult,
        dataset_name: str,
        fold: int,
        pipeline_name: str,
        split_name: str,
    ) -> None:
        path = self._get_result_path(
            dataset_name, fold, pipeline_name, split_name
        )
        result.save(path)

    def list_results(
        self,
        dataset_names: str | list[str] = '*',
        folds: int | str | list[int | str] = '*',
        pipeline_names: str | list[str] = '*',
        split_names: str | list[str] = '*',
    ) -> pd.DataFrame:

        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        if isinstance(folds, (str, int)):
            folds = [str(folds)]
        if isinstance(pipeline_names, str):
            pipeline_names = [pipeline_names]
        if isinstance(split_names, str):
            split_names = [split_names]

        results_list = []

        for path in self.root.glob('results/*/*/*/*'):
            dataset_name, fold, pipeline_name, split_name = path.parts[-4:]

            for pattern in dataset_names:
                if fnmatch.fnmatch(dataset_name, pattern):
                    break
            else:
                continue

            for pattern2 in folds:
                if fnmatch.fnmatch(str(fold), str(pattern2)):
                    break
            else:
                continue

            for pattern in pipeline_names:
                if fnmatch.fnmatch(pipeline_name, pattern):
                    break
            else:
                continue

            for pattern in split_names:
                if fnmatch.fnmatch(split_name, pattern):
                    break
            else:
                continue

            try:
                fold_int = int(fold)
            except ValueError:
                continue

            results_list.append(
                {
                    'dataset_name': dataset_name,
                    'fold': fold_int,
                    'pipeline_name': pipeline_name,
                    'split_name': split_name,
                }
            )

        results = pd.DataFrame(results_list)
        return results

    def plot_results_readiness(
        self,
        results_info: pd.DataFrame,
        folds: int | None = None,
        compact: bool = True,
        with_time: bool = True,
    ) -> None:
        """
        Plot results readiness using dataframe obtained by .list_results()
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sep = ' : '

        dataset_and_fold = (
            results_info['dataset_name']
            + sep
            + results_info['fold'].astype(str)
        )

        pipeline_and_split = (
            results_info['pipeline_name'] + sep + results_info['split_name']
        )

        from natsort import natsorted

        if folds is None:
            dataset_and_fold_unique = natsorted(dataset_and_fold.unique())
        else:
            dataset_and_fold_unique = natsorted(
                [
                    dataset_name + sep + str(fold)
                    for dataset_name in results_info['dataset_name'].unique()
                    for fold in range(folds)
                ]
            )
        pipeline_and_split_unique = natsorted(pipeline_and_split.unique())

        readiness = np.zeros(
            (len(dataset_and_fold_unique), len(pipeline_and_split_unique))
        )

        indices1 = dataset_and_fold.apply(
            dataset_and_fold_unique.index
        ).to_numpy()
        indices2 = pipeline_and_split.apply(
            pipeline_and_split_unique.index
        ).to_numpy()

        readiness[indices1, indices2] = 1

        if with_time:
            gen = product(
                enumerate(dataset_and_fold_unique),
                enumerate(pipeline_and_split_unique),
            )
            size = len(dataset_and_fold_unique) * len(
                pipeline_and_split_unique
            )
            gen = tqdm(gen, total=size)
            for (i, dataset_and_fold), (j, pipeline_and_split) in gen:
                dataset, fold = dataset_and_fold.split(sep)
                pipeline, split = pipeline_and_split.split(sep)
                timestamp = self.get_result_last_modified_time(
                    dataset, fold, pipeline, split
                )
                if timestamp is not None:
                    readiness[i, j] = timestamp
                else:
                    readiness[i, j] = np.nan
            readiness = np.clip(time.time() - readiness, 10, 84600 * 3)
            readiness = np.log(readiness)

        readiness_df = pd.DataFrame(
            data=readiness,
            index=dataset_and_fold_unique,
            columns=pipeline_and_split_unique,
        )

        figsize = (readiness.shape[1] / 3, readiness.shape[0] / 5)
        if compact:
            figsize = (figsize[0], figsize[1] / 4)

        fig, ax = plt.subplots(figsize=figsize)
        if compact:
            yticklabels = dataset_and_fold_unique.copy()
            for i in range(len(yticklabels)):
                if not yticklabels[i].endswith(sep + '0'):
                    yticklabels[i] = ''
        else:
            yticklabels = 'auto'  # type: ignore[assignment]
        sns.heatmap(readiness_df, ax=ax, cbar=False, yticklabels=yticklabels)
        plt.gcf().set_dpi(60)
        plt.show()

    def _get_result_path(
        self,
        dataset_name: str,
        fold: int,
        pipeline_name: str,
        split_name: str,
    ) -> Path:
        assert _validate_filename(
            pipeline_name
        ), f'bad filename {pipeline_name}'
        assert _validate_filename(split_name), f'bad filename {split_name}'
        return (
            self._get_results_folder(dataset_name, fold)
            / pipeline_name
            / split_name
        )

    def get_result_last_modified_time(
        self,
        dataset_name: str,
        fold: int,
        pipeline_name: str,
        split_name,
    ) -> float | None:
        path = self._get_result_path(
            dataset_name, fold, pipeline_name, split_name
        )
        return path.stat().st_mtime if PipelineResult.exists(path) else None

    def result_exists(
        self,
        dataset_name: str,
        fold: int,
        pipeline_name: str,
        split_name,
    ) -> bool:
        path = self._get_result_path(
            dataset_name, fold, pipeline_name, split_name
        )
        return PipelineResult.exists(path)

    def del_pipeline_result(
        self,
        dataset_name: str,
        fold: int,
        pipeline_name: str,
        split_name: str = 'none',
    ) -> None:
        path = self._get_result_path(
            dataset_name, fold, pipeline_name, split_name
        )
        if path.is_dir():
            shutil.rmtree(path)

    def load_pipeline_result(
        self,
        dataset_name: str,
        fold: int,
        pipeline_name: str,
        split_name: str,
        fields: str | list[str] | None = None,
    ) -> PipelineResult | None:
        path = self._get_result_path(
            dataset_name, fold, pipeline_name, split_name
        )
        if not PipelineResult.exists(path):
            return None
        return PipelineResult.load(path, fields=fields)

    def load_pipeline_field(
        self,
        dataset_name: str,
        fold: int,
        pipeline_name: str,
        split_name: str,
        field: str,
    ) -> PipelineResult | None:
        result = self.load_pipeline_result(
            dataset_name, fold, pipeline_name, split_name, field
        )
        if result is None:
            return None
        return result[field]

    def summary(
        self,
        fields: dict[str, tuple[str, str, str]] | None = None,
        folds: int | Sequence[int] | None = None,
        explode_folds: bool = False,
    ) -> pd.DataFrame:
        """
        fields: key is dataframe column to save in,
        value is (pipeline_name, split, field)
        """
        if folds is None:
            folds = range(10)
        elif isinstance(folds, int):
            folds = [folds]

        if fields is None:
            fields = {}

        summary = pd.DataFrame(
            [
                {'dataset_name': dataset_name, 'fold': fold}
                for dataset_name in self.list_datasets()
                for fold in folds
            ]
        )

        for col_name, (pipeline_name, split_name, field) in fields.items():
            summary[col_name] = summary.apply(
                lambda row: self.load_pipeline_field(
                    row.dataset_name,  # noqa: B023
                    row.fold,  # noqa: B023
                    pipeline_name,  # noqa: B023
                    split_name,  # noqa: B023
                    field,  # noqa: B023
                )
                or np.nan,
                axis=1,
            )

        if explode_folds:
            return summary
        else:
            pivot_cols = {}
            for col_name in fields:
                pivot_cols[col_name] = summary.groupby('dataset_name').apply(
                    lambda df: df[col_name].to_numpy()  # noqa: B023
                )
            return pd.DataFrame(pivot_cols)
