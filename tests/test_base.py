# ruff: noqa: PD011

import tempfile

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.utils import Bunch

from tabular_models.data_store import DataStore
from tabular_models.dataset import Dataset
from tabular_models.dataset_splitter import DatasetSplitter
from tabular_models.pipeline_result import PipelineResult
from tabular_models.predictions import Predictions
from tabular_models.task_type import TaskType
from tabular_models.utils import Timer, _validate_filename

data_store = DataStore('/data/tabular_data')


def test_datasets_list():
    list_datasets = list(data_store.list_datasets())
    assert 'eucalyptus' in list_datasets


def test_dataset():
    dataset = data_store.load_dataset('eucalyptus', 0)
    assert dataset.task_type == TaskType.MULTICLASS
    dataset.__str__()
    dataset.__repr__()
    assert dataset.X_train.shape == dataset.X_trainval.shape == (662, 19)
    assert dataset.y_train.shape == dataset.y_trainval.shape == (662,)
    assert dataset.X_test.shape == (74, 19)
    assert dataset.y_test.shape == (74,)
    assert dataset.y_trainval.dtype == 'category'
    assert dataset.y_trainval.dtype == 'category'
    assert is_numeric_dtype(dataset.y_trainval_numerical)
    assert set(dataset.iloc.train) == set(dataset.iloc.trainval)
    assert set(dataset.iloc.trainval).intersection(dataset.iloc.test) == set()
    assert set(dataset.iloc.trainval).union(dataset.iloc.test) == set(
        range(len(dataset.X))
    )


def test_dataset_subsets():
    dataset = Dataset(
        X=pd.DataFrame(
            index=[323, 222, 111, 321],
            columns=['input_feature'],
            data=[[1], [2], [3], [4]],
        ),
        y=pd.Series(index=[323, 222, 111, 321], data=[1, 2, 3, 4]),
        task_type=TaskType.REGRESSION,
        iloc=Bunch(
            **{
                'trainval': [0, 1, 3],
                'train': [3, 0],
                'val': [1],
                'test': [2],
            }
        ),
    )

    subset = dataset.train
    assert set(subset.y) == {4, 1}


test_dataset_subsets()


def test_dataset_rev():
    dataset = data_store.load_dataset('eucalyptus[rev]', 0)
    dataset_rev = data_store.load_dataset('eucalyptus', 0)
    assert dataset_rev.task_type == TaskType.MULTICLASS
    assert (dataset.y == dataset_rev.y).all()
    assert set(dataset.iloc.trainval).intersection(dataset.iloc.test) == set()
    assert (
        set(dataset_rev.iloc.trainval).intersection(dataset_rev.iloc.test)
        == set()
    )
    assert set(dataset.iloc.trainval) == set(dataset_rev.iloc.test)


def test_splitter():
    for dataset_name in ['eucalyptus', 'socmob']:
        dataset = data_store.load_dataset(dataset_name, 0)

        splitter = DatasetSplitter(
            method='holdout', val_ratio=0.1, random_seed=0
        )
        assert str(splitter) == 'holdout0.1_seed0'
        d = splitter.split(dataset)
        assert set(d.iloc.train).intersection(d.iloc.val) == set()
        assert set(d.iloc.train).union(d.iloc.val) == set(d.iloc.trainval)

        splitter = DatasetSplitter(
            method='holdout', val_ratio=0.1, random_seed=0
        )
        d_trial2 = splitter.split(dataset)
        assert set(d.iloc.train) == set(d_trial2.iloc.train)
        assert set(d.iloc.val) == set(d_trial2.iloc.val)

        splitter = DatasetSplitter(
            method='holdout', val_ratio=0.1, random_seed=1
        )
        assert str(splitter) == 'holdout0.1_seed1'
        d2 = splitter.split(dataset)
        assert len(set(d2.iloc.train).intersection(d.iloc.train)) > 0
        assert len(set(d2.iloc.val).intersection(d.iloc.val)) > 0

        splitter = DatasetSplitter(
            method='cv', n_folds=5, fold_idx=0, random_seed=2
        )
        assert str(splitter) == 'cv5_seed2_fold0'
        splitter.split(dataset)

        vals = [
            set(
                DatasetSplitter(
                    method='cv', n_folds=5, fold_idx=i, random_seed=1
                )
                .split(dataset)
                .iloc.val
            )
            for i in range(5)
        ]
        assert set.union(*vals) == set(dataset.iloc.trainval)
        for i in range(5):
            for j in range(5):
                if i != j:
                    assert set.intersection(vals[i], vals[j]) == set()


# def test_splitter_ensure_all_classes():
#     dataset_orig = data_store.load_dataset('yeast', 0)
#     dataset_orig = DatasetSplitter(
#         method='holdout',
#         val_ratio=0.65,
#         random_seed=0,
#         ensure_all_classes=False,
#         from_subset='trainval',
#         to_subsets=('trainval', '_'),
#     ).split(dataset_orig)

#     # now value counts in trainval are: {'cyt': 140, 'erl': 139, 'exc': 82,
#     # 'me1': 45, 'me3': 15, 'me2': 15, 'nuc': 14, 'mit': 9, 'pox': 6, 'vac': 2}
#     assert dataset_orig.y_trainval.value_counts()['vac'] == 2

#     dataset = DatasetSplitter(
#         method='holdout',
#         val_ratio=0.99,
#         stratified=False,
#         ensure_all_classes=True,
#     ).split(dataset_orig)
#     train_iloc_set = set(dataset.iloc['train'])
#     val_iloc_set = set(dataset.iloc['val'])
#     trainval_iloc_set = set(dataset.iloc['trainval'])
#     val_classes = set(dataset.y_val.unique())
#     train_classes = set(dataset.y_val.unique())
#     assert train_iloc_set.union(val_iloc_set) == trainval_iloc_set
#     assert train_iloc_set.intersection(val_iloc_set) == set()
#     assert val_classes.difference(train_classes) == set()

#     dataset = DatasetSplitter(
#         method='cv',
#         n_folds=4,
#         fold_idx=0,
#         stratified=False,
#         ensure_all_classes=False,
#     ).split(dataset_orig)
#     train_iloc_set = set(dataset.iloc['train'])
#     val_iloc_set = set(dataset.iloc['val'])
#     trainval_iloc_set = set(dataset.iloc['trainval'])
#     val_classes = set(dataset.y_val.unique())
#     train_classes = set(dataset.y_val.unique())
#     assert train_iloc_set.union(val_iloc_set) == trainval_iloc_set
#     assert train_iloc_set.intersection(val_iloc_set) == set()
#     assert val_classes.difference(train_classes) == set()


def test_filename_validation():
    # disallowed symbols: ? \ / : * < > |
    assert _validate_filename('aaa bbb')
    assert not _validate_filename('aaa bbb/ccc')
    assert _validate_filename('aaa bbb/ccc', allow_slash=True)
    assert not _validate_filename('aaa bbb?')
    assert not _validate_filename('aaa\\ bbb')
    assert not _validate_filename('aaa:bbb')
    assert not _validate_filename('aaa*bbb')
    assert not _validate_filename('aaa<bbb')
    assert not _validate_filename('aaa>bbb')
    assert not _validate_filename('aaa|bbb')


def test_timer():
    with Timer() as t:
        pass
    assert t.duration >= 0
    assert t.duration < 0.001


def test_regression_predictions():
    predictions = Predictions(
        predictions=np.array([0, 1, 2, 3, 4, 5]),
        truth=pd.Series(np.array([0, 1, 2, 3, 4, 4])),
        task_type=TaskType.REGRESSION,
        iloc={'train': [0, 1], 'trainval': [0, 1, 2], 'test': [3, 4, 5]},
    )
    assert np.isclose(predictions.score('r2'), 0.925)
    assert np.isclose(predictions.train.score('r2'), 1)
    assert np.isclose(predictions.trainval.score('r2'), 1)
    assert np.isclose(predictions.test.score('r2'), -0.5)


def test_multiclass_predictions():
    predictions = Predictions(
        predictions=np.array([[-1, 0, 1], [1, 0, -1]]),
        truth=pd.Series(
            pd.Categorical(['1', '1'], categories=['1', '2', '3'])
        ),
        task_type=TaskType.MULTICLASS,
    )
    assert np.allclose(
        predictions.probabilities,
        [
            [0.09003057, 0.24472847, 0.66524096],
            [0.66524096, 0.24472847, 0.09003057],
        ],
    )
    predictions.predictions += 10
    assert np.allclose(
        predictions.probabilities,
        [
            [0.09003057, 0.24472847, 0.66524096],
            [0.66524096, 0.24472847, 0.09003057],
        ],
    )


def test_binary_predictions():
    predictions = Predictions(
        predictions=np.array([-1, 1]),
        truth=pd.Series(pd.Categorical(['1', '2'], categories=['1', '2'])),
        task_type=TaskType.BINARY,
    )
    assert predictions.score('roc_auc') == 1
    assert np.allclose(predictions.probabilities, [0.26894142, 0.73105858])
    predictions.predictions = np.array([2, -1])
    assert np.allclose(predictions.probabilities, [0.88079708, 0.26894142])
    assert predictions.score('roc_auc') == 0


def test_predictions_averaging():
    predictions_list = [
        Predictions(
            predictions=np.array([0, 1, 2]),
            truth=pd.Series(np.array([0, 1, 1])),
            task_type=TaskType.REGRESSION,
            t=1,
        ),
        Predictions(
            predictions=np.array([0, 1, 0]),
            truth=pd.Series(np.array([0, 1, 1])),
            task_type=TaskType.REGRESSION,
            t=None,
        ),
    ]
    p12 = Predictions.average(predictions_list)
    assert np.isclose(p12.score('r2'), 1)
    assert p12.t is None

    p1, p12 = Predictions.cumulative_average(predictions_list)
    assert np.isclose(p1.score('r2'), -0.5)
    assert np.isclose(p12.score('r2'), 1)
    assert p1.t == 1
    assert p12.t is None


def test_pipeline_result():
    with tempfile.TemporaryDirectory() as tmpdirname:
        result = PipelineResult(
            a={'b': 1, 'c': [1, 2, 3]},
            preds=Predictions(
                predictions=np.array([0, 1, 2]),
                truth=pd.Series(np.array([0, 1, 1])),
                task_type=TaskType.REGRESSION,
                t=1,
            ),
            x=DatasetSplitter(method='holdout', val_ratio=0.1),
            y=b'01234',
        )
        result.save(tmpdirname)
        result_loaded = PipelineResult.load(tmpdirname)
        assert result_loaded.a['b'] == result.a['b']
        assert result_loaded.a['c'] == result.a['c']
        assert np.isclose(result.preds.score('r2'), -0.5)
        assert np.isclose(result_loaded.preds.score('r2'), -0.5)
        assert isinstance(result_loaded.x, DatasetSplitter)
        assert str(result_loaded.x) == 'holdout0.1_seed0'
        assert result_loaded.y == b'01234'
