import tempfile

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

from tabular_models.data_store import DataStore
from tabular_models.dataset_splitter import DatasetSplitter
from tabular_models.models.catboost import CatBoostModel
from tabular_models.models.linear import get_preproc_for_linear_model
from tabular_models.models.sklearn import SklearnModel
from tabular_models.pipeline_result import PipelineResult
from tabular_models.predictions import Predictions
from tabular_models.task_type import TaskType

data_store = DataStore('/data/tabular_data')
datasets = []
for name in ['socmob', 'blood-transfusion-service-center', 'eucalyptus']:
    dataset = data_store.load_dataset(name, 0)
    splitter = DatasetSplitter(method='holdout', val_ratio=0.95, random_seed=0)
    dataset = splitter.split(dataset)
    datasets.append(dataset)


def ds_to_metric(dataset):
    if dataset.task_type == TaskType.REGRESSION:
        return 'r2'
    elif dataset.task_type == TaskType.BINARY:
        return 'roc_auc'
    else:
        return 'balanced_accuracy'


def test_sklearn_linear_model():
    for dataset in datasets:
        if dataset.task_type == TaskType.REGRESSION:
            sklearn_model = LinearRegression()
        else:
            sklearn_model = LogisticRegression()
        sklearn_model = make_pipeline(
            get_preproc_for_linear_model(dataset.X_train), sklearn_model
        )
        model = SklearnModel(
            task_type=dataset.task_type, random_seed=0, model=sklearn_model
        )
        model.fit(dataset.X_train, dataset.y_train)
        assert model.sklearn_model._final_estimator.coef_ is not None
        assert (
            Predictions(
                model.predict(dataset.X_test),
                dataset.y_test,
                dataset.task_type,
            ).score(ds_to_metric(dataset))
            > 0
        )


def test_dummy_appender():
    data_store = DataStore('/data/tabular_data')
    # dataset with no numerical features
    dataset = data_store.load_dataset('Amazon_employee_access', 0)
    splitter = DatasetSplitter(method='holdout', val_ratio=0.95, random_seed=0)
    dataset = splitter.split(dataset)
    sklearn_model = make_pipeline(
        get_preproc_for_linear_model(
            dataset.X_train, use_cols=dataset.num_features
        ),
        LogisticRegression(),
    )
    model = SklearnModel(
        task_type=dataset.task_type, random_seed=0, model=sklearn_model
    )
    model.fit(dataset.X_train, dataset.y_train)
    assert (
        Predictions(
            model.predict(dataset.X_test), dataset.y_test, dataset.task_type
        ).score(ds_to_metric(dataset))
        == 0.5
    )


def test_catboost_1():
    # early stopping, use_best_model=True
    for dataset in datasets:
        model = CatBoostModel(
            task_type=dataset.task_type,
            random_seed=0,
            catboost_kwargs={'depth': 1},
        )
        model.fit(
            dataset.X_train,
            dataset.y_train,
            eval_set=(dataset.X_val, dataset.y_val),
            early_stopping_rounds=2,
            use_best_model=True,
        )
        assert (
            Predictions(
                model.predict(dataset.X_test),
                dataset.y_test,
                dataset.task_type,
            ).score(ds_to_metric(dataset))
            > 0
        )
        assert model.best_iteration == model.tree_count - 1


def test_catboost_2():
    # early stopping, use_best_model=False
    for dataset in datasets:
        model = CatBoostModel(
            task_type=dataset.task_type,
            random_seed=0,
            catboost_kwargs={'depth': 1},
        )
        model.fit(
            dataset.X_train,
            dataset.y_train,
            eval_set=(dataset.X_val, dataset.y_val),
            early_stopping_rounds=2,
            use_best_model=False,
        )
        assert model.best_iteration == model.tree_count - 3


def test_catboost_with_baseline_model():
    for dataset in datasets:
        model1 = CatBoostModel(
            task_type=dataset.task_type,
            random_seed=0,
            catboost_kwargs={'depth': 1},
        )
        model = CatBoostModel(
            task_type=dataset.task_type,
            random_seed=0,
            catboost_kwargs={'depth': 2},
            init_model=model1,
        )
        model1.fit(
            dataset.X_train,
            dataset.y_train,
            eval_set=(dataset.X_val, dataset.y_val),
            early_stopping_rounds=2,
            use_best_model=True,
        )
        model.fit(
            dataset.X_train,
            dataset.y_train,
            eval_set=(dataset.X_val, dataset.y_val),
            early_stopping_rounds=2,
            use_best_model=True,
            fit_init_model=False,
        )
        assert (
            Predictions(
                model.predict(dataset.X_test),
                dataset.y_test,
                dataset.task_type,
            ).score(ds_to_metric(dataset))
            > 0
        )


def test_catboost_with_init_predictions():
    for dataset in datasets:
        model = CatBoostModel(
            task_type=dataset.task_type,
            random_seed=0,
            catboost_kwargs={'depth': 1},
        )
        model.fit(dataset.X_train, dataset.y_train, max_iterations=10)
        preds = Predictions(
            model.predict(dataset.X),
            dataset.y,
            task_type=dataset.task_type,
            iloc=dataset.iloc,
        )

        score = preds.score(ds_to_metric(dataset))
        assert score > 0
        result = PipelineResult(preds=preds)
        with tempfile.TemporaryDirectory() as tmpdirname:
            result.save(tmpdirname)
            result = PipelineResult.load(tmpdirname)
        assert preds.score(ds_to_metric(dataset)) == score

        model = CatBoostModel(
            task_type=dataset.task_type,
            random_seed=0,
            catboost_kwargs={'depth': 6},
        )
        model.fit(
            dataset.X_train,
            dataset.y_train,
            eval_set=(dataset.X_val, dataset.y_val),
            max_iterations=10,
            init_predictions=result.preds.train.predictions,
            init_val_predictions=result.preds.val.predictions,
        )
        preds = Predictions(
            model.predict(
                dataset.X,
                init_predictions=result.preds.predictions,
            ),
            dataset.y,
            task_type=dataset.task_type,
            iloc=dataset.iloc,
        )

        score = preds.score(ds_to_metric(dataset))
        assert score > 0
