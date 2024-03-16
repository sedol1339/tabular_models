from collections.abc import Sequence
from typing import Literal

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor

from tabular_models.dataset import Dataset
from tabular_models.models.sklearn import SklearnModel
from tabular_models.pipeline_result import PipelineResult
from tabular_models.pipelines.base import Pipeline, PrevResults, Req
from tabular_models.predictions import Predictions, Scores
from tabular_models.task_type import TaskType

EVAL_METRICS = {
    TaskType.REGRESSION: ['r2', 'd2_absolute_error'],
    TaskType.BINARY: ['neg_log_loss', 'roc_auc'],
    TaskType.MULTICLASS: ['neg_log_loss', 'balanced_accuracy'],
}

SELECTION_METRIC = {
    # this will be used in sklearn *CV models as "scoring" argument,
    # as well as in Predictions.score() function as "metric" argument
    # so, both sklearn and Predictions.score should support the metric name
    TaskType.REGRESSION: 'r2',
    TaskType.BINARY: 'neg_log_loss',
    TaskType.MULTICLASS: 'neg_log_loss',
    # NOTE: higher values are considered better
}


def default_scoring(
    predictions: Predictions,
    subsets: Sequence[str | None] | Literal['default'] = 'default',
) -> Scores:
    if subsets == 'default':
        subsets = ['val', 'test']
    subsets = [s for s in subsets if s in predictions.iloc]
    return predictions.score_all(
        metrics=EVAL_METRICS[predictions.task_type],
        subsets=subsets,
        bootstrap_seeds=[None] + list(range(50)),
    )


class ConstantPredictor(Pipeline):
    def splits_and_requirements(self) -> dict[str, list[Req]]:
        return {'none': []}

    def run(self, dataset: Dataset, reqs: PrevResults) -> PipelineResult:
        model = SklearnModel(
            dataset.task_type,
            DummyRegressor(strategy='mean')
            if dataset.task_type == TaskType.REGRESSION
            else DummyClassifier(strategy='prior', random_state=0),
        )
        model.fit(dataset.X_train, dataset.y_train)
        # todo remove duplicates from below code
        scores = default_scoring(  # type: ignore[arg-type]
            Predictions(model.predict(dataset.X), dataset=dataset),
            subsets=['train', 'val', 'test'],
        )
        cross_val_preds = Predictions(
            model.cross_val_predict(
                dataset.X_trainval,
                dataset.y_trainval,
                n_folds=5,
                n_jobs=self.CPUS,
            ),
            dataset.y_trainval,
            dataset.task_type,
            iloc={'cross_val': np.arange(len(dataset.y_trainval))},
        )
        cross_val_scores = default_scoring(
            cross_val_preds, subsets=['cross_val']
        )

        return PipelineResult(
            model=model, scores=Scores.concat([scores, cross_val_scores])
        )
