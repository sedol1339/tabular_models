from copy import copy
from typing import Any

from sklearn.utils import Bunch

from tabular_models.dataset import Dataset
from tabular_models.models.catboost import (
    CatBoostModel,
    get_default_catboost_params,
    recalc_catboost_best_iter,
)
from tabular_models.pipeline_result import PipelineResult
from tabular_models.pipelines.base import GB, Pipeline, PrevResults, Req
from tabular_models.pipelines.scoring import (
    default_scoring,
)
from tabular_models.predictions import Predictions
from tabular_models.utils import Timer


def _fit_catboost_with_early_stopping(
    dataset: Dataset,
    model: CatBoostModel,
    fit_kwargs: dict[str, Any] | None = None,
) -> PipelineResult:
    with Timer() as fit_timer:
        model.fit(
            dataset.X_train,
            dataset.y_train,
            eval_set=(dataset.X_val, dataset.y_val),
            use_best_model=False,
            max_iterations=100_000,
            early_stopping_rounds=500,
            **(fit_kwargs or {}),
        )
    best_iter50 = recalc_catboost_best_iter(model, es_rounds=50)
    preds = Predictions(
        model.predict(dataset.X, ntree_end=best_iter50), dataset=dataset
    )
    return PipelineResult(
        fit_time=fit_timer.duration,
        model=model,
        scores=default_scoring(preds),
    )


class Catboost_baseline(Pipeline):
    CPUS = 8
    MEMORY = 10 * GB

    def splits_and_requirements(self) -> dict[str, list[Req]]:
        return {f'cv5_seed0_fold{i}': [] for i in range(5)}

    def run(self, dataset: Dataset, reqs: PrevResults) -> PipelineResult:
        dataset = dataset.fill_missing_categorical()
        hps = get_default_catboost_params(dataset.task_type, self.CPUS)
        model = CatBoostModel(dataset.task_type, catboost_kwargs=hps)
        return _fit_catboost_with_early_stopping(dataset, model)


class Catboost_init(Pipeline):
    CPUS = 8
    MEMORY = 10 * GB

    def __init__(self, init_from: str = 'Linear_l2_OHE_standard'):
        tokens = init_from.split(':')
        self._init_model_info = Bunch(
            pipeline=tokens[0],
            split=tokens[1] if len(tokens) > 1 else 'none',
            field=tokens[2] if len(tokens) > 2 else 'model',
        )

    def init_model_info(self, split: str) -> Bunch:
        info = copy(self._init_model_info)
        if info.split == 'same':
            info.split = split
        return info

    def splits_and_requirements(self) -> dict[str, list[Req]]:
        splits = [f'cv5_seed0_fold{i}' for i in range(5)]
        reqs = {}
        for split in splits:
            # requirements for the split
            info = self.init_model_info(split)
            reqs[split] = [Req(info.pipeline, info.split, info.field)]
        return reqs

    def run(self, dataset: Dataset, reqs: PrevResults) -> PipelineResult:
        dataset = dataset.fill_missing_categorical()
        hps = get_default_catboost_params(dataset.task_type, self.CPUS)

        # get init model
        split = dataset.metadata['split']
        init_model_info = self.init_model_info(split)
        init_model = reqs[(init_model_info.pipeline, init_model_info.split)][
            init_model_info.field
        ]

        # build and fit model
        model = CatBoostModel(
            dataset.task_type,
            catboost_kwargs=hps,
            init_model=init_model,
        )
        return _fit_catboost_with_early_stopping(
            dataset, model, fit_kwargs={'fit_init_model': False}
        )
