from typing import Literal

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from tabular_models.dataset import Dataset
from tabular_models.models.linear import LinearRegType
from tabular_models.models.preproc import PreprocType
from tabular_models.pipeline_result import PipelineResult
from tabular_models.pipelines.base import Pipeline, PrevResults, Req
from tabular_models.pipelines.linear_base import Abstract_LinearCV
from tabular_models.pipelines.scoring import (
    SELECTION_METRIC,
    default_scoring,
)
from tabular_models.predictions import HIGHER_IS_BETTER, Predictions


class Linear_l2_OHE_standard(Abstract_LinearCV):
    def get_num_transform(self) -> PreprocType | Literal['skip']:
        return PreprocType.STANDARD

    def get_cat_transform(self) -> PreprocType | Literal['skip', 'auto']:
        return PreprocType.ONE_HOT_ENCODING


class Linear_l2_TE_standard(Abstract_LinearCV):
    def get_num_transform(self) -> PreprocType | Literal['skip']:
        return PreprocType.STANDARD

    def get_cat_transform(self) -> PreprocType | Literal['skip', 'auto']:
        return PreprocType.TARGET_ENCODING


class Linear_l1_num_standard(Abstract_LinearCV):
    def get_reg_type(self) -> LinearRegType:
        return 'l1'

    def get_num_transform(self) -> PreprocType | Literal['skip']:
        return PreprocType.STANDARD

    def get_cat_transform(self) -> PreprocType | Literal['skip', 'auto']:
        return 'skip'


class Linear_l2_num_standard(Abstract_LinearCV):
    def get_num_transform(self) -> PreprocType | Literal['skip']:
        return PreprocType.STANDARD

    def get_cat_transform(self) -> PreprocType | Literal['skip', 'auto']:
        return 'skip'


class Linear_l2_num_quantileUniform(Abstract_LinearCV):
    def get_num_transform(self) -> PreprocType | Literal['skip']:
        return PreprocType.QUANTILE_UNIFORM

    def get_cat_transform(self) -> PreprocType | Literal['skip', 'auto']:
        return 'skip'


class Linear_l2_num_quantileNormal(Abstract_LinearCV):
    def get_num_transform(self) -> PreprocType | Literal['skip']:
        return PreprocType.QUANTILE_NORMAL

    def get_cat_transform(self) -> PreprocType | Literal['skip', 'auto']:
        return 'skip'


class Linear_l2_num_auto(Pipeline):
    def models_to_compare(self) -> dict[str, tuple[str, str]]:
        return {
            'standard': ('Linear_l2_num_standard', 'none'),
            'quantile_uniform': ('Linear_l2_num_quantileUniform', 'none'),
            'quantile_normal': ('Linear_l2_num_quantileNormal', 'none'),
        }

    def splits_and_requirements(self) -> dict[str, list[Req]]:
        return {'none': [Req(*x) for x in self.models_to_compare().values()]}

    def run(self, dataset: Dataset, reqs: PrevResults) -> PipelineResult:
        pipelines_results = {
            k: reqs[v] for k, v in self.models_to_compare().items()
        }
        selection_metric = SELECTION_METRIC[dataset.task_type]
        cross_val_scores = {
            k: result.scores.get('cross_val', selection_metric)
            for k, result in pipelines_results.items()
        }
        best_scaling = (max if HIGHER_IS_BETTER[selection_metric] else min)(
            cross_val_scores, key=cross_val_scores.get  # type: ignore[arg-type]
        )
        best_model = pipelines_results[best_scaling].model
        best_preds = Predictions(
            best_model.predict(dataset.X), dataset=dataset
        )

        return PipelineResult(
            scaling=best_scaling,
            model=best_model,
            scores=default_scoring(best_preds),
        )


class Linear_l2_num_featureWisePreproc(Abstract_LinearCV):
    def __init__(self):
        self.inferred_best_preprocs: dict[str, PreprocType] | None = None
        super().__init__()

    def get_preproc_pipelines(self) -> dict[PreprocType, str]:
        return {
            PreprocType.STANDARD: 'SingleFeature_Num_Linear_Standard',
            PreprocType.QUANTILE_UNIFORM: 'SingleFeature_Num_Linear_QuantileUniform',
            PreprocType.QUANTILE_NORMAL: 'SingleFeature_Num_Linear_QuantileNormal',
        }

    def splits_and_requirements(self) -> dict[str, list[Req]]:
        return {
            'none': [
                Req(pipeline_name, 'none')
                for pipeline_name in self.get_preproc_pipelines().values()
            ]
        }

    def get_preproc_types(self, dataset: Dataset) -> dict[str, PreprocType]:
        # this field gets filled in `.run()`
        assert self.inferred_best_preprocs is not None
        return self.inferred_best_preprocs

    @ignore_warnings(category=UserWarning)
    @ignore_warnings(category=RuntimeWarning)
    @ignore_warnings(category=ConvergenceWarning)
    def run(self, dataset: Dataset, reqs: PrevResults) -> PipelineResult:
        # selecting best preproc for each feature
        metric = SELECTION_METRIC[dataset.task_type]
        collected_scores_list = []
        for (
            preproc_type,
            pipeline_name,
        ) in self.get_preproc_pipelines().items():
            for feature_name, scores in reqs[
                (pipeline_name, 'none')
            ].scores.items():
                collected_scores_list.append(
                    {
                        'preproc_type': preproc_type,
                        'feature_name': feature_name,
                        'score': scores.get('cross_val', metric, None),
                    }
                )

        collected_scores = pd.DataFrame(collected_scores_list)
        self.inferred_best_preprocs = {}

        for feature_name, scores in collected_scores.groupby('feature_name'):
            best_preproc = scores['preproc_type'].iloc[
                scores['score'].argmax()
            ]
            self.inferred_best_preprocs[feature_name] = best_preproc

        # running pipeline
        result = super().run(dataset, reqs)
        result.inferred_best_preprocs = self.inferred_best_preprocs
        return result
