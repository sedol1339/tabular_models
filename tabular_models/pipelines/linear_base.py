import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from tabular_models.dataset import Dataset
from tabular_models.models.linear import LinearCV, LinearRegType
from tabular_models.models.preproc import AbstractPreproc
from tabular_models.models.sklearn import TargetTransformType
from tabular_models.pipeline_result import PipelineResult
from tabular_models.pipelines.base import Pipeline, PrevResults, Req
from tabular_models.pipelines.scoring import default_scoring
from tabular_models.predictions import Predictions, Scores


class Abstract_LinearCV(Pipeline, AbstractPreproc):
    def get_reg_type(self) -> LinearRegType:
        return 'l2'

    def get_y_transform(self) -> TargetTransformType:
        return 'standard'

    def get_model(self, dataset: Dataset, reqs: PrevResults) -> LinearCV:
        return LinearCV(
            task_type=dataset.task_type,
            reg_type=self.get_reg_type(),
            preproc=self.get_preproc(dataset, self.CPUS),
            y_transform=self.get_y_transform(),
            n_jobs=self.CPUS,
        )

    def splits_and_requirements(self) -> dict[str, list[Req]]:
        return {'none': []}

    @ignore_warnings(category=UserWarning)
    @ignore_warnings(category=RuntimeWarning)
    @ignore_warnings(category=ConvergenceWarning)
    def run(self, dataset: Dataset, reqs: PrevResults) -> PipelineResult:
        model = self.get_model(dataset, reqs)
        model.fit(dataset.X_train, dataset.y_train)

        scores = default_scoring(  # type: ignore[arg-type]
            Predictions(model.predict(dataset.X), dataset=dataset),
            subsets=['train', 'val', 'test'],
        )

        # model cross-val scores on trainval
        cross_val_preds = Predictions(
            model.cross_val_predict_with_fixed_coef(
                dataset.X_trainval, dataset.y_trainval, n_folds=5
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


# def default_linreg_selection_metric(
#     task_type: TaskType
# ) -> str | Callable:
#     if task_type.task_type == TaskType.REGRESSION:
#         return 'r2'
#     else:
#         # to reduce a chance to make a mistake, better not to use
#         # this complex logic (the pipeline will not work on some
#         # datasets with extreme minor classes)
#         # https://github.com/scikit-learn/scikit-learn/issues/28178
#         # scoring = make_scorer(
#         #     log_loss,
#         #     greater_is_better=False,
#         #     labels=dataset.y.cat.codes.unique(),
#         #     needs_proba=True
#         # )
#         return 'neg_log_loss'
