from typing import Literal

import numpy as np
import pandas as pd
import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    Ridge,
    RidgeCV,
)
from sklearn.pipeline import make_pipeline
from sklearn.utils import Bunch

from tabular_models.models.sklearn import (
    SklearnModel,
    TargetTransformType,
    make_transformed_target_regressor,
)
from tabular_models.task_type import TaskType

LinearRegType = Literal['l1', 'l2', 'elasticnet', 'omp']


def _make_RidgeCV(
    reg_coefs: list[float] | Literal['auto'] = 'auto',
    scoring: str | None = None,
) -> tuple[RidgeCV, Ridge]:
    default_alphas = 2.0 ** np.arange(-14, 15)
    linear_model_cv = RidgeCV(
        alphas=reg_coefs if reg_coefs != 'auto' else default_alphas,
        scoring=scoring or 'neg_mean_squared_error',
    )
    linear_model_nocv = Ridge()
    return linear_model_cv, linear_model_nocv


def _make_LassoCV(
    reg_coefs: list[float] | Literal['auto'] = 'auto',
    random_seed: int = 0,
    n_jobs: int = -1,
    max_iter: int = 100_000,
    n_folds: int = 5,
) -> tuple[LassoCV, Lasso]:
    default_alphas = 2.0 ** np.arange(-14, 15)
    kwargs = Bunch(
        random_state=random_seed,
        max_iter=max_iter,
    )
    linear_model_cv = LassoCV(
        **kwargs,
        alphas=reg_coefs if reg_coefs != 'auto' else default_alphas,
        cv=n_folds,
        n_jobs=n_jobs,
    )
    linear_model_nocv = Lasso(**kwargs)
    return linear_model_cv, linear_model_nocv


def _make_ElasticNetCV(
    reg_coefs: list[float] | Literal['auto'] = 'auto',
    random_seed: int = 0,
    n_jobs: int = -1,
    max_iter: int = 100_000,
    n_folds: int = 5,
    l1_ratio: float | list[float] = 0.5,
) -> tuple[ElasticNetCV, ElasticNet]:
    default_alphas = 2.0 ** np.arange(-14, 15)
    kwargs = Bunch(
        l1_ratio=l1_ratio,
        random_state=random_seed,
        max_iter=max_iter,
    )
    linear_model_cv = ElasticNetCV(
        **kwargs,
        alphas=reg_coefs if reg_coefs != 'auto' else default_alphas,
        cv=n_folds,
        n_jobs=n_jobs,
    )
    linear_model_nocv = ElasticNet(**kwargs)
    return linear_model_cv, linear_model_nocv


def _make_LogisticRegressionCV(
    task_type: TaskType,
    reg_type: Literal['l1', 'l2', 'elasticnet'] = 'l2',
    reg_coefs: list[float] | Literal['auto'] = 'auto',
    random_seed: int = 0,
    n_jobs: int = -1,
    max_iter: int = 100_000,
    scoring: str | None = None,
    n_folds: int = 5,
    multiclass: Literal['multinomial', 'ovr', 'auto'] = 'auto',
    l1_ratios: list[float] | None = None,
) -> tuple[LogisticRegressionCV, LogisticRegression]:
    # multinomial and OVR losses are different in multiclass
    # classification and may lead to different optimization results;
    # usually OVR is used for binary classification
    if multiclass == 'auto':
        multiclass = (
            'multinomial' if task_type == TaskType.MULTICLASS else 'ovr'
        )
    # newton-cholesky is faster and parallelizable,
    # but supports only OVR loss
    if reg_type == 'l2':
        solver = 'newton-cholesky' if multiclass == 'ovr' else 'lbfgs'
    else:
        solver = 'saga'
    logreg_kwargs = Bunch(
        penalty=reg_type,
        n_jobs=n_jobs,
        max_iter=max_iter,
        multi_class=multiclass,
        solver=solver,
        random_state=random_seed,
    )
    linear_model_cv = LogisticRegressionCV(
        **logreg_kwargs,
        # below are default Cs for LogisticRegressionCV
        Cs=(
            reg_coefs
            if reg_coefs != 'auto'
            else np.logspace(-4, 4, num=10, base=10)
        ),
        cv=n_folds,
        scoring=scoring or 'neg_log_loss',
        l1_ratios=l1_ratios if reg_type == 'elasticnet' else None,
    )
    linear_model_nocv = LogisticRegression(**logreg_kwargs)
    return linear_model_cv, linear_model_nocv


def _make_OrthogonalMatchingPursuitCV(
    max_nonzero_coefs: int | None = None,
    n_jobs: int = -1,
    n_folds: int = 5,
) -> tuple[RidgeCV, Ridge]:
    linear_model_cv = OrthogonalMatchingPursuitCV(
        max_iter=max_nonzero_coefs,  # this is correct, see docs
        n_jobs=n_jobs,
        cv=n_folds,
    )
    linear_model_nocv = OrthogonalMatchingPursuit()
    return linear_model_cv, linear_model_nocv


class LinearCV(SklearnModel):
    """
    Sklearn linear model that is enough for our experiments.

    It is able to do the following:
    - fit with auto selecting the best regulariation coefficient
    - get cross-val predictions with any regulariation coefficient

    This class uses *CV models (like RidgeCV, LogisticRegressionCV) to find optimal
    regularization coefficient.

    Additionally, it exposes method cross_val_predict which returns
    out-of-fold predictions with any regularization coefficient. For example, if *CV
    model is RidgeCV, then the method cross_val_predict will make
    Ridge model with the same preprocessing and specified regularization coefficient,
    and run cross_val_predict on it.

    Usually cross-val predictions and/or coefs are stored in fitted *CV models (like
    LogisticRegressionCV.coefs_paths_ or RidgeCV.cv_values_, but this depends on
    a concrete class; cross_val_predict is made for consistency across
    classes).
    """

    def __init__(
        self,
        task_type: TaskType,
        reg_type: LinearRegType = 'l2',
        reg_coefs: list[float] | Literal['auto'] = 'auto',
        preproc: TransformerMixin | sklearn.pipeline.Pipeline | None = None,
        y_transform: TargetTransformType = 'standard',
        random_seed: int = 0,
        n_jobs: int = -1,
        max_iter: int = 100_000,
        scoring: str | None = None,
        n_folds: int = 5,
        multiclass: Literal['multinomial', 'ovr', 'auto'] = 'auto',
        l1_ratio: float | list[float] = 0.5,
    ):
        self._n_jobs = n_jobs
        self._reg_type = reg_type
        if task_type == TaskType.REGRESSION:
            if reg_type == 'l2':
                linear_model_cv, linear_model_nocv = _make_RidgeCV(
                    reg_coefs=reg_coefs,
                    scoring=scoring,
                )
            elif reg_type == 'l1':
                assert scoring is None, 'scoring for LassoCV is not supported'
                linear_model_cv, linear_model_nocv = _make_LassoCV(
                    reg_coefs=reg_coefs,
                    random_seed=random_seed,
                    n_jobs=n_jobs,
                    max_iter=max_iter,
                    n_folds=n_folds,
                )
            elif reg_type == 'elasticnet':
                assert (
                    scoring is None
                ), 'scoring for ElasticNetCV is not supported'
                linear_model_cv, linear_model_nocv = _make_ElasticNetCV(
                    reg_coefs=reg_coefs,
                    random_seed=random_seed,
                    n_jobs=n_jobs,
                    max_iter=max_iter,
                    n_folds=n_folds,
                    l1_ratio=l1_ratio,
                )
            elif reg_type == 'omp':
                assert (
                    scoring is None
                ), 'scoring for OrthogonalMatchingPursuitCV is not supported'
                assert (
                    reg_coefs == 'auto'
                ), 'reg_coefs for OrthogonalMatchingPursuitCV are not supported'
                (
                    linear_model_cv,
                    linear_model_nocv,
                ) = _make_OrthogonalMatchingPursuitCV(
                    max_nonzero_coefs=None,  # will be set in .fit()
                    n_jobs=n_jobs,
                    n_folds=n_folds,
                )
            # apply y transform
            linear_model_cv = make_transformed_target_regressor(
                linear_model_cv, y_transform, random_seed
            )
            linear_model_nocv = make_transformed_target_regressor(
                linear_model_nocv, y_transform, random_seed
            )
        elif task_type in (TaskType.BINARY, TaskType.MULTICLASS):
            assert reg_type != 'omp', 'omp is not supported for classification'
            linear_model_cv, linear_model_nocv = _make_LogisticRegressionCV(
                task_type=task_type,
                reg_type=reg_type,
                reg_coefs=reg_coefs,
                random_seed=random_seed,
                n_jobs=n_jobs,
                max_iter=max_iter,
                scoring=scoring,
                n_folds=n_folds,
                multiclass=multiclass,
                l1_ratios=(
                    l1_ratio if isinstance(l1_ratio, list) else [l1_ratio]
                ),
            )

        self._has_preproc = preproc is not None
        maybe_preproc = [preproc] if self._has_preproc else []
        model = make_pipeline(*maybe_preproc, linear_model_cv)
        super().__init__(
            task_type=task_type, model=model, random_seed=random_seed
        )
        self._model_cv = self._model
        self._model_nocv = make_pipeline(*maybe_preproc, linear_model_nocv)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        if self._reg_type == 'omp':
            if self._has_preproc:
                preproc = self._get_preproc(cv=True)
                X_preprocessed = clone(preproc).fit_transform(X)
                n_features = X_preprocessed.shape[1]
            else:
                n_features = X.shape[1]
            model = self._get_linear_model(cv=True, fitted=False)
            model.max_iter = n_features
            print('set to', X.shape[1])
        return super().fit(X, y)

    def _get_preproc(
        self, cv: bool = True
    ) -> sklearn.pipeline.Pipeline | None:
        if self._has_preproc:
            pipeline = self._model_cv if cv else self._model_nocv
            return pipeline[0]
        else:
            return None

    def _get_linear_model(
        self, cv: bool, fitted: bool = True
    ) -> BaseEstimator:
        pipeline = self._model_cv if cv else self._model_nocv
        last_el = pipeline[-1]
        if isinstance(last_el, TransformedTargetRegressor):
            if fitted:
                return last_el.regressor_
            else:
                return last_el.regressor
        else:
            return last_el

    def get_best_coef(
        self, mean_for_multiclass: bool = True
    ) -> float | int | list[float] | list[int]:
        assert self._fitted, 'model is not fitted, best coef is unknown'
        if self._task_type == TaskType.REGRESSION:
            if self._reg_type == 'omp':
                coef = self._get_linear_model(cv=True).n_nonzero_coefs_
            else:
                coef = self._get_linear_model(cv=True).alpha_
        else:
            coef = self._get_linear_model(cv=True).C_
            if mean_for_multiclass:
                coef = np.mean(coef)
        return coef

    def set_coef(self, coef: float | int) -> None:
        linear_model = self._get_linear_model(cv=False, fitted=False)
        if self._task_type == TaskType.REGRESSION:
            if self._reg_type == 'omp':
                linear_model.n_nonzero_coefs = coef
            else:
                linear_model.alpha = coef
        else:
            linear_model.C = coef

    def get_best_l1_ratio(
        self, mean_for_multiclass: bool = True
    ) -> float | list[float]:
        assert self._fitted, 'model is not fitted, best coef is unknown'
        l1_ratio = self._get_linear_model(cv=True).l1_ratio_
        if self._task_type != TaskType.REGRESSION:
            if mean_for_multiclass:
                l1_ratio = np.mean(l1_ratio)
        return l1_ratio

    def set_l1_ratio(self, l1_ratio: float) -> None:
        linear_model = self._get_linear_model(cv=False, fitted=False)
        linear_model.l1_ratio = l1_ratio

    def cross_val_predict_with_fixed_coef(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        coef: float | int | Literal['best'] = 'best',
        l1_ratio: float | Literal['best'] = 'best',
        n_folds: int = 5,
    ) -> np.ndarray:
        """
        Do cross_val_predict for given (fixed) coef. If coef == 'best', obtain best
        coef (alpha or C) from fitted CV model. Same for l1_ratio.
        """
        # set coef for noCV model
        if coef == 'best':
            coef = self.get_best_coef()  # type: ignore[assignment]
        self.set_coef(coef)  # type: ignore[arg-type]

        # set l1_ratio for noCV model
        if self._reg_type == 'elasticnet':
            if l1_ratio == 'best':
                l1_ratio = self.get_best_l1_ratio()  # type: ignore[assignment]
            self.set_l1_ratio(l1_ratio)  # type: ignore[arg-type]

        saved_model = self._model
        self._model = self._model_nocv
        preds = SklearnModel.cross_val_predict(
            self, X, y, n_folds=n_folds, n_jobs=self._n_jobs
        )
        self._model = saved_model

        return preds


def make_linear_model(
    task_type: TaskType,
    l1: float = 0,
    l2: float = 0,
    max_iter: int = 100_000,
    ridge_solver: str | None = None,
    logreg_solver: str | None = None,
    multiclass: Literal['multinomial', 'ovr'] = 'multinomial',
    num_cpus: int = -1,
    random_seed: int = 0,
    y_transform: TargetTransformType = 'standard',
) -> (
    TransformedTargetRegressor
    | LinearRegression
    | Ridge
    | Lasso
    | ElasticNet
    | LogisticRegression
):
    """
    Returns the model that will minimize the following objective:
    ||y - Xw||^2_2 + 2 * _l1 * ||w||^2_2 + _l2 * ||w||_1

    For classification will return LogisticRegression with all required params set
    (including regularization coefs, solver, seed, random_state, n_jobs). For
    regression, depending on l1 > 0 and/or l2 > 0, will return LinearRegression,
    Ridge, Lasso or ElasticNet with all required params set. Max iter is set to
    100_000.

    "multiclass" param is used for LogisticRegression in multiclass case:

    LogisticRegression(
        ...,
        multi_class=multiclass if task_type == TaskType.MULTICLASS else "ovr"
    )

    Using "ovr" (one-vs-all) means fitting a binary classifier for each class.
    Denote logits as l_1, ..., l_N, and true class as C, then OVR is equievlent
    to train multi-output model with the following loss:

    loss = -l_C + sum[i] log(1 + exp l_i)        (1)

    Multinomial loss has the following equation:

    loss = -l_C + log sum[i] exp l_i             (2)

    In equation (1) the derivative of loss with respect to l_i does not depend
    on the values of other logits (unlike eq. (2)), so each binary classifier
    may be trained independently.
    """
    if task_type == TaskType.REGRESSION:
        if l1 == 0 and l2 == 0:
            model = LinearRegression(n_jobs=num_cpus)
        elif l1 == 0 and l2 > 0:
            model = Ridge(
                alpha=l2,
                max_iter=max_iter,
                random_state=random_seed,
                solver=ridge_solver or 'auto',
            )
        elif l1 > 0 and l2 == 0:
            model = Lasso(
                alpha=l1,
                max_iter=max_iter,
                random_state=random_seed,
            )
        else:  # l1 > 0 and l2 > 0
            model = ElasticNet(
                alpha=l1 + l2,
                l1_ratio=l1 / (l1 + l2),
                max_iter=max_iter,
                random_state=random_seed,
            )
        if y_transform != 'none':
            model = make_transformed_target_regressor(
                model, y_transform, random_seed
            )
        return model
    else:
        kwargs = Bunch()
        kwargs.multi_class = (
            'ovr' if task_type == TaskType.BINARY else multiclass
        )
        kwargs.random_state = random_seed
        kwargs.max_iter = max_iter
        kwargs.n_jobs = num_cpus
        default_l2_solver = (
            'newton-cholesky' if task_type == TaskType.BINARY else 'lbfgs'
        )
        if l1 == 0 and l2 == 0:
            kwargs.penalty = None
            kwargs.solver = logreg_solver or default_l2_solver
        elif l1 == 0 and l2 > 0:
            kwargs.penalty = 'l2'
            kwargs.C = 1 / (2 * l2)
            kwargs.solver = logreg_solver or default_l2_solver
        elif l1 > 0 and l2 == 0:
            kwargs.penalty = 'l1'
            kwargs.solver = logreg_solver or 'saga'
            kwargs.C = 1 / 2 / l1
        else:
            kwargs.penalty = 'elasticnet'
            kwargs.solver = logreg_solver or 'saga'
            kwargs.C = 1 / 2 / (l1 + l2)
            kwargs.l1_ratio = l1 / (l1 + l2)

        return LogisticRegression(**kwargs)


def should_use_newton_cholesky_ovr(X: pd.DataFrame, y: pd.Series) -> bool:
    """
    Returns true if n_samples * sqrt(n_features) * n_classes > 10**6

    For this large datasets, newton-cholesky solver (ovr) is better because
    it is faster, otherwise it's better to use lbfgs (multinomial).
    """
    n_samples, n_features = X.shape
    n_classes = len(y.cat.categories)
    return n_samples * np.sqrt(n_features) * n_classes > 10**6
