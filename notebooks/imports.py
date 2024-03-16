# ruff: noqa

import sys
from itertools import product

sys.path.append('/data/osedukhin/tabular-models')

from tabular_models import *
from tabular_models.models import *
from tabular_models.pipelines import *
import tabular_models.pipelines as pipelines

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from enum import Enum
from dataclasses import dataclass

from typing import Literal, Any

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import sklearn.pipeline
from sklearn.utils import Bunch
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    cross_val_score,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
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
