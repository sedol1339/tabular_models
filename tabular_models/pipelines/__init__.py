# ruff: noqa

import re
import sys

from .base import *
from .catboost import *
from .linear import *
from .scoring import *
from .single_feature import *


def get_pipeline_object(pipeline_name: str) -> Pipeline:
    pipelines = sys.modules[__name__]
    # this regex is searching for parametrized pipeline names,
    # where parameters are in brackets, like "MyPipeline[params_string]"
    if match := re.fullmatch(r'([^\[\]]+)\[([^\[\]]+)\]', pipeline_name):
        pipeline_name, pipeline_params = match.groups()
        pipeline_cls = getattr(pipelines, pipeline_name)
        pipeline_obj = pipeline_cls(pipeline_params)
    else:
        assert not '[' in pipeline_name, 'Bad pipeline name'
        assert not ']' in pipeline_name, 'Bad pipeline name'
        pipeline_cls = getattr(pipelines, pipeline_name)
        pipeline_obj = pipeline_cls()
    return pipeline_obj
