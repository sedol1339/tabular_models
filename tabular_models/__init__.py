from pathlib import Path

from .data_store import DataStore
from .dataset import Dataset
from .dataset_splitter import DatasetSplitter
from .parallel import chunk_parallel, sequence_parallel
from .pipeline_result import PipelineResult
from .predictions import Predictions
from .scores import (
    ScoreDeltas,
    Scores,
    bootstrap_gain,
    bootstrap_matrix,
    gains_summary_plot,
    plot_bootstrap_matrix,
)
from .task_type import TaskType
from .utils import (
    Timer,
    _validate_filename,
    display_full_dataframe,
    get_best_iter,
    visualize_feature_matrix,
)


def get_project_root() -> Path:
    """Returns a path to the project's root directory."""
    return Path(__file__).parent.parent.resolve()
