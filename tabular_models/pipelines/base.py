from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tabular_models.dataset import Dataset
from tabular_models.pipeline_result import PipelineResult

MB = 2**20
GB = 2**30


@dataclass
class Req:
    pipeline: str
    split: str
    fields: str | list[str] | None = None


PrevResults = dict[tuple[str, str], PipelineResult]  # pipeline and split


class Pipeline(ABC):
    CPUS: int = 1
    MEMORY: int = 1 * GB

    @abstractmethod
    def splits_and_requirements(self) -> dict[str, list[Req]]:
        ...

    @abstractmethod
    def run(self, dataset: Dataset, reqs: PrevResults) -> PipelineResult:
        ...
