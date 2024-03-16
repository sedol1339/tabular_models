from __future__ import annotations

from enum import Enum


class TaskType(Enum):
    """
    TaskType is associated with every dataset and is used to construct BaseModel and
    Predictions objects. See more details in Predictions docstring.
    """

    REGRESSION = 1
    BINARY = 2
    MULTICLASS = 3

    def __str__(self) -> str:
        return {
            TaskType.REGRESSION: 'regression',
            TaskType.BINARY: 'binary',
            TaskType.MULTICLASS: 'multiclass',
        }[self]

    @classmethod
    def from_string(cls, name: str) -> TaskType:
        return {
            'regression': TaskType.REGRESSION,
            'binary': TaskType.BINARY,
            'multiclass': TaskType.MULTICLASS,
        }[name]
