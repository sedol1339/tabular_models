from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sklearn.utils import Bunch
from yaml import CLoader

from .utils import _validate_filename


@dataclass
class PipelineResult(Bunch):
    """
    Stores, saves or loads pipeline results.

    .save(directory) is done as follows:
    - all fields are saved with suffix '.pkl'
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        kvs = ', '.join([f'{k}={v}' for k, v in self.items()])
        return f'PipelineResult({kvs})'

    def validate_filenames(self) -> None:
        all_names = [f'{x}.pkl' for x in self.keys()]
        # check special symbols
        for name in all_names:
            assert _validate_filename(name), f'bad filename {name}'
        # check special names
        for name in {'metadata.yaml'}:
            assert name not in all_names, f'Filename "{name}" is reserved'

    def save(self, directory: str | Path) -> None:
        """
        Saving to the specified directory is done as follows:
        - all fields are saved with suffix '.pkl'

        Creates the directory if needed. If the directory exists and not empty, will
        rewrite the existing files.
        """
        # make directory
        directory = Path(directory)
        if directory.is_file():
            raise AssertionError(f'"{directory}" is file')
        else:
            directory.mkdir(parents=True, exist_ok=True)

        self.validate_filenames()

        # save pickles
        for name, obj in self.items():
            with open(directory / f'{name}.pkl', 'wb') as h:
                pickle.dump(obj, h)

        # save metadata
        metadata: dict[str, Any] = {'pickles': list(self.keys())}
        with open(directory / 'metadata.yaml', 'w') as h:
            h.write(yaml.dump(metadata))

    @classmethod
    def exists(
        cls,
        directory: str | Path,
    ) -> bool:
        """
        Checks if PipelineResult exists in the folder
        """
        directory = Path(directory)
        return (directory / 'metadata.yaml').is_file()

    @classmethod
    def load(
        cls,
        directory: str | Path,
        fields: str | list[str] | None = None,
    ) -> PipelineResult:
        """
        Loads saved PipelineResult from the specified directory.

        If fields argument is passed, will load only specified fields
        """
        directory = Path(directory)

        with open(directory / 'metadata.yaml') as h:
            metadata = yaml.load(h, Loader=CLoader)

        pickles = {}
        if isinstance(fields, str):
            fields = [fields]
        for name in metadata['pickles']:
            if fields is None or name in fields:
                with open(directory / f'{name}.pkl', 'rb') as h:
                    pickles[name] = pickle.load(h)

        return PipelineResult(**pickles)

    # def plot(
    #     self,
    #     train: bool = True,
    #     val: bool = True,
    #     test: bool = True,
    #     xscale: str | None = None,
    #     h = 5,
    # ) -> None:
    #     """
    #     Plots train, val, test learning curves
    #     """
    #     assert self.scores is not None
    #     all_scores = self.scores[self.scores.scoring_type == 'full']

    #     def do_plot(scores: pd.DataFrame, subset: str, ax: plt.Axes):
    #         color = {
    #             'train': 'C0',
    #             'val': 'C2',
    #             'test': 'C1',
    #         }.get(subset, 'C4')

    #         if len(scores) == 0:
    #             return
    #         scores = scores.sort_values('step')
    #         ax.plot(
    #             scores.step.to_numpy() + 1,
    #             scores.score.to_numpy(),
    #             label=subset,
    #             c=color,
    #         )

    #     fig, axs = plt.subplots(
    #         ncols=all_scores.metric.nunique(), figsize=(9, h)
    #     )

    #     for (metric, scores), ax in zip(
    #         all_scores.groupby('metric'), axs.flat
    #     ):
    #         ax.set_title(metric)
    #         if train:
    #             do_plot(scores[scores.subset == 'train'], 'train', ax=ax)
    #         if val:
    #             do_plot(scores[scores.subset == 'val'], 'val', ax=ax)
    #         if test:
    #             do_plot(scores[scores.subset == 'test'], 'test', ax=ax)
    #         ax.legend()
    #         if 'best_iteration' in self.values:
    #             ax.axvline(self.values['best_iteration'], color='r')
    #         if xscale is not None:
    #             ax.set_xscale(xscale)

    #     plt.tight_layout()
    #     plt.show()
