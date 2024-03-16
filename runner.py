# mypy: disable_error_code="arg-type"

import argparse
import itertools
import re
import sys
from typing import Any

import ray

import tabular_models
from tabular_models import (
    Dataset,
    DatasetSplitter,
    DataStore,
    PipelineResult,
)
from tabular_models.pipelines import get_pipeline_object
from tabular_models.pipelines.base import Pipeline, PrevResults, Req

CLUSTER_NAMES = {
    'default_cluster': '10.198.126.80',
    'cpu_cluster': '10.198.127.107',
    'gpu_cluster': '10.198.127.211',
}
RUNTIME_ENV = {
    'pip': [
        'catboost==1.2.2',
        'scikit-learn==1.4.0',
        'pandas==2.1.3',
        'polars==0.19.15',
        'more_itertools',
        '--trusted-host 7.223.199.227',
        '-i http://7.223.199.227/pypi/simple',
    ],
    'py_modules': [tabular_models],
}


def init_ray(args):
    if not args.ray or args.ray == 'localhost':
        ray.init()
    else:
        if args.ray in CLUSTER_NAMES:
            print(f'Ray cluster name "{args.ray}"')
            ip = CLUSTER_NAMES[args.ray]
        else:
            ip = args.ray
        url = f'ray://{ip}:10001'
        print(f'Connecting to Ray URL: {url}')
        ray.init(address=url, runtime_env=RUNTIME_ENV)


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ray',
        type=str,
        help=(
            'Name or IP of the ray cluster head,'
            ' default IP is specified inside runner.py',
        ),
        default='default_cluster',
        required=False,
    )
    parser.add_argument(
        '--order',
        type=str,
        help='dataset-fold or fold-dataset',
        default='fold-dataset',
        required=False,
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Path to root dir with datasets and results',
        default='/data/tabular_data',
        required=False,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='A single dataset to run',
        default=[],
        required=False,
        action='append',
    )
    parser.add_argument(
        '--exclude_dataset',
        type=str,
        help='Exclude a dataset',
        default=[],
        required=False,
        action='append',
    )
    parser.add_argument(
        '--min_trainval_samples',
        type=int,
        help='Filter datasets with n_trainval_samples < min_trainval_samples',
        required=False,
    )
    parser.add_argument(
        '--max_trainval_samples',
        type=int,
        help='Filter datasets with n_trainval_samples >= max_trainval_samples',
        required=False,
    )
    parser.add_argument(
        '--max_features',
        type=int,
        help='Filter datasets with n_features >= max_features',
        required=False,
    )
    parser.add_argument(
        '--folds',
        type=str,
        help='Folds to run, in form of N or N1-N2 (incluive)',
        required=False,
        default='0-9',
    )
    parser.add_argument(
        '--pipeline',
        type=str,
        help='Name under which pipeline was registered in tabular_models',
        default=[],
        required=True,
        action='append',
    )
    parser.add_argument(
        '--try_once',
        action='store_true',
        help='Run the pipeline on first dataset and fold, then return',
    )
    args = parser.parse_args()

    if re.fullmatch(r'[0-9]+', args.folds):
        args.folds = [int(args.folds)]
    elif match := re.fullmatch(r'([0-9]+)-([0-9]+)', args.folds):
        _from, _to = match.groups()
        args.folds = list(range(int(_from), int(_to) + 1))

    return args


@ray.remote
def run(
    key: tuple[str, int, str, str],
    dataset: Dataset,
    pipeline: Pipeline,
    reqs: dict[tuple[str, str], PipelineResult],
) -> tuple[tuple[str, int, str, str], PipelineResult]:
    dataset_name, fold, _pipeline_name, split = key
    if dataset_name.endswith('[rev]'):
        dataset.metadata['reversed'] = True
    dataset.set_fold(fold)
    dataset = DatasetSplitter.from_string(split).split(dataset)
    result = pipeline.run(dataset=dataset, reqs=reqs)
    return key, result


DATASET_REFS: dict[str, ray.ObjectRef] = {}


def schedule_task(
    data_store: DataStore,
    dataset_name: str,
    fold: int,
    split: str,
    reqs: list[Req],
    pipeline_name: str,
    pipeline: Pipeline,
    cpus: int,
    memory: int,
) -> Any:   # return ref or None

    try:
        loaded_reqs: PrevResults = {}
        for req in reqs:
            loaded_req = data_store.load_pipeline_result(
                dataset_name=dataset_name,
                fold=fold,
                pipeline_name=req.pipeline,
                split_name=req.split,
                fields=req.fields,
            )
            if loaded_req is None:
                raise FileNotFoundError()
            loaded_reqs[(req.pipeline, req.split)] = loaded_req
    except FileNotFoundError:
        print(
            'No prev results for'
            f' {dataset_name}/{fold}/{req.pipeline}/{req.split}'
        )
        return None

    try:
        reqs_ref = ray.put(loaded_reqs)
    except Exception as e:
        print(f'Cannot ray.put required results {dataset_name}/{fold}: {e}')
        return None

    dataset_name_norev = dataset_name.replace('[rev]', '')
    if dataset_name_norev not in DATASET_REFS:
        try:
            dataset = data_store.load_dataset(dataset_name_norev, fold=None)
        except Exception as e:
            print(f'Exception while loading dataset {dataset_name_norev}: {e}')
            return None
        DATASET_REFS[dataset_name_norev] = ray.put(dataset)
    dataset_ref = DATASET_REFS[dataset_name_norev]

    result_ref = run.options(
        name=f'{dataset_name}/{fold}/{pipeline_name}/{split}',
        num_cpus=cpus,
        memory=memory,
    ).remote(
        key=(dataset_name, fold, pipeline_name, split),
        dataset=dataset_ref,
        pipeline=pipeline,
        reqs=reqs_ref,
    )
    print(
        f'Queued: {dataset_name}/{fold}/{pipeline_name}/{split}'
        f' (num cpus: {cpus})'
    )
    return result_ref


def finish_task(
    data_store: DataStore,
    key: tuple[str, int, str, str],
    result: PipelineResult,
) -> None:
    dataset_name, fold, pipeline_name, split = key
    data_store.save_pipeline_result(
        result, dataset_name, fold, pipeline_name, split
    )
    print(f'Finished: {dataset_name}/{fold}/{pipeline_name}/{split}')


def format_exception(e: ray.exceptions.RayError) -> str:
    return '\n'.join(['### ' + line for line in f'Exception: {e}'.split('\n')])


def main():
    args = get_cli_args()
    init_ray(args)

    data_store = DataStore(args.dir)
    if len(args.dataset) > 0:
        # run on a specified list of datasets
        dataset_names = args.dataset
    else:
        # run on all datasets
        dataset_names = list(
            data_store.list_datasets(
                exclude=args.exclude_dataset,
                min_trainval_samples=args.min_trainval_samples,
                max_trainval_samples=args.max_trainval_samples,
                max_features=args.max_features,
            )
        )

    if len(dataset_names) == 0:
        print('No datasets to run')
        sys.exit(0)

    print('Will run on datasets:')
    print('\n'.join([f'{i} {n}' for i, n in enumerate(dataset_names)]))

    if args.order == 'dataset-fold':
        datasets_and_folds = [
            (fold, dataset)
            for dataset, fold in list(
                itertools.product(dataset_names, args.folds)
            )
        ]
    elif args.order == 'fold-dataset':
        datasets_and_folds = list(itertools.product(args.folds, dataset_names))
    else:
        raise AssertionError()

    if args.try_once:
        datasets_and_folds = datasets_and_folds[:1]

    result_refs = []
    for fold, dataset_name in datasets_and_folds:
        for pipeline_name in args.pipeline:
            pipeline = get_pipeline_object(pipeline_name)
            split_and_reqs: dict[
                str, list[Req]
            ] = pipeline.splits_and_requirements()
            for split in split_and_reqs.keys():
                if data_store.result_exists(
                    dataset_name, fold, pipeline_name, split
                ):
                    print(
                        'Already exists:'
                        f' {dataset_name}/{fold}/{pipeline_name}/{split}'
                    )
                    continue
                ref = schedule_task(
                    data_store=data_store,
                    dataset_name=dataset_name,
                    fold=fold,
                    split=split,
                    reqs=split_and_reqs[split],
                    pipeline_name=pipeline_name,
                    pipeline=pipeline,
                    cpus=pipeline.CPUS,
                    memory=pipeline.MEMORY,
                )
                if ref is not None:
                    result_refs.append(ref)
                ready_result_refs, result_refs = ray.wait(
                    result_refs, num_returns=len(result_refs), timeout=0
                )
                for ready_result in ready_result_refs:
                    try:
                        key, result = ray.get(ready_result)
                        finish_task(data_store, key, result)
                    except ray.exceptions.RayError as e:
                        print(format_exception(e))

    while len(result_refs) > 0:
        ready_result_refs, result_refs = ray.wait(result_refs, num_returns=1)
        try:
            key, result = ray.get(ready_result_refs)[0]
            finish_task(data_store, key, result)
        except ray.exceptions.RayError as e:
            print(format_exception(e))


if __name__ == '__main__':
    main()
