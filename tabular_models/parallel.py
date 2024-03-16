from asyncio import ALL_COMPLETED, FIRST_COMPLETED
from collections.abc import Callable, Iterable
from concurrent import futures
from concurrent.futures import wait
from typing import Any

import numpy as np
from more_itertools import chunked


class KwargsWrapper:
    def __init__(self, function):
        self.function = function

    def __call__(self, kwargs):
        return self.function(**kwargs)


class _EnumerateFn:
    def __init__(self, function):
        self.function = function

    def __call__(self, args):
        i, inputs = args
        return (i, self.function(inputs))


class _ChunkFn:
    def __init__(self, function):
        self.function = function

    def __call__(self, chunk):
        return [self.function(x) for x in chunk]


def chunk_parallel(iterator: Iterable, function: Callable, n_chunks: int):
    """
    Turns iterator into list, then splits it into the specified number of chunks.
    Each chunk goes into its own process, when the specified function is applied
    to every element of the chunk. Finally, all results are combined.

    These calls will produce equal results:
        >>> import math
        >>> [math.sqrt(x) for x in range(10)]
        >>> chunk_parallel(range(10), math.sqrt, n_chunks=4)
    """
    executor = futures.ProcessPoolExecutor(max_workers=n_chunks)
    not_done = set()

    iterator_results = list(iterator)
    iterator_results_np = np.empty(len(iterator_results), dtype='object')
    for i in range(len(iterator_results)):
        iterator_results_np[i] = iterator_results[i]

    for chunk in enumerate(np.array_split(iterator_results_np, n_chunks)):
        not_done.add(executor.submit(_EnumerateFn(_ChunkFn(function)), chunk))

    ready_results, _ = wait(not_done, return_when=ALL_COMPLETED)
    ready_results_evaluated = [result.result() for result in ready_results]

    order = np.argsort([i for i, x in ready_results_evaluated])
    reordered_results = [ready_results_evaluated[i][1] for i in order]

    executor.shutdown()
    return sum(reordered_results, [])


def sequence_parallel(
    iterator: Iterable,
    function: Callable,
    max_workers: int,
    max_queued: int,
    chunk_size: int,
):
    """
    Applies function to the iterator results in a parallel way. Runs max_workers
    separate processes concurrently, sequentially evaluates iterator and calculates
    values of the function concurrently. Evaluates no more than max_queued elements
    of the iterator at every moment of time.

    Use this method instead of chunk_parallel when list(iterator) will not fit
    into memory.

    chunk_size is used to evaluate chunk_size iterator elements and pass them
    to the subprocess at once

    These calls will produce equal results:
        >>> import math
        >>> [math.sqrt(x) for x in range(10)]
        >>> sequence_parallel(range(10), math.sqrt, max_workers=3, max_queued=6)
    """
    executor = futures.ProcessPoolExecutor(max_workers=max_workers)
    ready_results: list[Any] = []
    not_done: set[Any] = set()

    for input_args_batch in chunked(enumerate(iterator), chunk_size):
        while len(not_done) >= max_queued:
            done, not_done = wait(not_done, return_when=FIRST_COMPLETED)
            ready_results.extend(done)
        not_done.add(
            executor.submit(_ChunkFn(_EnumerateFn(function)), input_args_batch)
        )

    done, _ = wait(not_done, return_when=ALL_COMPLETED)
    ready_results.extend(done)
    ready_results_evaluated = [result.result() for result in ready_results]
    ready_results_evaluated = sum(ready_results_evaluated, [])

    order = np.argsort([i for i, x in ready_results_evaluated])
    executor.shutdown()
    return [ready_results_evaluated[i][1] for i in order]
