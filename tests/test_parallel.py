import numpy as np
from sklearn.metrics import roc_auc_score

from tabular_models import chunk_parallel, sequence_parallel

size = 1_000
y_true = np.random.default_rng(0).integers(low=0, high=2, size=size)


def staged_predict(n_steps) -> np.ndarray:
    for step in range(n_steps):
        yield np.random.default_rng(step).normal(size=size).astype(np.float64)


def score(predictions: np.ndarray) -> tuple[int, float]:
    # start_time = time.time()
    score = roc_auc_score(y_true, predictions)
    # print(f'Score {score:.4f}, time {time.time() - start_time:.2f}')
    return score


def test_parallel():
    n_steps = 50

    results_no_parallel = [
        score(preds) for i, preds in enumerate(staged_predict(n_steps))
    ]

    for n_chunks in [1, 3, 7, 8, 49, 50, 100]:
        results_chunk_parallel = chunk_parallel(
            staged_predict(n_steps),
            score,
            n_chunks=n_chunks,
        )
        assert np.allclose(results_no_parallel, results_chunk_parallel)

    for max_workers, max_queued, chunk_size in [
        (9, 15, 1),
        (8, 8, 2),
        (1, 30, 99),
        (1, 1000, 1),
        (99, 5, 99),
    ]:
        results_sequence_parallel = sequence_parallel(
            staged_predict(n_steps),
            score,
            max_workers=max_workers,
            max_queued=max_queued,
            chunk_size=chunk_size,
        )
        assert np.allclose(results_no_parallel, results_sequence_parallel)
