"""Parallelization helpers for experiment-level work.

Strategy:
- Algorithm-level (inside game tree recursion): memoization (`functools.cache`)
  provides large speedups. Cross-process cache sharing is more expensive
  than the recursion itself, so we do NOT parallelize there.
- Experiment-level (over many independent game pairs / instances):
  joblib.Parallel gives near-linear scaling with cores.
- GPU: only relevant for embedding/ML workloads (not CGT core),
  deferred to potential ML notebooks.
"""

import os
from collections.abc import Callable, Iterable
from typing import TypeVar

from joblib import Parallel, delayed

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    n_jobs: int = -1,
    verbose: int = 0,
    backend: str = "loky",
) -> list[R]:
    """Parallel map over items using joblib.

    Args:
        func: Function applied to each item. Must be picklable.
        items: Iterable of inputs.
        n_jobs: Number of parallel workers. -1 = all cores.
        verbose: Joblib verbosity (0=silent, 10=progress).
        backend: 'loky' (default, robust), 'threading', or 'multiprocessing'.

    Returns:
        List of results in input order.

    Note: Each worker process builds its own memoization cache from scratch.
    For very fine-grained tasks, single-threaded with cache may be
    faster. Profile if unsure.
    """
    return Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
        delayed(func)(item) for item in items
    )


def detect_compute_backend() -> dict:
    """Report available compute resources for diagnostic purposes."""
    info = {
        "cpu_count": os.cpu_count(),
        "cpu_count_physical": None,
        "cuda_available": False,
        "mps_available": False,
        "accelerate_blas": False,
    }
    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        pass
    try:
        import numpy as np

        config = np.show_config(mode="dicts")
        if config and "Build Dependencies" in config:
            blas = config.get("Build Dependencies", {}).get("blas", {})
            info["accelerate_blas"] = "accelerate" in str(blas).lower()
    except Exception:
        pass
    return info
