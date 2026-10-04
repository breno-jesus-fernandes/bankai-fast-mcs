"""Compare the native Rust core with the pinned academic fastMCS source."""

import argparse
import importlib.util
import statistics
import time
from pathlib import Path

import numpy as np

from bankai_fast_mcs import _native


REFERENCE_PATH = Path(__file__).parent / "reference" / "fastMCS.py"
SPEC = importlib.util.spec_from_file_location("fastMCS_reference", REFERENCE_PATH)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


def make_inputs(observations, models, bootstraps):
    rng = np.random.default_rng(20261004)
    losses = rng.normal(size=(observations, models))
    losses += np.arange(models)[None, :] * 0.001
    indices = rng.integers(0, observations, size=(observations, bootstraps))
    return losses, indices


def run_reference(losses, indices, algorithm):
    REFERENCE.blockBootstrap = lambda rng, obs, count, block: indices
    model_set = REFERENCE.mcs(seed=1, verbose=False)
    model_set.addLosses(losses)
    model_set.run(
        B=indices.shape[1], b=10, bootstrap="block", algorithm=algorithm
    )
    return model_set.tScore, model_set.exclMods, model_set.tBootDist


def run_rust(losses, indices, algorithm):
    return _native._run_fast_mcs(losses, indices, algorithm)


def measure(function, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=int, default=100)
    parser.add_argument("--models", type=int, default=50)
    parser.add_argument("--bootstraps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--algorithm", choices=["1-pass", "2-pass"], default="2-pass")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()

    losses, indices = make_inputs(args.observations, args.models, args.bootstraps)
    reference_result = run_reference(losses, indices, args.algorithm)
    reference_seconds = measure(
        lambda: run_reference(losses, indices, args.algorithm), args.repeats
    )
    print(f"academic fastMCS median: {reference_seconds:.6f} s")

    if not args.reference_only:
        rust_result = run_rust(losses, indices, args.algorithm)
        np.testing.assert_allclose(rust_result[0], reference_result[0], rtol=1e-11, atol=1e-12)
        np.testing.assert_array_equal(rust_result[1], reference_result[1])
        np.testing.assert_allclose(
            np.asarray(rust_result[2]).reshape(indices.shape[1], losses.shape[1]),
            reference_result[2],
            rtol=1e-11,
            atol=1e-12,
        )
        rust_seconds = measure(lambda: run_rust(losses, indices, args.algorithm), args.repeats)
        print(f"Rust core median:        {rust_seconds:.6f} s")
        print(f"speedup:                  {reference_seconds / rust_seconds:.2f}x")
        if rust_seconds >= reference_seconds:
            raise SystemExit("Rust core benchmark must beat the academic Python version")


if __name__ == "__main__":
    main()
