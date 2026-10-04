"""End-to-end timing against the pinned academic fastMCS implementation."""

import argparse
import importlib.util
import statistics
import time
from pathlib import Path

import numpy as np


REFERENCE_PATH = Path(__file__).parent / "reference" / "fastMCS.py"
SPEC = importlib.util.spec_from_file_location(
    "fastMCS_public_api_reference", REFERENCE_PATH
)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


def make_inputs(observations, models):
    rng = np.random.default_rng(20261004)
    losses = rng.normal(size=(observations, models))
    losses += np.arange(models)[None, :] * 0.001
    return losses


def run_reference(losses, seed, bootstraps, block_size, bootstrap, algorithm):
    model_set = REFERENCE.mcs(seed=seed, verbose=False)
    model_set.addLosses(losses)
    model_set.run(
        B=bootstraps,
        b=block_size,
        bootstrap=bootstrap,
        algorithm=algorithm,
    )
    included, excluded = model_set.getMCS()
    return model_set, included, excluded


def run_native(losses, seed, bootstraps, block_size, bootstrap, algorithm):
    from bankai_fast_mcs import ModelConfidenceSet

    model_set = ModelConfidenceSet(losses, seed=seed, verbose=False)
    model_set.run(
        B=bootstraps,
        b=block_size,
        bootstrap=bootstrap,
        algorithm=algorithm,
    )
    included, excluded = model_set.get_mcs()
    return model_set, included, excluded


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
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--bootstrap", choices=["block", "stationary"], default="stationary")
    parser.add_argument("--algorithm", choices=["1-pass", "2-pass"], default="2-pass")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()

    losses = make_inputs(args.observations, args.models)
    expected, expected_included, expected_excluded = run_reference(
        losses,
        args.seed,
        args.bootstraps,
        args.block_size,
        args.bootstrap,
        args.algorithm,
    )
    reference_seconds = measure(
        lambda: run_reference(
            losses,
            args.seed,
            args.bootstraps,
            args.block_size,
            args.bootstrap,
            args.algorithm,
        ),
        args.repeats,
    )
    print(f"academic fastMCS median: {reference_seconds:.6f} s")

    if not args.reference_only:
        actual, actual_included, actual_excluded = run_native(
            losses,
            args.seed,
            args.bootstraps,
            args.block_size,
            args.bootstrap,
            args.algorithm,
        )
        np.testing.assert_allclose(actual.t_score, expected.tScore, rtol=1e-11, atol=1e-12)
        np.testing.assert_array_equal(actual.elimination_order, expected.exclMods)
        np.testing.assert_allclose(
            actual.t_boot_distribution, expected.tBootDist, rtol=1e-11, atol=1e-12
        )
        np.testing.assert_allclose(actual.p_values, expected.pVals, rtol=0, atol=0)
        np.testing.assert_array_equal(actual_included, expected_included)
        np.testing.assert_array_equal(actual_excluded, expected_excluded)
        rust_seconds = measure(
            lambda: run_native(
                losses,
                args.seed,
                args.bootstraps,
                args.block_size,
                args.bootstrap,
                args.algorithm,
            ),
            args.repeats,
        )
        print(f"Rust public API median:  {rust_seconds:.6f} s")
        print(f"speedup:                 {reference_seconds / rust_seconds:.2f}x")
        if rust_seconds >= reference_seconds:
            raise SystemExit("Rust implementation must beat the academic Python version")


if __name__ == "__main__":
    main()
