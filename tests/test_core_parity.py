import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bankai_fast_mcs import _native


REFERENCE_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "reference" / "fastMCS.py"
)
REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "fastMCS_reference", REFERENCE_PATH
)
REFERENCE = importlib.util.module_from_spec(REFERENCE_SPEC)
REFERENCE_SPEC.loader.exec_module(REFERENCE)


def _inputs():
    rng = np.random.default_rng(314159)
    observations, models, bootstraps = 12, 7, 19
    losses = rng.normal(size=(observations, models))
    losses += np.arange(models)[None, :] * 0.025
    indices = rng.integers(0, observations, size=(observations, bootstraps))
    return losses, indices


def _reference_result(losses, indices, algorithm):
    REFERENCE.blockBootstrap = lambda rng, obs, count, block: indices.copy()
    model_set = REFERENCE.mcs(seed=1, verbose=False)
    model_set.addLosses(losses)
    model_set.run(
        B=indices.shape[1],
        b=3,
        bootstrap="block",
        algorithm=algorithm,
    )
    return model_set.tScore, model_set.exclMods, model_set.tBootDist


@pytest.mark.parametrize("algorithm", ["1-pass", "2-pass"])
def test_core_matches_academic_reference(algorithm):
    losses, indices = _inputs()
    expected = _reference_result(losses, indices, algorithm)

    scores, elimination_order, boot_distribution = _native._run_fast_mcs(
        losses, indices, algorithm
    )

    np.testing.assert_allclose(scores, expected[0], rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(elimination_order, expected[1])
    np.testing.assert_allclose(
        np.asarray(boot_distribution).reshape(indices.shape[1], losses.shape[1]),
        expected[2],
        rtol=1e-11,
        atol=1e-12,
    )


def test_core_accepts_strided_numpy_views_without_changing_results():
    losses, indices = _inputs()
    expected = _native._run_fast_mcs(losses, indices, "2-pass")

    actual = _native._run_fast_mcs(
        np.asfortranarray(losses), np.asfortranarray(indices), "2-pass"
    )

    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[2], expected[2], rtol=0, atol=0)


def test_core_rejects_bootstrap_indices_with_wrong_observation_count():
    losses, indices = _inputs()

    with pytest.raises(ValueError, match="bootstrap indices"):
        _native._run_fast_mcs(losses, indices[:-1], "2-pass")
