import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bankai_fast_mcs import ModelConfidenceSet


REFERENCE_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "reference" / "fastMCS.py"
)
REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "fastMCS_public_api_reference", REFERENCE_PATH
)
REFERENCE = importlib.util.module_from_spec(REFERENCE_SPEC)
REFERENCE_SPEC.loader.exec_module(REFERENCE)


def _losses():
    rng = np.random.default_rng(271828)
    losses = rng.normal(size=(30, 9))
    losses += np.arange(losses.shape[1])[None, :] * 0.01
    return losses


def _reference(losses, seed, bootstraps, block_size, bootstrap, algorithm):
    model_set = REFERENCE.mcs(seed=seed, verbose=False)
    model_set.addLosses(losses)
    model_set.run(
        B=bootstraps,
        b=block_size,
        bootstrap=bootstrap,
        algorithm=algorithm,
    )
    included, excluded = model_set.getMCS(alpha=0.05)
    return model_set, included, excluded


@pytest.mark.parametrize("bootstrap", ["block", "stationary"])
@pytest.mark.parametrize("algorithm", ["1-pass", "2-pass"])
def test_model_confidence_set_matches_reference(
    bootstrap, algorithm
):
    losses = _losses()
    seed, bootstraps, block_size = 27, 80, 5
    expected, expected_included, expected_excluded = _reference(
        losses, seed, bootstraps, block_size, bootstrap, algorithm
    )

    result = ModelConfidenceSet(losses, seed=seed, verbose=False)
    result.run(
        B=bootstraps,
        b=block_size,
        bootstrap=bootstrap,
        algorithm=algorithm,
    )
    included, excluded = result.get_mcs(alpha=0.05)

    np.testing.assert_allclose(result.t_score, expected.tScore, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(result.elimination_order, expected.exclMods)
    np.testing.assert_allclose(
        result.t_boot_distribution, expected.tBootDist, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(result.p_values, expected.pVals, rtol=0, atol=0)
    np.testing.assert_array_equal(included, expected_included)
    np.testing.assert_array_equal(excluded, expected_excluded)


def test_model_confidence_set_borrows_loss_arrays_and_accepts_multiple_blocks():
    original = _losses()
    first = np.asfortranarray(original[:, :4])
    second = np.asfortranarray(original[:, 4:])
    result = ModelConfidenceSet(seed=3, verbose=False)
    result.add_losses(first)
    result.add_losses(second)

    first[0, 0] += 10.0
    expected_losses = np.concatenate((first, second), axis=1)
    expected, _, _ = _reference(
        expected_losses, 3, 40, 4, "stationary", "2-pass"
    )

    result.run(B=40, b=4, bootstrap="stationary", algorithm="2-pass")

    np.testing.assert_allclose(result.t_score, expected.tScore, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(result.elimination_order, expected.exclMods)


def test_model_confidence_set_rejects_mismatched_observation_counts():
    result = ModelConfidenceSet(_losses()[:, :3], seed=1, verbose=False)

    with pytest.raises(ValueError, match="observation"):
        result.add_losses(_losses()[:-1, 3:5])
