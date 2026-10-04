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


@pytest.mark.parametrize("shape", [(0, 3), (5, 0)])
def test_model_confidence_set_rejects_empty_loss_dimensions(shape):
    with pytest.raises(ValueError, match="at least one observation and one model"):
        ModelConfidenceSet(np.empty(shape, dtype=np.float64))


def test_unprocessed_model_exposes_empty_results_and_requires_run():
    result = ModelConfidenceSet(seed=8)

    assert result.verbose is True
    assert result.t_score.shape == (0,)
    assert result.elimination_order.shape == (0,)
    assert result.t_boot_distribution.shape == (0, 0)
    assert result.p_values.shape == (0,)
    with pytest.raises(ValueError, match="no losses have been added"):
        result.run()
    with pytest.raises(ValueError, match="run must complete"):
        result.get_mcs()


@pytest.mark.parametrize("bootstraps, block_size", [(0, 4), (4, 0)])
def test_run_rejects_non_positive_bootstrap_parameters(bootstraps, block_size):
    result = ModelConfidenceSet(_losses()[:, :3], seed=3, verbose=False)

    with pytest.raises(ValueError, match="B and b must both be positive"):
        result.run(B=bootstraps, b=block_size)


def test_run_rejects_unsupported_algorithm_and_bootstrap_method():
    result = ModelConfidenceSet(_losses()[:, :3], seed=3, verbose=False)

    with pytest.raises(ValueError, match="algorithm must be"):
        result.run(B=10, b=3, algorithm="approximate")
    with pytest.raises(ValueError, match="bootstrap must be"):
        result.run(B=10, b=3, bootstrap="moving")


@pytest.mark.parametrize("alpha", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_get_mcs_rejects_alpha_outside_the_probability_range(alpha):
    result = ModelConfidenceSet(_losses()[:, :3], seed=3, verbose=False)
    result.run(B=10, b=3)

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        result.get_mcs(alpha=alpha)


def test_adding_losses_after_run_invalidates_results_until_the_next_run():
    losses = _losses()[:, :3]
    result = ModelConfidenceSet(losses, seed=12, verbose=False)
    result.run(B=12, b=4)
    result.get_mcs()
    assert result.t_score.shape == (3,)
    assert result.p_values.shape == (3,)

    result.add_losses(_losses()[:, 3:5])

    assert result.t_score.shape == (0,)
    assert result.elimination_order.shape == (0,)
    assert result.t_boot_distribution.shape == (0, 5)
    assert result.p_values.shape == (0,)
    with pytest.raises(ValueError, match="run must complete"):
        result.get_mcs()

    result.run(B=12, b=4)
    assert result.t_score.shape == (5,)
    assert result.t_boot_distribution.shape == (12, 5)


def test_alpha_boundaries_preserve_the_complete_confidence_set():
    result = ModelConfidenceSet(_losses()[:, :4], seed=22, verbose=False)
    result.run(B=20, b=5)

    included_at_zero, excluded_at_zero = result.get_mcs(alpha=0.0)
    np.testing.assert_array_equal(np.sort(included_at_zero), np.arange(4))
    assert excluded_at_zero.size == 0

    included_at_one, excluded_at_one = result.get_mcs(alpha=1.0)
    assert included_at_one.size + excluded_at_one.size == 4
