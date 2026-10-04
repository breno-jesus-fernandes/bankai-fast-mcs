import numpy as np
import pytest

from bankai_fast_mcs._bootstrap import generate_indices


@pytest.mark.parametrize(
    "observations, bootstraps, block_size",
    [(0, 4, 2), (5, 0, 2), (5, 4, 0)],
)
def test_generate_indices_rejects_non_positive_sizes(
    observations, bootstraps, block_size
):
    with pytest.raises(ValueError, match="must be positive"):
        generate_indices(observations, bootstraps, block_size, "block", 17)


def test_block_bootstrap_returns_wrapping_blocks_and_trims_the_final_block():
    observations, bootstraps, block_size = 7, 5, 3
    indices = generate_indices(observations, bootstraps, block_size, "block", 91)

    assert indices.shape == (observations, bootstraps)
    assert indices.dtype == np.int64
    assert np.all((0 <= indices) & (indices < observations))
    for column in range(bootstraps):
        for start in (0, block_size, 2 * block_size):
            stop = min(start + block_size, observations)
            block = indices[start:stop, column]
            np.testing.assert_array_equal(
                block[1:], (block[:-1] + 1) % observations
            )


def test_stationary_bootstrap_without_jumps_continues_circular_blocks():
    observations, bootstraps = 9, 4
    indices = generate_indices(
        observations, bootstraps, 10**12, "stationary", seed=4
    )

    for column in range(bootstraps):
        np.testing.assert_array_equal(
            indices[:, column],
            (indices[0, column] + np.arange(observations)) % observations,
        )


def test_stationary_bootstrap_with_unit_block_size_resamples_every_observation():
    first = generate_indices(20, 8, 1, "stationary", seed=73)
    second = generate_indices(20, 8, 1, "stationary", seed=73)

    assert first.shape == (20, 8)
    assert first.dtype == np.int64
    assert np.all((0 <= first) & (first < 20))
    np.testing.assert_array_equal(first, second)


def test_generate_indices_rejects_unknown_bootstrap_method():
    with pytest.raises(ValueError, match="bootstrap must be"):
        generate_indices(5, 3, 2, "moving", seed=17)
