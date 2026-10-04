"""NumPy-compatible bootstrap index generation for the public API."""

import numpy as np


def generate_indices(observations, bootstraps, block_size, method, seed):
    if observations <= 0 or bootstraps <= 0 or block_size <= 0:
        raise ValueError("observations, bootstraps, and block_size must be positive")

    rng = np.random.default_rng(seed)
    if method == "block":
        blocks = int(np.ceil(observations / block_size))
        indices = np.zeros((blocks * block_size, bootstraps))
        for block in range(blocks):
            start = block * block_size
            indices[start, :] = np.ceil(observations * rng.random(bootstraps))
            for offset in range(1, block_size):
                indices[start + offset, :] = indices[start + offset - 1, :] + 1
        indices = indices[:observations, :]
        indices[indices > observations - 1] -= observations
        return indices.astype(np.int64)

    if method == "stationary":
        indices = np.zeros((observations, bootstraps))
        indices[0, :] = np.ceil(observations * rng.random(bootstraps))
        jumps = rng.random((observations, bootstraps)) < 1 / block_size
        jump_count = int(np.sum(jumps))
        indices[np.where(jumps)] = np.floor(rng.random((1, jump_count)) * observations)
        for observation in range(1, observations):
            continuing = ~jumps[observation, :]
            indices[observation, continuing] = indices[observation - 1, continuing] + 1
        indices[indices > observations - 1] -= observations
        return indices.astype(np.int64)

    raise ValueError("bootstrap must be 'block' or 'stationary'")
