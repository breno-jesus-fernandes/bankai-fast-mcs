# 🔥 Bankai Fast MCS

Bankai Fast MCS is a high-performance implementation of the **Model Confidence Set (MCS)** with a native Rust core and a simple Python API. It provides 1-pass and 2-pass algorithms for comparing the predictive performance of multiple models using bootstrap samples.

In our benchmark, the Rust 2-pass implementation ran up to **6.26× faster** than the academic Python reference, using 5,000 models, 250 observations, and 100 bootstrap replications ([results](benchmarks/RESULTS.md)).

⚠️ Bankai Fast MCS is currently in Alpha. The API may change.

## Install

Python 3.11 or newer is required. Install the alpha release from PyPI with:

```bash
python -m pip install bankai-fast-mcs
```

## Quick start

Pass a two-dimensional NumPy array of `float64` losses to `ModelConfidenceSet`. Rows are observations and columns are models.

```python
import numpy as np
from bankai_fast_mcs import ModelConfidenceSet

# Example losses for 3 models evaluated over 250 observations.
rng = np.random.default_rng(42)
losses = rng.random((250, 3), dtype=np.float64)

mcs = ModelConfidenceSet(losses, seed=42)
mcs.run(B=1_000, b=10, bootstrap="stationary", algorithm="2-pass")

included, excluded = mcs.get_mcs(alpha=0.05)
print("Included models:", included)
print("Excluded models:", excluded)
```

`included` and `excluded` contain the column indices of the corresponding models. The example uses random losses to demonstrate the API; replace them with the loss matrix from your model evaluation.

## Python API

```python
ModelConfidenceSet(losses=None, seed=None, verbose=True)
```

- `losses`: optional NumPy `float64` matrix with shape `(observations, models)`.
- `seed`: optional seed for reproducible bootstrap samples.
- `verbose`: accepted for API configuration.

### `run`

```python
mcs.run(B=1_000, b=10, bootstrap="stationary", algorithm="2-pass")
```

- `B`: number of bootstrap replications.
- `b`: block length parameter for bootstrap sampling.
- `bootstrap`: `"stationary"` or `"block"`.
- `algorithm`: `"1-pass"` or `"2-pass"`.

### `get_mcs`

```python
included, excluded = mcs.get_mcs(alpha=0.05)
```

Call `run` before `get_mcs`. The method returns two arrays of model column indices. The object also exposes `t_score`, `elimination_order`, `t_boot_distribution`, and `p_values` results.

## Development

To run the test suites from the repository root:

```bash
cargo test
uv run pytest
```

Benchmark results against the academic Python implementation are available in [benchmarks/RESULTS.md](benchmarks/RESULTS.md). The comparison uses the reference implementation at [fastMCS.py](https://github.com/Sylvain-Barde/fastMCS/blob/main/fastMCS.py).

## References

- Hansen, P. R., Lunde, A., and Nason, J. M. (2011). [The Model Confidence Set](https://doi.org/10.3982/ECTA5771). *Econometrica*, 79(2), 453–497.
- Academic implementation: [Sylvain Barde, fastMCS.py](https://github.com/Sylvain-Barde/fastMCS/blob/main/fastMCS.py).

## License

See [LICENSE](LICENSE).
