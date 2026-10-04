# fastMCS Benchmark Results

Release end-to-end timings against the pinned academic implementation in `reference/fastMCS.py`. Both implementations received the same deterministic loss matrix and seed. Each command was run once (`--repeats 1`); the script checked scores, rankings, bootstrap distributions, p-values, and MCS membership before reporting timings.

Environment: Apple M1, CPython 3.11.11, NumPy 2.4.6, Rust 1.97.1. `maturin develop --release` was used for the native extension. Inputs used 250 observations, 100 bootstrap replications, stationary bootstrap with block size 10.

| Algorithm | Models | Academic Python | Rust API | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 1-pass | 2,000 | 2.105 s | 0.644 s | 3.27× |
| 1-pass | 5,000 | 15.027 s | 4.680 s | 3.21× |
| 2-pass | 2,000 | 3.798 s | 0.637 s | 5.97× |
| 2-pass | 5,000 | 23.813 s | 3.801 s | 6.26× |

Reproduce each row with:

```sh
.venv/bin/python benchmarks/compare_public_api.py \
  --observations 250 --models 2000 --bootstraps 100 --repeats 1 \
  --bootstrap stationary --algorithm 2-pass
```

Change `--models` to `5000` or `--algorithm` to `1-pass` for the other rows. Timings are hardware- and load-dependent; rerun the comparison script to measure another system.
