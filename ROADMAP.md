## Phase 1: Setup & Infrastructure
- [x] Initialize Python environment with `uv` (Python >= 3.11).
- [x] Initialize Rust/PyO3 project with `maturin`.
- [x] Set up `cargo test`, `cargo bench` (Criterion), and `pytest`.

## Phase 2: Core Algorithm (Rust)
- [ ] Implement the 1-pass and 2-pass vector updating rules (O(M^2) time complexity).
- [ ] Implement cache-friendly dense matrix loops.
- [ ] Implement Rayon parallelization for the bootstrap block.

## Phase 3: Python Interop (NumPy Core)
- [ ] Expose the `ModelConfidenceSet` API class to Python.
- [ ] Implement zero-copy PyArray ingestion via `rust-numpy`.

## Phase 4: Parity & Benchmarking
- [ ] Validate statistical output parity (p-values, rankings) against the original academic Python repo.
- [ ] Benchmark at M=2000 and M=5000 to prove massive performance gains.

## Phase 5: Future Proofing (Polars)
- [ ] Design an endpoint to accept Polars DataFrames and extract dense matrices via Arrow with minimal cost.
