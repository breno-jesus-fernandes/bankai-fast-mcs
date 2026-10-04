## 1. Tech Stack & Boundaries
- Core: Rust (Edition 2024).
- Python Bindings: PyO3 + `rust-numpy`.
- Constraints: Strictly Python >= 3.11.
- Tooling: `uv` (Python dependency management) and `maturin` (build system).
- Performance limits: ZERO-COPY dense array transfers are mandatory. Never copy a PyArray to a Rust Vec if you can read it by reference.

## 2. Mandatory Execution Loop
Every single implementation step MUST follow this exact micro-loop. Do not skip steps:
0. **TDD ALWAYS:** Write tests before implementation code.
1. **TEST FIRST:** Create a Rust/Python test describing the behavior. Verify it FAILS.
2. **BENCHMARK FIRST:** Create a benchmark comparing our Rust implementation with the academic Python version (https://github.com/Sylvain-Barde/fastMCS/blob/main/fastMCS.py). Our code must NEVER be slower.
3. **OPTIMIZE:** Optimize using safe Rust, data-oriented layout, and cache-locality. Use `unsafe`/SIMD only with documented invariants and undeniable benchmark proof. Use `Rayon` for embarrassingly parallel tasks (e.g., bootstrap loops).
4. **REGRESSION:** Run all tests and benchmarks. Update `ROADMAP.md` only when passing.
5. **COMMIT & PUSH:** After each new feature is implemented and its tests pass, commit the feature and push the commit to `main`.
