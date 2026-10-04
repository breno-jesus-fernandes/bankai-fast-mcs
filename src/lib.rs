use pyo3::prelude::*;

/// Native extension module for `bankai_fast_mcs`.
#[pymodule]
fn _native(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
