mod core;

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "_run_fast_mcs")]
fn run_fast_mcs_py(
    losses: PyReadonlyArray2<'_, f64>,
    bootstrap_indices: PyReadonlyArray2<'_, i64>,
    algorithm: &str,
) -> PyResult<(Vec<f64>, Vec<usize>, Vec<f64>)> {
    let algorithm = core::Algorithm::parse(algorithm)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = core::run_fast_mcs(losses.as_array(), bootstrap_indices.as_array(), algorithm)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    Ok((
        result.t_score,
        result.elimination_order,
        result.t_boot_distribution,
    ))
}

/// Native extension module for `bankai_fast_mcs`.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_fast_mcs_py, module)?)?;
    Ok(())
}
