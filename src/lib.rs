mod api;
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
    let loss_view = losses.as_array();
    let result = core::run_fast_mcs(&loss_view, bootstrap_indices.as_array(), algorithm)
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
    module.add_class::<api::ModelConfidenceSet>()?;
    Ok(())
}

#[cfg(test)]
mod test_support {
    use pyo3::prelude::*;
    use std::path::PathBuf;

    pub fn add_project_python_paths(py: Python<'_>) {
        let mut candidates = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")];
        let mut virtual_environments =
            vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".venv")];
        if let Some(path) = std::env::var_os("VIRTUAL_ENV") {
            virtual_environments.push(PathBuf::from(path));
        }

        for virtual_environment in virtual_environments {
            if let Ok(entries) = std::fs::read_dir(virtual_environment.join("lib")) {
                for entry in entries.flatten() {
                    if entry.file_name().to_string_lossy().starts_with("python") {
                        candidates.push(entry.path().join("site-packages"));
                    }
                }
            }
            candidates.push(virtual_environment.join("Lib").join("site-packages"));
        }

        let path = py
            .import("sys")
            .expect("Python sys module is available")
            .getattr("path")
            .expect("Python sys.path is available");
        for candidate in candidates.into_iter().rev() {
            if candidate.is_dir() {
                path.call_method1("insert", (0, candidate.to_string_lossy().into_owned()))
                    .expect("Python import path accepts project paths");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use numpy::{IntoPyArray, PyArrayMethods};

    #[test]
    fn native_module_registers_the_public_python_api() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "bankai_fast_mcs._native").unwrap();

            _native(&module).unwrap();

            assert!(module.hasattr("_run_fast_mcs").unwrap());
            assert!(module.hasattr("ModelConfidenceSet").unwrap());
        });
    }

    #[test]
    fn native_function_returns_results_and_rejects_invalid_inputs() {
        Python::initialize();
        Python::attach(|py| {
            test_support::add_project_python_paths(py);
            let losses = array![[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]].into_pyarray(py);
            let indices = array![[0_i64, 1], [1, 2], [2, 0]].into_pyarray(py);

            let (scores, order, boot_distribution) =
                run_fast_mcs_py(losses.readonly(), indices.readonly(), "2-pass").unwrap();
            assert_eq!(scores.len(), 2);
            assert_eq!(order.len(), 2);
            assert_eq!(boot_distribution.len(), 4);

            let invalid_algorithm =
                run_fast_mcs_py(losses.readonly(), indices.readonly(), "invalid").unwrap_err();
            assert!(invalid_algorithm.to_string().contains("algorithm must be"));

            let invalid_indices = array![[-1_i64], [0], [1]].into_pyarray(py);
            let invalid_index =
                run_fast_mcs_py(losses.readonly(), invalid_indices.readonly(), "2-pass")
                    .unwrap_err();
            assert!(invalid_index.to_string().contains("outside 0..3"));
        });
    }
}
