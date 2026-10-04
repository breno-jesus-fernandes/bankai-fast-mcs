use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::{self, Algorithm, SegmentedLossMatrix};

#[pyclass(name = "ModelConfidenceSet")]
pub struct ModelConfidenceSet {
    loss_arrays: Vec<Py<PyArray2<f64>>>,
    observations: Option<usize>,
    seed: Option<u64>,
    verbose: bool,
    t_score_values: Vec<f64>,
    elimination_order_values: Vec<usize>,
    t_boot_distribution_values: Vec<f64>,
    p_value_values: Vec<f64>,
    bootstraps: usize,
    model_count: usize,
    is_processed: bool,
}

#[pymethods]
impl ModelConfidenceSet {
    #[new]
    #[pyo3(signature = (losses=None, seed=None, verbose=true))]
    fn new(
        py: Python<'_>,
        losses: Option<Py<PyArray2<f64>>>,
        seed: Option<u64>,
        verbose: bool,
    ) -> PyResult<Self> {
        let mut result = Self {
            loss_arrays: Vec::new(),
            observations: None,
            seed,
            verbose,
            t_score_values: Vec::new(),
            elimination_order_values: Vec::new(),
            t_boot_distribution_values: Vec::new(),
            p_value_values: Vec::new(),
            bootstraps: 0,
            model_count: 0,
            is_processed: false,
        };
        if let Some(losses) = losses {
            result.add_losses(py, losses)?;
        }
        Ok(result)
    }

    fn add_losses(&mut self, py: Python<'_>, losses: Py<PyArray2<f64>>) -> PyResult<()> {
        let array = losses.bind(py);
        let (observations, models) = {
            let readonly = array
                .try_readonly()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            readonly.as_array().dim()
        };
        if observations == 0 || models == 0 {
            return Err(PyValueError::new_err(
                "losses must contain at least one observation and one model",
            ));
        }
        if let Some(expected) = self.observations {
            if observations != expected {
                return Err(PyValueError::new_err(
                    "all losses arrays must have the same observation count",
                ));
            }
        } else {
            self.observations = Some(observations);
        }
        self.model_count += models;
        self.loss_arrays.push(losses);
        self.is_processed = false;
        self.bootstraps = 0;
        self.t_score_values.clear();
        self.elimination_order_values.clear();
        self.t_boot_distribution_values.clear();
        self.p_value_values.clear();
        Ok(())
    }

    #[allow(non_snake_case)]
    #[pyo3(signature = (B=1000, b=10, bootstrap="stationary", algorithm="2-pass"))]
    fn run(
        &mut self,
        py: Python<'_>,
        B: usize,
        b: usize,
        bootstrap: &str,
        algorithm: &str,
    ) -> PyResult<()> {
        let observations = self.observations.ok_or_else(|| {
            PyValueError::new_err("no losses have been added; call add_losses first")
        })?;
        if B == 0 || b == 0 {
            return Err(PyValueError::new_err(
                "B and b must both be positive integers",
            ));
        }
        let algorithm = Algorithm::parse(algorithm)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        let bootstrap_module = py.import("bankai_fast_mcs._bootstrap")?;
        let generated_indices = bootstrap_module.getattr("generate_indices")?.call1((
            observations,
            B,
            b,
            bootstrap,
            self.seed,
        ))?;
        let bootstrap_indices = generated_indices.extract::<PyReadonlyArray2<'_, i64>>()?;

        let mut readonly_arrays = Vec::with_capacity(self.loss_arrays.len());
        for losses in &self.loss_arrays {
            readonly_arrays.push(
                losses
                    .bind(py)
                    .try_readonly()
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            );
        }
        let views = readonly_arrays
            .iter()
            .map(|losses| losses.as_array())
            .collect::<Vec<_>>();
        let result = if views.len() == 1 {
            core::run_fast_mcs(&views[0], bootstrap_indices.as_array(), algorithm)
        } else {
            let segmented_losses = SegmentedLossMatrix::new(views)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            core::run_fast_mcs(&segmented_losses, bootstrap_indices.as_array(), algorithm)
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

        self.t_score_values = result.t_score;
        self.elimination_order_values = result.elimination_order;
        self.t_boot_distribution_values = result.t_boot_distribution;
        self.bootstraps = B;
        self.is_processed = true;
        self.p_value_values.clear();
        Ok(())
    }

    #[pyo3(signature = (alpha=0.05))]
    fn get_mcs<'py>(
        &mut self,
        py: Python<'py>,
        alpha: f64,
    ) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)> {
        if !self.is_processed {
            return Err(PyValueError::new_err("run must complete before get_mcs"));
        }
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(PyValueError::new_err("alpha must be between 0 and 1"));
        }

        let mut p_values = Vec::with_capacity(self.model_count);
        for &model in &self.elimination_order_values {
            let exceedances = (0..self.bootstraps)
                .filter(|&bootstrap| {
                    self.t_boot_distribution_values[bootstrap * self.model_count + model]
                        >= self.t_score_values[model]
                })
                .count();
            p_values.push(exceedances as f64 / self.bootstraps as f64);
        }
        let mut maximum = 0.0_f64;
        for p_value in &mut p_values {
            maximum = maximum.max(*p_value);
            *p_value = maximum;
        }
        self.p_value_values = p_values;

        let mut included = Vec::new();
        let mut excluded = Vec::new();
        for (position, &model) in self.elimination_order_values.iter().enumerate() {
            if self.p_value_values[position] >= alpha {
                included.push(model);
            } else {
                excluded.push(model);
            }
        }
        Ok((included.into_pyarray(py), excluded.into_pyarray(py)))
    }

    #[getter]
    fn t_score<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.t_score_values.clone().into_pyarray(py)
    }

    #[getter]
    fn elimination_order<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        self.elimination_order_values.clone().into_pyarray(py)
    }

    #[getter]
    fn t_boot_distribution<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let values = self.t_boot_distribution_values.clone();
        Array2::from_shape_vec((self.bootstraps, self.model_count), values)
            .expect("MCS bootstrap distribution shape is tracked with its data")
            .into_pyarray(py)
    }

    #[getter]
    fn p_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.p_value_values.clone().into_pyarray(py)
    }

    #[getter]
    fn verbose(&self) -> bool {
        self.verbose
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn constructor_and_getters_expose_empty_unprocessed_state() {
        Python::initialize();
        Python::attach(|py| {
            crate::test_support::add_project_python_paths(py);
            let mut model = ModelConfidenceSet::new(py, None, None, false).unwrap();

            assert!(!model.verbose());
            assert_eq!(model.t_score(py).readonly().as_array().len(), 0);
            assert_eq!(model.elimination_order(py).readonly().as_array().len(), 0);
            assert_eq!(
                model.t_boot_distribution(py).readonly().as_array().dim(),
                (0, 0)
            );
            assert_eq!(model.p_values(py).readonly().as_array().len(), 0);
            assert!(
                model
                    .get_mcs(py, 0.05)
                    .unwrap_err()
                    .to_string()
                    .contains("run must complete")
            );
            assert!(
                model
                    .run(py, 10, 2, "block", "2-pass")
                    .unwrap_err()
                    .to_string()
                    .contains("no losses have been added")
            );
        });
    }

    #[test]
    fn adding_losses_rejects_empty_arrays_and_inconsistent_observation_counts() {
        Python::initialize();
        Python::attach(|py| {
            crate::test_support::add_project_python_paths(py);
            let mut model = ModelConfidenceSet::new(py, None, Some(11), true).unwrap();
            let empty_observations = Array2::<f64>::zeros((0, 2)).into_pyarray(py).unbind();
            assert!(
                model
                    .add_losses(py, empty_observations)
                    .unwrap_err()
                    .to_string()
                    .contains("at least one observation and one model")
            );

            let empty_models = Array2::<f64>::zeros((3, 0)).into_pyarray(py).unbind();
            assert!(
                model
                    .add_losses(py, empty_models)
                    .unwrap_err()
                    .to_string()
                    .contains("at least one observation and one model")
            );

            let first = array![[1.0, 2.0], [3.0, 4.0]].into_pyarray(py).unbind();
            model.add_losses(py, first).unwrap();
            let mismatched = array![[1.0], [2.0], [3.0]].into_pyarray(py).unbind();
            assert!(
                model
                    .add_losses(py, mismatched)
                    .unwrap_err()
                    .to_string()
                    .contains("same observation count")
            );
        });
    }

    #[test]
    fn run_validates_parameters_and_algorithm_before_bootstrapping() {
        Python::initialize();
        Python::attach(|py| {
            crate::test_support::add_project_python_paths(py);
            let losses = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]]
                .into_pyarray(py)
                .unbind();
            let mut model = ModelConfidenceSet::new(py, Some(losses), Some(11), true).unwrap();

            for (bootstraps, block_size) in [(0, 2), (10, 0)] {
                assert!(
                    model
                        .run(py, bootstraps, block_size, "block", "2-pass")
                        .unwrap_err()
                        .to_string()
                        .contains("B and b must both be positive")
                );
            }
            assert!(
                model
                    .run(py, 10, 2, "block", "approximate")
                    .unwrap_err()
                    .to_string()
                    .contains("algorithm must be")
            );
        });
    }

    #[test]
    fn run_and_get_mcs_populate_results_and_reject_invalid_alpha() {
        Python::initialize();
        Python::attach(|py| {
            crate::test_support::add_project_python_paths(py);
            let losses = array![
                [1.0, 2.1, 3.2],
                [1.4, 2.5, 3.0],
                [0.8, 2.8, 3.5],
                [1.7, 2.0, 3.1],
                [1.2, 2.3, 3.7],
            ]
            .into_pyarray(py)
            .unbind();
            let mut model = ModelConfidenceSet::new(py, Some(losses), Some(7), false).unwrap();
            model.run(py, 24, 2, "stationary", "1-pass").unwrap();

            assert_eq!(model.t_score(py).readonly().as_array().len(), 3);
            assert_eq!(model.elimination_order(py).readonly().as_array().len(), 3);
            assert_eq!(
                model.t_boot_distribution(py).readonly().as_array().dim(),
                (24, 3)
            );
            assert!(
                model
                    .get_mcs(py, f64::NAN)
                    .unwrap_err()
                    .to_string()
                    .contains("alpha must be between 0 and 1")
            );

            let (included, excluded) = model.get_mcs(py, 0.0).unwrap();
            assert_eq!(included.readonly().as_array().len(), 3);
            assert_eq!(excluded.readonly().as_array().len(), 0);
            assert_eq!(model.p_values(py).readonly().as_array().len(), 3);
        });
    }

    #[test]
    fn get_mcs_uses_cumulative_p_values_to_split_included_and_excluded_models() {
        Python::initialize();
        Python::attach(|py| {
            crate::test_support::add_project_python_paths(py);
            let mut model = ModelConfidenceSet::new(py, None, None, false).unwrap();
            model.model_count = 3;
            model.bootstraps = 2;
            model.is_processed = true;
            model.elimination_order_values = vec![0, 1, 2];
            model.t_score_values = vec![2.0, 2.0, 1.0];
            model.t_boot_distribution_values = vec![1.0, 1.0, 1.0, 3.0, 1.5, 2.0];

            let (included, excluded) = model.get_mcs(py, 0.75).unwrap();

            assert_eq!(included.readonly().as_array().to_vec(), vec![2]);
            assert_eq!(excluded.readonly().as_array().to_vec(), vec![0, 1]);
            assert_eq!(
                model.p_values(py).readonly().as_array().to_vec(),
                vec![0.5, 0.5, 1.0]
            );
        });
    }

    #[test]
    fn adding_more_models_after_run_clears_previous_results() {
        Python::initialize();
        Python::attach(|py| {
            crate::test_support::add_project_python_paths(py);
            let first = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]]
                .into_pyarray(py)
                .unbind();
            let mut model = ModelConfidenceSet::new(py, Some(first), Some(19), false).unwrap();
            model.run(py, 12, 2, "block", "2-pass").unwrap();
            model.get_mcs(py, 0.05).unwrap();
            assert_eq!(model.p_values(py).readonly().as_array().len(), 2);

            let second = array![[4.0], [2.0], [3.0]].into_pyarray(py).unbind();
            model.add_losses(py, second).unwrap();

            assert_eq!(model.t_score(py).readonly().as_array().len(), 0);
            assert_eq!(model.elimination_order(py).readonly().as_array().len(), 0);
            assert_eq!(
                model.t_boot_distribution(py).readonly().as_array().dim(),
                (0, 3)
            );
            assert_eq!(model.p_values(py).readonly().as_array().len(), 0);
        });
    }
}
