use std::error::Error;
use std::fmt::{Display, Formatter};

use ndarray::ArrayView2;
use rayon::prelude::*;

pub trait LossMatrix: Sync {
    fn observations(&self) -> usize;
    fn models(&self) -> usize;
    fn get(&self, observation: usize, model: usize) -> f64;
}

impl LossMatrix for ArrayView2<'_, f64> {
    #[inline]
    fn observations(&self) -> usize {
        self.nrows()
    }

    #[inline]
    fn models(&self) -> usize {
        self.ncols()
    }

    #[inline]
    fn get(&self, observation: usize, model: usize) -> f64 {
        self[(observation, model)]
    }
}

pub struct SegmentedLossMatrix<'a> {
    matrices: Vec<ArrayView2<'a, f64>>,
    model_locations: Vec<(usize, usize)>,
    observations: usize,
}

impl<'a> SegmentedLossMatrix<'a> {
    pub fn new(matrices: Vec<ArrayView2<'a, f64>>) -> Result<Self, CoreError> {
        let observations = matrices
            .first()
            .ok_or(CoreError::InvalidShape(
                "at least one losses array is required",
            ))?
            .nrows();
        let mut model_locations = Vec::new();
        for (matrix_index, matrix) in matrices.iter().enumerate() {
            if matrix.nrows() != observations {
                return Err(CoreError::InvalidShape(
                    "all losses arrays must have the same observation count",
                ));
            }
            model_locations.extend((0..matrix.ncols()).map(|model| (matrix_index, model)));
        }
        Ok(Self {
            matrices,
            model_locations,
            observations,
        })
    }
}

impl LossMatrix for SegmentedLossMatrix<'_> {
    fn observations(&self) -> usize {
        self.observations
    }

    fn models(&self) -> usize {
        self.model_locations.len()
    }

    #[inline]
    fn get(&self, observation: usize, model: usize) -> f64 {
        let (matrix, local_model) = self.model_locations[model];
        self.matrices[matrix][(observation, local_model)]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Algorithm {
    OnePass,
    TwoPass,
}

impl Algorithm {
    pub fn parse(value: &str) -> Result<Self, CoreError> {
        match value {
            "1-pass" => Ok(Self::OnePass),
            "2-pass" => Ok(Self::TwoPass),
            _ => Err(CoreError::InvalidAlgorithm(value.to_owned())),
        }
    }
}

#[derive(Debug)]
pub enum CoreError {
    InvalidAlgorithm(String),
    InvalidShape(&'static str),
    InvalidBootstrapIndex { index: i64, observations: usize },
}

impl Display for CoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAlgorithm(value) => write!(
                formatter,
                "algorithm must be '1-pass' or '2-pass', got {value:?}"
            ),
            Self::InvalidShape(message) => formatter.write_str(message),
            Self::InvalidBootstrapIndex {
                index,
                observations,
            } => write!(
                formatter,
                "bootstrap index {index} is outside 0..{observations}"
            ),
        }
    }
}

impl Error for CoreError {}

#[derive(Debug)]
pub struct McsResult {
    pub t_score: Vec<f64>,
    pub elimination_order: Vec<usize>,
    /// Flattened in bootstrap-major order with shape `(bootstraps, models)`.
    pub t_boot_distribution: Vec<f64>,
}

/// Run the fast MCS updating algorithm on borrowed dense loss and bootstrap views.
pub fn run_fast_mcs<L: LossMatrix>(
    losses: &L,
    bootstrap_indices: ArrayView2<'_, i64>,
    algorithm: Algorithm,
) -> Result<McsResult, CoreError> {
    let observations = losses.observations();
    let models = losses.models();
    let (index_observations, bootstraps) = bootstrap_indices.dim();
    if observations == 0 || models == 0 {
        return Err(CoreError::InvalidShape(
            "losses must contain at least one observation and one model",
        ));
    }
    if index_observations != observations || bootstraps == 0 {
        return Err(CoreError::InvalidShape(
            "bootstrap indices must have shape (observations, bootstraps) with nonzero bootstraps",
        ));
    }
    for &index in bootstrap_indices.iter() {
        if index < 0 || index as usize >= observations {
            return Err(CoreError::InvalidBootstrapIndex {
                index,
                observations,
            });
        }
    }

    let mut bootstrap_means = vec![0.0; bootstraps * models];
    bootstrap_means
        .par_chunks_mut(models)
        .enumerate()
        .for_each(|(bootstrap, means)| {
            for observation in 0..observations {
                let sampled_observation = bootstrap_indices[(observation, bootstrap)] as usize;
                for model in 0..models {
                    means[model] += losses.get(sampled_observation, model);
                }
            }
            let inverse_observations = 1.0 / observations as f64;
            for mean in means {
                *mean *= inverse_observations;
            }
        });

    let mut t_score = vec![0.0; models];
    let mut t_boot_distribution = vec![0.0; bootstraps * models];
    let mut processed = Vec::with_capacity(models);
    let mut previous_ranking = Vec::new();
    let mut processing_order: Vec<usize> = (0..models).collect();

    let mut model_means = vec![0.0; models];
    for observation in 0..observations {
        for model in 0..models {
            model_means[model] += losses.get(observation, model);
        }
    }
    for mean in &mut model_means {
        *mean /= observations as f64;
    }

    if algorithm == Algorithm::OnePass {
        processing_order.sort_by(|left, right| {
            model_means[*left]
                .total_cmp(&model_means[*right])
                .then_with(|| left.cmp(right))
        });
    }

    let mut ranking = Vec::with_capacity(models);
    for model in processing_order {
        let (pair_scores, pair_boot_t) = pair_statistics(
            &model_means,
            &bootstrap_means,
            models,
            bootstraps,
            model,
            &processed,
        );

        let current_score = pair_scores
            .iter()
            .fold(0.0_f64, |score, value| score.max(-value));
        t_score[model] = current_score;

        for (column, &previous_model) in processed.iter().enumerate() {
            let value = pair_scores[column];
            if value > t_score[previous_model] && value >= current_score {
                t_score[previous_model] = value;
            }
        }
        processed.push(model);

        ranking.clone_from(&processed);
        ranking.sort_by(|left, right| {
            t_score[*left]
                .total_cmp(&t_score[*right])
                .then_with(|| left.cmp(right))
        });

        if algorithm == Algorithm::OnePass && processed.len() > 1 {
            update_one_pass_distribution(
                model,
                &processed,
                &ranking,
                &previous_ranking,
                &pair_boot_t,
                &t_score,
                &mut t_boot_distribution,
                bootstraps,
                models,
            );
        }
        previous_ranking.clone_from(&ranking);
    }

    if algorithm == Algorithm::TwoPass {
        let mut rank_by_model = vec![0; models];
        for (rank, &model) in ranking.iter().enumerate() {
            rank_by_model[model] = rank;
        }
        for rank_position in 1..models {
            let model = ranking[rank_position];
            let better_models: Vec<usize> = (0..models)
                .filter(|&candidate| rank_by_model[candidate] < rank_position)
                .collect();
            let maxima = pair_max_boot_statistics(
                &model_means,
                &bootstrap_means,
                models,
                bootstraps,
                model,
                &better_models,
            );
            for bootstrap in 0..bootstraps {
                let previous = t_boot_distribution[bootstrap * models + ranking[rank_position - 1]];
                t_boot_distribution[bootstrap * models + model] = maxima[bootstrap].max(previous);
            }
        }
    }

    ranking.reverse();
    Ok(McsResult {
        t_score,
        elimination_order: ranking,
        t_boot_distribution,
    })
}

/// Pairwise statistics for one candidate against the listed existing models.
/// Bootstrap t-statistics are returned in row-major `[bootstrap][existing model]` order.
fn pair_statistics(
    model_means: &[f64],
    bootstrap_means: &[f64],
    models: usize,
    bootstraps: usize,
    model: usize,
    existing_models: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let pair_count = existing_models.len();
    let mean_differences: Vec<f64> = existing_models
        .iter()
        .map(|&existing_model| model_means[existing_model] - model_means[model])
        .collect();

    let mut variances = vec![0.0; pair_count];
    for bootstrap in 0..bootstraps {
        let model_mean = bootstrap_means[bootstrap * models + model];
        for (column, &existing_model) in existing_models.iter().enumerate() {
            let existing_mean = bootstrap_means[bootstrap * models + existing_model];
            let difference = existing_mean - model_mean - mean_differences[column];
            variances[column] += difference * difference;
        }
    }

    let mut scores = vec![0.0; pair_count];
    let mut inverse_standard_deviations = vec![0.0; pair_count];
    for column in 0..pair_count {
        let standard_deviation = (variances[column] / bootstraps as f64).sqrt();
        inverse_standard_deviations[column] = 1.0 / standard_deviation;
        scores[column] = mean_differences[column] * inverse_standard_deviations[column];
    }

    let mut boot_t = vec![0.0; bootstraps * pair_count];
    for bootstrap in 0..bootstraps {
        let model_mean = bootstrap_means[bootstrap * models + model];
        for (column, &existing_model) in existing_models.iter().enumerate() {
            let existing_mean = bootstrap_means[bootstrap * models + existing_model];
            let difference = existing_mean - model_mean - mean_differences[column];
            boot_t[bootstrap * pair_count + column] =
                difference * inverse_standard_deviations[column];
        }
    }

    (scores, boot_t)
}

fn pair_max_boot_statistics(
    model_means: &[f64],
    bootstrap_means: &[f64],
    models: usize,
    bootstraps: usize,
    model: usize,
    existing_models: &[usize],
) -> Vec<f64> {
    let mean_differences: Vec<f64> = existing_models
        .iter()
        .map(|&existing_model| model_means[existing_model] - model_means[model])
        .collect();
    let variance_for_column = |(column, &existing_model): (usize, &usize)| {
        let mut variance = 0.0;
        for bootstrap in 0..bootstraps {
            let offset = bootstrap * models;
            let difference = bootstrap_means[offset + existing_model]
                - bootstrap_means[offset + model]
                - mean_differences[column];
            variance += difference * difference;
        }
        variance
    };
    let work = bootstraps.saturating_mul(existing_models.len());
    let variances: Vec<f64> = if work >= 16_384 {
        existing_models
            .par_iter()
            .enumerate()
            .map(variance_for_column)
            .collect()
    } else {
        existing_models
            .iter()
            .enumerate()
            .map(variance_for_column)
            .collect()
    };
    let mut inverse_standard_deviations = vec![0.0; existing_models.len()];
    for (column, variance) in variances.into_iter().enumerate() {
        let standard_deviation = (variance / bootstraps as f64).sqrt();
        inverse_standard_deviations[column] = 1.0 / standard_deviation;
    }

    let maximum_for_bootstrap = |bootstrap: usize| {
        let offset = bootstrap * models;
        let mut maximum = 0.0_f64;
        for (column, &existing_model) in existing_models.iter().enumerate() {
            let difference = bootstrap_means[offset + existing_model]
                - bootstrap_means[offset + model]
                - mean_differences[column];
            maximum = maximum.max((difference * inverse_standard_deviations[column]).abs());
        }
        maximum
    };
    let maxima: Vec<f64> = if work >= 16_384 {
        (0..bootstraps)
            .into_par_iter()
            .map(maximum_for_bootstrap)
            .collect()
    } else {
        (0..bootstraps).map(maximum_for_bootstrap).collect()
    };
    maxima
}

#[allow(clippy::too_many_arguments)]
fn update_one_pass_distribution(
    current_model: usize,
    processed: &[usize],
    ranking: &[usize],
    previous_ranking: &[usize],
    pair_boot_t: &[f64],
    t_score: &[f64],
    t_boot_distribution: &mut [f64],
    bootstraps: usize,
    models: usize,
) {
    let location = ranking
        .iter()
        .position(|&model| model == current_model)
        .expect("current model is present in the ranking");
    let mut predicted_previous_ranking = previous_ranking.to_vec();
    predicted_previous_ranking.insert(location, current_model);
    let mut swaps: Vec<bool> = predicted_previous_ranking
        .iter()
        .zip(ranking)
        .map(|(old, new)| old != new)
        .collect();
    if let (Some(first), Some(last)) = (
        swaps.iter().position(|&swap| swap),
        swaps.iter().rposition(|&swap| swap),
    ) {
        swaps[first..=last].fill(true);
    }

    let existing_models = &processed[..processed.len() - 1];
    let pair_count = existing_models.len();
    let better_columns: Vec<usize> = existing_models
        .iter()
        .enumerate()
        .filter_map(|(column, &model)| (t_score[model] < t_score[current_model]).then_some(column))
        .collect();
    let worse_models: Vec<(usize, usize)> = existing_models
        .iter()
        .enumerate()
        .filter_map(|(column, &model)| {
            (t_score[model] > t_score[current_model]).then_some((column, model))
        })
        .collect();

    let mut maximum = vec![0.0_f64; bootstraps];
    if location != 0 {
        for bootstrap in 0..bootstraps {
            let row_start = bootstrap * pair_count;
            maximum[bootstrap] = better_columns
                .iter()
                .map(|&column| pair_boot_t[row_start + column].abs())
                .fold(0.0_f64, f64::max);
            let previous_best = t_boot_distribution[bootstrap * models + ranking[location - 1]];
            t_boot_distribution[bootstrap * models + current_model] =
                maximum[bootstrap].max(previous_best);
        }
    }

    for (offset, &(column, _worse_model)) in worse_models.iter().enumerate() {
        let rank_position = offset + location + 1;
        for bootstrap in 0..bootstraps {
            maximum[bootstrap] =
                maximum[bootstrap].max(pair_boot_t[bootstrap * pair_count + column].abs());

            let distribution_index = bootstrap * models + ranking[rank_position];
            if swaps[rank_position] {
                let lower = maximum[bootstrap]
                    .max(t_boot_distribution[bootstrap * models + ranking[rank_position - 1]]);
                let upper = lower.max(t_boot_distribution[distribution_index]);
                t_boot_distribution[distribution_index] = 0.5 * (lower + upper);
            } else {
                t_boot_distribution[distribution_index] =
                    maximum[bootstrap].max(t_boot_distribution[distribution_index]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn parses_supported_algorithms_and_reports_unknown_names() {
        assert_eq!(Algorithm::parse("1-pass").unwrap(), Algorithm::OnePass);
        assert_eq!(Algorithm::parse("2-pass").unwrap(), Algorithm::TwoPass);

        let error = Algorithm::parse("other").unwrap_err();
        assert_eq!(
            error.to_string(),
            "algorithm must be '1-pass' or '2-pass', got \"other\""
        );
    }

    #[test]
    fn segmented_loss_matrix_validates_and_maps_columns_across_blocks() {
        let first = array![[1.0, 2.0], [3.0, 4.0]];
        let second = array![[5.0], [6.0]];
        let segmented = SegmentedLossMatrix::new(vec![first.view(), second.view()]).unwrap();

        assert_eq!(segmented.observations(), 2);
        assert_eq!(segmented.models(), 3);
        assert_eq!(segmented.get(1, 0), 3.0);
        assert_eq!(segmented.get(0, 1), 2.0);
        assert_eq!(segmented.get(1, 2), 6.0);

        assert!(matches!(
            SegmentedLossMatrix::new(Vec::new()),
            Err(CoreError::InvalidShape(
                "at least one losses array is required"
            ))
        ));

        let different_length = array![[7.0], [8.0], [9.0]];
        assert!(matches!(
            SegmentedLossMatrix::new(vec![first.view(), different_length.view()]),
            Err(CoreError::InvalidShape(
                "all losses arrays must have the same observation count"
            ))
        ));
    }

    #[test]
    fn run_fast_mcs_rejects_empty_loss_dimensions_and_invalid_bootstrap_shapes() {
        let no_observations = Array2::<f64>::zeros((0, 2));
        let no_observation_indices = Array2::<i64>::zeros((0, 1));
        assert!(matches!(
            run_fast_mcs(
                &no_observations.view(),
                no_observation_indices.view(),
                Algorithm::TwoPass
            ),
            Err(CoreError::InvalidShape(
                "losses must contain at least one observation and one model"
            ))
        ));

        let no_models = Array2::<f64>::zeros((2, 0));
        let valid_indices = array![[0_i64], [1_i64]];
        assert!(matches!(
            run_fast_mcs(&no_models.view(), valid_indices.view(), Algorithm::TwoPass),
            Err(CoreError::InvalidShape(
                "losses must contain at least one observation and one model"
            ))
        ));

        let losses = array![[1.0], [2.0]];
        let wrong_observation_count = array![[0_i64]];
        assert!(matches!(
            run_fast_mcs(
                &losses.view(),
                wrong_observation_count.view(),
                Algorithm::TwoPass
            ),
            Err(CoreError::InvalidShape(_))
        ));

        let no_bootstraps = Array2::<i64>::zeros((2, 0));
        assert!(matches!(
            run_fast_mcs(&losses.view(), no_bootstraps.view(), Algorithm::TwoPass),
            Err(CoreError::InvalidShape(_))
        ));

        let shape_error = run_fast_mcs(
            &losses.view(),
            wrong_observation_count.view(),
            Algorithm::TwoPass,
        )
        .unwrap_err();
        assert_eq!(
            shape_error.to_string(),
            "bootstrap indices must have shape (observations, bootstraps) with nonzero bootstraps"
        );
    }

    #[test]
    fn run_fast_mcs_rejects_negative_and_out_of_range_indices() {
        let losses = array![[1.0], [2.0]];
        for index in [-1_i64, 2_i64] {
            let indices = array![[index], [0_i64]];
            assert!(matches!(
                run_fast_mcs(&losses.view(), indices.view(), Algorithm::TwoPass),
                Err(CoreError::InvalidBootstrapIndex {
                    index: invalid,
                    observations: 2
                }) if invalid == index
            ));
        }
    }

    #[test]
    fn single_model_result_has_one_rank_and_zero_test_statistics() {
        let losses = array![[1.0], [2.0], [4.0]];
        let indices = array![[0_i64, 1], [1, 2], [2, 0]];

        for algorithm in [Algorithm::OnePass, Algorithm::TwoPass] {
            let result = run_fast_mcs(&losses.view(), indices.view(), algorithm).unwrap();
            assert_eq!(result.t_score, vec![0.0]);
            assert_eq!(result.elimination_order, vec![0]);
            assert_eq!(result.t_boot_distribution, vec![0.0, 0.0]);
        }
    }

    #[test]
    fn pairwise_scores_update_an_earlier_model_that_is_outperformed_later() {
        let losses = array![[2.0, 1.0], [4.0, 1.5], [3.0, 2.0]];
        let indices = array![[0_i64, 1], [1, 2], [2, 0]];

        let result = run_fast_mcs(&losses.view(), indices.view(), Algorithm::TwoPass).unwrap();

        assert!(result.t_score[0] > 0.0);
        assert_eq!(result.elimination_order.len(), 2);
    }

    #[test]
    fn pair_statistics_standardizes_mean_differences_and_bootstrap_deviations() {
        let model_means = [1.0, 2.0];
        let bootstrap_means = [0.0, 2.0, 2.0, 3.0];

        let (scores, bootstrap_t) = pair_statistics(&model_means, &bootstrap_means, 2, 2, 0, &[1]);

        let expected = 2.0_f64.sqrt();
        assert!((scores[0] - expected).abs() < 1e-12);
        assert!((bootstrap_t[0] - expected).abs() < 1e-12);
        assert_eq!(bootstrap_t[1], 0.0);
    }

    #[test]
    fn two_pass_parallel_pair_max_matches_direct_bootstrap_maxima() {
        let models = 130;
        let bootstraps = 128;
        let model_means: Vec<f64> = (0..models).map(|model| model as f64 * 0.03).collect();
        let mut bootstrap_means = Vec::with_capacity(bootstraps * models);
        for bootstrap in 0..bootstraps {
            for (model, mean) in model_means.iter().enumerate() {
                bootstrap_means.push(
                    mean + (bootstrap as f64 - (bootstraps as f64 - 1.0) / 2.0)
                        * (model as f64 + 1.0)
                        * 0.001,
                );
            }
        }
        let candidate = models - 1;
        let existing_models: Vec<usize> = (0..candidate).collect();

        let maxima = pair_max_boot_statistics(
            &model_means,
            &bootstrap_means,
            models,
            bootstraps,
            candidate,
            &existing_models,
        );

        let expected: Vec<f64> = (0..bootstraps)
            .map(|bootstrap| {
                existing_models
                    .iter()
                    .map(|&existing| {
                        let mean_difference = model_means[existing] - model_means[candidate];
                        let variance: f64 = (0..bootstraps)
                            .map(|sample| {
                                let offset = sample * models;
                                let difference = bootstrap_means[offset + existing]
                                    - bootstrap_means[offset + candidate]
                                    - mean_difference;
                                difference * difference
                            })
                            .sum();
                        let offset = bootstrap * models;
                        let difference = bootstrap_means[offset + existing]
                            - bootstrap_means[offset + candidate]
                            - mean_difference;
                        (difference / (variance / bootstraps as f64).sqrt()).abs()
                    })
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        for (actual, expected) in maxima.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn one_pass_distribution_keeps_previous_maximum_when_rank_does_not_change() {
        let processed = [0, 1];
        let ranking = [1, 0];
        let previous_ranking = [0];
        let pair_boot_t = [2.0, -1.0];
        let t_score = [3.0, 1.0];
        let mut distribution = [4.0, 0.0, 0.0, 0.0];

        update_one_pass_distribution(
            1,
            &processed,
            &ranking,
            &previous_ranking,
            &pair_boot_t,
            &t_score,
            &mut distribution,
            2,
            2,
        );

        assert_eq!(distribution, [4.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn one_pass_distribution_interpolates_when_ranking_positions_swap() {
        let processed = [0, 1, 2];
        let ranking = [1, 2, 0];
        let previous_ranking = [0, 1];
        let pair_boot_t = [2.0, 1.0, 1.0, -3.0];
        let t_score = [2.0, 0.0, 1.0];
        let mut distribution = [10.0, 6.0, 7.0, 7.0, 1.0, 3.0];

        update_one_pass_distribution(
            2,
            &processed,
            &ranking,
            &previous_ranking,
            &pair_boot_t,
            &t_score,
            &mut distribution,
            2,
            3,
        );

        assert_eq!(distribution[2], 6.0);
        assert_eq!(distribution[5], 3.0);
        assert_eq!(distribution[0], 8.0);
        assert_eq!(distribution[3], 5.0);
    }
}
