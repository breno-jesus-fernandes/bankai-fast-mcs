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

    if algorithm == Algorithm::OnePass {
        let mut means = vec![0.0; models];
        for observation in 0..observations {
            for model in 0..models {
                means[model] += losses.get(observation, model);
            }
        }
        for mean in &mut means {
            *mean /= observations as f64;
        }
        processing_order.sort_by(|left, right| {
            means[*left]
                .total_cmp(&means[*right])
                .then_with(|| left.cmp(right))
        });
    }

    let mut ranking = Vec::with_capacity(models);
    for model in processing_order {
        let (pair_scores, pair_boot_t) =
            pair_statistics(losses, &bootstrap_means, bootstraps, model, &processed);

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
        for rank_position in 1..models {
            let model = ranking[rank_position];
            let better_models = &ranking[..rank_position];
            let (_, pair_boot_t) =
                pair_statistics(losses, &bootstrap_means, bootstraps, model, better_models);
            let pair_count = better_models.len();
            for bootstrap in 0..bootstraps {
                let row = &pair_boot_t[bootstrap * pair_count..(bootstrap + 1) * pair_count];
                let maximum = row
                    .iter()
                    .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
                let previous = t_boot_distribution[bootstrap * models + ranking[rank_position - 1]];
                t_boot_distribution[bootstrap * models + model] = maximum.max(previous);
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
fn pair_statistics<L: LossMatrix>(
    losses: &L,
    bootstrap_means: &[f64],
    bootstraps: usize,
    model: usize,
    existing_models: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let observations = losses.observations();
    let models = losses.models();
    let pair_count = existing_models.len();
    let mut mean_differences = vec![0.0; pair_count];

    for observation in 0..observations {
        let candidate_loss = losses.get(observation, model);
        for (column, &existing_model) in existing_models.iter().enumerate() {
            mean_differences[column] += losses.get(observation, existing_model) - candidate_loss;
        }
    }
    for difference in &mut mean_differences {
        *difference /= observations as f64;
    }

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
