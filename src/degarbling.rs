use serde::{Deserialize, Serialize};

use crate::types::MAX_FEATURES;
use crate::{Observation, Result, SdkError};

const MAX_COMPONENTS: usize = 32;

pub trait DegarblingModel: Send + Sync {
    fn separate(&self, observation: &Observation) -> Result<DegarblingResult>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DegarblingResult {
    components: Vec<Observation>,
    reconstruction_error: f32,
    separation_confidence: f32,
}

impl DegarblingResult {
    pub fn components(&self) -> &[Observation] {
        &self.components
    }
    pub fn reconstruction_error(&self) -> f32 {
        self.reconstruction_error
    }
    pub fn separation_confidence(&self) -> f32 {
        self.separation_confidence
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityDegarbler;

impl DegarblingModel for IdentityDegarbler {
    fn separate(&self, observation: &Observation) -> Result<DegarblingResult> {
        validate_observation(observation)?;
        Ok(DegarblingResult {
            components: vec![observation.clone()],
            reconstruction_error: 0.0,
            separation_confidence: 1.0,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrototypeMaskDegarbler {
    templates: Vec<Vec<f32>>,
    temperature: f32,
}

impl PrototypeMaskDegarbler {
    pub fn new(templates: Vec<Vec<f32>>, temperature: f32) -> Result<Self> {
        let model = Self {
            templates,
            temperature,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn fit(
        samples: &[(usize, Vec<f32>)],
        component_count: usize,
        temperature: f32,
    ) -> Result<Self> {
        if samples.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        if component_count == 0 || component_count > MAX_COMPONENTS {
            return Err(SdkError::DimensionLimit {
                actual: component_count,
                max: MAX_COMPONENTS,
            });
        }
        let dim = samples[0].1.len();
        if dim == 0 || dim > MAX_FEATURES {
            return Err(SdkError::DimensionLimit {
                actual: dim,
                max: MAX_FEATURES,
            });
        }
        let mut sums = vec![vec![0.0_f64; dim]; component_count];
        let mut counts = vec![0_u64; component_count];
        for (label, features) in samples {
            if *label >= component_count {
                return Err(SdkError::InvalidArgument(
                    "degarbling component label is out of range".into(),
                ));
            }
            if features.len() != dim {
                return Err(SdkError::DimensionMismatch {
                    expected: dim,
                    actual: features.len(),
                });
            }
            for (index, value) in features.iter().enumerate() {
                if !value.is_finite() {
                    return Err(SdkError::NonFiniteValue { index });
                }
                sums[*label][index] += f64::from(value.abs());
            }
            counts[*label] = counts[*label].saturating_add(1);
        }
        let mut templates = Vec::with_capacity(component_count);
        for (component_sums, count) in sums.iter().zip(&counts) {
            if *count == 0 {
                return Err(SdkError::InvalidArgument(
                    "every degarbling component requires at least one training sample".into(),
                ));
            }
            let denominator = *count as f64;
            templates.push(
                component_sums
                    .iter()
                    .map(|sum| (*sum / denominator) as f32)
                    .collect(),
            );
        }
        Self::new(templates, temperature)
    }

    pub fn component_count(&self) -> usize {
        self.templates.len()
    }
    pub fn input_dim(&self) -> usize {
        match self.templates.first() {
            Some(template) => template.len(),
            None => 0,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.templates.is_empty() {
            return Err(SdkError::InvalidArgument(
                "degarbling templates must not be empty".into(),
            ));
        }
        if self.templates.len() > MAX_COMPONENTS {
            return Err(SdkError::DimensionLimit {
                actual: self.templates.len(),
                max: MAX_COMPONENTS,
            });
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "degarbling temperature must be finite and positive".into(),
            ));
        }
        let dim = self.templates[0].len();
        if dim == 0 || dim > MAX_FEATURES {
            return Err(SdkError::DimensionLimit {
                actual: dim,
                max: MAX_FEATURES,
            });
        }
        for template in &self.templates {
            if template.len() != dim {
                return Err(SdkError::DimensionMismatch {
                    expected: dim,
                    actual: template.len(),
                });
            }
            if let Some((index, _)) = template
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite() || **value < 0.0)
            {
                return Err(SdkError::InvalidArgument(format!(
                    "degarbling template value at index {index} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }
}

impl DegarblingModel for PrototypeMaskDegarbler {
    fn separate(&self, observation: &Observation) -> Result<DegarblingResult> {
        self.validate()?;
        validate_observation(observation)?;
        if observation.features().len() != self.input_dim() {
            return Err(SdkError::DimensionMismatch {
                expected: self.input_dim(),
                actual: observation.features().len(),
            });
        }

        let mut separated = vec![vec![0.0_f32; self.input_dim()]; self.component_count()];
        let mut entropy_sum = 0.0_f64;
        for feature_index in 0..self.input_dim() {
            let max_template = self
                .templates
                .iter()
                .map(|template| template[feature_index])
                .fold(f32::NEG_INFINITY, f32::max);
            let mut weights = Vec::with_capacity(self.component_count());
            let mut weight_sum = 0.0_f64;
            for template in &self.templates {
                let exponent =
                    f64::from((template[feature_index] - max_template) / self.temperature);
                let weight = exponent.exp();
                weights.push(weight);
                weight_sum += weight;
            }
            if !weight_sum.is_finite() || weight_sum <= 0.0 {
                return Err(SdkError::InvalidArgument(
                    "degarbling produced an invalid assignment normalization".into(),
                ));
            }
            let mut entropy = 0.0_f64;
            for (component_index, weight) in weights.into_iter().enumerate() {
                let probability = weight / weight_sum;
                separated[component_index][feature_index] =
                    observation.features()[feature_index] * (probability as f32);
                if probability > 0.0 {
                    entropy -= probability * probability.ln();
                }
            }
            let normalizer = (self.component_count() as f64).ln();
            if normalizer > 0.0 {
                entropy_sum += entropy / normalizer;
            }
        }

        let mut components = Vec::with_capacity(self.component_count());
        for (index, features) in separated.into_iter().enumerate() {
            components.push(Observation::new(
                format!("{}#component-{index}", observation.id()),
                observation.sensor_id().to_string(),
                observation.timestamp_ms(),
                features,
            )?);
        }

        let reconstruction_error = reconstruction_error(observation.features(), &components)?;
        let mean_entropy = if self.input_dim() == 0 {
            0.0
        } else {
            entropy_sum / (self.input_dim() as f64)
        };
        let separation_confidence = (1.0 - mean_entropy).clamp(0.0, 1.0) as f32;
        Ok(DegarblingResult {
            components,
            reconstruction_error,
            separation_confidence,
        })
    }
}

fn reconstruction_error(original: &[f32], components: &[Observation]) -> Result<f32> {
    if components.is_empty() {
        return Err(SdkError::InvalidArgument(
            "degarbling must produce at least one component".into(),
        ));
    }
    let mut absolute_error = 0.0_f64;
    for (index, &original_value) in original.iter().enumerate() {
        let reconstructed: f64 = components
            .iter()
            .map(|component| f64::from(component.features()[index]))
            .sum();
        absolute_error += (reconstructed - f64::from(original_value)).abs();
    }
    Ok((absolute_error / (original.len() as f64)) as f32)
}

fn validate_observation(observation: &Observation) -> Result<()> {
    Observation::new(
        observation.id().to_string(),
        observation.sensor_id().to_string(),
        observation.timestamp_ms(),
        observation.features().to_vec(),
    )
    .map(|_| ())
}
