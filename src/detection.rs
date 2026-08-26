use serde::{Deserialize, Serialize};

use crate::linear::{fit_logistic, predict_logistic};
use crate::{Detection, Detector, Embedding, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryLogisticDetector {
    weights: Vec<f32>,
    bias: f32,
    threshold: f32,
}

impl BinaryLogisticDetector {
    pub fn fit(
        samples: &[Vec<f32>],
        labels: &[bool],
        epochs: usize,
        learning_rate: f32,
        l2: f32,
    ) -> Result<Self> {
        let (weights, bias) = fit_logistic(samples, labels, epochs, learning_rate, l2)?;
        Ok(Self { weights, bias, threshold: 0.5 })
    }

    pub fn validate(&self) -> Result<()> {
        if self.weights.is_empty() || self.weights.len() > 8_192 || self.weights.iter().any(|v| !v.is_finite()) || !self.bias.is_finite() {
            return Err(SdkError::InvalidArgument("invalid detector payload".into()));
        }
        if !self.threshold.is_finite() || !(0.0..=1.0).contains(&self.threshold) { return Err(SdkError::InvalidProbability(self.threshold)); }
        Ok(())
    }

    pub fn with_threshold(mut self, threshold: f32) -> Result<Self> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(SdkError::InvalidProbability(threshold));
        }
        self.threshold = threshold;
        Ok(self)
    }

    pub fn probability(&self, embedding: &Embedding) -> Result<f32> {
        predict_logistic(&self.weights, self.bias, embedding.values())
    }
}

impl Detector for BinaryLogisticDetector {
    fn detect(&self, embedding: &Embedding) -> Result<Detection> {
        Detection::new(self.probability(embedding)?, self.threshold)
    }
    fn input_dim(&self) -> usize { self.weights.len() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementQualityModel {
    weights: Vec<f32>,
    bias: f32,
}

impl MeasurementQualityModel {
    pub fn fit(
        samples: &[Vec<f32>],
        acceptable: &[bool],
        epochs: usize,
        learning_rate: f32,
        l2: f32,
    ) -> Result<Self> {
        let (weights, bias) = fit_logistic(samples, acceptable, epochs, learning_rate, l2)?;
        Ok(Self { weights, bias })
    }

    pub fn score(&self, features: &[f32]) -> Result<f32> {
        predict_logistic(&self.weights, self.bias, features)
    }
}
