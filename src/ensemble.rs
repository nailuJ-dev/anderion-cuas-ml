use std::collections::BTreeMap;

use crate::temporal::normalize;
use crate::{ClassScore, Detection, Result, SdkError};

#[derive(Debug, Clone)]
pub struct WeightedPerceptionEnsemble {
    weights: Vec<f32>,
}

impl WeightedPerceptionEnsemble {
    pub fn new(weights: Vec<f32>) -> Result<Self> {
        if weights.is_empty()
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
            || weights.iter().sum::<f32>() <= 0.0
        {
            return Err(SdkError::InvalidArgument(
                "ensemble weights must be finite, non-negative, and have positive mass".into(),
            ));
        }
        Ok(Self { weights })
    }

    pub fn combine_classifications(
        &self,
        predictions: &[Vec<ClassScore>],
    ) -> Result<Vec<ClassScore>> {
        if predictions.len() != self.weights.len() {
            return Err(SdkError::DimensionMismatch {
                expected: self.weights.len(),
                actual: predictions.len(),
            });
        }
        let total_weight: f32 = self.weights.iter().sum();
        let mut totals: BTreeMap<String, f32> = BTreeMap::new();
        for (prediction, weight) in predictions.iter().zip(&self.weights) {
            for score in prediction {
                *totals.entry(score.label.clone()).or_insert(0.0) += *weight * score.probability;
            }
        }
        normalize(
            totals
                .into_iter()
                .map(|(label, value)| (label, value / total_weight))
                .collect(),
        )
    }

    pub fn combine_detections(
        &self,
        detections: &[Detection],
        threshold: f32,
    ) -> Result<Detection> {
        if detections.len() != self.weights.len() {
            return Err(SdkError::DimensionMismatch {
                expected: self.weights.len(),
                actual: detections.len(),
            });
        }
        let total_weight: f32 = self.weights.iter().sum();
        let probability = detections
            .iter()
            .zip(&self.weights)
            .map(|(detection, weight)| detection.probability * *weight)
            .sum::<f32>()
            / total_weight;
        Detection::new(probability, threshold)
    }
}
