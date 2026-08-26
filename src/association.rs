use serde::{Deserialize, Serialize};

use crate::linear::{fit_logistic, predict_logistic};
use crate::{AssociationModel, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedAssociation {
    weights: Vec<f32>,
    bias: f32,
}

impl LearnedAssociation {
    pub fn fit(
        training: &[(Vec<f32>, bool)],
        epochs: usize,
        learning_rate: f32,
        l2: f32,
    ) -> Result<Self> {
        let samples: Vec<Vec<f32>> = training.iter().map(|(features, _)| features.clone()).collect();
        let labels: Vec<bool> = training.iter().map(|(_, label)| *label).collect();
        let (weights, bias) = fit_logistic(&samples, &labels, epochs, learning_rate, l2)?;
        Ok(Self { weights, bias })
    }
}

impl AssociationModel for LearnedAssociation {
    fn score_features(&self, features: &[f32]) -> Result<f32> {
        predict_logistic(&self.weights, self.bias, features)
    }
    fn feature_dim(&self) -> usize { self.weights.len() }
}
