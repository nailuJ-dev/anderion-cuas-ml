use serde::{Deserialize, Serialize};

use crate::{Embedding, Localization, Localizer, Position3, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLocalizer {
    weights: Vec<[f64; 3]>,
    bias: [f64; 3],
    residual_sigma_m: f64,
}

impl LinearLocalizer {
    pub fn fit(
        samples: &[(Embedding, Position3)],
        epochs: usize,
        learning_rate: f64,
        l2: f64,
    ) -> Result<Self> {
        if samples.is_empty() { return Err(SdkError::EmptyDataset); }
        if epochs == 0 || !learning_rate.is_finite() || learning_rate <= 0.0 || !l2.is_finite() || l2 < 0.0 {
            return Err(SdkError::InvalidArgument("invalid localizer training hyperparameters".into()));
        }
        let dim = samples[0].0.dim();
        if samples.iter().any(|(e, _)| e.dim() != dim) {
            return Err(SdkError::DimensionMismatch { expected: dim, actual: samples.iter().map(|(e, _)| e.dim()).max().unwrap_or(dim) });
        }
        let mut weights = vec![[0.0_f64; 3]; dim];
        let mut bias = [0.0_f64; 3];
        let n = samples.len() as f64;
        for _ in 0..epochs {
            let mut grad_w = vec![[0.0_f64; 3]; dim];
            let mut grad_b = [0.0_f64; 3];
            for (embedding, position) in samples {
                let pred = predict_raw(&weights, bias, embedding.values())?;
                let target = [position.x, position.y, position.z];
                for axis in 0..3 {
                    let error = pred[axis] - target[axis];
                    grad_b[axis] += error;
                    for (index, feature) in embedding.values().iter().enumerate() {
                        grad_w[index][axis] += error * f64::from(*feature);
                    }
                }
            }
            for axis in 0..3 { bias[axis] -= learning_rate * grad_b[axis] / n; }
            for index in 0..dim {
                for axis in 0..3 {
                    weights[index][axis] -= learning_rate * (grad_w[index][axis] / n + l2 * weights[index][axis]);
                }
            }
        }
        let mut squared_error = 0.0_f64;
        for (embedding, position) in samples {
            let pred = predict_raw(&weights, bias, embedding.values())?;
            squared_error += (pred[0] - position.x).powi(2) + (pred[1] - position.y).powi(2) + (pred[2] - position.z).powi(2);
        }
        let residual_sigma_m = (squared_error / samples.len() as f64).sqrt();
        Ok(Self { weights, bias, residual_sigma_m })
    }

    pub fn validate(&self) -> Result<()> {
        if self.weights.is_empty() || self.weights.len() > 8_192 || self.weights.iter().flatten().any(|v| !v.is_finite()) { return Err(SdkError::InvalidArgument("invalid localizer weights".into())); }
        if self.bias.iter().any(|v| !v.is_finite()) || !self.residual_sigma_m.is_finite() || self.residual_sigma_m < 0.0 { return Err(SdkError::InvalidArgument("invalid localizer payload".into())); }
        Ok(())
    }

    pub fn residual_sigma_m(&self) -> f64 { self.residual_sigma_m }
}

impl Localizer for LinearLocalizer {
    fn localize(&self, embedding: &Embedding) -> Result<Localization> {
        if embedding.dim() != self.weights.len() {
            return Err(SdkError::DimensionMismatch { expected: self.weights.len(), actual: embedding.dim() });
        }
        let raw = predict_raw(&self.weights, self.bias, embedding.values())?;
        let position = Position3::new(raw[0], raw[1], raw[2])?;
        let confidence = (1.0 / (1.0 + self.residual_sigma_m)) as f32;
        Localization::new(position, self.residual_sigma_m, confidence.clamp(0.0, 1.0))
    }

    fn input_dim(&self) -> usize { self.weights.len() }
}

fn predict_raw(weights: &[[f64; 3]], bias: [f64; 3], features: &[f32]) -> Result<[f64; 3]> {
    if weights.len() != features.len() {
        return Err(SdkError::DimensionMismatch { expected: weights.len(), actual: features.len() });
    }
    let mut out = bias;
    for (w, feature) in weights.iter().zip(features) {
        for axis in 0..3 { out[axis] += w[axis] * f64::from(*feature); }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(SdkError::InvalidArgument("localizer prediction became non-finite".into()));
    }
    Ok(out)
}
