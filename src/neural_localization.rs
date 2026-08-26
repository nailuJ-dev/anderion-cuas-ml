use serde::{Deserialize, Serialize};

use crate::{Embedding, Localization, Localizer, Position3, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralLocalizer {
    input_dim: usize,
    hidden_dim: usize,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<[f32; 3]>,
    b2: [f32; 3],
    residual_sigma_m: f64,
}

impl NeuralLocalizer {
    pub fn fit(
        samples: &[(Embedding, Position3)],
        hidden_dim: usize,
        epochs: usize,
        learning_rate: f32,
        l2: f32,
        seed: u64,
    ) -> Result<Self> {
        if samples.is_empty() { return Err(SdkError::EmptyDataset); }
        let input_dim = samples[0].0.dim();
        if hidden_dim == 0 || hidden_dim > 256 {
            return Err(SdkError::InvalidArgument("hidden_dim must be in 1..=256".into()));
        }
        if input_dim == 0 || input_dim > 8_192 || input_dim.saturating_mul(hidden_dim) > 1_048_576 {
            return Err(SdkError::DimensionLimit { actual: input_dim.saturating_mul(hidden_dim), max: 1_048_576 });
        }
        if epochs == 0 || epochs > 100_000 || !learning_rate.is_finite() || learning_rate <= 0.0 || learning_rate > 1.0 || !l2.is_finite() || l2 < 0.0 || l2 > 1_000.0 {
            return Err(SdkError::InvalidArgument("invalid neural localizer hyperparameters".into()));
        }
        if samples.iter().any(|(embedding, _)| embedding.dim() != input_dim) {
            return Err(SdkError::InvalidArgument("localizer sample dimensions must match".into()));
        }
        let mut w1 = vec![vec![0.0_f32; input_dim]; hidden_dim];
        for (hidden, row) in w1.iter_mut().enumerate() {
            for (input, value) in row.iter_mut().enumerate() {
                *value = 0.05 * pseudo_weight(seed, hidden, input);
            }
        }
        let mut b1 = vec![0.0_f32; hidden_dim];
        let mut w2 = vec![[0.0_f32; 3]; hidden_dim];
        for (hidden, weights) in w2.iter_mut().enumerate() {
            for (axis, value) in weights.iter_mut().enumerate() {
                *value = 0.05 * pseudo_weight(seed ^ 0xA5A5_A5A5_A5A5_A5A5, axis, hidden);
            }
        }
        let mut b2 = [0.0_f32; 3];
        let n = samples.len() as f32;
        for _ in 0..epochs {
            let mut gw1 = vec![vec![0.0_f32; input_dim]; hidden_dim];
            let mut gb1 = vec![0.0_f32; hidden_dim];
            let mut gw2 = vec![[0.0_f32; 3]; hidden_dim];
            let mut gb2 = [0.0_f32; 3];
            for (embedding, position) in samples {
                let (hidden, prediction) = forward(&w1, &b1, &w2, b2, embedding.values())?;
                let target = position_as_f32(*position)?;
                let mut output_error = [0.0_f32; 3];
                for axis in 0..3 {
                    output_error[axis] = prediction[axis] - target[axis];
                    gb2[axis] += output_error[axis];
                }
                let mut hidden_error = vec![0.0_f32; hidden_dim];
                for hidden_index in 0..hidden_dim {
                    for axis in 0..3 {
                        gw2[hidden_index][axis] += output_error[axis] * hidden[hidden_index];
                        hidden_error[hidden_index] += output_error[axis] * w2[hidden_index][axis];
                    }
                    hidden_error[hidden_index] *= 1.0 - hidden[hidden_index] * hidden[hidden_index];
                    gb1[hidden_index] += hidden_error[hidden_index];
                    for input in 0..input_dim {
                        gw1[hidden_index][input] += hidden_error[hidden_index] * embedding.values()[input];
                    }
                }
            }
            for axis in 0..3 { b2[axis] -= learning_rate * gb2[axis] / n; }
            for hidden in 0..hidden_dim {
                b1[hidden] -= learning_rate * gb1[hidden] / n;
                for axis in 0..3 {
                    w2[hidden][axis] -= learning_rate * (gw2[hidden][axis] / n + l2 * w2[hidden][axis]);
                }
                for input in 0..input_dim {
                    w1[hidden][input] -= learning_rate * (gw1[hidden][input] / n + l2 * w1[hidden][input]);
                }
            }
        }
        let mut squared_error = 0.0_f64;
        for (embedding, position) in samples {
            let (_, prediction) = forward(&w1, &b1, &w2, b2, embedding.values())?;
            squared_error += (f64::from(prediction[0]) - position.x).powi(2)
                + (f64::from(prediction[1]) - position.y).powi(2)
                + (f64::from(prediction[2]) - position.z).powi(2);
        }
        let residual_sigma_m = (squared_error / samples.len() as f64).sqrt();
        Ok(Self { input_dim, hidden_dim, w1, b1, w2, b2, residual_sigma_m })
    }

    pub fn validate(&self) -> Result<()> {
        if self.input_dim == 0 || self.hidden_dim == 0 || self.hidden_dim > 256 || self.w1.len() != self.hidden_dim || self.w2.len() != self.hidden_dim || self.b1.len() != self.hidden_dim {
            return Err(SdkError::InvalidArgument("invalid neural localizer dimensions".into()));
        }
        if self.w1.iter().any(|row| row.len() != self.input_dim || row.iter().any(|value| !value.is_finite()))
            || self.w2.iter().flatten().any(|value| !value.is_finite())
            || self.b1.iter().any(|value| !value.is_finite())
            || self.b2.iter().any(|value| !value.is_finite())
            || !self.residual_sigma_m.is_finite()
            || self.residual_sigma_m < 0.0
        {
            return Err(SdkError::InvalidArgument("invalid neural localizer payload".into()));
        }
        Ok(())
    }
}

impl Localizer for NeuralLocalizer {
    fn localize(&self, embedding: &Embedding) -> Result<Localization> {
        self.validate()?;
        if embedding.dim() != self.input_dim {
            return Err(SdkError::DimensionMismatch { expected: self.input_dim, actual: embedding.dim() });
        }
        let (_, prediction) = forward(&self.w1, &self.b1, &self.w2, self.b2, embedding.values())?;
        let position = Position3::new(f64::from(prediction[0]), f64::from(prediction[1]), f64::from(prediction[2]))?;
        let confidence = (1.0 / (1.0 + self.residual_sigma_m)) as f32;
        Localization::new(position, self.residual_sigma_m, confidence.clamp(0.0, 1.0))
    }

    fn input_dim(&self) -> usize { self.input_dim }
}

fn forward(w1: &[Vec<f32>], b1: &[f32], w2: &[[f32; 3]], b2: [f32; 3], features: &[f32]) -> Result<(Vec<f32>, [f32; 3])> {
    if w1.len() != b1.len() || w1.len() != w2.len() || w1.iter().any(|row| row.len() != features.len()) {
        return Err(SdkError::InvalidArgument("neural localizer matrix dimensions are inconsistent".into()));
    }
    let hidden: Vec<f32> = w1.iter().zip(b1).map(|(row, bias)| (row.iter().zip(features).map(|(weight, value)| weight * value).sum::<f32>() + *bias).tanh()).collect();
    let mut output = b2;
    for (activation, weights) in hidden.iter().zip(w2) {
        for axis in 0..3 { output[axis] += activation * weights[axis]; }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(SdkError::InvalidArgument("neural localizer produced non-finite output".into()));
    }
    Ok((hidden, output))
}

fn position_as_f32(position: Position3) -> Result<[f32; 3]> {
    let max = f64::from(f32::MAX);
    if position.x.abs() > max || position.y.abs() > max || position.z.abs() > max {
        return Err(SdkError::InvalidArgument("position magnitude exceeds neural localizer range".into()));
    }
    Ok([position.x as f32, position.y as f32, position.z as f32])
}

fn pseudo_weight(seed: u64, row: usize, column: usize) -> f32 {
    let mut x = seed ^ (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (column as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let unit = x as f64 / u64::MAX as f64;
    (unit.mul_add(2.0, -1.0)) as f32
}
