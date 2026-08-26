use serde::{Deserialize, Serialize};

use crate::{Position3, Result, SdkError};

#[derive(Debug, Clone, Copy)]
pub struct TrajectorySample {
    current: Position3,
    velocity: [f64; 3],
    horizon_s: f64,
    target: Position3,
}

impl TrajectorySample {
    pub fn new(current: Position3, velocity: [f64; 3], horizon_s: f64, target: Position3) -> Result<Self> {
        if velocity.iter().any(|value| !value.is_finite()) || !horizon_s.is_finite() || horizon_s <= 0.0 || horizon_s > 3_600.0 {
            return Err(SdkError::InvalidArgument("trajectory sample requires finite velocity and horizon in (0,3600]".into()));
        }
        Ok(Self { current, velocity, horizon_s, target })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoregressiveTrajectoryPredictor {
    weights: Vec<[f64; 3]>,
}

impl AutoregressiveTrajectoryPredictor {
    pub fn fit(samples: &[TrajectorySample], epochs: usize, learning_rate: f64, l2: f64) -> Result<Self> {
        if samples.is_empty() { return Err(SdkError::EmptyDataset); }
        if epochs == 0 || epochs > 100_000 || !learning_rate.is_finite() || learning_rate <= 0.0 || !l2.is_finite() || l2 < 0.0 {
            return Err(SdkError::InvalidArgument("invalid trajectory training hyperparameters".into()));
        }
        let mut weights = vec![[0.0_f64; 3]; 7];
        let n = samples.len() as f64;
        for _ in 0..epochs {
            let mut gradients = vec![[0.0_f64; 3]; 7];
            for sample in samples {
                let features = feature_vector(sample.current, sample.velocity, sample.horizon_s)?;
                let prediction = predict_raw(&weights, &features)?;
                let target = [sample.target.x, sample.target.y, sample.target.z];
                for axis in 0..3 {
                    let error = prediction[axis] - target[axis];
                    for index in 0..features.len() { gradients[index][axis] += error * features[index]; }
                }
            }
            for index in 0..weights.len() {
                for axis in 0..3 {
                    weights[index][axis] -= learning_rate * (gradients[index][axis] / n + l2 * weights[index][axis]);
                }
            }
        }
        Ok(Self { weights })
    }

    pub fn validate(&self) -> Result<()> {
        if self.weights.len() != 7 || self.weights.iter().flatten().any(|value| !value.is_finite()) {
            return Err(SdkError::InvalidArgument("invalid trajectory predictor payload".into()));
        }
        Ok(())
    }

    pub fn predict(&self, current: Position3, velocity: [f64; 3], horizon_s: f64) -> Result<Position3> {
        self.validate()?;
        let features = feature_vector(current, velocity, horizon_s)?;
        let output = predict_raw(&self.weights, &features)?;
        Position3::new(output[0], output[1], output[2])
    }
}

fn feature_vector(current: Position3, velocity: [f64; 3], horizon_s: f64) -> Result<[f64; 7]> {
    if velocity.iter().any(|value| !value.is_finite()) || !horizon_s.is_finite() || horizon_s <= 0.0 || horizon_s > 3_600.0 {
        return Err(SdkError::InvalidArgument("prediction requires finite velocity and horizon in (0,3600]".into()));
    }
    Ok([current.x, current.y, current.z, velocity[0] * horizon_s, velocity[1] * horizon_s, velocity[2] * horizon_s, 1.0])
}

fn predict_raw(weights: &[[f64; 3]], features: &[f64; 7]) -> Result<[f64; 3]> {
    if weights.len() != features.len() {
        return Err(SdkError::DimensionMismatch { expected: features.len(), actual: weights.len() });
    }
    let mut output = [0.0_f64; 3];
    for (weights_for_feature, feature) in weights.iter().zip(features) {
        for axis in 0..3 { output[axis] += weights_for_feature[axis] * *feature; }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(SdkError::InvalidArgument("trajectory predictor produced non-finite output".into()));
    }
    Ok(output)
}
