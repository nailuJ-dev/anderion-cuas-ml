use serde::{Deserialize, Serialize};

use crate::{Result, SdkError};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SyntheticTrajectoryConfig {
    pub steps: usize,
    pub feature_dim: usize,
    pub velocity: f32,
    pub curvature: f32,
    pub noise_amplitude: f32,
    pub seed: u64,
}

pub fn synthetic_feature_trajectory(config: SyntheticTrajectoryConfig) -> Result<Vec<Vec<f32>>> {
    if config.steps == 0 || config.feature_dim == 0 || config.feature_dim > 4096 {
        return Err(SdkError::InvalidArgument(
            "invalid synthetic trajectory dimensions".into(),
        ));
    }
    let elements = config
        .steps
        .checked_mul(config.feature_dim)
        .ok_or_else(|| {
            SdkError::InvalidArgument("synthetic trajectory element count overflow".into())
        })?;
    if elements > 4_194_304 {
        return Err(SdkError::DimensionLimit {
            actual: elements,
            max: 4_194_304,
        });
    }
    if !config.velocity.is_finite()
        || !config.curvature.is_finite()
        || !config.noise_amplitude.is_finite()
        || config.noise_amplitude < 0.0
    {
        return Err(SdkError::InvalidArgument(
            "synthetic trajectory parameters must be finite".into(),
        ));
    }
    let mut state = config.seed;
    let mut out = Vec::with_capacity(config.steps);
    for step in 0..config.steps {
        let t = step as f32;
        let mut features = Vec::with_capacity(config.feature_dim);
        for dim in 0..config.feature_dim {
            state = xorshift64(state.wrapping_add(dim as u64).wrapping_add(1));
            let noise =
                (((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0) as f32 * config.noise_amplitude;
            let phase = dim as f32 * 0.173;
            features
                .push(config.velocity * t + config.curvature * (t * 0.05 + phase).sin() + noise);
        }
        out.push(features);
    }
    Ok(out)
}

fn xorshift64(mut x: u64) -> u64 {
    if x == 0 {
        x = 0x9E37_79B9_7F4A_7C15;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}
