use serde::{Deserialize, Serialize};

use crate::{ClassScore, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureScaler {
    temperature: f32,
}

impl TemperatureScaler {
    pub fn new(temperature: f32) -> Result<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SdkError::InvalidArgument("temperature must be finite and positive".into()));
        }
        Ok(Self { temperature })
    }


    pub fn fit(calibration: &[(Vec<ClassScore>, String)]) -> Result<Self> {
        if calibration.is_empty() { return Err(SdkError::EmptyDataset); }
        let candidates = [0.5_f32, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0];
        let mut best = (f32::INFINITY, 1.0_f32);
        for temperature in candidates {
            let scaler = Self::new(temperature)?;
            let mut nll = 0.0_f32;
            for (scores, label) in calibration {
                let calibrated = scaler.calibrate(scores)?;
                let probability = calibrated.iter().find(|score| &score.label == label)
                    .map(|score| score.probability).unwrap_or(f32::MIN_POSITIVE).max(f32::MIN_POSITIVE);
                nll -= probability.ln();
            }
            nll /= calibration.len() as f32;
            if nll < best.0 { best = (nll, temperature); }
        }
        Self::new(best.1)
    }

    pub fn calibrate(&self, scores: &[ClassScore]) -> Result<Vec<ClassScore>> {
        if scores.is_empty() { return Err(SdkError::EmptyDataset); }
        let values: Vec<f32> = scores.iter().map(|s| s.probability.max(f32::MIN_POSITIVE).powf(1.0 / self.temperature)).collect();
        let sum: f32 = values.iter().sum();
        if !sum.is_finite() || sum <= 0.0 { return Err(SdkError::InvalidArgument("calibration normalization failed".into())); }
        let mut out = Vec::with_capacity(scores.len());
        for (score, value) in scores.iter().zip(values) { out.push(ClassScore::new(score.label.clone(), value / sum)?); }
        out.sort_by(|a, b| b.probability.total_cmp(&a.probability));
        Ok(out)
    }
}
