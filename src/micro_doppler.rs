use crate::{Result, SdkError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MicroDopplerFeatures {
    pub centroid_hz: f64,
    pub bandwidth_hz: f64,
    pub spectral_entropy: f32,
    pub harmonicity: f32,
    pub sideband_symmetry: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MicroDopplerExtractor {
    pub peak_fraction: f32,
}
impl Default for MicroDopplerExtractor {
    fn default() -> Self {
        Self {
            peak_fraction: 0.25,
        }
    }
}

impl MicroDopplerExtractor {
    pub fn extract(&self, doppler_bins_hz: &[f64], power: &[f32]) -> Result<MicroDopplerFeatures> {
        if doppler_bins_hz.len() != power.len() {
            return Err(SdkError::DimensionMismatch {
                expected: doppler_bins_hz.len(),
                actual: power.len(),
            });
        }
        if power.len() < 5 {
            return Err(SdkError::InvalidArgument(
                "micro-Doppler spectrum requires at least five bins".into(),
            ));
        }
        if doppler_bins_hz.iter().any(|v| !v.is_finite())
            || power.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(SdkError::InvalidArgument(
                "micro-Doppler inputs must be finite and power non-negative".into(),
            ));
        }
        let total: f64 = power.iter().map(|v| f64::from(*v)).sum();
        if total <= f64::EPSILON {
            return Err(SdkError::EmptyFeatures);
        }
        let centroid = doppler_bins_hz
            .iter()
            .zip(power)
            .map(|(f, p)| *f * f64::from(*p))
            .sum::<f64>()
            / total;
        let variance = doppler_bins_hz
            .iter()
            .zip(power)
            .map(|(f, p)| (*f - centroid).powi(2) * f64::from(*p))
            .sum::<f64>()
            / total;
        let bandwidth = 2.0 * variance.sqrt();
        let entropy_raw = power
            .iter()
            .filter_map(|v| {
                let p = f64::from(*v) / total;
                (p > 0.0).then_some(-p * p.ln())
            })
            .sum::<f64>();
        let entropy = (entropy_raw / (power.len() as f64).ln()).clamp(0.0, 1.0) as f32;
        let max_power = power.iter().copied().fold(0.0_f32, f32::max);
        let cutoff = max_power * self.peak_fraction;
        let mut peak_freqs = Vec::new();
        for i in 1..power.len() - 1 {
            if power[i] >= cutoff && power[i] >= power[i - 1] && power[i] >= power[i + 1] {
                peak_freqs.push(doppler_bins_hz[i]);
            }
        }
        let harmonicity = harmonicity(&peak_freqs);
        let symmetry = sideband_symmetry(doppler_bins_hz, power, centroid);
        let snr_proxy = (1.0 - entropy).max(0.0);
        let confidence = (0.45 * snr_proxy + 0.35 * harmonicity + 0.20 * symmetry).clamp(0.0, 1.0);
        Ok(MicroDopplerFeatures {
            centroid_hz: centroid,
            bandwidth_hz: bandwidth,
            spectral_entropy: entropy,
            harmonicity,
            sideband_symmetry: symmetry,
            confidence,
        })
    }
}

fn harmonicity(peaks: &[f64]) -> f32 {
    if peaks.len() < 3 {
        return 0.0;
    }
    let mut sorted = peaks.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let spacings: Vec<f64> = sorted
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 0.0)
        .collect();
    if spacings.is_empty() {
        return 0.0;
    }
    let mean = spacings.iter().sum::<f64>() / spacings.len() as f64;
    let variance =
        spacings.iter().map(|d| (*d - mean).powi(2)).sum::<f64>() / spacings.len() as f64;
    (1.0 / (1.0 + variance.sqrt() / mean.max(1e-9))).clamp(0.0, 1.0) as f32
}

fn sideband_symmetry(freqs: &[f64], power: &[f32], center: f64) -> f32 {
    let mut left = 0.0_f64;
    let mut right = 0.0_f64;
    for (f, p) in freqs.iter().zip(power) {
        if *f < center {
            left += f64::from(*p);
        } else if *f > center {
            right += f64::from(*p);
        }
    }
    let sum = left + right;
    if sum <= f64::EPSILON {
        return 1.0;
    }
    (1.0 - (left - right).abs() / sum).clamp(0.0, 1.0) as f32
}
