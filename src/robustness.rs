use serde::{Deserialize, Serialize};

use crate::{ClassScore, Result, SdkError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdversarialReport {
    pub baseline_probability: f32,
    pub perturbed_probability: f32,
    pub confidence_drop: f32,
    pub class_changed: bool,
}

pub fn bounded_perturbation(features: &[f32], deltas: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    if features.len() != deltas.len() {
        return Err(SdkError::DimensionMismatch { expected: features.len(), actual: deltas.len() });
    }
    if !epsilon.is_finite() || epsilon < 0.0 { return Err(SdkError::InvalidArgument("epsilon must be non-negative".into())); }
    let mut out = Vec::with_capacity(features.len());
    for (feature, delta) in features.iter().zip(deltas) {
        if !feature.is_finite() || !delta.is_finite() { return Err(SdkError::InvalidArgument("perturbation inputs must be finite".into())); }
        out.push(*feature + delta.clamp(-epsilon, epsilon));
    }
    Ok(out)
}

pub fn adversarial_evaluate(target: &str, baseline: &[ClassScore], perturbed: &[ClassScore]) -> Result<AdversarialReport> {
    if baseline.is_empty() || perturbed.is_empty() { return Err(SdkError::EmptyDataset); }
    let baseline_probability = probability_for(target, baseline);
    let perturbed_probability = probability_for(target, perturbed);
    let baseline_top = baseline.iter().max_by(|a, b| a.probability.total_cmp(&b.probability)).map(|s| s.label.as_str());
    let perturbed_top = perturbed.iter().max_by(|a, b| a.probability.total_cmp(&b.probability)).map(|s| s.label.as_str());
    Ok(AdversarialReport {
        baseline_probability,
        perturbed_probability,
        confidence_drop: (baseline_probability - perturbed_probability).max(0.0),
        class_changed: baseline_top != perturbed_top,
    })
}

fn probability_for(target: &str, scores: &[ClassScore]) -> f32 {
    scores.iter().find(|s| s.label == target).map(|s| s.probability).unwrap_or(0.0)
}
