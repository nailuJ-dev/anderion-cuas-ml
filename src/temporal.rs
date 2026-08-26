use std::collections::BTreeMap;

use crate::{ClassScore, Result, SdkError};

#[derive(Debug, Clone)]
pub struct TemporalClassifier {
    decay: f32,
}

impl TemporalClassifier {
    pub fn new(decay: f32) -> Result<Self> {
        if !decay.is_finite() || !(0.0..=1.0).contains(&decay) { return Err(SdkError::InvalidProbability(decay)); }
        Ok(Self { decay })
    }

    pub fn aggregate(&self, history: &[Vec<ClassScore>]) -> Result<Vec<ClassScore>> {
        if history.is_empty() { return Err(SdkError::EmptyDataset); }
        let mut totals: BTreeMap<String, f32> = BTreeMap::new();
        let mut total_weight = 0.0;
        for (age, scores) in history.iter().rev().enumerate() {
            let weight = self.decay.powi(age as i32);
            total_weight += weight;
            for score in scores { *totals.entry(score.label.clone()).or_insert(0.0) += weight * score.probability; }
        }
        normalize(totals.into_iter().map(|(label, value)| (label, value / total_weight.max(f32::MIN_POSITIVE))).collect())
    }
}

pub(crate) fn normalize(raw: Vec<(String, f32)>) -> Result<Vec<ClassScore>> {
    let sum: f32 = raw.iter().map(|(_, value)| *value).sum();
    if !sum.is_finite() || sum <= 0.0 { return Err(SdkError::InvalidArgument("normalization failed".into())); }
    let mut out = Vec::with_capacity(raw.len());
    for (label, value) in raw { out.push(ClassScore::new(label, value / sum)?); }
    out.sort_by(|a, b| b.probability.total_cmp(&a.probability));
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct TemporalSelfAttention {
    temperature: f32,
}

impl TemporalSelfAttention {
    pub fn new(temperature: f32) -> Result<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SdkError::InvalidArgument("attention temperature must be finite and positive".into()));
        }
        Ok(Self { temperature })
    }

    pub fn aggregate(&self, history: &[Vec<ClassScore>]) -> Result<Vec<ClassScore>> {
        if history.is_empty() { return Err(SdkError::EmptyDataset); }
        let labels: Vec<String> = history[0].iter().map(|s| s.label.clone()).collect();
        if labels.is_empty() { return Err(SdkError::EmptyDataset); }
        let mut vectors = Vec::with_capacity(history.len());
        for scores in history {
            let mut vector = Vec::with_capacity(labels.len());
            for label in &labels {
                let p = scores.iter().find(|s| &s.label == label).map(|s| s.probability).unwrap_or(0.0);
                vector.push(p);
            }
            vectors.push(vector);
        }
        let query = vectors.last().ok_or(SdkError::EmptyDataset)?;
        let logits: Vec<f32> = vectors.iter().map(|key| dot(query, key) / self.temperature).collect();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|v| (*v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if !sum.is_finite() || sum <= 0.0 { return Err(SdkError::InvalidArgument("attention normalization failed".into())); }
        let weights: Vec<f32> = exps.into_iter().map(|v| v / sum).collect();
        let mut pooled = vec![0.0_f32; labels.len()];
        for (weight, vector) in weights.iter().zip(&vectors) {
            for (dst, value) in pooled.iter_mut().zip(vector) { *dst += *weight * *value; }
        }
        normalize(labels.into_iter().zip(pooled).collect())
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
