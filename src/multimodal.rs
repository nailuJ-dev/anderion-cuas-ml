use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, Copy)]
pub struct MultimodalSelfAttention {
    temperature: f32,
}

impl MultimodalSelfAttention {
    pub fn new(temperature: f32) -> Result<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SdkError::InvalidArgument("attention temperature must be finite and positive".into()));
        }
        Ok(Self { temperature })
    }

    pub fn aggregate(&self, modalities: &[(String, Embedding)]) -> Result<Embedding> {
        if modalities.is_empty() { return Err(SdkError::EmptyDataset); }
        if modalities.len() > 1_024 { return Err(SdkError::DimensionLimit { actual: modalities.len(), max: 1_024 }); }
        let dim = modalities[0].1.dim();
        if modalities.iter().any(|(sensor, embedding)| sensor.trim().is_empty() || embedding.dim() != dim) {
            return Err(SdkError::InvalidArgument("modalities require non-empty sensor ids and matching dimensions".into()));
        }
        let mut query = vec![0.0_f32; dim];
        for (_, embedding) in modalities {
            for (dst, value) in query.iter_mut().zip(embedding.values()) { *dst += *value; }
        }
        for value in &mut query { *value /= modalities.len() as f32; }
        let normalization = (dim as f32).sqrt().max(1.0) * self.temperature;
        let logits: Vec<f32> = modalities.iter().map(|(_, embedding)| dot(&query, embedding.values()) / normalization).collect();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|value| (*value - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        if !sum.is_finite() || sum <= 0.0 { return Err(SdkError::InvalidArgument("multimodal attention normalization failed".into())); }
        let mut pooled = vec![0.0_f32; dim];
        for ((_, embedding), weight) in modalities.iter().zip(exp.iter().map(|value| *value / sum)) {
            for (dst, value) in pooled.iter_mut().zip(embedding.values()) { *dst += weight * *value; }
        }
        Embedding::new(pooled)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

#[derive(Debug, Clone, Copy)]
pub struct MultimodalTransformerEncoder {
    attention: MultimodalSelfAttention,
    residual_weight: f32,
}

impl MultimodalTransformerEncoder {
    pub fn new(temperature: f32, residual_weight: f32) -> Result<Self> {
        if !residual_weight.is_finite() || !(0.0..=1.0).contains(&residual_weight) {
            return Err(SdkError::InvalidProbability(residual_weight));
        }
        Ok(Self { attention: MultimodalSelfAttention::new(temperature)?, residual_weight })
    }

    pub fn aggregate(&self, modalities: &[(String, Embedding)]) -> Result<Embedding> {
        let attended = self.attention.aggregate(modalities)?;
        let mut mean = vec![0.0_f32; attended.dim()];
        for (_, embedding) in modalities {
            for (dst, value) in mean.iter_mut().zip(embedding.values()) { *dst += *value; }
        }
        for value in &mut mean { *value /= modalities.len() as f32; }
        let mixed: Vec<f32> = attended.values().iter().zip(&mean).map(|(attention, residual)| {
            ((1.0 - self.residual_weight) * *attention + self.residual_weight * *residual).tanh()
        }).collect();
        Embedding::new(layer_normalize(mixed))
    }
}

fn layer_normalize(mut values: Vec<f32>) -> Vec<f32> {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|value| { let delta = *value - mean; delta * delta }).sum::<f32>() / values.len() as f32;
    let denom = (variance + 1.0e-5).sqrt();
    for value in &mut values { *value = (*value - mean) / denom; }
    values
}
