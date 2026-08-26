use serde::{Deserialize, Serialize};

use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAdapter {
    source_mean: Vec<f32>,
    source_std: Vec<f32>,
    target_mean: Vec<f32>,
    target_std: Vec<f32>,
    epsilon: f32,
}

impl DomainAdapter {
    pub fn fit(source: &[Embedding], target: &[Embedding]) -> Result<Self> {
        if source.is_empty() || target.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let dim = source[0].dim();
        if source.iter().chain(target).any(|item| item.dim() != dim) {
            return Err(SdkError::InvalidArgument(
                "domain embeddings must share a dimension".into(),
            ));
        }
        let (source_mean, source_std) = moments(source, dim);
        let (target_mean, target_std) = moments(target, dim);
        Ok(Self {
            source_mean,
            source_std,
            target_mean,
            target_std,
            epsilon: 1e-6,
        })
    }

    pub fn transform(&self, embedding: &Embedding) -> Result<Embedding> {
        if embedding.dim() != self.source_mean.len() {
            return Err(SdkError::DimensionMismatch {
                expected: self.source_mean.len(),
                actual: embedding.dim(),
            });
        }
        Embedding::new(
            embedding
                .values()
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    let z = (*value - self.source_mean[i]) / (self.source_std[i] + self.epsilon);
                    z * self.target_std[i] + self.target_mean[i]
                })
                .collect(),
        )
    }
}

fn moments(samples: &[Embedding], dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0_f32; dim];
    for sample in samples {
        for (dst, src) in mean.iter_mut().zip(sample.values()) {
            *dst += *src;
        }
    }
    let n = samples.len() as f32;
    for value in &mut mean {
        *value /= n;
    }
    let mut std = vec![0.0_f32; dim];
    for sample in samples {
        for ((dst, src), avg) in std.iter_mut().zip(sample.values()).zip(&mean) {
            let d = *src - *avg;
            *dst += d * d;
        }
    }
    for value in &mut std {
        *value = (*value / n).sqrt();
    }
    (mean, std)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSensorAdapter {
    adapters: std::collections::BTreeMap<String, DomainAdapter>,
}

impl MultiSensorAdapter {
    pub fn fit(
        source_domains: &std::collections::BTreeMap<String, Vec<Embedding>>,
        target_domain: &[Embedding],
    ) -> Result<Self> {
        if source_domains.is_empty() || target_domain.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let mut adapters = std::collections::BTreeMap::new();
        for (sensor, samples) in source_domains {
            if sensor.trim().is_empty() {
                return Err(SdkError::InvalidArgument(
                    "sensor id must be non-empty".into(),
                ));
            }
            adapters.insert(sensor.clone(), DomainAdapter::fit(samples, target_domain)?);
        }
        Ok(Self { adapters })
    }

    pub fn transform(&self, sensor_id: &str, embedding: &Embedding) -> Result<Embedding> {
        self.adapters
            .get(sensor_id)
            .ok_or_else(|| {
                SdkError::InvalidArgument(format!("no domain adapter for sensor '{sensor_id}'"))
            })?
            .transform(embedding)
    }
}
