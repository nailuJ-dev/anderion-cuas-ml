use std::collections::BTreeMap;

use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone)]
pub struct PairedModalAligner {
    dimension: usize,
    global_mean: Vec<f32>,
    sensor_means: BTreeMap<String, Vec<f32>>,
}

impl PairedModalAligner {
    pub fn fit(samples: &[(String, Embedding)]) -> Result<Self> {
        if samples.is_empty() { return Err(SdkError::EmptyDataset); }
        let dimension = samples[0].1.dim();
        let mut global_mean = vec![0.0_f32; dimension];
        let mut sensor_sums: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let mut sensor_counts: BTreeMap<String, u64> = BTreeMap::new();
        for (sensor, embedding) in samples {
            if sensor.trim().is_empty() { return Err(SdkError::InvalidArgument("sensor id must be non-empty".into())); }
            if embedding.dim() != dimension { return Err(SdkError::DimensionMismatch { expected: dimension, actual: embedding.dim() }); }
            for (dst, value) in global_mean.iter_mut().zip(embedding.values()) { *dst += *value; }
            let sum = sensor_sums.entry(sensor.clone()).or_insert_with(|| vec![0.0; dimension]);
            for (dst, value) in sum.iter_mut().zip(embedding.values()) { *dst += *value; }
            *sensor_counts.entry(sensor.clone()).or_insert(0) += 1;
        }
        for value in &mut global_mean { *value /= samples.len() as f32; }
        let mut sensor_means = BTreeMap::new();
        for (sensor, mut sum) in sensor_sums {
            let count = sensor_counts.get(&sensor).copied().unwrap_or(0);
            if count == 0 { return Err(SdkError::InvalidArgument("sensor count cannot be zero".into())); }
            for value in &mut sum { *value /= count as f32; }
            sensor_means.insert(sensor, sum);
        }
        Ok(Self { dimension, global_mean, sensor_means })
    }

    pub fn align(&self, sensor: &str, embedding: &Embedding) -> Result<Embedding> {
        if embedding.dim() != self.dimension {
            return Err(SdkError::DimensionMismatch { expected: self.dimension, actual: embedding.dim() });
        }
        let mean = self.sensor_means.get(sensor).ok_or_else(|| SdkError::InvalidArgument("unknown sensor id for aligner".into()))?;
        Embedding::new(embedding.values().iter().zip(mean).zip(&self.global_mean).map(|((value, sensor_mean), global_mean)| value - sensor_mean + global_mean).collect())
    }
}
