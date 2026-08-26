use serde::{Deserialize, Serialize};

use crate::{Result, SdkError};

pub const MAX_FEATURES: usize = 65_536;
pub const MAX_EMBEDDING_DIM: usize = 8_192;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    id: String,
    sensor_id: String,
    timestamp_ms: u64,
    features: Vec<f32>,
}

impl Observation {
    pub fn new(
        id: impl Into<String>,
        sensor_id: impl Into<String>,
        timestamp_ms: u64,
        features: Vec<f32>,
    ) -> Result<Self> {
        validate_vector(&features, MAX_FEATURES)?;
        let id = id.into();
        let sensor_id = sensor_id.into();
        if id.trim().is_empty() || sensor_id.trim().is_empty() {
            return Err(SdkError::InvalidArgument("observation id and sensor_id must be non-empty".into()));
        }
        Ok(Self { id, sensor_id, timestamp_ms, features })
    }

    pub fn id(&self) -> &str { &self.id }
    pub fn sensor_id(&self) -> &str { &self.sensor_id }
    pub fn timestamp_ms(&self) -> u64 { self.timestamp_ms }
    pub fn features(&self) -> &[f32] { &self.features }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    values: Vec<f32>,
}

impl Embedding {
    pub fn new(values: Vec<f32>) -> Result<Self> {
        validate_vector(&values, MAX_EMBEDDING_DIM)?;
        Ok(Self { values })
    }
    pub fn values(&self) -> &[f32] { &self.values }
    pub fn dim(&self) -> usize { self.values.len() }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Position3 {
    pub fn new(x: f64, y: f64, z: f64) -> Result<Self> {
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(SdkError::InvalidArgument("position coordinates must be finite".into()));
        }
        Ok(Self { x, y, z })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassScore {
    pub label: String,
    pub probability: f32,
}

impl ClassScore {
    pub fn new(label: impl Into<String>, probability: f32) -> Result<Self> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(SdkError::InvalidProbability(probability));
        }
        let label = label.into();
        if label.trim().is_empty() { return Err(SdkError::InvalidArgument("label must be non-empty".into())); }
        Ok(Self { label, probability })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub probability: f32,
    pub detected: bool,
    pub uncertainty: f32,
}

impl Detection {
    pub fn new(probability: f32, threshold: f32) -> Result<Self> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(SdkError::InvalidProbability(probability));
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(SdkError::InvalidProbability(threshold));
        }
        Ok(Self { probability, detected: probability >= threshold, uncertainty: 1.0 - (2.0 * (probability - 0.5).abs()).clamp(0.0, 1.0) })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Localization {
    pub position: Position3,
    pub sigma_m: f64,
    pub confidence: f32,
}

impl Localization {
    pub fn new(position: Position3, sigma_m: f64, confidence: f32) -> Result<Self> {
        if !sigma_m.is_finite() || sigma_m < 0.0 {
            return Err(SdkError::InvalidArgument("sigma_m must be finite and non-negative".into()));
        }
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(SdkError::InvalidProbability(confidence));
        }
        Ok(Self { position, sigma_m, confidence })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub id: u64,
    pub first_seen_ms: u64,
    pub last_seen_ms: u64,
    pub observations: u64,
    pub embedding: Embedding,
    pub position: Option<Position3>,
}

pub(crate) fn validate_vector(values: &[f32], max: usize) -> Result<()> {
    if values.is_empty() { return Err(SdkError::EmptyFeatures); }
    if values.len() > max { return Err(SdkError::DimensionLimit { actual: values.len(), max }); }
    if let Some((index, _)) = values.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(SdkError::NonFiniteValue { index });
    }
    Ok(())
}
