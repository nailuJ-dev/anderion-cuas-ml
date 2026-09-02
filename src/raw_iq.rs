use serde::{Deserialize, Serialize};

use crate::{Result, SdkError};

pub const MAX_RAW_IQ_SAMPLES: usize = 4_194_304;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComplexSample {
    i: f32,
    q: f32,
}

impl ComplexSample {
    pub fn new(i: f32, q: f32) -> Result<Self> {
        if !i.is_finite() || !q.is_finite() {
            return Err(SdkError::InvalidArgument(
                "raw I/Q samples must contain finite values".into(),
            ));
        }
        Ok(Self { i, q })
    }

    pub const fn from_parts_unchecked(i: f32, q: f32) -> Self {
        Self { i, q }
    }

    pub fn i(self) -> f32 {
        self.i
    }

    pub fn q(self) -> f32 {
        self.q
    }

    pub fn norm_sqr(self) -> f32 {
        self.i.mul_add(self.i, self.q * self.q)
    }

    pub fn magnitude(self) -> f32 {
        self.norm_sqr().sqrt()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawIqCapture {
    id: String,
    timestamp_ms: u64,
    sample_rate_hz: f64,
    center_frequency_hz: f64,
    samples: Vec<ComplexSample>,
}

impl RawIqCapture {
    pub fn new(
        id: impl Into<String>,
        timestamp_ms: u64,
        sample_rate_hz: f64,
        center_frequency_hz: f64,
        samples: Vec<ComplexSample>,
    ) -> Result<Self> {
        let value = Self {
            id: id.into(),
            timestamp_ms,
            sample_rate_hz,
            center_frequency_hz,
            samples,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() || self.id.len() > 4_096 {
            return Err(SdkError::InvalidArgument(
                "raw I/Q capture id must be non-empty and bounded".into(),
            ));
        }
        if !self.sample_rate_hz.is_finite() || self.sample_rate_hz <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "sample_rate_hz must be finite and positive".into(),
            ));
        }
        if !self.center_frequency_hz.is_finite() || self.center_frequency_hz < 0.0 {
            return Err(SdkError::InvalidArgument(
                "center_frequency_hz must be finite and non-negative".into(),
            ));
        }
        if self.samples.is_empty() {
            return Err(SdkError::EmptyFeatures);
        }
        if self.samples.len() > MAX_RAW_IQ_SAMPLES {
            return Err(SdkError::DimensionLimit {
                actual: self.samples.len(),
                max: MAX_RAW_IQ_SAMPLES,
            });
        }
        if let Some((index, _)) = self
            .samples
            .iter()
            .enumerate()
            .find(|(_, sample)| !sample.i.is_finite() || !sample.q.is_finite())
        {
            return Err(SdkError::NonFiniteValue { index });
        }
        Ok(())
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }

    pub fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }

    pub fn center_frequency_hz(&self) -> f64 {
        self.center_frequency_hz
    }

    pub fn samples(&self) -> &[ComplexSample] {
        &self.samples
    }

    pub fn sample_period_s(&self) -> f64 {
        1.0 / self.sample_rate_hz
    }

    pub fn duration_s(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate_hz
    }

    pub fn mean_power(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples
            .iter()
            .map(|sample| f64::from(sample.norm_sqr()))
            .sum::<f64>()
            / self.samples.len() as f64
    }

    pub fn dc_removed(&self) -> Result<Self> {
        self.validate()?;
        let n = self.samples.len() as f64;
        let mean_i = self
            .samples
            .iter()
            .map(|sample| f64::from(sample.i))
            .sum::<f64>()
            / n;
        let mean_q = self
            .samples
            .iter()
            .map(|sample| f64::from(sample.q))
            .sum::<f64>()
            / n;
        let samples = self
            .samples
            .iter()
            .map(|sample| {
                ComplexSample::from_parts_unchecked(
                    (f64::from(sample.i) - mean_i) as f32,
                    (f64::from(sample.q) - mean_q) as f32,
                )
            })
            .collect();
        Self::new(
            self.id.clone(),
            self.timestamp_ms,
            self.sample_rate_hz,
            self.center_frequency_hz,
            samples,
        )
    }

    pub fn with_samples(&self, samples: Vec<ComplexSample>) -> Result<Self> {
        Self::new(
            self.id.clone(),
            self.timestamp_ms,
            self.sample_rate_hz,
            self.center_frequency_hz,
            samples,
        )
    }
}

pub(crate) fn samples_for_duration(sample_rate_hz: f64, duration_s: f64) -> usize {
    (sample_rate_hz * duration_s).round().max(1.0) as usize
}
