use crate::{Result, SdkError};

#[derive(Debug, Clone)]
pub struct FederatedDelta {
    pub client_id: String,
    pub values: Vec<f32>,
    pub sample_count: u64,
}

impl FederatedDelta {
    pub fn new(client_id: impl Into<String>, values: Vec<f32>, sample_count: u64) -> Result<Self> {
        let client_id = client_id.into();
        if client_id.trim().is_empty() { return Err(SdkError::InvalidArgument("client id must be non-empty".into())); }
        if values.is_empty() || values.len() > 1_048_576 { return Err(SdkError::DimensionLimit { actual: values.len(), max: 1_048_576 }); }
        if values.iter().any(|value| !value.is_finite()) { return Err(SdkError::InvalidArgument("federated delta must be finite".into())); }
        if sample_count == 0 { return Err(SdkError::InvalidArgument("sample_count must be positive".into())); }
        Ok(Self { client_id, values, sample_count })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FederatedAverager {
    max_clients: usize,
    max_dimension: usize,
    clip_l2: f32,
}

impl FederatedAverager {
    pub fn new(max_clients: usize, max_dimension: usize, clip_l2: f32) -> Result<Self> {
        if max_clients == 0 || max_clients > 65_536 || max_dimension == 0 || max_dimension > 1_048_576 {
            return Err(SdkError::InvalidArgument("invalid federated aggregation bounds".into()));
        }
        if !clip_l2.is_finite() || clip_l2 <= 0.0 { return Err(SdkError::InvalidArgument("clip_l2 must be finite and positive".into())); }
        Ok(Self { max_clients, max_dimension, clip_l2 })
    }

    pub fn aggregate(&self, deltas: &[FederatedDelta]) -> Result<Vec<f32>> {
        if deltas.is_empty() { return Err(SdkError::EmptyDataset); }
        if deltas.len() > self.max_clients { return Err(SdkError::DimensionLimit { actual: deltas.len(), max: self.max_clients }); }
        let dimension = deltas[0].values.len();
        if dimension > self.max_dimension { return Err(SdkError::DimensionLimit { actual: dimension, max: self.max_dimension }); }
        let elements = deltas.len().checked_mul(dimension).ok_or_else(|| SdkError::InvalidArgument("federated element count overflow".into()))?;
        if elements > 16_777_216 {
            return Err(SdkError::DimensionLimit { actual: elements, max: 16_777_216 });
        }
        if deltas.iter().any(|delta| delta.values.len() != dimension) { return Err(SdkError::InvalidArgument("federated delta dimensions must match".into())); }
        let mut aggregate = vec![0.0_f64; dimension];
        let mut total_samples = 0_u64;
        for delta in deltas {
            let norm = delta.values.iter().map(|value| value * value).sum::<f32>().sqrt();
            let scale = if norm > self.clip_l2 { self.clip_l2 / norm } else { 1.0 };
            total_samples = total_samples.checked_add(delta.sample_count).ok_or_else(|| SdkError::InvalidArgument("sample count overflow".into()))?;
            for (dst, value) in aggregate.iter_mut().zip(&delta.values) { *dst += f64::from(*value * scale) * delta.sample_count as f64; }
        }
        if total_samples == 0 { return Err(SdkError::InvalidArgument("total sample count must be positive".into())); }
        Ok(aggregate.into_iter().map(|value| (value / total_samples as f64) as f32).collect())
    }
}
