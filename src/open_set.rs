use serde::{Deserialize, Serialize};

use crate::classification::squared_distance;
use crate::{Embedding, OpenSetModel, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearestPrototypeOod {
    prototypes: Vec<Embedding>,
    max_distance: f32,
}

impl NearestPrototypeOod {
    pub fn new(prototypes: Vec<Embedding>, max_distance: f32) -> Result<Self> {
        if prototypes.is_empty() { return Err(SdkError::EmptyDataset); }
        if !max_distance.is_finite() || max_distance < 0.0 {
            return Err(SdkError::InvalidArgument("max_distance must be finite and non-negative".into()));
        }
        let dim = prototypes[0].dim();
        if prototypes.iter().any(|p| p.dim() != dim) {
            return Err(SdkError::InvalidArgument("OOD prototypes must share a dimension".into()));
        }
        Ok(Self { prototypes, max_distance })
    }

    pub fn validate(&self) -> Result<()> {
        if self.prototypes.is_empty() || !self.max_distance.is_finite() || self.max_distance < 0.0 { return Err(SdkError::InvalidArgument("invalid OOD payload".into())); }
        let dim = self.prototypes[0].dim();
        if dim == 0 || self.prototypes.iter().any(|p| p.dim() != dim) { return Err(SdkError::InvalidArgument("invalid OOD prototype dimensions".into())); }
        Ok(())
    }

    pub fn nearest_distance(&self, embedding: &Embedding) -> Result<f32> {
        if embedding.dim() != self.prototypes[0].dim() {
            return Err(SdkError::DimensionMismatch { expected: self.prototypes[0].dim(), actual: embedding.dim() });
        }
        let mut best = f32::INFINITY;
        for prototype in &self.prototypes {
            best = best.min(squared_distance(embedding.values(), prototype.values())?.sqrt());
        }
        Ok(best)
    }
}

impl OpenSetModel for NearestPrototypeOod {
    fn is_unknown(&self, embedding: &Embedding) -> Result<bool> { Ok(self.nearest_distance(embedding)? > self.max_distance) }
    fn input_dim(&self) -> usize { self.prototypes[0].dim() }
}
