use serde::{Deserialize, Serialize};
use crate::{Embedding, Position3, Result, SdkError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReacquisitionEnvelope {
    pub track_id: u64,
    pub expires_ms: u64,
    pub predicted_position: Position3,
    pub sigma_m: f64,
    pub embedding: Embedding,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReacquisitionScore {
    pub kinematic: f32,
    pub embedding: f32,
    pub temporal: f32,
    pub total: f32,
}

impl ReacquisitionEnvelope {
    pub fn new(track_id: u64, expires_ms: u64, predicted_position: Position3, sigma_m: f64, embedding: Embedding) -> Result<Self> {
        if expires_ms == 0 || !sigma_m.is_finite() || sigma_m <= 0.0 { return Err(SdkError::InvalidArgument("reacquisition envelope requires positive expiry and sigma".into())); }
        Ok(Self { track_id, expires_ms, predicted_position, sigma_m, embedding })
    }

    pub fn score(&self, timestamp_ms: u64, position: Position3, embedding: &Embedding) -> Result<ReacquisitionScore> {
        if timestamp_ms > self.expires_ms { return Ok(ReacquisitionScore { kinematic: 0.0, embedding: 0.0, temporal: 0.0, total: 0.0 }); }
        if embedding.dim() != self.embedding.dim() { return Err(SdkError::DimensionMismatch { expected: self.embedding.dim(), actual: embedding.dim() }); }
        let dx=position.x-self.predicted_position.x; let dy=position.y-self.predicted_position.y; let dz=position.z-self.predicted_position.z;
        let distance=(dx*dx+dy*dy+dz*dz).sqrt();
        let kinematic=(-0.5*(distance/self.sigma_m).powi(2)).exp().clamp(0.0,1.0) as f32;
        let embedding_score=cosine(self.embedding.values(), embedding.values());
        let remaining=(self.expires_ms-timestamp_ms) as f64/self.expires_ms as f64;
        let temporal=remaining.clamp(0.0,1.0) as f32;
        let total=(0.50*kinematic+0.35*embedding_score+0.15*temporal).clamp(0.0,1.0);
        Ok(ReacquisitionScore { kinematic, embedding: embedding_score, temporal, total })
    }
}

fn cosine(a:&[f32], b:&[f32])->f32 {
    let mut dot=0.0; let mut aa=0.0; let mut bb=0.0;
    for (x,y) in a.iter().zip(b) { dot+=x*y; aa+=x*x; bb+=y*y; }
    if aa<=f32::EPSILON || bb<=f32::EPSILON { return 0.0; }
    (dot/(aa.sqrt()*bb.sqrt())).clamp(0.0,1.0)
}
