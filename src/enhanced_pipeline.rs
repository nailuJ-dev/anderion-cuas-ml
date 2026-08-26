use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};

use crate::{
    CandidateKinematics, CooperativeCorrelation, CooperativeCorrelator, CooperativeDisposition,
    CooperativeTrack, DegarblingModel, DegarblingResult, Digest32, IdentityDegarbler, Observation,
    PerceptionPipeline, PerceptionResult, Result,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptionComponentResult {
    source_component_index: usize,
    perception: PerceptionResult,
    cooperative_correlations: Vec<CooperativeCorrelation>,
    cooperative_disposition: CooperativeDisposition,
}

impl PerceptionComponentResult {
    pub fn source_component_index(&self) -> usize { self.source_component_index }
    pub fn perception(&self) -> &PerceptionResult { &self.perception }
    pub fn cooperative_correlations(&self) -> &[CooperativeCorrelation] { &self.cooperative_correlations }
    pub fn cooperative_disposition(&self) -> CooperativeDisposition { self.cooperative_disposition }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnhancedPerceptionResult {
    source_observation_id: String,
    separation: DegarblingResult,
    components: Vec<PerceptionComponentResult>,
}

impl EnhancedPerceptionResult {
    pub fn source_observation_id(&self) -> &str { &self.source_observation_id }
    pub fn separation(&self) -> &DegarblingResult { &self.separation }
    pub fn components(&self) -> &[PerceptionComponentResult] { &self.components }

    pub fn evidence_digest(&self) -> Digest32 {
        let mut hasher = Sha256::new();
        update_bytes(&mut hasher, b"anderion-cuas-enhanced-evidence-v1");
        update_bytes(&mut hasher, self.source_observation_id.as_bytes());
        update_f32(&mut hasher, self.separation.reconstruction_error());
        update_f32(&mut hasher, self.separation.separation_confidence());
        hasher.update((self.separation.components().len() as u64).to_le_bytes());
        for component in self.separation.components() {
            update_bytes(&mut hasher, component.id().as_bytes());
            update_bytes(&mut hasher, component.sensor_id().as_bytes());
            hasher.update(component.timestamp_ms().to_le_bytes());
            update_f32_slice(&mut hasher, component.features());
        }
        hasher.update((self.components.len() as u64).to_le_bytes());
        for component in &self.components {
            hasher.update((component.source_component_index as u64).to_le_bytes());
            update_perception(&mut hasher, &component.perception);
            hasher.update([match component.cooperative_disposition {
                CooperativeDisposition::MatchedCooperative => 1,
                CooperativeDisposition::Unmatched => 0,
            }]);
            hasher.update((component.cooperative_correlations.len() as u64).to_le_bytes());
            for correlation in &component.cooperative_correlations {
                hasher.update([cooperative_kind_byte(correlation.kind())]);
                update_bytes(&mut hasher, correlation.identity().as_bytes());
                update_f64(&mut hasher, correlation.spatial_distance_m());
                hasher.update(correlation.time_delta_ms().to_le_bytes());
                match correlation.velocity_delta_mps() {
                    Some(value) => { hasher.update([1]); update_f64(&mut hasher, value); }
                    None => hasher.update([0]),
                }
                update_f32(&mut hasher, correlation.score());
            }
        }
        let finalized = hasher.finalize();
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&finalized);
        Digest32::from_digest_bytes(bytes)
    }
}

#[derive(Clone)]
pub struct EnhancedPerceptionPipeline {
    pipeline: PerceptionPipeline,
    degarbler: Option<Arc<dyn DegarblingModel>>,
    correlator: Option<CooperativeCorrelator>,
}

impl EnhancedPerceptionPipeline {
    pub fn new(pipeline: PerceptionPipeline) -> Self {
        Self { pipeline, degarbler: None, correlator: None }
    }

    pub fn with_degarbler(mut self, model: Arc<dyn DegarblingModel>) -> Self {
        self.degarbler = Some(model);
        self
    }

    pub fn with_correlator(mut self, correlator: CooperativeCorrelator) -> Self {
        self.correlator = Some(correlator);
        self
    }

    pub fn infer(
        &self,
        observation: &Observation,
        candidate: Option<&CandidateKinematics>,
        cooperative_tracks: &[CooperativeTrack],
    ) -> Result<EnhancedPerceptionResult> {
        let separation = match &self.degarbler {
            Some(model) => model.separate(observation)?,
            None => IdentityDegarbler.separate(observation)?,
        };
        let correlations = match (&self.correlator, candidate) {
            (Some(correlator), Some(kinematics)) => correlator.correlate(kinematics, cooperative_tracks)?,
            _ => Vec::new(),
        };
        let disposition = match self.correlator.as_ref() {
            Some(correlator) => correlator.disposition(&correlations),
            None => CooperativeDisposition::Unmatched,
        };
        let mut components = Vec::with_capacity(separation.components().len());
        for (index, component) in separation.components().iter().enumerate() {
            components.push(PerceptionComponentResult {
                source_component_index: index,
                perception: self.pipeline.infer(component)?,
                cooperative_correlations: correlations.clone(),
                cooperative_disposition: disposition,
            });
        }
        Ok(EnhancedPerceptionResult {
            source_observation_id: observation.id().to_string(),
            separation,
            components,
        })
    }
}

fn update_perception(hasher: &mut Sha256, result: &PerceptionResult) {
    update_bytes(hasher, result.observation_id.as_bytes());
    update_bytes(hasher, result.sensor_id.as_bytes());
    update_f32(hasher, result.detection.probability);
    hasher.update([u8::from(result.detection.detected)]);
    update_f32(hasher, result.detection.uncertainty);
    hasher.update((result.classification.len() as u64).to_le_bytes());
    for score in &result.classification {
        update_bytes(hasher, score.label.as_bytes());
        update_f32(hasher, score.probability);
    }
    hasher.update([u8::from(result.unknown)]);
    match &result.localization {
        Some(localization) => {
            hasher.update([1]);
            update_f64(hasher, localization.position.x);
            update_f64(hasher, localization.position.y);
            update_f64(hasher, localization.position.z);
            update_f64(hasher, localization.sigma_m);
            update_f32(hasher, localization.confidence);
        }
        None => hasher.update([0]),
    }
    update_f32_slice(hasher, result.embedding.values());
    update_f32(hasher, result.classification_uncertainty);
}

fn cooperative_kind_byte(kind: crate::CooperativeIdentityKind) -> u8 {
    match kind {
        crate::CooperativeIdentityKind::Ais => 1,
        crate::CooperativeIdentityKind::Adsb => 2,
        crate::CooperativeIdentityKind::RemoteId => 3,
        crate::CooperativeIdentityKind::Custom => 4,
    }
}

fn update_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn update_f32_slice(hasher: &mut Sha256, values: &[f32]) {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values { update_f32(hasher, *value); }
}

fn update_f32(hasher: &mut Sha256, value: f32) {
    let bits = if value == 0.0 { 0_u32 } else { value.to_bits() };
    hasher.update(bits.to_le_bytes());
}

fn update_f64(hasher: &mut Sha256, value: f64) {
    let bits = if value == 0.0 { 0_u64 } else { value.to_bits() };
    hasher.update(bits.to_le_bytes());
}
