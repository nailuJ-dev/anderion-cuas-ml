use serde::{Deserialize, Serialize};
use crate::{CandidateKinematics, CooperativeIdentityKind, CooperativeTrack, GeoPosition, Result, SdkError, VelocityNed};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CooperativeTrustVerdict { Consistent, WeaklyConsistent, Conflict, InsufficientEvidence }

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CooperativeTrustPolicy {
    pub position_scale_m: f64,
    pub velocity_scale_mps: f64,
    pub time_scale_ms: u64,
    pub consistent_threshold: f32,
    pub conflict_threshold: f32,
}

impl Default for CooperativeTrustPolicy {
    fn default() -> Self {
        Self { position_scale_m: 150.0, velocity_scale_mps: 12.0, time_scale_ms: 2_000, consistent_threshold: 0.75, conflict_threshold: 0.35 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CooperativeTrustAssessment {
    pub source: CooperativeIdentityKind,
    pub identity: String,
    pub identity_consistency: f32,
    pub position_consistency: f32,
    pub velocity_consistency: Option<f32>,
    pub temporal_consistency: f32,
    pub physical_signature_consistency: Option<f32>,
    pub aggregate_consistency: f32,
    pub verdict: CooperativeTrustVerdict,
}

pub fn assess_cooperative_trust(
    candidate: &CandidateKinematics,
    track: &CooperativeTrack,
    policy: &CooperativeTrustPolicy,
    physical_signature_consistency: Option<f32>,
) -> Result<CooperativeTrustAssessment> {
    if !policy.position_scale_m.is_finite() || policy.position_scale_m <= 0.0
        || !policy.velocity_scale_mps.is_finite() || policy.velocity_scale_mps <= 0.0
        || policy.time_scale_ms == 0
    { return Err(SdkError::InvalidArgument("cooperative trust policy scales must be positive".into())); }
    if let Some(value) = physical_signature_consistency {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) { return Err(SdkError::InvalidProbability(value)); }
    }
    let distance = geo_distance_m(candidate.position(), track.position());
    let position_consistency = (-distance / policy.position_scale_m).exp().clamp(0.0, 1.0) as f32;
    let dt = candidate.timestamp_ms().abs_diff(track.timestamp_ms());
    let temporal_consistency = (-(dt as f64) / policy.time_scale_ms as f64).exp().clamp(0.0, 1.0) as f32;
    let velocity_consistency = match (candidate.velocity(), track.velocity()) {
        (Some(a), Some(b)) => {
            let delta = velocity_distance(a, b);
            Some((-delta / policy.velocity_scale_mps).exp().clamp(0.0, 1.0) as f32)
        }
        _ => None,
    };
    let identity_consistency = track.source_confidence();
    let mut weighted = 0.30 * position_consistency + 0.20 * temporal_consistency + 0.20 * identity_consistency;
    let mut weight = 0.70_f32;
    if let Some(value) = velocity_consistency { weighted += 0.15 * value; weight += 0.15; }
    if let Some(value) = physical_signature_consistency { weighted += 0.15 * value; weight += 0.15; }
    let aggregate = (weighted / weight).clamp(0.0, 1.0);
    let evidence_count = 3 + usize::from(velocity_consistency.is_some()) + usize::from(physical_signature_consistency.is_some());
    let verdict = if evidence_count < 3 { CooperativeTrustVerdict::InsufficientEvidence }
        else if aggregate >= policy.consistent_threshold { CooperativeTrustVerdict::Consistent }
        else if aggregate <= policy.conflict_threshold { CooperativeTrustVerdict::Conflict }
        else { CooperativeTrustVerdict::WeaklyConsistent };
    Ok(CooperativeTrustAssessment {
        source: track.kind(), identity: track.identity().to_string(), identity_consistency,
        position_consistency, velocity_consistency, temporal_consistency,
        physical_signature_consistency, aggregate_consistency: aggregate, verdict,
    })
}

fn geo_distance_m(a: GeoPosition, b: GeoPosition) -> f64 {
    const R: f64 = 6_371_008.8;
    let lat1 = a.latitude_deg().to_radians(); let lat2 = b.latitude_deg().to_radians();
    let dlat = lat2 - lat1; let dlon = (b.longitude_deg() - a.longitude_deg()).to_radians();
    let hav = (dlat * 0.5).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon * 0.5).sin().powi(2);
    let horizontal = 2.0 * R * hav.sqrt().asin();
    horizontal.hypot(b.altitude_m() - a.altitude_m())
}

fn velocity_distance(a: VelocityNed, b: VelocityNed) -> f64 {
    (a.north_mps() - b.north_mps()).hypot(a.east_mps() - b.east_mps()).hypot(a.down_mps() - b.down_mps())
}
