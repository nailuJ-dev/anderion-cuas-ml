use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::{
    ConceptKind, OntologyGraph, OntologyNode, OntologyRelation, RelationKind, Result, SdkError,
};

const EARTH_RADIUS_M: f64 = 6_371_008.8;
const MAX_TRACKS: usize = 65_536;
const MAX_ID_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CooperativeIdentityKind {
    Ais,
    Adsb,
    RemoteId,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoPosition {
    latitude_deg: f64,
    longitude_deg: f64,
    altitude_m: f64,
}

impl GeoPosition {
    pub fn new(latitude_deg: f64, longitude_deg: f64, altitude_m: f64) -> Result<Self> {
        if !latitude_deg.is_finite() || !(-90.0..=90.0).contains(&latitude_deg) {
            return Err(SdkError::InvalidArgument(
                "latitude_deg must be finite and in -90..=90".into(),
            ));
        }
        if !longitude_deg.is_finite() || !(-180.0..=180.0).contains(&longitude_deg) {
            return Err(SdkError::InvalidArgument(
                "longitude_deg must be finite and in -180..=180".into(),
            ));
        }
        if !altitude_m.is_finite() {
            return Err(SdkError::InvalidArgument(
                "altitude_m must be finite".into(),
            ));
        }
        Ok(Self {
            latitude_deg,
            longitude_deg,
            altitude_m,
        })
    }

    pub fn latitude_deg(&self) -> f64 {
        self.latitude_deg
    }
    pub fn longitude_deg(&self) -> f64 {
        self.longitude_deg
    }
    pub fn altitude_m(&self) -> f64 {
        self.altitude_m
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VelocityNed {
    north_mps: f64,
    east_mps: f64,
    down_mps: f64,
}

impl VelocityNed {
    pub fn new(north_mps: f64, east_mps: f64, down_mps: f64) -> Result<Self> {
        if !north_mps.is_finite() || !east_mps.is_finite() || !down_mps.is_finite() {
            return Err(SdkError::InvalidArgument(
                "velocity components must be finite".into(),
            ));
        }
        Ok(Self {
            north_mps,
            east_mps,
            down_mps,
        })
    }

    pub fn north_mps(&self) -> f64 {
        self.north_mps
    }
    pub fn east_mps(&self) -> f64 {
        self.east_mps
    }
    pub fn down_mps(&self) -> f64 {
        self.down_mps
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateKinematics {
    timestamp_ms: u64,
    position: GeoPosition,
    velocity: Option<VelocityNed>,
}

impl CandidateKinematics {
    pub fn new(
        timestamp_ms: u64,
        position: GeoPosition,
        velocity: Option<VelocityNed>,
    ) -> Result<Self> {
        validate_geo(position)?;
        if let Some(value) = velocity {
            validate_velocity(value)?;
        }
        Ok(Self {
            timestamp_ms,
            position,
            velocity,
        })
    }

    pub fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
    pub fn position(&self) -> GeoPosition {
        self.position
    }
    pub fn velocity(&self) -> Option<VelocityNed> {
        self.velocity
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CooperativeTrack {
    kind: CooperativeIdentityKind,
    identity: String,
    timestamp_ms: u64,
    position: GeoPosition,
    velocity: Option<VelocityNed>,
    source_confidence: f32,
}

impl CooperativeTrack {
    pub fn new(
        kind: CooperativeIdentityKind,
        identity: impl Into<String>,
        timestamp_ms: u64,
        position: GeoPosition,
        velocity: Option<VelocityNed>,
        source_confidence: f32,
    ) -> Result<Self> {
        let identity = identity.into();
        validate_identity(&identity)?;
        validate_geo(position)?;
        if let Some(value) = velocity {
            validate_velocity(value)?;
        }
        validate_probability(source_confidence)?;
        Ok(Self {
            kind,
            identity,
            timestamp_ms,
            position,
            velocity,
            source_confidence,
        })
    }

    pub fn kind(&self) -> CooperativeIdentityKind {
        self.kind
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
    pub fn position(&self) -> GeoPosition {
        self.position
    }
    pub fn velocity(&self) -> Option<VelocityNed> {
        self.velocity
    }
    pub fn source_confidence(&self) -> f32 {
        self.source_confidence
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrelationPolicy {
    max_spatial_distance_m: f64,
    max_time_delta_ms: u64,
    max_velocity_delta_mps: f64,
    min_score: f32,
}

impl CorrelationPolicy {
    pub fn new(
        max_spatial_distance_m: f64,
        max_time_delta_ms: u64,
        max_velocity_delta_mps: f64,
        min_score: f32,
    ) -> Result<Self> {
        if !max_spatial_distance_m.is_finite() || max_spatial_distance_m <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "max_spatial_distance_m must be finite and positive".into(),
            ));
        }
        if max_time_delta_ms == 0 {
            return Err(SdkError::InvalidArgument(
                "max_time_delta_ms must be positive".into(),
            ));
        }
        if !max_velocity_delta_mps.is_finite() || max_velocity_delta_mps <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "max_velocity_delta_mps must be finite and positive".into(),
            ));
        }
        validate_probability(min_score)?;
        Ok(Self {
            max_spatial_distance_m,
            max_time_delta_ms,
            max_velocity_delta_mps,
            min_score,
        })
    }

    pub fn min_score(&self) -> f32 {
        self.min_score
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CooperativeCorrelation {
    kind: CooperativeIdentityKind,
    identity: String,
    spatial_distance_m: f64,
    time_delta_ms: u64,
    velocity_delta_mps: Option<f64>,
    score: f32,
}

impl CooperativeCorrelation {
    pub fn kind(&self) -> CooperativeIdentityKind {
        self.kind
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn spatial_distance_m(&self) -> f64 {
        self.spatial_distance_m
    }
    pub fn time_delta_ms(&self) -> u64 {
        self.time_delta_ms
    }
    pub fn velocity_delta_mps(&self) -> Option<f64> {
        self.velocity_delta_mps
    }
    pub fn score(&self) -> f32 {
        self.score
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CooperativeDisposition {
    MatchedCooperative,
    Unmatched,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CooperativeCorrelator {
    policy: CorrelationPolicy,
}

impl CooperativeCorrelator {
    pub fn new(policy: CorrelationPolicy) -> Result<Self> {
        validate_policy(&policy)?;
        Ok(Self { policy })
    }

    pub fn policy(&self) -> &CorrelationPolicy {
        &self.policy
    }

    pub fn correlate(
        &self,
        candidate: &CandidateKinematics,
        tracks: &[CooperativeTrack],
    ) -> Result<Vec<CooperativeCorrelation>> {
        validate_candidate(candidate)?;
        validate_policy(&self.policy)?;
        if tracks.len() > MAX_TRACKS {
            return Err(SdkError::DimensionLimit {
                actual: tracks.len(),
                max: MAX_TRACKS,
            });
        }
        let mut correlations = Vec::new();
        for track in tracks {
            validate_track(track)?;
            let time_delta_ms = candidate.timestamp_ms.abs_diff(track.timestamp_ms);
            if time_delta_ms > self.policy.max_time_delta_ms {
                continue;
            }
            let horizontal_distance_m = haversine_distance_m(candidate.position, track.position);
            let altitude_delta_m = candidate.position.altitude_m - track.position.altitude_m;
            let spatial_distance_m = (horizontal_distance_m * horizontal_distance_m
                + altitude_delta_m * altitude_delta_m)
                .sqrt();
            if spatial_distance_m > self.policy.max_spatial_distance_m {
                continue;
            }
            let velocity_delta_mps = match (candidate.velocity, track.velocity) {
                (Some(left), Some(right)) => Some(velocity_distance(left, right)),
                _ => None,
            };
            if velocity_delta_mps.is_some_and(|value| value > self.policy.max_velocity_delta_mps) {
                continue;
            }

            let spatial_score =
                1.0 - (spatial_distance_m / self.policy.max_spatial_distance_m).clamp(0.0, 1.0);
            let temporal_score = 1.0
                - ((time_delta_ms as f64) / (self.policy.max_time_delta_ms as f64)).clamp(0.0, 1.0);
            let (velocity_score, velocity_weight) = match velocity_delta_mps {
                Some(delta) => (
                    1.0 - (delta / self.policy.max_velocity_delta_mps).clamp(0.0, 1.0),
                    0.2,
                ),
                None => (0.0, 0.0),
            };
            let weighted =
                0.5 * spatial_score + 0.3 * temporal_score + velocity_weight * velocity_score;
            let total_weight = 0.8 + velocity_weight;
            let score = ((weighted / total_weight) * f64::from(track.source_confidence))
                .clamp(0.0, 1.0) as f32;
            if score >= self.policy.min_score {
                correlations.push(CooperativeCorrelation {
                    kind: track.kind,
                    identity: track.identity.clone(),
                    spatial_distance_m,
                    time_delta_ms,
                    velocity_delta_mps,
                    score,
                });
            }
        }
        correlations.sort_by(compare_correlations);
        Ok(correlations)
    }

    pub fn disposition(&self, correlations: &[CooperativeCorrelation]) -> CooperativeDisposition {
        if correlations
            .first()
            .is_some_and(|value| value.score >= self.policy.min_score)
        {
            CooperativeDisposition::MatchedCooperative
        } else {
            CooperativeDisposition::Unmatched
        }
    }
}

pub fn augment_graph_with_cooperative_correlations(
    graph: &mut OntologyGraph,
    correlations: &[CooperativeCorrelation],
) -> Result<()> {
    if correlations.len() > 1_024 {
        return Err(SdkError::DimensionLimit {
            actual: correlations.len(),
            max: 1_024,
        });
    }
    if !graph.nodes().contains_key("candidate:primary") {
        return Err(SdkError::InvalidArgument(
            "cooperative ontology augmentation requires candidate:primary".into(),
        ));
    }
    for (index, correlation) in correlations.iter().enumerate() {
        let id = format!("cooperative:{index}");
        let label = format!(
            "{:?}:{};score={:.6}",
            correlation.kind, correlation.identity, correlation.score
        );
        graph.add_node(OntologyNode::new(
            id.clone(),
            ConceptKind::CooperativeIdentity,
            Some(label),
        )?)?;
        graph.add_relation(OntologyRelation::new(
            "candidate:primary",
            RelationKind::CorrelatesWith,
            id,
        )?)?;
    }
    Ok(())
}

fn compare_correlations(left: &CooperativeCorrelation, right: &CooperativeCorrelation) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.kind.cmp(&right.kind))
        .then_with(|| left.identity.cmp(&right.identity))
        .then_with(|| left.spatial_distance_m.total_cmp(&right.spatial_distance_m))
        .then_with(|| left.time_delta_ms.cmp(&right.time_delta_ms))
}

fn validate_track(track: &CooperativeTrack) -> Result<()> {
    validate_identity(&track.identity)?;
    validate_geo(track.position)?;
    if let Some(value) = track.velocity {
        validate_velocity(value)?;
    }
    validate_probability(track.source_confidence)
}

fn validate_candidate(candidate: &CandidateKinematics) -> Result<()> {
    validate_geo(candidate.position)?;
    if let Some(value) = candidate.velocity {
        validate_velocity(value)?;
    }
    Ok(())
}

fn validate_policy(policy: &CorrelationPolicy) -> Result<()> {
    CorrelationPolicy::new(
        policy.max_spatial_distance_m,
        policy.max_time_delta_ms,
        policy.max_velocity_delta_mps,
        policy.min_score,
    )
    .map(|_| ())
}

fn validate_identity(identity: &str) -> Result<()> {
    if identity.trim().is_empty() {
        return Err(SdkError::InvalidArgument(
            "cooperative identity must be non-empty".into(),
        ));
    }
    if identity.len() > MAX_ID_BYTES {
        return Err(SdkError::DimensionLimit {
            actual: identity.len(),
            max: MAX_ID_BYTES,
        });
    }
    Ok(())
}

fn validate_geo(position: GeoPosition) -> Result<()> {
    GeoPosition::new(
        position.latitude_deg,
        position.longitude_deg,
        position.altitude_m,
    )
    .map(|_| ())
}

fn validate_velocity(velocity: VelocityNed) -> Result<()> {
    VelocityNed::new(velocity.north_mps, velocity.east_mps, velocity.down_mps).map(|_| ())
}

fn validate_probability(value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(SdkError::InvalidProbability(value));
    }
    Ok(())
}

fn haversine_distance_m(left: GeoPosition, right: GeoPosition) -> f64 {
    let left_lat = left.latitude_deg.to_radians();
    let right_lat = right.latitude_deg.to_radians();
    let delta_lat = (right.latitude_deg - left.latitude_deg).to_radians();
    let delta_lon = (right.longitude_deg - left.longitude_deg).to_radians();
    let a = (delta_lat / 2.0).sin().powi(2)
        + left_lat.cos() * right_lat.cos() * (delta_lon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * a.sqrt().atan2((1.0 - a).max(0.0).sqrt())
}

fn velocity_distance(left: VelocityNed, right: VelocityNed) -> f64 {
    let north = left.north_mps - right.north_mps;
    let east = left.east_mps - right.east_mps;
    let down = left.down_mps - right.down_mps;
    (north * north + east * east + down * down).sqrt()
}
