use crate::{AssociationModel, Embedding, Position3, Result, SdkError, Track};

#[derive(Debug, Clone, PartialEq)]
pub struct TrackObservation {
    pub timestamp_ms: u64,
    pub embedding: Embedding,
    pub position: Option<Position3>,
}

impl TrackObservation {
    pub fn new(timestamp_ms: u64, embedding: Embedding, position: Option<Position3>) -> Result<Self> {
        if embedding.dim() == 0 { return Err(SdkError::EmptyFeatures); }
        Ok(Self { timestamp_ms, embedding, position })
    }
}

pub struct TrackManager {
    association: Box<dyn AssociationModel>,
    threshold: f32,
    max_tracks: usize,
    stale_after_ms: u64,
    next_id: u64,
    tracks: Vec<Track>,
}

impl TrackManager {
    pub fn new(
        association: Box<dyn AssociationModel>,
        threshold: f32,
        max_tracks: usize,
        stale_after_ms: u64,
    ) -> Result<Self> {
        if association.feature_dim() != 3 {
            return Err(SdkError::DimensionMismatch { expected: 3, actual: association.feature_dim() });
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(SdkError::InvalidProbability(threshold));
        }
        if max_tracks == 0 || stale_after_ms == 0 {
            return Err(SdkError::InvalidArgument("max_tracks and stale_after_ms must be positive".into()));
        }
        Ok(Self { association, threshold, max_tracks, stale_after_ms, next_id: 1, tracks: Vec::new() })
    }

    pub fn update(&mut self, observation: TrackObservation) -> Result<u64> {
        self.tracks.retain(|track| observation.timestamp_ms.saturating_sub(track.last_seen_ms) <= self.stale_after_ms);
        let mut best: Option<(usize, f32)> = None;
        for (index, track) in self.tracks.iter().enumerate() {
            if track.embedding.dim() != observation.embedding.dim() { continue; }
            let features = association_features(track, &observation)?;
            let score = self.association.score_features(&features)?;
            if score >= self.threshold && best.map_or(true, |(_, best_score)| score > best_score) {
                best = Some((index, score));
            }
        }
        if let Some((index, _)) = best {
            let track = &mut self.tracks[index];
            let next_count = track.observations.saturating_add(1);
            let alpha = 1.0 / next_count as f32;
            let mut values = track.embedding.values().to_vec();
            for (dst, src) in values.iter_mut().zip(observation.embedding.values()) { *dst += alpha * (*src - *dst); }
            track.embedding = Embedding::new(values)?;
            if let Some(position) = observation.position {
                track.position = Some(match track.position {
                    Some(existing) => Position3::new(
                        existing.x + (position.x - existing.x) / next_count as f64,
                        existing.y + (position.y - existing.y) / next_count as f64,
                        existing.z + (position.z - existing.z) / next_count as f64,
                    )?,
                    None => position,
                });
            }
            track.last_seen_ms = observation.timestamp_ms;
            track.observations = next_count;
            return Ok(track.id);
        }
        if self.tracks.len() >= self.max_tracks {
            return Err(SdkError::DimensionLimit { actual: self.tracks.len() + 1, max: self.max_tracks });
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.tracks.push(Track {
            id,
            first_seen_ms: observation.timestamp_ms,
            last_seen_ms: observation.timestamp_ms,
            observations: 1,
            embedding: observation.embedding,
            position: observation.position,
        });
        Ok(id)
    }

    pub fn tracks(&self) -> &[Track] { &self.tracks }
}

fn association_features(track: &Track, observation: &TrackObservation) -> Result<Vec<f32>> {
    let embedding_distance = euclidean(track.embedding.values(), observation.embedding.values())?;
    let dt_seconds = observation.timestamp_ms.saturating_sub(track.last_seen_ms) as f32 / 1000.0;
    let position_distance_km = match (track.position, observation.position) {
        (Some(a), Some(b)) => (((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt() / 1000.0) as f32,
        _ => 0.0,
    };
    Ok(vec![embedding_distance, dt_seconds, position_distance_km])
}

fn euclidean(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() { return Err(SdkError::DimensionMismatch { expected: a.len(), actual: b.len() }); }
    Ok(a.iter().zip(b).map(|(x, y)| { let d = x - y; d * d }).sum::<f32>().sqrt())
}
