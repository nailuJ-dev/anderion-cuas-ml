#![cfg(feature = "remoteid")]

use anderion_cuas_ml::{
    CandidateKinematics, CooperativeCorrelator, CooperativeTrack, CorrelationPolicy, GeoPosition,
    Result,
};
use remoteid::{
    BasicId, HorizontalAccuracy, IdType, Location, Message, MessagePackBuilder, Tracker,
    TrackerConfig, Transmitter, UaType, Verdict,
};

const SERIAL: &str = "1596F35074529ABCDEFG";

fn frame(lat: f64, lon: f64, speed: f32, direction: f32, ts: f32) -> Vec<u8> {
    let id = Message::BasicId(
        BasicId::new(IdType::SerialNumber, UaType::HelicopterOrMultirotor, SERIAL).unwrap(),
    );
    let location = Message::Location(Location {
        latitude: lat,
        longitude: lon,
        speed_horizontal: Some(speed),
        direction: Some(direction),
        altitude_geo: Some(100.0),
        timestamp: Some(ts),
        horizontal_accuracy: HorizontalAccuracy(10),
        ..Location::default()
    });
    let mut pack = MessagePackBuilder::new();
    pack.push_raw(&id.encode().unwrap()).unwrap();
    pack.push_raw(&location.encode().unwrap()).unwrap();
    pack.as_bytes().to_vec()
}

fn correlate(track: CooperativeTrack, lat: f64, lon: f64) -> Result<usize> {
    let candidate = CandidateKinematics::new(19_000, GeoPosition::new(lat, lon, 100.0)?, None)?;
    let correlator = CooperativeCorrelator::new(CorrelationPolicy::new(250.0, 2_000, 15.0, 0.1)?)?;
    Ok(correlator.correlate(&candidate, &[track])?.len())
}

#[test]
fn steady_remote_id_broadcast_becomes_a_track_that_correlates() -> Result<()> {
    let mut tracker = Tracker::new(TrackerConfig::default());
    let mut lat = 0.0;
    for i in 0..20u64 {
        // 10 m/s due north, one fix per second.
        lat = 47.0 + (10.0 * i as f64) / 111_195.0;
        let payload = frame(lat, 8.0, 10.0, 0.0, 600.0 + i as f32);
        tracker.ingest(Transmitter::Ble([1; 6]), i as u8, &payload, 1_000 * i);
    }
    let uas = tracker.get(&Transmitter::Ble([1; 6])).unwrap();
    assert_eq!(uas.assessment.verdict, Verdict::Plausible);

    let track = CooperativeTrack::from_remote_id(uas)?.unwrap();
    assert_eq!(track.identity(), SERIAL);
    assert_eq!(track.timestamp_ms(), 19_000);
    assert!((track.velocity().unwrap().north_mps() - 10.0).abs() < 0.5);
    assert!(track.source_confidence() > 0.0);
    assert_eq!(correlate(track, lat, 8.0)?, 1);
    Ok(())
}

#[test]
fn spoofed_remote_id_broadcast_does_not_correlate() -> Result<()> {
    let mut tracker = Tracker::new(TrackerConfig::default());
    let mut lon = 0.0;
    for i in 0..20u64 {
        // Jumps 2 km east every second while claiming 5 m/s north.
        lon = 8.0 + (2_000.0 * i as f64) / 75_800.0;
        let payload = frame(47.0, lon, 5.0, 0.0, 600.0 + i as f32);
        tracker.ingest(Transmitter::Ble([2; 6]), i as u8, &payload, 1_000 * i);
    }
    let uas = tracker.get(&Transmitter::Ble([2; 6])).unwrap();
    assert_eq!(uas.assessment.verdict, Verdict::Spoofed);

    let track = CooperativeTrack::from_remote_id(uas)?.unwrap();
    assert_eq!(track.source_confidence(), 0.0);
    assert_eq!(correlate(track, 47.0, lon)?, 0);
    Ok(())
}
