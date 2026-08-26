use anderion_cuas_ml::{
    CandidateKinematics, CooperativeCorrelator, CooperativeDisposition, CooperativeIdentityKind,
    CooperativeTrack, CorrelationPolicy, GeoPosition, Result, VelocityNed,
};

fn position(lat: f64, lon: f64) -> Result<GeoPosition> {
    GeoPosition::new(lat, lon, 40.0)
}

#[test]
fn correlator_matches_ais_adsb_and_remote_id_with_stable_ordering() -> Result<()> {
    let candidate = CandidateKinematics::new(
        10_000,
        position(48.8566, 2.3522)?,
        Some(VelocityNed::new(5.0, 1.0, 0.0)?),
    )?;
    let tracks = vec![
        CooperativeTrack::new(
            CooperativeIdentityKind::Adsb,
            "aircraft-1",
            9_900,
            position(48.85661, 2.35221)?,
            Some(VelocityNed::new(5.1, 1.0, 0.0)?),
            0.98,
        )?,
        CooperativeTrack::new(
            CooperativeIdentityKind::Ais,
            "vessel-1",
            9_900,
            position(48.8567, 2.3523)?,
            Some(VelocityNed::new(5.0, 1.2, 0.0)?),
            0.90,
        )?,
        CooperativeTrack::new(
            CooperativeIdentityKind::RemoteId,
            "uas-1",
            9_950,
            position(48.8566, 2.3522)?,
            Some(VelocityNed::new(5.0, 1.0, 0.0)?),
            1.0,
        )?,
    ];
    let correlator = CooperativeCorrelator::new(CorrelationPolicy::new(250.0, 1_000, 20.0, 0.1)?)?;
    let first = correlator.correlate(&candidate, &tracks)?;
    let mut reversed = tracks.clone();
    reversed.reverse();
    let second = correlator.correlate(&candidate, &reversed)?;
    assert_eq!(first, second);
    assert_eq!(first.len(), 3);
    assert_eq!(first[0].kind(), CooperativeIdentityKind::RemoteId);
    assert_eq!(correlator.disposition(&first), CooperativeDisposition::MatchedCooperative);
    Ok(())
}

#[test]
fn correlator_rejects_stale_and_distant_tracks() -> Result<()> {
    let candidate = CandidateKinematics::new(20_000, position(48.0, 2.0)?, None)?;
    let stale = CooperativeTrack::new(
        CooperativeIdentityKind::Adsb,
        "stale",
        1_000,
        position(48.0, 2.0)?,
        None,
        1.0,
    )?;
    let distant = CooperativeTrack::new(
        CooperativeIdentityKind::RemoteId,
        "distant",
        20_000,
        position(49.0, 3.0)?,
        None,
        1.0,
    )?;
    let correlator = CooperativeCorrelator::new(CorrelationPolicy::new(100.0, 500, 10.0, 0.2)?)?;
    assert!(correlator.correlate(&candidate, &[stale, distant])?.is_empty());
    Ok(())
}

#[test]
fn cooperative_types_validate_external_values() {
    assert!(GeoPosition::new(91.0, 0.0, 0.0).is_err());
    assert!(GeoPosition::new(0.0, 181.0, 0.0).is_err());
    assert!(VelocityNed::new(f64::NAN, 0.0, 0.0).is_err());
    assert!(CorrelationPolicy::new(-1.0, 100, 1.0, 0.5).is_err());
}
