use anderion_cuas_ml::{
    CandidateKinematics, CooperativeIdentityKind, CooperativeTrack, CooperativeTrustPolicy,
    CooperativeTrustVerdict, GeoPosition, VelocityNed, assess_cooperative_trust,
};

#[test]
fn physically_consistent_remote_id_is_consistent() {
    let candidate = CandidateKinematics::new(
        1_000,
        GeoPosition::new(48.0, -1.0, 120.0).unwrap(),
        Some(VelocityNed::new(10.0, 0.0, 0.0).unwrap()),
    )
    .unwrap();
    let track = CooperativeTrack::new(
        CooperativeIdentityKind::RemoteId,
        "RID-1",
        1_020,
        GeoPosition::new(48.00001, -1.0, 121.0).unwrap(),
        Some(VelocityNed::new(10.2, 0.0, 0.0).unwrap()),
        0.95,
    )
    .unwrap();
    let result = assess_cooperative_trust(
        &candidate,
        &track,
        &CooperativeTrustPolicy::default(),
        Some(0.92),
    )
    .unwrap();
    assert_eq!(result.verdict, CooperativeTrustVerdict::Consistent);
}

#[test]
fn large_physical_mismatch_is_reported_as_conflict() {
    let candidate =
        CandidateKinematics::new(1_000, GeoPosition::new(48.0, -1.0, 120.0).unwrap(), None)
            .unwrap();
    let track = CooperativeTrack::new(
        CooperativeIdentityKind::RemoteId,
        "RID-1",
        1_000,
        GeoPosition::new(48.02, -1.0, 120.0).unwrap(),
        None,
        1.0,
    )
    .unwrap();
    let result = assess_cooperative_trust(
        &candidate,
        &track,
        &CooperativeTrustPolicy::default(),
        Some(0.1),
    )
    .unwrap();
    assert_eq!(result.verdict, CooperativeTrustVerdict::Conflict);
}

#[test]
fn isolated_position_mismatch_without_second_physical_conflict_is_not_forced_to_conflict() {
    let candidate = CandidateKinematics::new(
        1_000,
        GeoPosition::new(48.0, -1.0, 120.0).unwrap(),
        None,
    )
    .unwrap();

    let track = CooperativeTrack::new(
        CooperativeIdentityKind::RemoteId,
        "RID-2",
        1_000,
        GeoPosition::new(48.02, -1.0, 120.0).unwrap(),
        None,
        1.0,
    )
    .unwrap();

    let result = assess_cooperative_trust(
        &candidate,
        &track,
        &CooperativeTrustPolicy::default(),
        Some(0.95),
    )
    .unwrap();

    assert_eq!(
        result.verdict,
        CooperativeTrustVerdict::WeaklyConsistent
    );
}