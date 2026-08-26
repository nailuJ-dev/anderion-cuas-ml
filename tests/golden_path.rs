use anderion_cuas_ml::{
    CandidateKinematics, CooperativeIdentityKind, CooperativeTrack, GeoPosition, GoldenCuasModel,
    GoldenCuasScenario, GoldenCuasSource, GoldenCuasTrainingSample, ReferenceCuasFeatureAdapter,
    ReferenceSensorMeasurements, RecordedSensorFrame, ReplayStatus, VelocityNed,
};

fn measurements(snr_db: f32, micro: f32, rf: f32) -> ReferenceSensorMeasurements {
    ReferenceSensorMeasurements::new(
        220.0, 14.0, 35.0, 8.0, snr_db, 42.0, micro, 0.35, rf, 0.55, 12.0,
    )
    .expect("measurements")
}

fn frame(id: &str, snr_db: f32, micro: f32, rf: f32) -> RecordedSensorFrame {
    RecordedSensorFrame::new(id, "reference-sensor", 10_000, measurements(snr_db, micro, rf))
        .expect("frame")
}

#[test]
fn reference_feature_adapter_is_deterministic() {
    let adapter = ReferenceCuasFeatureAdapter::default();
    let source = frame("drone", 22.0, 0.8, 0.7);
    let first = adapter.to_observation(&source).expect("observation");
    let second = adapter.to_observation(&source).expect("observation");
    assert_eq!(first.features(), second.features());
    assert_eq!(first.features().len(), adapter.feature_dim());
}

#[test]
fn golden_cuas_pipeline_correlates_remote_id_and_replays_exactly() {
    let drone_position = anderion_cuas_ml::Position3::new(170.0, 120.0, 32.0).expect("position");
    let background_position = anderion_cuas_ml::Position3::new(20.0, 5.0, 2.0).expect("position");
    let training = vec![
        GoldenCuasTrainingSample::new(frame("drone-a", 22.0, 0.85, 0.75), true, "drone", drone_position).expect("sample"),
        GoldenCuasTrainingSample::new(frame("drone-b", 20.0, 0.78, 0.68), true, "drone", drone_position).expect("sample"),
        GoldenCuasTrainingSample::new(frame("background-a", 4.0, 0.05, 0.08), false, "background", background_position).expect("sample"),
        GoldenCuasTrainingSample::new(frame("background-b", 5.0, 0.08, 0.10), false, "background", background_position).expect("sample"),
    ];
    let model = GoldenCuasModel::fit(&training, 11).expect("fit");
    let geo = GeoPosition::new(48.1173, -1.6778, 80.0).expect("geo");
    let velocity = VelocityNed::new(4.0, 1.0, 0.0).expect("velocity");
    let candidate = CandidateKinematics::new(10_000, geo, Some(velocity)).expect("candidate");
    let track = CooperativeTrack::new(
        CooperativeIdentityKind::RemoteId,
        "RID-DEMO-001",
        10_020,
        GeoPosition::new(48.11731, -1.67779, 80.5).expect("geo"),
        Some(velocity),
        0.98,
    )
    .expect("track");
    let scenario = GoldenCuasScenario::new(
        "remote-id-match",
        GoldenCuasSource::Recorded { frame: frame("eval", 21.0, 0.82, 0.72) },
        Some(candidate),
        vec![track],
        Some("drone".into()),
    )
    .expect("scenario");
    let report = model.infer(&scenario).expect("infer");
    assert!(report.components().first().is_some_and(|component| !component.cooperative_correlations().is_empty()));
    let (_, replay) = model.replay(&scenario, &report).expect("replay");
    assert_eq!(replay, ReplayStatus::Exact);
}
