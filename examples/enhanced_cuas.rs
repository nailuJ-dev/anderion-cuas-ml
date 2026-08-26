use std::{collections::BTreeSet, sync::Arc};

use anderion_cuas_ml::{
    ActiveSensingProvider, BinaryLogisticDetector, CandidateKinematics, CooperativeCorrelator,
    CooperativeIdentityKind, CooperativeTrack, CorrelationPolicy, Digest32,
    EnhancedPerceptionPipeline, GeoPosition, HashProjectionEncoder, OperatorAuthorization,
    PerceptionPipeline, PrototypeClassifier, PrototypeMaskDegarbler, RecordedSensingProvider,
    Result, SensingCapabilities, SensingFrame, SensingMode, SensingRequest,
};

fn main() -> Result<()> {
    let encoder = HashProjectionEncoder::new(2, 2, 23)?;
    let drone = encoder.encode_features(&[1.0, 0.0])?;
    let other = encoder.encode_features(&[0.0, 1.0])?;
    let detector = BinaryLogisticDetector::fit(
        &[drone.values().to_vec(), other.values().to_vec()],
        &[true, false],
        128,
        0.1,
        0.001,
    )?;
    let classifier = PrototypeClassifier::fit(&[(drone, "drone".into()), (other, "other".into())])?;
    let base = PerceptionPipeline::new(Arc::new(encoder), Arc::new(detector), Arc::new(classifier), 0.2)?;

    let degarbler = PrototypeMaskDegarbler::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]], 0.1)?;
    let correlator = CooperativeCorrelator::new(CorrelationPolicy::new(300.0, 2_000, 30.0, 0.2)?)?;
    let pipeline = EnhancedPerceptionPipeline::new(base)
        .with_degarbler(Arc::new(degarbler))
        .with_correlator(correlator);

    let capabilities = SensingCapabilities::new(
        "ran-site-01",
        BTreeSet::from([SensingMode::OperatorManagedActive]),
        2,
    )?;
    let authorization = OperatorAuthorization::new(
        "operator-a",
        "ran-site-01",
        1_000,
        2_000,
        BTreeSet::from([SensingMode::OperatorManagedActive]),
        Digest32::from_bytes(b"externally-validated-authorization"),
    )?;
    let request = SensingRequest::new(
        "ran-site-01",
        SensingMode::OperatorManagedActive,
        2,
        Digest32::from_bytes(b"operator-approved-sensing-config"),
    )?;
    let frame = SensingFrame::new(1_500, "ran-site-01", SensingMode::OperatorManagedActive, vec![0.8, 0.3])?;
    let mut provider = RecordedSensingProvider::new(capabilities, vec![frame])?;
    let session = provider.prepare(&authorization, &request, 1_500)?;
    let acquired = provider.acquire(&session)?.ok_or_else(|| anderion_cuas_ml::SdkError::InvalidArgument("missing frame".into()))?;
    provider.stop(session)?;

    let observation = acquired.to_observation("isac-1")?;
    let candidate = CandidateKinematics::new(1_500, GeoPosition::new(48.8566, 2.3522, 40.0)?, None)?;
    let remote_id = CooperativeTrack::new(
        CooperativeIdentityKind::RemoteId,
        "rid-demo",
        1_500,
        GeoPosition::new(48.8566, 2.3522, 40.0)?,
        None,
        1.0,
    )?;
    let result = pipeline.infer(&observation, Some(&candidate), &[remote_id])?;
    println!("components={} cooperative={:?}", result.components().len(), result.components()[0].cooperative_disposition());
    Ok(())
}
