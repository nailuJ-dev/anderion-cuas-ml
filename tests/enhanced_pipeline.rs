use std::sync::Arc;

use anderion_cuas_ml::{
    BinaryLogisticDetector, CandidateKinematics, CooperativeCorrelator, CooperativeDisposition,
    CooperativeIdentityKind, CooperativeTrack, CorrelationPolicy, EnhancedPerceptionPipeline,
    GeoPosition, HashProjectionEncoder, IdentityDegarbler, Observation, PerceptionPipeline,
    PrototypeClassifier, PrototypeMaskDegarbler, Result,
};

fn reference_pipeline() -> Result<PerceptionPipeline> {
    let encoder = HashProjectionEncoder::new(2, 2, 19)?;
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
    PerceptionPipeline::new(Arc::new(encoder), Arc::new(detector), Arc::new(classifier), 0.2)
}

#[test]
fn enhanced_pipeline_keeps_degarbling_optional() -> Result<()> {
    let pipeline = EnhancedPerceptionPipeline::new(reference_pipeline()?)
        .with_degarbler(Arc::new(IdentityDegarbler));
    let observation = Observation::new("obs", "sensor", 100, vec![1.0, 0.0])?;
    let result = pipeline.infer(&observation, None, &[])?;
    assert_eq!(result.components().len(), 1);
    assert_eq!(result.components()[0].source_component_index(), 0);
    Ok(())
}

#[test]
fn enhanced_pipeline_handles_multiple_degarbled_components() -> Result<()> {
    let degarbler = PrototypeMaskDegarbler::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]], 0.05)?;
    let pipeline = EnhancedPerceptionPipeline::new(reference_pipeline()?)
        .with_degarbler(Arc::new(degarbler));
    let observation = Observation::new("mix", "sensor", 100, vec![0.8, 0.7])?;
    let result = pipeline.infer(&observation, None, &[])?;
    assert_eq!(result.components().len(), 2);
    Ok(())
}

#[test]
fn enhanced_pipeline_attaches_cooperative_context_without_mutating_ml_scores() -> Result<()> {
    let correlator = CooperativeCorrelator::new(CorrelationPolicy::new(300.0, 2_000, 30.0, 0.1)?)?;
    let pipeline = EnhancedPerceptionPipeline::new(reference_pipeline()?).with_correlator(correlator);
    let observation = Observation::new("obs", "sensor", 100, vec![1.0, 0.0])?;
    let candidate = CandidateKinematics::new(100, GeoPosition::new(48.0, 2.0, 50.0)?, None)?;
    let cooperative = CooperativeTrack::new(
        CooperativeIdentityKind::RemoteId,
        "rid-1",
        100,
        GeoPosition::new(48.0, 2.0, 50.0)?,
        None,
        1.0,
    )?;
    let result = pipeline.infer(&observation, Some(&candidate), &[cooperative])?;
    let component = &result.components()[0];
    assert_eq!(component.cooperative_disposition(), CooperativeDisposition::MatchedCooperative);
    assert!(!component.perception().classification.is_empty());
    Ok(())
}

#[test]
fn enhanced_evidence_digest_is_order_independent_for_cooperative_input() -> Result<()> {
    let correlator = CooperativeCorrelator::new(CorrelationPolicy::new(500.0, 2_000, 30.0, 0.1)?)?;
    let pipeline = EnhancedPerceptionPipeline::new(reference_pipeline()?).with_correlator(correlator);
    let observation = Observation::new("obs-digest", "sensor", 100, vec![1.0, 0.0])?;
    let candidate = CandidateKinematics::new(100, GeoPosition::new(48.0, 2.0, 50.0)?, None)?;
    let a = CooperativeTrack::new(
        CooperativeIdentityKind::Ais,
        "ais-1",
        100,
        GeoPosition::new(48.0001, 2.0001, 0.0)?,
        None,
        0.9,
    )?;
    let b = CooperativeTrack::new(
        CooperativeIdentityKind::RemoteId,
        "rid-1",
        100,
        GeoPosition::new(48.0, 2.0, 50.0)?,
        None,
        1.0,
    )?;
    let first = pipeline.infer(&observation, Some(&candidate), &[a.clone(), b.clone()])?;
    let second = pipeline.infer(&observation, Some(&candidate), &[b, a])?;
    assert_eq!(first.evidence_digest(), second.evidence_digest());
    Ok(())
}
