use std::sync::Arc;

use anderion_cuas_ml::{
    BinaryLogisticDetector, ConceptKind, ConsistencyReport, DeterministicVerifier, Digest32,
    HashProjectionEncoder, OntologyGraph, OntologyNode, OntologyRelation, PatternEngine,
    PatternEvent, PatternToken, PerceptionPipeline, PerceptionVerificationPolicy,
    PrototypeClassifier, RelationKind, ReplayStatus, Result, VerificationContext,
    VerifiedPerceptionPipeline,
};

fn reference_pipeline() -> Result<PerceptionPipeline> {
    let encoder = HashProjectionEncoder::new(2, 2, 11)?;
    let a = encoder.encode_features(&[1.0, 0.0])?;
    let b = encoder.encode_features(&[0.0, 1.0])?;
    let classifier =
        PrototypeClassifier::fit(&[(a.clone(), "drone".into()), (b.clone(), "other".into())])?;
    let detector = BinaryLogisticDetector::fit(
        &[a.values().to_vec(), b.values().to_vec()],
        &[true, false],
        64,
        0.1,
        0.001,
    )?;
    PerceptionPipeline::new(
        Arc::new(encoder),
        Arc::new(detector),
        Arc::new(classifier),
        0.15,
    )
}

fn context() -> Result<VerificationContext> {
    VerificationContext::new(
        Digest32::from_bytes(b"reference-model"),
        Digest32::from_bytes(b"reference-config"),
        "cuas-ontology-v1",
        "perception-v1",
        11,
    )
}

#[test]
fn ontology_detects_conflicting_candidate_classes_deterministically() -> Result<()> {
    let mut graph = OntologyGraph::new("cuas-ontology-v1")?;
    graph.add_node(OntologyNode::new(
        "candidate:1",
        ConceptKind::Candidate,
        None,
    )?)?;
    graph.add_node(OntologyNode::new(
        "class:a",
        ConceptKind::ObjectClass,
        Some("drone".into()),
    )?)?;
    graph.add_node(OntologyNode::new(
        "class:b",
        ConceptKind::ObjectClass,
        Some("bird".into()),
    )?)?;
    graph.add_relation(OntologyRelation::new(
        "candidate:1",
        RelationKind::ClassifiedAs,
        "class:a",
    )?)?;
    graph.add_relation(OntologyRelation::new(
        "candidate:1",
        RelationKind::ClassifiedAs,
        "class:b",
    )?)?;

    let first = graph.validate_reference_schema();
    let second = graph.validate_reference_schema();
    assert!(!first.is_valid());
    assert_eq!(first, second);
    assert!(first.violations().iter().any(|v| v.code() == "cardinality"));
    Ok(())
}

#[test]
fn recurring_behaviour_patterns_are_deterministic() -> Result<()> {
    let approach = PatternToken::new(
        ConceptKind::BehaviourPattern,
        "approach",
        Some("cluster-1".into()),
    )?;
    let hover = PatternToken::new(ConceptKind::BehaviourPattern, "hover", None)?;
    let events = vec![
        PatternEvent::new(400, hover.clone()),
        PatternEvent::new(100, approach.clone()),
        PatternEvent::new(300, approach),
        PatternEvent::new(200, hover),
    ];
    let engine = PatternEngine::default();
    let first = engine.detect_sequences(&events, 2, 2)?;
    let mut reordered = events.clone();
    reordered.rotate_left(1);
    let second = engine.detect_sequences(&reordered, 2, 2)?;
    assert_eq!(first, second);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].occurrences(), 2);
    Ok(())
}

#[test]
fn verified_perception_pipeline_replays_exactly() -> Result<()> {
    let policy = PerceptionVerificationPolicy::new(0.2, 0.98, 0.98, 0.1, 1_000_000, true)?;
    let verified_pipeline =
        VerifiedPerceptionPipeline::new(reference_pipeline()?, context()?, policy);
    let observation =
        anderion_cuas_ml::Observation::new("obs-1", "sensor-a", 1_234, vec![1.0, 0.0])?;

    let first = verified_pipeline.infer(&observation)?;
    let second = verified_pipeline.infer(&observation)?;
    assert_eq!(first.certificate(), second.certificate());

    let (replayed, status) = verified_pipeline.replay(&observation, first.certificate())?;
    assert_eq!(status, ReplayStatus::Exact);
    assert_eq!(replayed.certificate(), first.certificate());
    Ok(())
}

#[test]
fn semantic_contradiction_forces_review() -> Result<()> {
    let pipeline = reference_pipeline()?;
    let observation =
        anderion_cuas_ml::Observation::new("obs-2", "sensor-a", 2_000, vec![0.0, 1.0])?;
    let result = pipeline.infer(&observation)?;
    let report =
        ConsistencyReport::from_violation("cardinality", "conflicting semantic assertions")?;
    let policy = PerceptionVerificationPolicy::new(0.2, 0.98, 0.98, 0.1, 1_000_000, true)?;
    let certificate = DeterministicVerifier::verify_perception(
        &observation,
        &result,
        &context()?,
        &policy,
        &report,
    )?;
    assert_eq!(
        certificate.decision(),
        anderion_cuas_ml::VerificationDecision::Review
    );
    Ok(())
}

#[test]
fn replay_detects_model_context_drift() -> Result<()> {
    let policy = PerceptionVerificationPolicy::new(0.2, 0.98, 0.98, 0.1, 1_000_000, true)?;
    let observation =
        anderion_cuas_ml::Observation::new("obs-3", "sensor-a", 3_000, vec![1.0, 0.0])?;
    let first_pipeline =
        VerifiedPerceptionPipeline::new(reference_pipeline()?, context()?, policy.clone());
    let first = first_pipeline.infer(&observation)?;
    let changed_context = VerificationContext::new(
        Digest32::from_bytes(b"reference-model-v2"),
        Digest32::from_bytes(b"reference-config"),
        "cuas-ontology-v1",
        "perception-v1",
        11,
    )?;
    let second_pipeline =
        VerifiedPerceptionPipeline::new(reference_pipeline()?, changed_context, policy);
    let second = second_pipeline.infer(&observation)?;
    assert_eq!(
        DeterministicVerifier::compare_replay(first.certificate(), second.certificate()),
        ReplayStatus::NonReproducible
    );
    Ok(())
}

#[test]
fn digest_hex_round_trip_is_stable() -> Result<()> {
    let digest = Digest32::from_bytes(b"stable");
    assert_eq!(Digest32::from_hex(&digest.to_hex())?, digest);
    Ok(())
}
