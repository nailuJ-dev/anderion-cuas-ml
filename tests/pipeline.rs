use std::sync::Arc;
use anderion_cuas_ml::{
    BinaryLogisticDetector, HashProjectionEncoder, Observation, PerceptionPipeline,
    PrototypeClassifier,
};

#[test]
fn perception_pipeline_is_deterministic_and_batchable() {
    let encoder = HashProjectionEncoder::new(2, 2, 5).unwrap();
    let drone = encoder.encode_features(&[1.0, 0.0]).unwrap();
    let bird = encoder.encode_features(&[0.0, 1.0]).unwrap();
    let detector = BinaryLogisticDetector::fit(
        &[drone.values().to_vec(), bird.values().to_vec()],
        &[true, false], 800, 0.1, 1e-4,
    ).unwrap();
    let classifier = PrototypeClassifier::fit(&[(drone, "drone".into()), (bird, "bird".into())]).unwrap();
    let pipeline = PerceptionPipeline::new(Arc::new(encoder), Arc::new(detector), Arc::new(classifier), 0.4).unwrap();
    let obs = Observation::new("o", "s", 0, vec![1.0, 0.0]).unwrap();
    let first = pipeline.infer(&obs).unwrap();
    let second = pipeline.infer(&obs).unwrap();
    assert_eq!(first, second);
    assert_eq!(pipeline.infer_batch(&[obs]).unwrap().len(), 1);
}
