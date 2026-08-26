use std::collections::BTreeMap;
use std::sync::Arc;
use anderion_cuas_ml::{
    BenchmarkConfig, BinaryLogisticDetector, Embedding, HashProjectionEncoder, MultiSensorAdapter,
    Observation, PerceptionPipeline, PrototypeClassifier, benchmark_pipeline,
};

#[test]
fn multi_sensor_adapter_maps_each_sensor_to_common_domain() {
    let mut sources = BTreeMap::new();
    sources.insert("s1".to_string(), vec![Embedding::new(vec![0.0]).unwrap(), Embedding::new(vec![2.0]).unwrap()]);
    sources.insert("s2".to_string(), vec![Embedding::new(vec![20.0]).unwrap(), Embedding::new(vec![24.0]).unwrap()]);
    let target = vec![Embedding::new(vec![10.0]).unwrap(), Embedding::new(vec![14.0]).unwrap()];
    let adapters = MultiSensorAdapter::fit(&sources, &target).unwrap();
    let out = adapters.transform("s1", &Embedding::new(vec![1.0]).unwrap()).unwrap();
    assert!((out.values()[0] - 12.0).abs() < 1e-4);
}

#[test]
fn benchmark_reports_expected_prediction_count() {
    let encoder = HashProjectionEncoder::new(2, 2, 5).unwrap();
    let drone = encoder.encode_features(&[1.0, 0.0]).unwrap();
    let bird = encoder.encode_features(&[0.0, 1.0]).unwrap();
    let detector = BinaryLogisticDetector::fit(
        &[drone.values().to_vec(), bird.values().to_vec()], &[true, false], 500, 0.1, 1e-4,
    ).unwrap();
    let classifier = PrototypeClassifier::fit(&[(drone, "drone".into()), (bird, "bird".into())]).unwrap();
    let pipeline = PerceptionPipeline::new(Arc::new(encoder), Arc::new(detector), Arc::new(classifier), 0.4).unwrap();
    let observations = vec![Observation::new("o", "s", 0, vec![1.0, 0.0]).unwrap()];
    let report = benchmark_pipeline(&pipeline, &observations, BenchmarkConfig { warmup_runs: 1, measured_runs: 3 }).unwrap();
    assert_eq!(report.inferences, 3);
}
