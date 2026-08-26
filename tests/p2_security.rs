use anderion_cuas_ml::{
    Embedding, FederatedAverager, FederatedDelta, GraphEdge, GraphMessagePasser, NeuralLocalizer,
    Position3, SymmetricQuantizer,
};

#[test]
fn graph_rejects_out_of_range_edges() {
    let model = GraphMessagePasser::new(0.5, 0.5).expect("graph");
    let nodes = vec![Embedding::new(vec![1.0, 0.0]).expect("embedding")];
    let edges = vec![GraphEdge::new(0, 1, 1.0).expect("edge")];
    assert!(model.propagate(&nodes, &edges).is_err());
}

#[test]
fn federated_aggregation_enforces_client_and_dimension_bounds() {
    let averager = FederatedAverager::new(1, 2, 1.0).expect("averager");
    let deltas = vec![
        FederatedDelta::new("a", vec![1.0, 0.0], 1).expect("delta"),
        FederatedDelta::new("b", vec![0.0, 1.0], 1).expect("delta"),
    ];
    assert!(averager.aggregate(&deltas).is_err());
}

#[test]
fn neural_localizer_rejects_unbounded_hidden_width() {
    let samples = vec![(
        Embedding::new(vec![0.0, 0.0]).expect("embedding"),
        Position3::new(0.0, 0.0, 0.0).expect("position"),
    )];
    assert!(NeuralLocalizer::fit(&samples, 257, 1, 0.01, 0.0, 1).is_err());
}

#[test]
fn quantizer_rejects_unsupported_precision() {
    assert!(SymmetricQuantizer::new(9).is_err());
}

#[test]
fn deserialized_trajectory_predictor_revalidates_shape_before_use() {
    let malformed = r#"{"weights":[[0.0,0.0,0.0]]}"#;
    let model: anderion_cuas_ml::AutoregressiveTrajectoryPredictor =
        serde_json::from_str(malformed).expect("json shape deserializes");
    let position = Position3::new(0.0, 0.0, 0.0).expect("position");
    assert!(model.predict(position, [1.0, 0.0, 0.0], 1.0).is_err());
}
