use anderion_cuas_ml::{
    AutoregressiveTrajectoryPredictor, Embedding, FederatedAverager, FederatedDelta,
    GraphEdge, GraphMessagePasser, MultimodalTransformerEncoder, Position3, TrajectorySample,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nodes = vec![Embedding::new(vec![1.0, 0.0])?, Embedding::new(vec![0.0, 1.0])?];
    let edges = vec![GraphEdge::new(0, 1, 1.0)?, GraphEdge::new(1, 0, 1.0)?];
    let graph = GraphMessagePasser::new(0.6, 0.4)?;
    let propagated = graph.propagate(&nodes, &edges)?;

    let transformer = MultimodalTransformerEncoder::new(1.0, 0.25)?;
    let fused = transformer.aggregate(&[
        ("sensor-a".to_string(), propagated[0].clone()),
        ("sensor-b".to_string(), propagated[1].clone()),
    ])?;
    println!("multimodal dimension: {}", fused.dim());

    let samples = vec![TrajectorySample::new(
        Position3::new(0.0, 0.0, 0.0)?,
        [1.0, 0.0, 0.0],
        1.0,
        Position3::new(1.0, 0.0, 0.0)?,
    )?];
    let predictor = AutoregressiveTrajectoryPredictor::fit(&samples, 10, 0.05, 0.0)?;
    let predicted = predictor.predict(Position3::new(0.0, 0.0, 0.0)?, [1.0, 0.0, 0.0], 1.0)?;
    println!("predicted x: {}", predicted.x);

    let averager = FederatedAverager::new(4, 4, 1.0)?;
    let aggregate = averager.aggregate(&[
        FederatedDelta::new("client-a", vec![0.5, 0.0], 10)?,
        FederatedDelta::new("client-b", vec![0.0, 0.5], 10)?,
    ])?;
    println!("aggregated delta dimension: {}", aggregate.len());
    Ok(())
}
