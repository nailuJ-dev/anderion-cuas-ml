use std::sync::Arc;

use anderion_cuas_ml::{
    BinaryLogisticDetector, HashProjectionEncoder, Observation, PerceptionPipeline,
    PrototypeClassifier,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encoder = HashProjectionEncoder::new(4, 8, 21)?;
    let drone = encoder.encode_features(&[1.0, 0.1, 0.0, 0.1])?;
    let bird = encoder.encode_features(&[0.0, 1.0, 0.2, 0.0])?;
    let detector = BinaryLogisticDetector::fit(
        &[drone.values().to_vec(), bird.values().to_vec()],
        &[true, false],
        1200,
        0.08,
        1e-4,
    )?;
    let classifier =
        PrototypeClassifier::fit(&[(drone, "drone".to_string()), (bird, "bird".to_string())])?;
    let pipeline = PerceptionPipeline::new(
        Arc::new(encoder),
        Arc::new(detector),
        Arc::new(classifier),
        0.55,
    )?;
    let observation = Observation::new("demo", "sensor-a", 0, vec![0.9, 0.1, 0.0, 0.1])?;
    let result = pipeline.infer(&observation)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
