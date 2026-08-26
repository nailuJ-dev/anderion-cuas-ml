use std::fs;

use anderion_cuas_ml::service::PerceptionBundle;
use anderion_cuas_ml::{
    ArtifactManifest, BinaryLogisticDetector, HashProjectionEncoder, PrototypeClassifier,
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
    let classifier = PrototypeClassifier::fit(&[(drone, "drone".into()), (bird, "bird".into())])?;
    let bundle = PerceptionBundle {
        encoder,
        detector,
        classifier,
        open_set: None,
        localizer: None,
        calibrator: None,
        unknown_confidence_threshold: 0.55,
    };
    let payload = serde_json::to_vec_pretty(&bundle)?;
    let manifest =
        ArtifactManifest::for_payload("demo-cuas-model", "perception_bundle", 1, &payload)?;
    fs::create_dir_all("artifacts")?;
    fs::write("artifacts/perception-model.json", &payload)?;
    fs::write(
        "artifacts/perception-model.manifest.json",
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}
