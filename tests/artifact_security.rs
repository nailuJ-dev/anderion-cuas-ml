use anderion_cuas_ml::{ArtifactManifest, ArtifactPolicy, verify_payload};

#[test]
fn artifact_rejects_tampering() {
    let payload = b"model";
    let manifest = ArtifactManifest::for_payload("m1", "perception_bundle", 1, payload).unwrap();
    assert!(verify_payload(&manifest, payload, &ArtifactPolicy::default()).is_ok());
    assert!(verify_payload(&manifest, b"tamper", &ArtifactPolicy::default()).is_err());
}
