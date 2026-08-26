use anderion_cuas_ml::{Observation, Position3, SdkError};

#[test]
fn observation_rejects_empty_features() {
    assert!(matches!(Observation::new("o", "sensor-a", 0, vec![]).unwrap_err(), SdkError::EmptyFeatures));
}

#[test]
fn observation_rejects_non_finite_feature() {
    assert!(matches!(Observation::new("o", "sensor-a", 0, vec![f32::NAN]).unwrap_err(), SdkError::NonFiniteValue { .. }));
}

#[test]
fn position_rejects_non_finite_coordinate() {
    assert!(Position3::new(0.0, f64::NAN, 0.0).is_err());
}
