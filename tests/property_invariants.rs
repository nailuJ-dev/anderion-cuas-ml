use anderion_cuas_ml::{ClassScore, Observation};
use proptest::prelude::*;

proptest! {
    #[test]
    fn valid_feature_vectors_are_accepted(values in prop::collection::vec(-1000.0f32..1000.0f32, 1..256)) {
        let obs = Observation::new("o", "s", 0, values.clone()).unwrap();
        prop_assert_eq!(obs.features(), values.as_slice());
    }

    #[test]
    fn probability_is_bounded(p in 0.0f32..=1.0f32) {
        let score = ClassScore::new("x", p).unwrap();
        prop_assert!((0.0..=1.0).contains(&score.probability));
    }
}
