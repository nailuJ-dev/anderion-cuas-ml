use anderion_cuas_ml::{
    BinaryLogisticDetector, Classifier, Detector, Embedding, PrototypeClassifier,
};

#[test]
fn logistic_detector_learns_separable_examples() {
    let x = vec![
        vec![-2.0, -1.0],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        vec![2.0, 1.0],
    ];
    let y = vec![false, false, true, true];
    let model = BinaryLogisticDetector::fit(&x, &y, 800, 0.05, 1e-4).unwrap();
    assert!(
        model
            .detect(&Embedding::new(vec![2.0, 2.0]).unwrap())
            .unwrap()
            .probability
            > 0.7
    );
    assert!(
        model
            .detect(&Embedding::new(vec![-2.0, -2.0]).unwrap())
            .unwrap()
            .probability
            < 0.3
    );
}

#[test]
fn prototype_classifier_supports_few_shot_classes() {
    let model = PrototypeClassifier::fit(&[
        (Embedding::new(vec![1.0, 0.0]).unwrap(), "quad".into()),
        (Embedding::new(vec![0.0, 1.0]).unwrap(), "bird".into()),
    ])
    .unwrap();
    let scores = model
        .classify(&Embedding::new(vec![0.9, 0.1]).unwrap())
        .unwrap();
    assert_eq!(scores[0].label, "quad");
}
