use anderion_cuas_ml::{ClassScore, LearnedSensorFusion, TemporalClassifier};

#[test]
fn sensor_fusion_favors_reliable_sensor() {
    let fusion = LearnedSensorFusion::fit(&[
        ("s1".into(), true),
        ("s1".into(), true),
        ("s1".into(), true),
        ("s2".into(), true),
        ("s2".into(), false),
        ("s2".into(), false),
    ])
    .unwrap();
    let fused = fusion
        .fuse(&[
            (
                "s1".into(),
                vec![
                    ClassScore::new("drone", 0.9).unwrap(),
                    ClassScore::new("bird", 0.1).unwrap(),
                ],
            ),
            (
                "s2".into(),
                vec![
                    ClassScore::new("drone", 0.1).unwrap(),
                    ClassScore::new("bird", 0.9).unwrap(),
                ],
            ),
        ])
        .unwrap();
    assert_eq!(fused[0].label, "drone");
}

#[test]
fn temporal_classifier_smooths_short_flip() {
    let temporal = TemporalClassifier::new(0.8).unwrap();
    let out = temporal
        .aggregate(&[
            vec![
                ClassScore::new("drone", 0.9).unwrap(),
                ClassScore::new("bird", 0.1).unwrap(),
            ],
            vec![
                ClassScore::new("drone", 0.8).unwrap(),
                ClassScore::new("bird", 0.2).unwrap(),
            ],
            vec![
                ClassScore::new("drone", 0.4).unwrap(),
                ClassScore::new("bird", 0.6).unwrap(),
            ],
        ])
        .unwrap();
    assert_eq!(out[0].label, "drone");
}
