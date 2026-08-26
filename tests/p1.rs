use anderion_cuas_ml::{
    ClassScore, DomainAdapter, DriftMonitor, Embedding, WeakLabelModel, adversarial_evaluate,
};

#[test]
fn domain_adapter_aligns_mean() {
    let source = vec![
        Embedding::new(vec![0.0]).unwrap(),
        Embedding::new(vec![2.0]).unwrap(),
    ];
    let target = vec![
        Embedding::new(vec![10.0]).unwrap(),
        Embedding::new(vec![14.0]).unwrap(),
    ];
    let adapter = DomainAdapter::fit(&source, &target).unwrap();
    let out = adapter
        .transform(&Embedding::new(vec![1.0]).unwrap())
        .unwrap();
    assert!((out.values()[0] - 12.0).abs() < 1e-4);
}

#[test]
fn weak_label_model_prefers_consensus() {
    let model = WeakLabelModel::fit(&[
        vec![
            Some("drone".into()),
            Some("drone".into()),
            Some("bird".into()),
        ],
        vec![
            Some("bird".into()),
            Some("bird".into()),
            Some("drone".into()),
        ],
        vec![
            Some("drone".into()),
            Some("drone".into()),
            Some("drone".into()),
        ],
    ])
    .unwrap();
    let label = model
        .predict(&[
            Some("drone".into()),
            Some("drone".into()),
            Some("bird".into()),
        ])
        .unwrap();
    assert_eq!(label, "drone");
}

#[test]
fn drift_monitor_detects_shift() {
    let base = vec![
        Embedding::new(vec![0.0]).unwrap(),
        Embedding::new(vec![0.1]).unwrap(),
    ];
    let monitor = DriftMonitor::fit(&base, 3.0).unwrap();
    let current = vec![
        Embedding::new(vec![5.0]).unwrap(),
        Embedding::new(vec![5.1]).unwrap(),
    ];
    assert!(monitor.evaluate(&current).unwrap().drifted);
}

#[test]
fn adversarial_evaluation_reports_confidence_drop() {
    let baseline = vec![
        ClassScore::new("drone", 0.9).unwrap(),
        ClassScore::new("bird", 0.1).unwrap(),
    ];
    let perturbed = vec![
        ClassScore::new("drone", 0.6).unwrap(),
        ClassScore::new("bird", 0.4).unwrap(),
    ];
    let report = adversarial_evaluate("drone", &baseline, &perturbed).unwrap();
    assert!(report.confidence_drop > 0.0);
}
