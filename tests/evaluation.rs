use anderion_cuas_ml::{DatasetRow, classification_metrics, grouped_split};

#[test]
fn grouped_split_prevents_group_leakage() {
    let rows = vec![
        DatasetRow::new("1", "g1", "s1", "drone", vec![1.0]).unwrap(),
        DatasetRow::new("2", "g1", "s1", "drone", vec![1.1]).unwrap(),
        DatasetRow::new("3", "g2", "s2", "bird", vec![2.0]).unwrap(),
        DatasetRow::new("4", "g3", "s2", "bird", vec![2.1]).unwrap(),
    ];
    let split = grouped_split(&rows, 0.5, 11).unwrap();
    for train in &split.train {
        assert!(!split.test.iter().any(|test| test.group_id == train.group_id));
    }
}

#[test]
fn classification_metrics_compute_accuracy() {
    let metrics = classification_metrics(&["drone", "bird", "drone"], &["drone", "bird", "bird"]).unwrap();
    assert!((metrics.accuracy - 2.0 / 3.0).abs() < 1e-6);
}
