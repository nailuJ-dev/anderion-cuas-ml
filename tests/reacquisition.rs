use anderion_cuas_ml::{Embedding, Position3, ReacquisitionEnvelope};

#[test]
fn matching_reappearance_scores_higher_than_distant_competitor() {
    let envelope = ReacquisitionEnvelope::new(42, 10_000, Position3::new(100.0, 50.0, 20.0).unwrap(), 30.0, Embedding::new(vec![1.0,0.0]).unwrap()).unwrap();
    let near = envelope.score(8_000, Position3::new(105.0, 49.0, 20.0).unwrap(), &Embedding::new(vec![0.99,0.02]).unwrap()).unwrap();
    let far = envelope.score(8_000, Position3::new(500.0, 500.0, 20.0).unwrap(), &Embedding::new(vec![0.0,1.0]).unwrap()).unwrap();
    assert!(near.total > far.total);
    assert!(near.total > 0.8);
}
