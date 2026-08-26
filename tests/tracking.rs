use anderion_cuas_ml::{
    AssociationModel, Embedding, LearnedAssociation, Position3, TrackManager, TrackObservation,
};

#[test]
fn learned_association_scores_similar_embeddings_higher() {
    let model = LearnedAssociation::fit(
        &[
            (vec![0.1, 0.1, 0.1], true),
            (vec![0.2, 0.2, 0.1], true),
            (vec![3.0, 2.0, 1.0], false),
            (vec![4.0, 3.0, 2.0], false),
        ],
        600,
        0.05,
        1e-4,
    )
    .unwrap();
    assert!(
        model.score_features(&[0.1, 0.1, 0.1]).unwrap()
            > model.score_features(&[4.0, 3.0, 2.0]).unwrap()
    );
}

#[test]
fn track_manager_preserves_track_for_similar_observations() {
    let association = LearnedAssociation::fit(
        &[
            (vec![0.05, 0.1, 0.0], true),
            (vec![0.2, 0.2, 0.1], true),
            (vec![3.0, 2.0, 2.0], false),
            (vec![4.0, 3.0, 3.0], false),
        ],
        800,
        0.05,
        1e-4,
    )
    .unwrap();
    let mut manager = TrackManager::new(Box::new(association), 0.5, 16, 10_000).unwrap();
    let first = TrackObservation::new(
        1,
        Embedding::new(vec![1.0, 0.0]).unwrap(),
        Some(Position3::new(1.0, 1.0, 0.0).unwrap()),
    )
    .unwrap();
    let second = TrackObservation::new(
        2,
        Embedding::new(vec![0.95, 0.05]).unwrap(),
        Some(Position3::new(1.1, 1.0, 0.0).unwrap()),
    )
    .unwrap();
    let id1 = manager.update(first).unwrap();
    let id2 = manager.update(second).unwrap();
    assert_eq!(id1, id2);
}
