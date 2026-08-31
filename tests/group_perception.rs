use anderion_cuas_ml::{perceive_group, FormationType, GroupMember, Position3};

#[test]
fn converging_tracks_are_described_as_converging() {
    let members = vec![
        GroupMember::new(1, Position3::new(-100.0,0.0,0.0).unwrap(), [10.0,0.0,0.0]).unwrap(),
        GroupMember::new(2, Position3::new(100.0,0.0,0.0).unwrap(), [-10.0,0.0,0.0]).unwrap(),
        GroupMember::new(3, Position3::new(0.0,100.0,0.0).unwrap(), [0.0,-10.0,0.0]).unwrap(),
    ];
    let group = perceive_group(&members).unwrap();
    assert_eq!(group.formation, FormationType::Converging);
    assert!(group.convergence_score > 0.8);
}
