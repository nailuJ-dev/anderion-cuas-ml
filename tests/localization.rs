use anderion_cuas_ml::{Embedding, LinearLocalizer, Localizer, Position3};

#[test]
fn linear_localizer_learns_supervised_mapping() {
    let samples = vec![
        (
            Embedding::new(vec![0.0, 0.0]).unwrap(),
            Position3::new(0.0, 0.0, 0.0).unwrap(),
        ),
        (
            Embedding::new(vec![1.0, 0.0]).unwrap(),
            Position3::new(2.0, 0.0, 1.0).unwrap(),
        ),
        (
            Embedding::new(vec![0.0, 1.0]).unwrap(),
            Position3::new(0.0, 3.0, 1.0).unwrap(),
        ),
        (
            Embedding::new(vec![1.0, 1.0]).unwrap(),
            Position3::new(2.0, 3.0, 2.0).unwrap(),
        ),
    ];
    let model = LinearLocalizer::fit(&samples, 1500, 0.03, 1e-4).unwrap();
    let location = model
        .localize(&Embedding::new(vec![0.5, 0.5]).unwrap())
        .unwrap();
    assert!((location.position.x - 1.0).abs() < 0.25);
    assert!((location.position.y - 1.5).abs() < 0.25);
    assert!(location.sigma_m >= 0.0);
}
