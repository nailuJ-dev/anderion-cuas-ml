use anderion_cuas_ml::{
    DegarblingModel, IdentityDegarbler, Observation, PrototypeMaskDegarbler, Result,
};

#[test]
fn identity_degarbler_preserves_observation_exactly() -> Result<()> {
    let observation = Observation::new("mix", "sensor", 100, vec![0.4, -0.2, 0.8])?;
    let result = IdentityDegarbler.separate(&observation)?;
    assert_eq!(result.components().len(), 1);
    assert_eq!(result.components()[0], observation);
    assert_eq!(result.reconstruction_error(), 0.0);
    assert_eq!(result.separation_confidence(), 1.0);
    Ok(())
}

#[test]
fn prototype_mask_degarbler_separates_components_deterministically() -> Result<()> {
    let model = PrototypeMaskDegarbler::new(
        vec![vec![1.0, 0.1, 0.0, 0.0], vec![0.0, 0.0, 0.2, 1.0]],
        0.05,
    )?;
    let observation = Observation::new("mix", "sensor", 100, vec![1.0, 0.5, 0.4, 0.8])?;
    let first = model.separate(&observation)?;
    let second = model.separate(&observation)?;
    assert_eq!(first, second);
    assert_eq!(first.components().len(), 2);
    assert!(first.reconstruction_error() <= 1e-6);
    assert!((0.0..=1.0).contains(&first.separation_confidence()));

    let reconstructed: Vec<f32> = (0..observation.features().len())
        .map(|index| {
            first
                .components()
                .iter()
                .map(|component| component.features()[index])
                .sum()
        })
        .collect();
    for (actual, expected) in reconstructed.iter().zip(observation.features()) {
        assert!((actual - expected).abs() <= 1e-6);
    }
    Ok(())
}

#[test]
fn prototype_mask_degarbler_rejects_invalid_templates() {
    assert!(PrototypeMaskDegarbler::new(Vec::new(), 0.1).is_err());
    assert!(PrototypeMaskDegarbler::new(vec![vec![1.0], vec![1.0, 2.0]], 0.1).is_err());
    assert!(PrototypeMaskDegarbler::new(vec![vec![-1.0, 0.0]], 0.1).is_err());
}

#[test]
fn prototype_mask_degarbler_can_fit_reference_templates() -> Result<()> {
    let samples = vec![
        (0_usize, vec![1.0, 0.1]),
        (0_usize, vec![0.8, 0.0]),
        (1_usize, vec![0.0, 0.9]),
        (1_usize, vec![0.1, 1.0]),
    ];
    let model = PrototypeMaskDegarbler::fit(&samples, 2, 0.1)?;
    assert_eq!(model.component_count(), 2);
    assert_eq!(model.input_dim(), 2);
    let observation = Observation::new("mix-fit", "sensor", 200, vec![0.9, 0.9])?;
    assert_eq!(model.separate(&observation)?.components().len(), 2);
    Ok(())
}
