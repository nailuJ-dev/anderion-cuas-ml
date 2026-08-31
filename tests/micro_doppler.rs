use anderion_cuas_ml::MicroDopplerExtractor;

#[test]
fn harmonic_rotor_spectrum_has_high_periodicity() {
    let bins: Vec<f64> = (-20..=20).map(|i| i as f64 * 10.0).collect();
    let mut power = vec![0.01_f32; bins.len()];
    for target in [-100.0, -50.0, 0.0, 50.0, 100.0] {
        let index = bins
            .iter()
            .position(|v| (*v - target).abs() < 1e-6)
            .unwrap();
        power[index] = 1.0;
    }
    let features = MicroDopplerExtractor::default()
        .extract(&bins, &power)
        .unwrap();
    assert!(features.harmonicity > 0.5);
    assert!(features.bandwidth_hz > 100.0);
}
