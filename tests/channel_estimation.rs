use anderion_cuas_ml::{
    ChannelEstimatorConfig, ComplexSample, KnownReferenceChannelEstimator, RawIqCapture, Result,
    estimate_known_reference_cfo,
};

const FS: f64 = 1_000_000.0;

fn capture(id: &str, samples: Vec<ComplexSample>) -> Result<RawIqCapture> {
    RawIqCapture::new(id, 0, FS, 2_400_000_000.0, samples)
}

#[test]
fn estimates_known_reference_cfo_from_raw_complex_samples() -> Result<()> {
    let frequency = 12_500.0;
    let mut reference = Vec::new();
    let mut received = Vec::new();
    for index in 0..512 {
        let sign = if ((index * 73 + 19) & 1) == 0 {
            1.0
        } else {
            -1.0
        };
        reference.push(ComplexSample::new(sign, 0.0)?);
        let phase = std::f64::consts::TAU * frequency * index as f64 / FS;
        received.push(ComplexSample::new(
            sign * phase.cos() as f32,
            sign * phase.sin() as f32,
        )?);
    }
    let estimated = estimate_known_reference_cfo(
        &capture("reference", reference)?,
        &capture("received", received)?,
    );
    assert!(
        (estimated - frequency).abs() < 50.0,
        "estimated={estimated}"
    );
    Ok(())
}

#[test]
fn impulse_reference_recovers_multipath_delays() -> Result<()> {
    let mut reference = vec![ComplexSample::default(); 64];
    reference[0] = ComplexSample::new(1.0, 0.0)?;
    let mut received = vec![ComplexSample::default(); 64];
    received[7] = ComplexSample::new(0.8, 0.0)?;
    received[19] = ComplexSample::new(0.3, 0.1)?;
    let estimator = KnownReferenceChannelEstimator::new(ChannelEstimatorConfig::new(
        1.0e-9,
        Some(64),
        false,
        8,
        -20.0,
    )?)?;
    let estimate = estimator.estimate(
        &capture("reference", reference)?,
        &capture("received", received)?,
    )?;
    let delays = estimate
        .taps()
        .iter()
        .map(|tap| tap.delay_samples())
        .collect::<Vec<_>>();
    assert!(delays.contains(&7), "delays={delays:?}");
    assert!(delays.contains(&19), "delays={delays:?}");
    assert!(estimate.normalized_frequency_domain_error() < 1.0e-3);
    Ok(())
}

#[test]
fn cfo_estimation_is_stable_with_unknown_integer_delay() -> Result<()> {
    let frequency = 18_750.0;
    let delay = 17_usize;
    let mut state = 0xD17F_42A1_u32;
    let mut reference = Vec::new();
    for _ in 0..768 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let value = if state & 1 == 0 { 1.0 } else { -1.0 };
        reference.push(ComplexSample::new(value, 0.0)?);
    }
    let mut received = vec![ComplexSample::default(); reference.len()];
    for (index, sample) in reference
        .iter()
        .copied()
        .enumerate()
        .take(reference.len() - delay)
    {
        let target = index + delay;
        let phase = std::f64::consts::TAU * frequency * target as f64 / FS;
        received[target] = ComplexSample::new(
            sample.i() * phase.cos() as f32 - sample.q() * phase.sin() as f32,
            sample.i() * phase.sin() as f32 + sample.q() * phase.cos() as f32,
        )?;
    }
    let estimated = estimate_known_reference_cfo(
        &capture("reference-delay", reference)?,
        &capture("received-delay", received)?,
    );
    assert!(
        (estimated - frequency).abs() < 100.0,
        "estimated={estimated}"
    );
    Ok(())
}

#[test]
fn channel_estimator_rejects_center_frequency_mismatch() -> Result<()> {
    let reference = RawIqCapture::new(
        "reference-grid",
        0,
        FS,
        2_400_000_000.0,
        vec![ComplexSample::new(1.0, 0.0)?; 16],
    )?;
    let received = RawIqCapture::new(
        "received-grid",
        0,
        FS,
        2_401_000_000.0,
        vec![ComplexSample::new(1.0, 0.0)?; 16],
    )?;
    assert!(
        KnownReferenceChannelEstimator::default()
            .estimate(&reference, &received)
            .is_err()
    );
    Ok(())
}
