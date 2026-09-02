use anderion_cuas_ml::{
    CaCfar2DConfig, ComplexSample, PropagationGeometry, RangeDopplerConfig, RangeDopplerProcessor,
    RawIqCapture, Result, ca_cfar_2d,
};

const FS: f64 = 1_000_000.0;
const FC: f64 = 10_000_000_000.0;
const PRI: f64 = 1.0e-3;

fn reference_samples() -> Vec<ComplexSample> {
    let mut state = 0xA5A5_1F3D_u32;
    (0..32)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = if state & 1 == 0 { 1.0 } else { -1.0 };
            ComplexSample::from_parts_unchecked(value, 0.0)
        })
        .collect()
}

fn capture(id: &str, samples: Vec<ComplexSample>) -> Result<RawIqCapture> {
    RawIqCapture::new(id, 0, FS, FC, samples)
}

fn add_target(
    pulse: &mut [ComplexSample],
    reference: &[ComplexSample],
    delay: usize,
    amplitude: f32,
    doppler_hz: f64,
    pulse_index: usize,
) {
    let phase = std::f64::consts::TAU * doppler_hz * pulse_index as f64 * PRI;
    let (sin, cos) = phase.sin_cos();
    for (index, reference_sample) in reference.iter().copied().enumerate() {
        let target = delay + index;
        if target >= pulse.len() {
            break;
        }
        let ri = f64::from(reference_sample.i());
        let rq = f64::from(reference_sample.q());
        let i = amplitude as f64 * (ri * cos - rq * sin);
        let q = amplitude as f64 * (ri * sin + rq * cos);
        pulse[target] = ComplexSample::from_parts_unchecked(
            pulse[target].i() + i as f32,
            pulse[target].q() + q as f32,
        );
    }
}

#[test]
fn range_doppler_peak_tracks_known_delay_and_doppler() -> Result<()> {
    let reference = reference_samples();
    let reference_capture = capture("reference", reference.clone())?;
    let mut pulses = Vec::new();
    for pulse_index in 0..32 {
        let mut samples = vec![ComplexSample::default(); 128];
        add_target(&mut samples, &reference, 20, 1.0, 125.0, pulse_index);
        pulses.push(capture(&format!("pulse-{pulse_index}"), samples)?);
    }
    let config = RangeDopplerConfig::new(
        None,
        Some(32),
        PropagationGeometry::MonostaticRoundTrip,
        false,
        true,
    )?;
    let map = RangeDopplerProcessor::new(config)?.process(&reference_capture, &pulses, PRI)?;
    let (doppler_bin, range_bin, _) = map.strongest_cell().ok_or_else(|| {
        anderion_cuas_ml::SdkError::InvalidArgument("empty range-Doppler map".into())
    })?;
    assert!(range_bin.abs_diff(20) <= 1, "range_bin={range_bin}");
    assert!(
        (map.doppler_bins_hz()[doppler_bin] - 125.0).abs() <= 32.0,
        "doppler={} Hz",
        map.doppler_bins_hz()[doppler_bin]
    );
    Ok(())
}

#[test]
fn cfar_finds_two_synthetic_targets() -> Result<()> {
    let reference = reference_samples();
    let reference_capture = capture("reference", reference.clone())?;
    let mut pulses = Vec::new();
    for pulse_index in 0..32 {
        let mut samples = vec![ComplexSample::default(); 160];
        add_target(&mut samples, &reference, 24, 1.0, 125.0, pulse_index);
        add_target(&mut samples, &reference, 72, 0.75, -187.5, pulse_index);
        pulses.push(capture(&format!("pulse-{pulse_index}"), samples)?);
    }
    let config = RangeDopplerConfig::new(
        None,
        Some(32),
        PropagationGeometry::MonostaticRoundTrip,
        false,
        true,
    )?;
    let map = RangeDopplerProcessor::new(config)?.process(&reference_capture, &pulses, PRI)?;
    let detections = ca_cfar_2d(&map, CaCfar2DConfig::new(6, 3, 2, 1, 1.0e-3)?)?;
    assert!(
        detections
            .iter()
            .any(|detection| detection.range_bin().abs_diff(24) <= 1),
        "detections={detections:?}"
    );
    assert!(
        detections
            .iter()
            .any(|detection| detection.range_bin().abs_diff(72) <= 1),
        "detections={detections:?}"
    );
    Ok(())
}

#[test]
fn signed_doppler_axis_separates_positive_and_negative_targets() -> Result<()> {
    let reference = reference_samples();
    let reference_capture = capture("reference-signed", reference.clone())?;
    let mut pulses = Vec::new();
    for pulse_index in 0..32 {
        let mut samples = vec![ComplexSample::default(); 160];
        add_target(&mut samples, &reference, 28, 1.0, 156.25, pulse_index);
        add_target(&mut samples, &reference, 84, 0.9, -218.75, pulse_index);
        pulses.push(capture(&format!("signed-{pulse_index}"), samples)?);
    }
    let config = RangeDopplerConfig::new(
        None,
        Some(32),
        PropagationGeometry::MonostaticRoundTrip,
        false,
        true,
    )?;
    let map = RangeDopplerProcessor::new(config)?.process(&reference_capture, &pulses, PRI)?;
    let detections = ca_cfar_2d(&map, CaCfar2DConfig::new(6, 3, 2, 1, 1.0e-3)?)?;
    assert!(detections.iter().any(|detection| {
        detection.range_bin().abs_diff(28) <= 1 && (detection.doppler_hz() - 156.25).abs() <= 32.0
    }));
    assert!(detections.iter().any(|detection| {
        detection.range_bin().abs_diff(84) <= 1 && (detection.doppler_hz() + 218.75).abs() <= 32.0
    }));
    Ok(())
}

#[test]
fn default_fast_time_window_does_not_create_range_dependent_receive_loss() -> Result<()> {
    let reference = reference_samples();
    let reference_capture = capture("reference-window", reference.clone())?;
    let mut pulses = Vec::new();
    for pulse_index in 0..16 {
        let mut samples = vec![ComplexSample::default(); 160];
        add_target(&mut samples, &reference, 16, 1.0, 0.0, pulse_index);
        add_target(&mut samples, &reference, 96, 1.0, 0.0, pulse_index);
        pulses.push(capture(&format!("window-{pulse_index}"), samples)?);
    }
    let map = RangeDopplerProcessor::default().process(&reference_capture, &pulses, PRI)?;
    let zero_bin = map
        .doppler_bins_hz()
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
        .map(|(index, _)| index)
        .ok_or_else(|| {
            anderion_cuas_ml::SdkError::InvalidArgument("missing Doppler bins".into())
        })?;
    let near = map.power_at(zero_bin, 16).unwrap_or(0.0);
    let far = map.power_at(zero_bin, 96).unwrap_or(0.0);
    let ratio = if near.max(far) <= f32::EPSILON {
        0.0
    } else {
        near.min(far) / near.max(far)
    };
    assert!(ratio > 0.85, "near={near} far={far} ratio={ratio}");
    Ok(())
}

#[test]
fn cfar_wraps_doppler_training_cells_at_fft_edges() -> Result<()> {
    let reference = reference_samples();
    let reference_capture = capture("reference-edge", reference.clone())?;
    let mut pulses = Vec::new();
    for pulse_index in 0..32 {
        let mut samples = vec![ComplexSample::default(); 128];
        add_target(&mut samples, &reference, 40, 1.0, 437.5, pulse_index);
        pulses.push(capture(&format!("edge-{pulse_index}"), samples)?);
    }
    let config = RangeDopplerConfig::new(
        None,
        Some(32),
        PropagationGeometry::MonostaticRoundTrip,
        false,
        false,
    )?;
    let map = RangeDopplerProcessor::new(config)?.process(&reference_capture, &pulses, PRI)?;
    let detections = ca_cfar_2d(&map, CaCfar2DConfig::new(6, 3, 2, 1, 1.0e-3)?)?;
    assert!(detections.iter().any(|detection| {
        detection.range_bin().abs_diff(40) <= 1 && (detection.doppler_hz() - 437.5).abs() <= 32.0
    }));
    Ok(())
}

#[test]
fn range_doppler_rejects_excessive_map_allocation_before_power_map_creation() -> Result<()> {
    let reference = reference_samples();
    let reference_capture = capture("reference-limit", reference)?;
    let pulse = capture("pulse-limit", vec![ComplexSample::default(); 1024])?;
    let config = RangeDopplerConfig::new(
        None,
        Some(16_384),
        PropagationGeometry::MonostaticRoundTrip,
        false,
        false,
    )?;
    assert!(
        RangeDopplerProcessor::new(config)?
            .process(&reference_capture, &[pulse], PRI)
            .is_err()
    );
    Ok(())
}

#[test]
fn range_doppler_rejects_center_frequency_mismatch() -> Result<()> {
    let reference = RawIqCapture::new("reference-grid", 0, FS, FC, reference_samples())?;
    let pulse = RawIqCapture::new(
        "pulse-grid",
        0,
        FS,
        FC + 1_000_000.0,
        vec![ComplexSample::default(); 128],
    )?;
    assert!(
        RangeDopplerProcessor::default()
            .process(&reference, &[pulse], PRI)
            .is_err()
    );
    Ok(())
}
