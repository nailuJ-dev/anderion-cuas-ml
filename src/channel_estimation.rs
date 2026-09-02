use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use serde::{Deserialize, Serialize};

use crate::raw_iq::{ComplexSample, RawIqCapture};
use crate::{Result, SdkError};

const MAX_FFT_LEN: usize = 4_194_304;
const MAX_CFO_CORRELATION_FFT_LEN: usize = 8_388_608;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChannelEstimatorConfig {
    regularization: f32,
    fft_len: Option<usize>,
    estimate_cfo: bool,
    max_taps: usize,
    tap_threshold_db: f32,
}

impl Default for ChannelEstimatorConfig {
    fn default() -> Self {
        Self {
            regularization: 1.0e-6,
            fft_len: None,
            estimate_cfo: true,
            max_taps: 16,
            tap_threshold_db: -24.0,
        }
    }
}

impl ChannelEstimatorConfig {
    pub fn new(
        regularization: f32,
        fft_len: Option<usize>,
        estimate_cfo: bool,
        max_taps: usize,
        tap_threshold_db: f32,
    ) -> Result<Self> {
        let value = Self {
            regularization,
            fft_len,
            estimate_cfo,
            max_taps,
            tap_threshold_db,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(self) -> Result<()> {
        if !self.regularization.is_finite() || self.regularization <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "channel regularization must be finite and positive".into(),
            ));
        }
        if let Some(fft_len) = self.fft_len {
            if fft_len == 0 || !fft_len.is_power_of_two() || fft_len > MAX_FFT_LEN {
                return Err(SdkError::InvalidArgument(format!(
                    "channel fft_len must be a power of two in 1..={MAX_FFT_LEN}"
                )));
            }
        }
        if self.max_taps == 0 || self.max_taps > 256 {
            return Err(SdkError::InvalidArgument(
                "max_taps must be in 1..=256".into(),
            ));
        }
        if !self.tap_threshold_db.is_finite()
            || self.tap_threshold_db > 0.0
            || self.tap_threshold_db < -120.0
        {
            return Err(SdkError::InvalidArgument(
                "tap_threshold_db must be finite and in [-120,0]".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChannelTap {
    delay_samples: usize,
    delay_s: f64,
    power: f32,
    relative_power_db: f32,
}

impl ChannelTap {
    pub fn delay_samples(self) -> usize {
        self.delay_samples
    }

    pub fn delay_s(self) -> f64 {
        self.delay_s
    }

    pub fn power(self) -> f32 {
        self.power
    }

    pub fn relative_power_db(self) -> f32 {
        self.relative_power_db
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelEstimate {
    frequency_response: Vec<ComplexSample>,
    impulse_response: Vec<ComplexSample>,
    taps: Vec<ChannelTap>,
    estimated_cfo_hz: f64,
    normalized_frequency_domain_error: f32,
}

impl ChannelEstimate {
    pub fn frequency_response(&self) -> &[ComplexSample] {
        &self.frequency_response
    }

    pub fn impulse_response(&self) -> &[ComplexSample] {
        &self.impulse_response
    }

    pub fn taps(&self) -> &[ChannelTap] {
        &self.taps
    }

    pub fn estimated_cfo_hz(&self) -> f64 {
        self.estimated_cfo_hz
    }

    pub fn normalized_frequency_domain_error(&self) -> f32 {
        self.normalized_frequency_domain_error
    }
}

#[derive(Debug, Clone, Default)]
pub struct KnownReferenceChannelEstimator {
    config: ChannelEstimatorConfig,
}

impl KnownReferenceChannelEstimator {
    pub fn new(config: ChannelEstimatorConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn estimate(
        &self,
        reference: &RawIqCapture,
        received: &RawIqCapture,
    ) -> Result<ChannelEstimate> {
        reference.validate()?;
        received.validate()?;
        self.config.validate()?;
        validate_compatible_grid(reference, received)?;

        let required = reference.samples().len().max(received.samples().len());
        let fft_len = match self.config.fft_len {
            Some(value) => {
                if value < required {
                    return Err(SdkError::DimensionMismatch {
                        expected: required,
                        actual: value,
                    });
                }
                value
            }
            None => required.next_power_of_two(),
        };
        if fft_len > MAX_FFT_LEN {
            return Err(SdkError::DimensionLimit {
                actual: fft_len,
                max: MAX_FFT_LEN,
            });
        }

        let estimated_cfo_hz = if self.config.estimate_cfo {
            estimate_known_reference_cfo(reference, received)
        } else {
            0.0
        };
        let mut x = vec![Complex::new(0.0_f32, 0.0_f32); fft_len];
        let mut y = vec![Complex::new(0.0_f32, 0.0_f32); fft_len];
        for (target, sample) in x.iter_mut().zip(reference.samples()) {
            *target = to_complex(*sample);
        }
        for (index, (target, sample)) in y.iter_mut().zip(received.samples()).enumerate() {
            let phase = -std::f64::consts::TAU * estimated_cfo_hz * index as f64
                / received.sample_rate_hz();
            let rotation = Complex::new(phase.cos() as f32, phase.sin() as f32);
            *target = to_complex(*sample) * rotation;
        }

        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(fft_len);
        forward.process(&mut x);
        forward.process(&mut y);

        let mut h = Vec::with_capacity(fft_len);
        let mut squared_error = 0.0_f64;
        let mut squared_received = 0.0_f64;
        for (x_bin, y_bin) in x.iter().zip(&y) {
            let denominator = x_bin.norm_sqr() + self.config.regularization;
            let h_bin = *y_bin * x_bin.conj() / denominator;
            let reconstructed = h_bin * *x_bin;
            squared_error += f64::from((*y_bin - reconstructed).norm_sqr());
            squared_received += f64::from(y_bin.norm_sqr());
            h.push(h_bin);
        }
        let normalized_error = if squared_received <= f64::EPSILON {
            0.0
        } else {
            (squared_error / squared_received).sqrt().clamp(0.0, 1.0) as f32
        };

        let frequency_response = h.iter().copied().map(from_complex).collect::<Vec<_>>();
        let inverse = planner.plan_fft_inverse(fft_len);
        inverse.process(&mut h);
        let normalization = 1.0 / fft_len as f32;
        for sample in &mut h {
            *sample *= normalization;
        }
        let impulse_response = h.iter().copied().map(from_complex).collect::<Vec<_>>();
        let taps = extract_taps(
            &h,
            reference.sample_rate_hz(),
            self.config.max_taps,
            self.config.tap_threshold_db,
        );

        Ok(ChannelEstimate {
            frequency_response,
            impulse_response,
            taps,
            estimated_cfo_hz,
            normalized_frequency_domain_error: normalized_error,
        })
    }
}

pub fn estimate_known_reference_cfo(reference: &RawIqCapture, received: &RawIqCapture) -> f64 {
    let reference_len = reference.samples().len();
    let received_len = received.samples().len();
    if reference_len < 2 || received_len < 2 {
        return 0.0;
    }

    let reference_peak_power = reference
        .samples()
        .iter()
        .map(|sample| sample.norm_sqr())
        .fold(0.0_f32, f32::max);
    if reference_peak_power <= f32::EPSILON {
        return 0.0;
    }
    let reference_threshold = reference_peak_power * 1.0e-4;

    // Differential correlation removes the unknown complex gain and converts CFO
    // into a constant phase rotation. Cross-correlating these differential
    // sequences makes the CFO estimate insensitive to an unknown integer delay.
    let reference_diff_len = reference_len - 1;
    let received_diff_len = received_len - 1;
    let correlation_len = reference_diff_len
        .saturating_add(received_diff_len)
        .saturating_sub(1);
    let fft_len = correlation_len.next_power_of_two();
    if fft_len == 0 || fft_len > MAX_CFO_CORRELATION_FFT_LEN {
        return 0.0;
    }

    let mut reference_diff = vec![Complex::new(0.0_f64, 0.0_f64); fft_len];
    let mut received_diff = vec![Complex::new(0.0_f64, 0.0_f64); fft_len];
    for index in 1..reference_len {
        let previous = reference.samples()[index - 1];
        let current = reference.samples()[index];
        if previous.norm_sqr() < reference_threshold || current.norm_sqr() < reference_threshold {
            continue;
        }
        let previous = Complex::new(f64::from(previous.i()), f64::from(previous.q()));
        let current = Complex::new(f64::from(current.i()), f64::from(current.q()));
        reference_diff[index - 1] = current * previous.conj();
    }
    for index in 1..received_len {
        let previous = received.samples()[index - 1];
        let current = received.samples()[index];
        let previous = Complex::new(f64::from(previous.i()), f64::from(previous.q()));
        let current = Complex::new(f64::from(current.i()), f64::from(current.q()));
        received_diff[index - 1] = current * previous.conj();
    }

    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(fft_len);
    forward.process(&mut reference_diff);
    forward.process(&mut received_diff);
    for (received_bin, reference_bin) in received_diff.iter_mut().zip(&reference_diff) {
        *received_bin *= reference_bin.conj();
    }
    let inverse = planner.plan_fft_inverse(fft_len);
    inverse.process(&mut received_diff);

    let Some((_, peak)) = received_diff
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.norm_sqr().is_finite())
        .max_by(|left, right| left.1.norm_sqr().total_cmp(&right.1.norm_sqr()))
    else {
        return 0.0;
    };
    if peak.norm_sqr() <= f64::EPSILON {
        0.0
    } else {
        peak.arg() * reference.sample_rate_hz() / std::f64::consts::TAU
    }
}

fn validate_compatible_grid(reference: &RawIqCapture, received: &RawIqCapture) -> Result<()> {
    let rate_delta = (reference.sample_rate_hz() - received.sample_rate_hz()).abs();
    let rate_tolerance = reference.sample_rate_hz().abs().max(1.0) * 1.0e-9;
    if rate_delta > rate_tolerance {
        return Err(SdkError::InvalidArgument(
            "reference and received captures must use the same sample rate".into(),
        ));
    }
    let frequency_delta = (reference.center_frequency_hz() - received.center_frequency_hz()).abs();
    let frequency_tolerance = reference.center_frequency_hz().abs().max(1.0) * 1.0e-9;
    if frequency_delta > frequency_tolerance {
        return Err(SdkError::InvalidArgument(
            "reference and received captures must use the same center frequency".into(),
        ));
    }
    Ok(())
}

fn extract_taps(
    impulse_response: &[Complex<f32>],
    sample_rate_hz: f64,
    max_taps: usize,
    threshold_db: f32,
) -> Vec<ChannelTap> {
    let powers = impulse_response
        .iter()
        .map(|sample| sample.norm_sqr())
        .collect::<Vec<_>>();
    let max_power = powers.iter().copied().fold(0.0_f32, f32::max);
    if max_power <= f32::EPSILON {
        return Vec::new();
    }
    let linear_threshold = max_power * 10.0_f32.powf(threshold_db / 10.0);
    let mut candidates = Vec::new();
    for (index, power) in powers.iter().copied().enumerate() {
        if power < linear_threshold {
            continue;
        }
        let left = if index == 0 { 0.0 } else { powers[index - 1] };
        let right = powers.get(index + 1).copied().unwrap_or(0.0);
        if power >= left && power >= right {
            candidates.push((index, power));
        }
    }
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    candidates.truncate(max_taps);
    candidates.sort_by_key(|(index, _)| *index);
    candidates
        .into_iter()
        .map(|(index, power)| ChannelTap {
            delay_samples: index,
            delay_s: index as f64 / sample_rate_hz,
            power,
            relative_power_db: 10.0 * (power / max_power).max(f32::MIN_POSITIVE).log10(),
        })
        .collect()
}

fn to_complex(sample: ComplexSample) -> Complex<f32> {
    Complex::new(sample.i(), sample.q())
}

fn from_complex(sample: Complex<f32>) -> ComplexSample {
    ComplexSample::from_parts_unchecked(sample.re, sample.im)
}
