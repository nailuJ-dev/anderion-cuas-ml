use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use serde::{Deserialize, Serialize};

use crate::raw_iq::RawIqCapture;
use crate::{Result, SdkError};

const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;
const MAX_FFT_LEN: usize = 4_194_304;
const MAX_DOPPLER_FFT_LEN: usize = 16_384;
const MAX_RANGE_DOPPLER_CELLS: usize = 8_388_608;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropagationGeometry {
    OneWay,
    MonostaticRoundTrip,
}

impl PropagationGeometry {
    fn path_factor(self) -> f64 {
        match self {
            Self::OneWay => 1.0,
            Self::MonostaticRoundTrip => 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RangeDopplerConfig {
    fast_time_fft_len: Option<usize>,
    doppler_fft_len: Option<usize>,
    geometry: PropagationGeometry,
    fast_time_hann: bool,
    slow_time_hann: bool,
}

impl Default for RangeDopplerConfig {
    fn default() -> Self {
        Self {
            fast_time_fft_len: None,
            doppler_fft_len: None,
            geometry: PropagationGeometry::MonostaticRoundTrip,
            fast_time_hann: true,
            slow_time_hann: true,
        }
    }
}

impl RangeDopplerConfig {
    pub fn new(
        fast_time_fft_len: Option<usize>,
        doppler_fft_len: Option<usize>,
        geometry: PropagationGeometry,
        fast_time_hann: bool,
        slow_time_hann: bool,
    ) -> Result<Self> {
        let value = Self {
            fast_time_fft_len,
            doppler_fft_len,
            geometry,
            fast_time_hann,
            slow_time_hann,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(self) -> Result<()> {
        if let Some(value) = self.fast_time_fft_len {
            if value == 0 || !value.is_power_of_two() || value > MAX_FFT_LEN {
                return Err(SdkError::InvalidArgument(format!(
                    "fast_time_fft_len must be a power of two in 1..={MAX_FFT_LEN}"
                )));
            }
        }
        if let Some(value) = self.doppler_fft_len {
            if value == 0 || !value.is_power_of_two() || value > MAX_DOPPLER_FFT_LEN {
                return Err(SdkError::InvalidArgument(format!(
                    "doppler_fft_len must be a power of two in 1..={MAX_DOPPLER_FFT_LEN}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RangeDopplerMap {
    range_bins_m: Vec<f64>,
    doppler_bins_hz: Vec<f64>,
    radial_velocity_bins_m_s: Option<Vec<f64>>,
    power: Vec<f32>,
    range_bin_count: usize,
    doppler_bin_count: usize,
}

impl RangeDopplerMap {
    pub fn range_bins_m(&self) -> &[f64] {
        &self.range_bins_m
    }

    pub fn doppler_bins_hz(&self) -> &[f64] {
        &self.doppler_bins_hz
    }

    pub fn radial_velocity_bins_m_s(&self) -> Option<&[f64]> {
        self.radial_velocity_bins_m_s.as_deref()
    }

    pub fn power(&self) -> &[f32] {
        &self.power
    }

    pub fn range_bin_count(&self) -> usize {
        self.range_bin_count
    }

    pub fn doppler_bin_count(&self) -> usize {
        self.doppler_bin_count
    }

    pub fn power_at(&self, doppler_bin: usize, range_bin: usize) -> Option<f32> {
        if doppler_bin >= self.doppler_bin_count || range_bin >= self.range_bin_count {
            return None;
        }
        self.power
            .get(doppler_bin * self.range_bin_count + range_bin)
            .copied()
    }

    pub fn strongest_cell(&self) -> Option<(usize, usize, f32)> {
        self.power
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(index, power)| {
                (
                    index / self.range_bin_count,
                    index % self.range_bin_count,
                    power,
                )
            })
    }
}

#[derive(Debug, Clone, Default)]
pub struct RangeDopplerProcessor {
    config: RangeDopplerConfig,
}

impl RangeDopplerProcessor {
    pub fn new(config: RangeDopplerConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn process(
        &self,
        reference: &RawIqCapture,
        received_pulses: &[RawIqCapture],
        pulse_repetition_interval_s: f64,
    ) -> Result<RangeDopplerMap> {
        reference.validate()?;
        self.config.validate()?;
        if received_pulses.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        if !pulse_repetition_interval_s.is_finite() || pulse_repetition_interval_s <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "pulse_repetition_interval_s must be finite and positive".into(),
            ));
        }
        for pulse in received_pulses {
            pulse.validate()?;
            validate_capture_grid(reference, pulse)?;
        }

        let max_received_len = received_pulses
            .iter()
            .map(|pulse| pulse.samples().len())
            .max()
            .ok_or(SdkError::EmptyDataset)?;
        let required_fast_len = reference
            .samples()
            .len()
            .saturating_add(max_received_len)
            .saturating_sub(1)
            .next_power_of_two();
        let fast_fft_len = self.config.fast_time_fft_len.unwrap_or(required_fast_len);
        if fast_fft_len < required_fast_len {
            return Err(SdkError::DimensionMismatch {
                expected: required_fast_len,
                actual: fast_fft_len,
            });
        }
        if fast_fft_len > MAX_FFT_LEN {
            return Err(SdkError::DimensionLimit {
                actual: fast_fft_len,
                max: MAX_FFT_LEN,
            });
        }

        let range_bin_count = max_received_len;
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(fast_fft_len);
        let inverse = planner.plan_fft_inverse(fast_fft_len);

        let mut reference_spectrum = vec![Complex::new(0.0_f32, 0.0_f32); fast_fft_len];
        let reference_len = reference.samples().len();
        for (index, sample) in reference.samples().iter().copied().enumerate() {
            let window = if self.config.fast_time_hann {
                hann(index, reference_len)
            } else {
                1.0
            };
            reference_spectrum[index] = Complex::new(sample.i() * window, sample.q() * window);
        }
        forward.process(&mut reference_spectrum);

        let mut range_profiles = Vec::with_capacity(received_pulses.len());
        for pulse in received_pulses {
            let mut received_spectrum = vec![Complex::new(0.0_f32, 0.0_f32); fast_fft_len];
            for (index, sample) in pulse.samples().iter().copied().enumerate() {
                received_spectrum[index] = Complex::new(sample.i(), sample.q());
            }
            forward.process(&mut received_spectrum);
            for (received_bin, reference_bin) in
                received_spectrum.iter_mut().zip(&reference_spectrum)
            {
                *received_bin *= reference_bin.conj();
            }
            inverse.process(&mut received_spectrum);
            let normalization = 1.0 / fast_fft_len as f32;
            for value in &mut received_spectrum {
                *value *= normalization;
            }
            received_spectrum.truncate(range_bin_count);
            range_profiles.push(received_spectrum);
        }

        let pulse_count = range_profiles.len();
        let required_doppler_len = pulse_count.next_power_of_two();
        let doppler_fft_len = self.config.doppler_fft_len.unwrap_or(required_doppler_len);
        if doppler_fft_len < pulse_count {
            return Err(SdkError::DimensionMismatch {
                expected: pulse_count,
                actual: doppler_fft_len,
            });
        }
        if doppler_fft_len > MAX_DOPPLER_FFT_LEN {
            return Err(SdkError::DimensionLimit {
                actual: doppler_fft_len,
                max: MAX_DOPPLER_FFT_LEN,
            });
        }
        let map_cells =
            range_bin_count
                .checked_mul(doppler_fft_len)
                .ok_or(SdkError::DimensionLimit {
                    actual: usize::MAX,
                    max: MAX_RANGE_DOPPLER_CELLS,
                })?;
        if map_cells > MAX_RANGE_DOPPLER_CELLS {
            return Err(SdkError::DimensionLimit {
                actual: map_cells,
                max: MAX_RANGE_DOPPLER_CELLS,
            });
        }

        let doppler_fft = planner.plan_fft_forward(doppler_fft_len);
        let mut power = vec![0.0_f32; doppler_fft_len * range_bin_count];
        for range_bin in 0..range_bin_count {
            let mut slow_time = vec![Complex::new(0.0_f32, 0.0_f32); doppler_fft_len];
            for pulse_index in 0..pulse_count {
                let window = if self.config.slow_time_hann {
                    hann(pulse_index, pulse_count)
                } else {
                    1.0
                };
                slow_time[pulse_index] = range_profiles[pulse_index][range_bin] * window;
            }
            doppler_fft.process(&mut slow_time);
            for shifted_bin in 0..doppler_fft_len {
                let source_bin = (shifted_bin + doppler_fft_len / 2) % doppler_fft_len;
                power[shifted_bin * range_bin_count + range_bin] =
                    slow_time[source_bin].norm_sqr() / doppler_fft_len as f32;
            }
        }

        let path_factor = self.config.geometry.path_factor();
        let range_bins_m = (0..range_bin_count)
            .map(|index| {
                index as f64 / reference.sample_rate_hz() * SPEED_OF_LIGHT_M_S / path_factor
            })
            .collect::<Vec<_>>();
        let doppler_bins_hz = (0..doppler_fft_len)
            .map(|index| {
                let signed = index as isize - (doppler_fft_len / 2) as isize;
                signed as f64 / (doppler_fft_len as f64 * pulse_repetition_interval_s)
            })
            .collect::<Vec<_>>();
        let radial_velocity_bins_m_s = if reference.center_frequency_hz() > 0.0 {
            Some(
                doppler_bins_hz
                    .iter()
                    .map(|frequency| {
                        frequency * SPEED_OF_LIGHT_M_S
                            / (path_factor * reference.center_frequency_hz())
                    })
                    .collect(),
            )
        } else {
            None
        };

        Ok(RangeDopplerMap {
            range_bins_m,
            doppler_bins_hz,
            radial_velocity_bins_m_s,
            power,
            range_bin_count,
            doppler_bin_count: doppler_fft_len,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CaCfar2DConfig {
    training_range: usize,
    training_doppler: usize,
    guard_range: usize,
    guard_doppler: usize,
    false_alarm_probability: f64,
}

impl Default for CaCfar2DConfig {
    fn default() -> Self {
        Self {
            training_range: 8,
            training_doppler: 4,
            guard_range: 2,
            guard_doppler: 1,
            false_alarm_probability: 1.0e-4,
        }
    }
}

impl CaCfar2DConfig {
    pub fn new(
        training_range: usize,
        training_doppler: usize,
        guard_range: usize,
        guard_doppler: usize,
        false_alarm_probability: f64,
    ) -> Result<Self> {
        let value = Self {
            training_range,
            training_doppler,
            guard_range,
            guard_doppler,
            false_alarm_probability,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(self) -> Result<()> {
        if self.training_range == 0 || self.training_doppler == 0 {
            return Err(SdkError::InvalidArgument(
                "CFAR training windows must be non-zero".into(),
            ));
        }
        if !self.false_alarm_probability.is_finite()
            || self.false_alarm_probability <= 0.0
            || self.false_alarm_probability >= 1.0
        {
            return Err(SdkError::InvalidArgument(
                "false_alarm_probability must be finite and in (0,1)".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RangeDopplerDetection {
    doppler_bin: usize,
    range_bin: usize,
    range_m: f64,
    doppler_hz: f64,
    radial_velocity_m_s: Option<f64>,
    power: f32,
    threshold: f32,
    signal_to_threshold_ratio: f32,
}

impl RangeDopplerDetection {
    pub fn doppler_bin(self) -> usize {
        self.doppler_bin
    }

    pub fn range_bin(self) -> usize {
        self.range_bin
    }

    pub fn range_m(self) -> f64 {
        self.range_m
    }

    pub fn doppler_hz(self) -> f64 {
        self.doppler_hz
    }

    pub fn radial_velocity_m_s(self) -> Option<f64> {
        self.radial_velocity_m_s
    }

    pub fn power(self) -> f32 {
        self.power
    }

    pub fn threshold(self) -> f32 {
        self.threshold
    }

    pub fn signal_to_threshold_ratio(self) -> f32 {
        self.signal_to_threshold_ratio
    }
}

pub fn ca_cfar_2d(
    map: &RangeDopplerMap,
    config: CaCfar2DConfig,
) -> Result<Vec<RangeDopplerDetection>> {
    config.validate()?;
    let range_margin = config.training_range + config.guard_range;
    let doppler_margin = config.training_doppler + config.guard_doppler;
    if map.range_bin_count <= 2 * range_margin || map.doppler_bin_count <= 2 * doppler_margin {
        return Err(SdkError::InvalidArgument(
            "range-Doppler map is too small for configured CFAR windows".into(),
        ));
    }

    let total_window_cells = (2 * range_margin + 1) * (2 * doppler_margin + 1);
    let guard_window_cells = (2 * config.guard_range + 1) * (2 * config.guard_doppler + 1);
    let training_cells = total_window_cells.saturating_sub(guard_window_cells);
    if training_cells == 0 {
        return Err(SdkError::InvalidArgument(
            "CFAR configuration has zero training cells".into(),
        ));
    }
    let n = training_cells as f64;
    let alpha = n * (config.false_alarm_probability.powf(-1.0 / n) - 1.0);
    let mut detections = Vec::new();

    for doppler in 0..map.doppler_bin_count {
        for range in range_margin..map.range_bin_count - range_margin {
            let mut noise_sum = 0.0_f64;
            let mut noise_count = 0_usize;
            for doppler_offset in -(doppler_margin as isize)..=(doppler_margin as isize) {
                let d = (doppler as isize + doppler_offset)
                    .rem_euclid(map.doppler_bin_count as isize) as usize;
                for r in range - range_margin..=range + range_margin {
                    let inside_guard = doppler_offset.unsigned_abs() <= config.guard_doppler
                        && r.abs_diff(range) <= config.guard_range;
                    if inside_guard {
                        continue;
                    }
                    if let Some(value) = map.power_at(d, r) {
                        noise_sum += f64::from(value);
                        noise_count = noise_count.saturating_add(1);
                    }
                }
            }
            if noise_count == 0 {
                continue;
            }
            let threshold = (noise_sum / noise_count as f64 * alpha) as f32;
            let Some(cell_power) = map.power_at(doppler, range) else {
                continue;
            };
            if cell_power <= threshold || threshold <= 0.0 {
                continue;
            }
            let radial_velocity_m_s = map
                .radial_velocity_bins_m_s
                .as_ref()
                .and_then(|values| values.get(doppler))
                .copied();
            detections.push(RangeDopplerDetection {
                doppler_bin: doppler,
                range_bin: range,
                range_m: map.range_bins_m[range],
                doppler_hz: map.doppler_bins_hz[doppler],
                radial_velocity_m_s,
                power: cell_power,
                threshold,
                signal_to_threshold_ratio: cell_power / threshold,
            });
        }
    }
    detections.sort_by(|left, right| {
        right
            .signal_to_threshold_ratio
            .total_cmp(&left.signal_to_threshold_ratio)
    });
    Ok(detections)
}

fn validate_capture_grid(reference: &RawIqCapture, actual: &RawIqCapture) -> Result<()> {
    let sample_rate_tolerance = reference.sample_rate_hz().abs().max(1.0) * 1.0e-9;
    if (reference.sample_rate_hz() - actual.sample_rate_hz()).abs() > sample_rate_tolerance {
        return Err(SdkError::InvalidArgument(
            "all range-Doppler captures must use the reference sample rate".into(),
        ));
    }
    let center_frequency_tolerance = reference.center_frequency_hz().abs().max(1.0) * 1.0e-9;
    if (reference.center_frequency_hz() - actual.center_frequency_hz()).abs()
        > center_frequency_tolerance
    {
        return Err(SdkError::InvalidArgument(
            "all range-Doppler captures must use the reference center frequency".into(),
        ));
    }
    Ok(())
}

fn hann(index: usize, len: usize) -> f32 {
    if len <= 1 {
        1.0
    } else {
        (0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / (len - 1) as f64).cos()) as f32
    }
}
