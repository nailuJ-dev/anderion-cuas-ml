use serde::{Deserialize, Serialize};

use crate::raw_iq::{ComplexSample, RawIqCapture, samples_for_duration};
use crate::{Result, SdkError};

const MODE_S_PREAMBLE_S: f64 = 8.0e-6;
const MODE_S_PULSE_S: f64 = 0.5e-6;
const MODE_S_BIT_S: f64 = 1.0e-6;
const MODE_S_PREAMBLE_PULSES_S: [f64; 4] = [0.0, 1.0e-6, 3.5e-6, 4.5e-6];
const MODE_S_CRC_POLY: u32 = 0x00FF_F409;
const MODE_S_CRC_MASK: u32 = 0x00FF_FFFF;

const ATCRBS_PULSE_S: f64 = 0.45e-6;
const ATCRBS_F2_S: f64 = 20.3e-6;
const ATCRBS_SPI_S: f64 = 24.65e-6;
const ATCRBS_INFORMATION_OFFSETS_S: [f64; 13] = [
    1.45e-6, 2.90e-6, 4.35e-6, 5.80e-6, 7.25e-6, 8.70e-6, 10.15e-6, 11.60e-6, 13.05e-6, 14.50e-6,
    15.95e-6, 17.40e-6, 18.85e-6,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransponderProtocol {
    ModeS,
    Atcrbs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModeSParityKind {
    DirectCrc,
    AddressParity,
    AddressOrDataParity,
    InterrogatorParity,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModeSParity {
    kind: ModeSParityKind,
    syndrome: u32,
}

impl ModeSParity {
    pub fn kind(self) -> ModeSParityKind {
        self.kind
    }

    pub fn syndrome(self) -> u32 {
        self.syndrome
    }

    pub fn direct_crc_valid(self) -> bool {
        self.kind == ModeSParityKind::DirectCrc && self.syndrome == 0
    }

    pub fn recovered_address(self) -> Option<u32> {
        if self.kind == ModeSParityKind::AddressParity {
            Some(self.syndrome)
        } else {
            None
        }
    }

    pub fn overlay_syndrome(self) -> u32 {
        self.syndrome
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModeSReply {
    start_sample: usize,
    bits: Vec<bool>,
    bytes: Vec<u8>,
    downlink_format: u8,
    parity: ModeSParity,
    confidence: f32,
    estimated_cfo_hz: f64,
}

impl ModeSReply {
    pub fn start_sample(&self) -> usize {
        self.start_sample
    }

    pub fn bits(&self) -> &[bool] {
        &self.bits
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn downlink_format(&self) -> u8 {
        self.downlink_format
    }

    pub fn parity(&self) -> ModeSParity {
        self.parity
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    pub fn estimated_cfo_hz(&self) -> f64 {
        self.estimated_cfo_hz
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtcrbsReply {
    start_sample: usize,
    information_pulses: [bool; 13],
    mode_a_digits: [u8; 4],
    mode_c_gillham_bits: [bool; 11],
    spi: bool,
    confidence: f32,
    estimated_cfo_hz: f64,
}

impl AtcrbsReply {
    pub fn start_sample(&self) -> usize {
        self.start_sample
    }

    pub fn information_pulses(&self) -> &[bool; 13] {
        &self.information_pulses
    }

    pub fn mode_a_digits(&self) -> [u8; 4] {
        self.mode_a_digits
    }

    pub fn mode_c_gillham_bits(&self) -> [bool; 11] {
        self.mode_c_gillham_bits
    }

    pub fn spi(&self) -> bool {
        self.spi
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    pub fn estimated_cfo_hz(&self) -> f64 {
        self.estimated_cfo_hz
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransponderReply {
    ModeS(ModeSReply),
    Atcrbs(AtcrbsReply),
}

impl TransponderReply {
    pub fn protocol(&self) -> TransponderProtocol {
        match self {
            Self::ModeS(_) => TransponderProtocol::ModeS,
            Self::Atcrbs(_) => TransponderProtocol::Atcrbs,
        }
    }

    pub fn start_sample(&self) -> usize {
        match self {
            Self::ModeS(reply) => reply.start_sample,
            Self::Atcrbs(reply) => reply.start_sample,
        }
    }

    pub fn confidence(&self) -> f32 {
        match self {
            Self::ModeS(reply) => reply.confidence,
            Self::Atcrbs(reply) => reply.confidence,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TransponderDegarblerConfig {
    max_replies: usize,
    min_sample_rate_hz: f64,
    min_mode_s_preamble_score: f32,
    min_mode_s_bit_confidence: f32,
    min_atcrbs_score: f32,
    cancellation_fraction: f32,
}

impl Default for TransponderDegarblerConfig {
    fn default() -> Self {
        Self {
            max_replies: 8,
            min_sample_rate_hz: 2_000_000.0,
            min_mode_s_preamble_score: 0.42,
            min_mode_s_bit_confidence: 0.18,
            min_atcrbs_score: 0.42,
            cancellation_fraction: 1.0,
        }
    }
}

impl TransponderDegarblerConfig {
    pub fn new(
        max_replies: usize,
        min_sample_rate_hz: f64,
        min_mode_s_preamble_score: f32,
        min_mode_s_bit_confidence: f32,
        min_atcrbs_score: f32,
        cancellation_fraction: f32,
    ) -> Result<Self> {
        let value = Self {
            max_replies,
            min_sample_rate_hz,
            min_mode_s_preamble_score,
            min_mode_s_bit_confidence,
            min_atcrbs_score,
            cancellation_fraction,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(self) -> Result<()> {
        if self.max_replies == 0 || self.max_replies > 64 {
            return Err(SdkError::InvalidArgument(
                "max_replies must be in 1..=64".into(),
            ));
        }
        if !self.min_sample_rate_hz.is_finite() || self.min_sample_rate_hz < 2_000_000.0 {
            return Err(SdkError::InvalidArgument(
                "min_sample_rate_hz must be finite and at least 2 MHz".into(),
            ));
        }
        for (name, value) in [
            ("min_mode_s_preamble_score", self.min_mode_s_preamble_score),
            ("min_mode_s_bit_confidence", self.min_mode_s_bit_confidence),
            ("min_atcrbs_score", self.min_atcrbs_score),
            ("cancellation_fraction", self.cancellation_fraction),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(SdkError::InvalidArgument(format!(
                    "{name} must be finite and in [0,1]"
                )));
            }
        }
        if self.cancellation_fraction <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "cancellation_fraction must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransponderDegarblingResult {
    replies: Vec<TransponderReply>,
    residual: RawIqCapture,
    initial_mean_power: f64,
    residual_mean_power: f64,
}

impl TransponderDegarblingResult {
    pub fn replies(&self) -> &[TransponderReply] {
        &self.replies
    }

    pub fn residual(&self) -> &RawIqCapture {
        &self.residual
    }

    pub fn initial_mean_power(&self) -> f64 {
        self.initial_mean_power
    }

    pub fn residual_mean_power(&self) -> f64 {
        self.residual_mean_power
    }

    pub fn cancelled_power_fraction(&self) -> f64 {
        if self.initial_mean_power <= f64::EPSILON {
            0.0
        } else {
            (1.0 - self.residual_mean_power / self.initial_mean_power).clamp(0.0, 1.0)
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TransponderDegarbler {
    config: TransponderDegarblerConfig,
}

impl TransponderDegarbler {
    pub fn new(config: TransponderDegarblerConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> TransponderDegarblerConfig {
        self.config
    }

    pub fn separate(&self, capture: &RawIqCapture) -> Result<TransponderDegarblingResult> {
        capture.validate()?;
        self.config.validate()?;
        if capture.sample_rate_hz() < self.config.min_sample_rate_hz {
            return Err(SdkError::InvalidArgument(format!(
                "raw transponder degarbling requires at least {:.0} Hz sample rate",
                self.config.min_sample_rate_hz
            )));
        }

        let initial_mean_power = capture.mean_power();
        let mut residual = capture.samples().to_vec();
        let mut replies = Vec::new();
        let mut blocked = Vec::new();

        for _ in 0..self.config.max_replies {
            let envelope = PowerEnvelope::new(&residual);
            let mode_s = find_best_mode_s(
                &residual,
                capture.sample_rate_hz(),
                &envelope,
                self.config,
                &blocked,
            );
            let atcrbs = find_best_atcrbs(
                &residual,
                capture.sample_rate_hz(),
                &envelope,
                self.config,
                &blocked,
            );
            let Some(mut candidate) = choose_candidate(mode_s, atcrbs) else {
                break;
            };

            let mask = candidate.pulse_mask(residual.len(), capture.sample_rate_hz());
            let estimated_cfo_hz =
                estimate_gated_cfo_hz(&residual, &mask, capture.sample_rate_hz());
            candidate.set_cfo(estimated_cfo_hz);
            cancel_candidate(
                &mut residual,
                &mask,
                capture.sample_rate_hz(),
                estimated_cfo_hz,
                self.config.cancellation_fraction,
            );

            blocked.push((candidate.protocol(), candidate.start_sample()));
            replies.push(candidate.into_reply());
        }

        let residual_capture = capture.with_samples(residual)?;
        let residual_mean_power = residual_capture.mean_power();
        Ok(TransponderDegarblingResult {
            replies,
            residual: residual_capture,
            initial_mean_power,
            residual_mean_power,
        })
    }
}

#[derive(Debug, Clone)]
enum Candidate {
    ModeS(ModeSCandidate),
    Atcrbs(AtcrbsCandidate),
}

impl Candidate {
    fn protocol(&self) -> TransponderProtocol {
        match self {
            Self::ModeS(_) => TransponderProtocol::ModeS,
            Self::Atcrbs(_) => TransponderProtocol::Atcrbs,
        }
    }

    fn start_sample(&self) -> usize {
        match self {
            Self::ModeS(value) => value.reply.start_sample,
            Self::Atcrbs(value) => value.reply.start_sample,
        }
    }

    fn score(&self) -> f32 {
        match self {
            Self::ModeS(value) => value.score,
            Self::Atcrbs(value) => value.score,
        }
    }

    fn signal_power(&self) -> f64 {
        match self {
            Self::ModeS(value) => value.signal_power,
            Self::Atcrbs(value) => value.signal_power,
        }
    }

    fn selection_metric(&self) -> f64 {
        f64::from(self.score()) * self.signal_power().max(0.0).sqrt()
    }

    fn set_cfo(&mut self, cfo_hz: f64) {
        match self {
            Self::ModeS(value) => value.reply.estimated_cfo_hz = cfo_hz,
            Self::Atcrbs(value) => value.reply.estimated_cfo_hz = cfo_hz,
        }
    }

    fn pulse_mask(&self, len: usize, sample_rate_hz: f64) -> Vec<bool> {
        match self {
            Self::ModeS(value) => mode_s_pulse_mask(&value.reply, len, sample_rate_hz),
            Self::Atcrbs(value) => atcrbs_pulse_mask(&value.reply, len, sample_rate_hz),
        }
    }

    fn into_reply(self) -> TransponderReply {
        match self {
            Self::ModeS(value) => TransponderReply::ModeS(value.reply),
            Self::Atcrbs(value) => TransponderReply::Atcrbs(value.reply),
        }
    }
}

#[derive(Debug, Clone)]
struct ModeSCandidate {
    reply: ModeSReply,
    score: f32,
    signal_power: f64,
}

#[derive(Debug, Clone)]
struct AtcrbsCandidate {
    reply: AtcrbsReply,
    score: f32,
    signal_power: f64,
}

fn choose_candidate(mode_s: Option<Candidate>, atcrbs: Option<Candidate>) -> Option<Candidate> {
    match (mode_s, atcrbs) {
        (Some(left), Some(right)) => {
            if left.selection_metric() >= right.selection_metric() {
                Some(left)
            } else {
                Some(right)
            }
        }
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

struct PowerEnvelope {
    prefix: Vec<f64>,
}

impl PowerEnvelope {
    fn new(samples: &[ComplexSample]) -> Self {
        let mut prefix = Vec::with_capacity(samples.len() + 1);
        prefix.push(0.0);
        let mut total = 0.0;
        for sample in samples {
            total += f64::from(sample.norm_sqr());
            prefix.push(total);
        }
        Self { prefix }
    }

    fn mean(&self, start: usize, end: usize) -> f64 {
        if start >= end || end >= self.prefix.len() {
            return 0.0;
        }
        (self.prefix[end] - self.prefix[start]) / (end - start) as f64
    }

    fn interval_mean(
        &self,
        start_sample: usize,
        sample_rate_hz: f64,
        offset_s: f64,
        width_s: f64,
    ) -> f64 {
        let start =
            start_sample.saturating_add((offset_s * sample_rate_hz).round().max(0.0) as usize);
        let width = samples_for_duration(sample_rate_hz, width_s);
        let max_sample = self.prefix.len().saturating_sub(1);
        let end = start.saturating_add(width).min(max_sample);
        self.mean(start, end)
    }
}

fn find_best_mode_s(
    samples: &[ComplexSample],
    sample_rate_hz: f64,
    envelope: &PowerEnvelope,
    config: TransponderDegarblerConfig,
    blocked: &[(TransponderProtocol, usize)],
) -> Option<Candidate> {
    let min_len = samples_for_duration(sample_rate_hz, MODE_S_PREAMBLE_S + 56.0e-6);
    if samples.len() < min_len {
        return None;
    }
    let max_start = samples.len().saturating_sub(min_len);
    let mut best: Option<ModeSCandidate> = None;
    for start in 0..=max_start {
        if is_blocked(blocked, TransponderProtocol::ModeS, start, sample_rate_hz) {
            continue;
        }
        let preamble_score = mode_s_preamble_score(envelope, start, sample_rate_hz);
        if preamble_score < config.min_mode_s_preamble_score {
            continue;
        }
        let Some((bits, bit_confidence)) =
            decode_mode_s_bits(envelope, start, sample_rate_hz, samples.len())
        else {
            continue;
        };
        if bit_confidence < config.min_mode_s_bit_confidence {
            continue;
        }
        let bytes = bits_to_bytes(&bits);
        if bytes.is_empty() {
            continue;
        }
        let raw_downlink_format = bytes[0] >> 3;
        let downlink_format = if raw_downlink_format >= 24 {
            24
        } else {
            raw_downlink_format
        };
        let parity_kind = mode_s_parity_kind(downlink_format);
        let syndrome = match parity_kind {
            ModeSParityKind::DirectCrc => mode_s_crc_syndrome(&bytes),
            ModeSParityKind::AddressParity
            | ModeSParityKind::AddressOrDataParity
            | ModeSParityKind::InterrogatorParity => {
                mode_s_overlay_syndrome(&bytes).unwrap_or(MODE_S_CRC_MASK)
            }
            ModeSParityKind::Unknown => mode_s_crc_syndrome(&bytes),
        };
        let parity = ModeSParity {
            kind: parity_kind,
            syndrome,
        };
        let base_score = (0.55 * preamble_score + 0.45 * bit_confidence).clamp(0.0, 1.0);
        let score = match parity.kind {
            ModeSParityKind::DirectCrc => {
                if parity.syndrome != 0 {
                    continue;
                }
                base_score
            }
            ModeSParityKind::AddressParity
            | ModeSParityKind::AddressOrDataParity
            | ModeSParityKind::InterrogatorParity => {
                if preamble_score < 0.70 || bit_confidence < 0.55 {
                    continue;
                }
                base_score * 0.92
            }
            ModeSParityKind::Unknown => continue,
        };
        let signal_power = mode_s_preamble_signal_power(envelope, start, sample_rate_hz);
        let candidate = ModeSCandidate {
            reply: ModeSReply {
                start_sample: start,
                bits,
                bytes,
                downlink_format,
                parity,
                confidence: score,
                estimated_cfo_hz: 0.0,
            },
            score,
            signal_power,
        };
        let replace = match best.as_ref() {
            Some(current) => {
                f64::from(candidate.score) * candidate.signal_power.max(0.0).sqrt()
                    > f64::from(current.score) * current.signal_power.max(0.0).sqrt()
            }
            None => true,
        };
        if replace {
            best = Some(candidate);
        }
    }
    best.map(Candidate::ModeS)
}

fn mode_s_preamble_signal_power(
    envelope: &PowerEnvelope,
    start: usize,
    sample_rate_hz: f64,
) -> f64 {
    MODE_S_PREAMBLE_PULSES_S
        .iter()
        .map(|offset| envelope.interval_mean(start, sample_rate_hz, *offset, MODE_S_PULSE_S))
        .sum::<f64>()
        / MODE_S_PREAMBLE_PULSES_S.len() as f64
}

fn mode_s_preamble_score(envelope: &PowerEnvelope, start: usize, sample_rate_hz: f64) -> f32 {
    let pulse = mode_s_preamble_signal_power(envelope, start, sample_rate_hz);
    let quiet = [
        (0.5e-6, 0.5e-6),
        (1.5e-6, 1.5e-6),
        (4.0e-6, 0.5e-6),
        (5.0e-6, 2.5e-6),
    ]
    .iter()
    .map(|(offset, width)| envelope.interval_mean(start, sample_rate_hz, *offset, *width))
    .sum::<f64>()
        / 4.0;
    contrast_score(pulse, quiet)
}

fn decode_mode_s_bits(
    envelope: &PowerEnvelope,
    start: usize,
    sample_rate_hz: f64,
    sample_len: usize,
) -> Option<(Vec<bool>, f32)> {
    let mut first_five = Vec::with_capacity(5);
    let mut initial_confidence = 0.0_f64;
    for bit_index in 0..5 {
        let (bit, confidence) = decode_ppm_bit(envelope, start, sample_rate_hz, bit_index)?;
        first_five.push(bit);
        initial_confidence += f64::from(confidence);
    }
    let df = bits_to_u8(&first_five);
    let bit_count = if df & 0x10 != 0 { 112 } else { 56 };
    let required = samples_for_duration(
        sample_rate_hz,
        MODE_S_PREAMBLE_S + bit_count as f64 * MODE_S_BIT_S,
    );
    if start.saturating_add(required) > sample_len {
        return None;
    }

    let mut bits = first_five;
    let mut confidence_sum = initial_confidence;
    for bit_index in 5..bit_count {
        let (bit, confidence) = decode_ppm_bit(envelope, start, sample_rate_hz, bit_index)?;
        bits.push(bit);
        confidence_sum += f64::from(confidence);
    }
    Some((bits, (confidence_sum / bit_count as f64) as f32))
}

fn decode_ppm_bit(
    envelope: &PowerEnvelope,
    start: usize,
    sample_rate_hz: f64,
    bit_index: usize,
) -> Option<(bool, f32)> {
    let bit_offset = MODE_S_PREAMBLE_S + bit_index as f64 * MODE_S_BIT_S;
    let first = envelope.interval_mean(start, sample_rate_hz, bit_offset, MODE_S_PULSE_S);
    let second = envelope.interval_mean(
        start,
        sample_rate_hz,
        bit_offset + MODE_S_PULSE_S,
        MODE_S_PULSE_S,
    );
    let total = first + second;
    if total <= f64::EPSILON {
        return None;
    }
    let confidence = ((first - second).abs() / total).clamp(0.0, 1.0) as f32;
    Some((first >= second, confidence))
}

fn bits_to_u8(bits: &[bool]) -> u8 {
    bits.iter()
        .take(8)
        .fold(0_u8, |acc, bit| (acc << 1) | u8::from(*bit))
}

fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| {
            chunk
                .iter()
                .fold(0_u8, |acc, bit| (acc << 1) | u8::from(*bit))
        })
        .collect()
}

fn mode_s_parity_kind(df: u8) -> ModeSParityKind {
    match df {
        17..=19 => ModeSParityKind::DirectCrc,
        0 | 4 | 5 | 16 | 24 => ModeSParityKind::AddressParity,
        20 | 21 => ModeSParityKind::AddressOrDataParity,
        11 => ModeSParityKind::InterrogatorParity,
        _ => ModeSParityKind::Unknown,
    }
}

pub fn mode_s_crc_syndrome(bytes: &[u8]) -> u32 {
    let mut remainder = 0_u32;
    for byte in bytes {
        for bit_index in (0..8).rev() {
            let incoming = u32::from((byte >> bit_index) & 1);
            let top = (remainder >> 23) & 1;
            remainder = (remainder << 1) & MODE_S_CRC_MASK;
            if top ^ incoming != 0 {
                remainder ^= MODE_S_CRC_POLY;
            }
        }
    }
    remainder & MODE_S_CRC_MASK
}

fn mode_s_overlay_syndrome(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 4 {
        return None;
    }

    let payload_len = bytes.len() - 3;

    let expected_parity = mode_s_crc_syndrome(&bytes[..payload_len]);

    let transmitted_parity = (u32::from(bytes[payload_len]) << 16)
        | (u32::from(bytes[payload_len + 1]) << 8)
        | u32::from(bytes[payload_len + 2]);

    Some((expected_parity ^ transmitted_parity) & MODE_S_CRC_MASK)
}

fn find_best_atcrbs(
    samples: &[ComplexSample],
    sample_rate_hz: f64,
    envelope: &PowerEnvelope,
    config: TransponderDegarblerConfig,
    blocked: &[(TransponderProtocol, usize)],
) -> Option<Candidate> {
    let required = samples_for_duration(sample_rate_hz, ATCRBS_F2_S + ATCRBS_PULSE_S);
    if samples.len() < required {
        return None;
    }
    let max_start = samples.len().saturating_sub(required);
    let mut best: Option<AtcrbsCandidate> = None;
    for start in 0..=max_start {
        if is_blocked(blocked, TransponderProtocol::Atcrbs, start, sample_rate_hz) {
            continue;
        }
        let f1 = envelope.interval_mean(start, sample_rate_hz, 0.0, ATCRBS_PULSE_S);
        let f2 = envelope.interval_mean(start, sample_rate_hz, ATCRBS_F2_S, ATCRBS_PULSE_S);
        let pre_width = samples_for_duration(sample_rate_hz, 1.8e-6);
        let pre_start = start.saturating_sub(pre_width);
        let pre_quiet = envelope.mean(pre_start, start);
        let quiet = (envelope.interval_mean(start, sample_rate_hz, 0.65e-6, 0.45e-6)
            + envelope.interval_mean(start, sample_rate_hz, 19.45e-6, 0.45e-6)
            + pre_quiet)
            / 3.0;
        let framing = f1.min(f2);
        let framing_score = contrast_score(framing, quiet);
        if framing_score < config.min_atcrbs_score {
            continue;
        }
        let threshold = (quiet * 3.0).max(0.22 * 0.5 * (f1 + f2));
        let mut information_pulses = [false; 13];
        let mut confidence_sum = 0.0_f64;
        for (index, offset) in ATCRBS_INFORMATION_OFFSETS_S.iter().enumerate() {
            let energy = envelope.interval_mean(start, sample_rate_hz, *offset, ATCRBS_PULSE_S);
            information_pulses[index] = energy >= threshold;
            confidence_sum += if energy + threshold <= f64::EPSILON {
                0.0
            } else {
                (energy - threshold).abs() / (energy + threshold)
            };
        }
        if information_pulses[6] {
            // The X position is reserved and is not used in Mode A/C replies.
            continue;
        }
        let spi_energy =
            envelope.interval_mean(start, sample_rate_hz, ATCRBS_SPI_S, ATCRBS_PULSE_S);
        let spi = spi_energy >= threshold;
        let slot_confidence = (confidence_sum / 13.0).clamp(0.0, 1.0) as f32;
        let score = (0.70 * framing_score + 0.30 * slot_confidence).clamp(0.0, 1.0);
        let signal_power = 0.5 * (f1 + f2);
        let candidate = AtcrbsCandidate {
            reply: AtcrbsReply {
                start_sample: start,
                information_pulses,
                mode_a_digits: decode_mode_a_digits(&information_pulses),
                mode_c_gillham_bits: extract_mode_c_gillham_bits(&information_pulses),
                spi,
                confidence: score,
                estimated_cfo_hz: 0.0,
            },
            score,
            signal_power,
        };
        let replace = match best.as_ref() {
            Some(current) => {
                f64::from(candidate.score) * candidate.signal_power.max(0.0).sqrt()
                    > f64::from(current.score) * current.signal_power.max(0.0).sqrt()
            }
            None => true,
        };
        if replace {
            best = Some(candidate);
        }
    }
    best.map(Candidate::Atcrbs)
}

fn extract_mode_c_gillham_bits(bits: &[bool; 13]) -> [bool; 11] {
    // Order: C1, A1, C2, A2, C4, A4, B1, B2, D2, B4, D4.
    // X and D1 are excluded because they are not used as Mode C altitude bits.
    [
        bits[0], bits[1], bits[2], bits[3], bits[4], bits[5], bits[7], bits[9], bits[10], bits[11],
        bits[12],
    ]
}

fn decode_mode_a_digits(bits: &[bool; 13]) -> [u8; 4] {
    let c = u8::from(bits[0]) + 2 * u8::from(bits[2]) + 4 * u8::from(bits[4]);
    let a = u8::from(bits[1]) + 2 * u8::from(bits[3]) + 4 * u8::from(bits[5]);
    let b = u8::from(bits[7]) + 2 * u8::from(bits[9]) + 4 * u8::from(bits[11]);
    let d = u8::from(bits[8]) + 2 * u8::from(bits[10]) + 4 * u8::from(bits[12]);
    [a, b, c, d]
}

fn contrast_score(signal: f64, background: f64) -> f32 {
    let total = signal + background;
    if total <= f64::EPSILON || signal <= background {
        0.0
    } else {
        ((signal - background) / total).clamp(0.0, 1.0) as f32
    }
}

fn is_blocked(
    blocked: &[(TransponderProtocol, usize)],
    protocol: TransponderProtocol,
    start: usize,
    sample_rate_hz: f64,
) -> bool {
    let tolerance = samples_for_duration(sample_rate_hz, 0.75e-6);
    blocked.iter().any(|(blocked_protocol, blocked_start)| {
        *blocked_protocol == protocol && blocked_start.abs_diff(start) <= tolerance
    })
}

fn mode_s_pulse_mask(reply: &ModeSReply, len: usize, sample_rate_hz: f64) -> Vec<bool> {
    let mut mask = vec![false; len];
    for offset in MODE_S_PREAMBLE_PULSES_S {
        mark_interval(
            &mut mask,
            reply.start_sample,
            sample_rate_hz,
            offset,
            MODE_S_PULSE_S,
        );
    }
    for (index, bit) in reply.bits.iter().enumerate() {
        let offset = MODE_S_PREAMBLE_S
            + index as f64 * MODE_S_BIT_S
            + if *bit { 0.0 } else { MODE_S_PULSE_S };
        mark_interval(
            &mut mask,
            reply.start_sample,
            sample_rate_hz,
            offset,
            MODE_S_PULSE_S,
        );
    }
    mask
}

fn atcrbs_pulse_mask(reply: &AtcrbsReply, len: usize, sample_rate_hz: f64) -> Vec<bool> {
    let mut mask = vec![false; len];
    mark_interval(
        &mut mask,
        reply.start_sample,
        sample_rate_hz,
        0.0,
        ATCRBS_PULSE_S,
    );
    mark_interval(
        &mut mask,
        reply.start_sample,
        sample_rate_hz,
        ATCRBS_F2_S,
        ATCRBS_PULSE_S,
    );
    for (present, offset) in reply
        .information_pulses
        .iter()
        .zip(ATCRBS_INFORMATION_OFFSETS_S)
    {
        if *present {
            mark_interval(
                &mut mask,
                reply.start_sample,
                sample_rate_hz,
                offset,
                ATCRBS_PULSE_S,
            );
        }
    }
    if reply.spi {
        mark_interval(
            &mut mask,
            reply.start_sample,
            sample_rate_hz,
            ATCRBS_SPI_S,
            ATCRBS_PULSE_S,
        );
    }
    mask
}

fn mark_interval(
    mask: &mut [bool],
    start_sample: usize,
    sample_rate_hz: f64,
    offset_s: f64,
    width_s: f64,
) {
    let start = start_sample.saturating_add((offset_s * sample_rate_hz).round() as usize);
    let end = start
        .saturating_add(samples_for_duration(sample_rate_hz, width_s))
        .min(mask.len());
    for value in mask.iter_mut().take(end).skip(start) {
        *value = true;
    }
}

fn estimate_gated_cfo_hz(samples: &[ComplexSample], mask: &[bool], sample_rate_hz: f64) -> f64 {
    let mut phase_steps = Vec::new();
    for index in 1..samples.len() {
        if !mask[index] || !mask[index - 1] {
            continue;
        }
        let previous = samples[index - 1];
        let current = samples[index];
        let real = f64::from(previous.i()) * f64::from(current.i())
            + f64::from(previous.q()) * f64::from(current.q());
        let imag = f64::from(previous.i()) * f64::from(current.q())
            - f64::from(previous.q()) * f64::from(current.i());
        if real.abs() + imag.abs() > f64::EPSILON {
            phase_steps.push(imag.atan2(real));
        }
    }
    median_f64(&mut phase_steps)
        .map(|phase_step| phase_step * sample_rate_hz / std::f64::consts::TAU)
        .unwrap_or(0.0)
}

fn cancel_candidate(
    residual: &mut [ComplexSample],
    mask: &[bool],
    sample_rate_hz: f64,
    cfo_hz: f64,
    cancellation_fraction: f32,
) {
    let omega = std::f64::consts::TAU * cfo_hz / sample_rate_hz;
    let mut corrected_i = Vec::new();
    let mut corrected_q = Vec::new();
    let mut gated_indices = Vec::new();
    for (index, sample) in residual.iter().enumerate() {
        if !mask[index] {
            continue;
        }
        let phase = omega * index as f64;
        let (sin, cos) = phase.sin_cos();
        corrected_i.push(f64::from(sample.i()) * cos + f64::from(sample.q()) * sin);
        corrected_q.push(f64::from(sample.q()) * cos - f64::from(sample.i()) * sin);
        gated_indices.push(index);
    }
    let Some(gain_i) = median_f64(&mut corrected_i) else {
        return;
    };
    let Some(gain_q) = median_f64(&mut corrected_q) else {
        return;
    };
    let scale = f64::from(cancellation_fraction);
    let gain_i = gain_i * scale;
    let gain_q = gain_q * scale;

    for index in gated_indices {
        let sample = residual[index];
        let phase = omega * index as f64;
        let (sin, cos) = phase.sin_cos();
        let model_i = gain_i * cos - gain_q * sin;
        let model_q = gain_i * sin + gain_q * cos;
        residual[index] = ComplexSample::from_parts_unchecked(
            (f64::from(sample.i()) - model_i) as f32,
            (f64::from(sample.q()) - model_q) as f32,
        );
    }
}

fn median_f64(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        Some(0.5 * (values[middle - 1] + values[middle]))
    } else {
        Some(values[middle])
    }
}
