use anderion_cuas_ml::{
    ComplexSample, ModeSParityKind, RawIqCapture, Result, TransponderDegarbler,
    TransponderProtocol, TransponderReply, mode_s_crc_syndrome,
};

const FS: f64 = 8_000_000.0;
const FC: f64 = 1_090_000_000.0;

fn add_pulse(
    samples: &mut [ComplexSample],
    start: usize,
    offset_s: f64,
    width_s: f64,
    amplitude: f32,
    cfo_hz: f64,
) {
    let first = start + (offset_s * FS).round() as usize;
    let width = (width_s * FS).round().max(1.0) as usize;
    let end = first.saturating_add(width).min(samples.len());
    for (index, sample) in samples.iter_mut().enumerate().take(end).skip(first) {
        let phase = std::f64::consts::TAU * cfo_hz * index as f64 / FS;
        let i = sample.i() + amplitude * phase.cos() as f32;
        let q = sample.q() + amplitude * phase.sin() as f32;
        *sample = ComplexSample::from_parts_unchecked(i, q);
    }
}

fn bits_from_bytes(bytes: &[u8]) -> Vec<bool> {
    bytes
        .iter()
        .flat_map(|byte| (0..8).rev().map(move |bit| ((byte >> bit) & 1) != 0))
        .collect()
}

fn add_mode_s(
    samples: &mut [ComplexSample],
    start_us: f64,
    bytes: &[u8],
    amplitude: f32,
    cfo_hz: f64,
) {
    let start = (start_us * 1.0e-6 * FS).round() as usize;
    for offset_us in [0.0_f64, 1.0, 3.5, 4.5] {
        add_pulse(
            samples,
            start,
            offset_us * 1.0e-6,
            0.5e-6,
            amplitude,
            cfo_hz,
        );
    }
    for (index, bit) in bits_from_bytes(bytes).iter().enumerate() {
        let offset_s = 8.0e-6 + index as f64 * 1.0e-6 + if *bit { 0.0 } else { 0.5e-6 };
        add_pulse(samples, start, offset_s, 0.5e-6, amplitude, cfo_hz);
    }
}

fn add_atcrbs(
    samples: &mut [ComplexSample],
    start_us: f64,
    digits: [u8; 4],
    amplitude: f32,
    cfo_hz: f64,
    spi: bool,
) {
    let start = (start_us * 1.0e-6 * FS).round() as usize;
    add_pulse(samples, start, 0.0, 0.45e-6, amplitude, cfo_hz);
    add_pulse(samples, start, 20.3e-6, 0.45e-6, amplitude, cfo_hz);
    let [a, b, c, d] = digits;
    let pulses = [
        (1.45, c & 1 != 0),
        (2.90, a & 1 != 0),
        (4.35, c & 2 != 0),
        (5.80, a & 2 != 0),
        (7.25, c & 4 != 0),
        (8.70, a & 4 != 0),
        (10.15, false),
        (11.60, b & 1 != 0),
        (13.05, d & 1 != 0),
        (14.50, b & 2 != 0),
        (15.95, d & 2 != 0),
        (17.40, b & 4 != 0),
        (18.85, d & 4 != 0),
    ];
    for (offset_us, present) in pulses {
        if present {
            add_pulse(
                samples,
                start,
                offset_us * 1.0e-6,
                0.45e-6,
                amplitude,
                cfo_hz,
            );
        }
    }
    if spi {
        add_pulse(samples, start, 24.65e-6, 0.45e-6, amplitude, cfo_hz);
    }
}

fn capture(samples: Vec<ComplexSample>) -> Result<RawIqCapture> {
    RawIqCapture::new("transponder-test", 0, FS, FC, samples)
}

#[test]
fn known_df17_frame_has_zero_crc_syndrome() {
    let bytes = hex_bytes("8D40621D58C382D690C8AC2863A7");
    assert_eq!(mode_s_crc_syndrome(&bytes), 0);
}

#[test]
fn decodes_isolated_mode_s_df17_from_raw_iq() -> Result<()> {
    let bytes = hex_bytes("8D40621D58C382D690C8AC2863A7");
    let mut samples = vec![ComplexSample::default(); (180.0e-6 * FS) as usize];
    add_mode_s(&mut samples, 20.0, &bytes, 1.0, 35_000.0);
    let result = TransponderDegarbler::default().separate(&capture(samples)?)?;
    let reply = result
        .replies()
        .iter()
        .find_map(|reply| match reply {
            TransponderReply::ModeS(value) => Some(value),
            TransponderReply::Atcrbs(_) => None,
        })
        .ok_or_else(|| {
            anderion_cuas_ml::SdkError::InvalidArgument("missing Mode S reply".into())
        })?;
    assert_eq!(reply.bytes(), bytes.as_slice());
    assert_eq!(reply.downlink_format(), 17);
    assert_eq!(reply.parity().kind(), ModeSParityKind::DirectCrc);
    assert!(reply.parity().direct_crc_valid());
    assert!((reply.estimated_cfo_hz() - 35_000.0).abs() < 8_000.0);
    Ok(())
}

#[test]
fn successive_interference_cancellation_recovers_overlapping_mode_s_replies() -> Result<()> {
    let bytes = hex_bytes("8D40621D58C382D690C8AC2863A7");
    let mut samples = vec![ComplexSample::default(); (260.0e-6 * FS) as usize];
    add_mode_s(&mut samples, 20.0, &bytes, 1.0, 22_000.0);
    add_mode_s(&mut samples, 82.0, &bytes, 0.42, -31_000.0);
    let result = TransponderDegarbler::default().separate(&capture(samples)?)?;
    let mode_s = result
        .replies()
        .iter()
        .filter(|reply| reply.protocol() == TransponderProtocol::ModeS)
        .count();
    assert!(mode_s >= 2, "replies={:?}", result.replies());
    assert!(result.cancelled_power_fraction() > 0.25);
    Ok(())
}

#[test]
fn decodes_atcrbs_mode_a_slots_and_spi_from_raw_iq() -> Result<()> {
    let mut samples = vec![ComplexSample::default(); (100.0e-6 * FS) as usize];
    add_atcrbs(&mut samples, 25.0, [1, 2, 3, 4], 1.0, 18_000.0, true);
    let result = TransponderDegarbler::default().separate(&capture(samples)?)?;
    let reply = result
        .replies()
        .iter()
        .find_map(|reply| match reply {
            TransponderReply::Atcrbs(value) => Some(value),
            TransponderReply::ModeS(_) => None,
        })
        .ok_or_else(|| {
            anderion_cuas_ml::SdkError::InvalidArgument("missing ATCRBS reply".into())
        })?;
    assert_eq!(reply.mode_a_digits(), [1, 2, 3, 4]);
    assert_eq!(
        reply.mode_c_gillham_bits(),
        [
            true, true, true, false, false, false, false, true, false, false, true
        ]
    );
    assert!(reply.spi());
    assert!((reply.estimated_cfo_hz() - 18_000.0).abs() < 8_000.0);
    Ok(())
}

#[test]
fn mixed_mode_s_and_atcrbs_overlap_are_recovered_as_distinct_protocols() -> Result<()> {
    let bytes = hex_bytes("8D40621D58C382D690C8AC2863A7");
    let mut samples = vec![ComplexSample::default(); (220.0e-6 * FS) as usize];
    add_mode_s(&mut samples, 20.0, &bytes, 1.0, 20_000.0);
    add_atcrbs(&mut samples, 62.0, [7, 0, 0, 1], 0.55, -15_000.0, false);
    let result = TransponderDegarbler::default().separate(&capture(samples)?)?;
    assert!(
        result
            .replies()
            .iter()
            .any(|reply| reply.protocol() == TransponderProtocol::ModeS)
    );
    assert!(
        result
            .replies()
            .iter()
            .any(|reply| reply.protocol() == TransponderProtocol::Atcrbs)
    );
    Ok(())
}

fn hex_bytes(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_nibble(pair[0]);
            let low = hex_nibble(pair[1]);
            (high << 4) | low
        })
        .collect()
}

fn hex_nibble(value: u8) -> u8 {
    match value {
        b'0'..=b'9' => value - b'0',
        b'A'..=b'F' => value - b'A' + 10,
        b'a'..=b'f' => value - b'a' + 10,
        _ => 0,
    }
}

#[test]
fn df20_parity_is_not_misreported_as_a_recovered_icao_address() -> Result<()> {
    let bytes = hex_bytes("A0001838CA380031440000F24177");
    let mut samples = vec![ComplexSample::default(); (180.0e-6 * FS) as usize];
    add_mode_s(&mut samples, 20.0, &bytes, 1.0, 0.0);
    let result = TransponderDegarbler::default().separate(&capture(samples)?)?;
    let reply = result
        .replies()
        .iter()
        .find_map(|reply| match reply {
            TransponderReply::ModeS(value) if value.downlink_format() == 20 => Some(value),
            _ => None,
        })
        .ok_or_else(|| anderion_cuas_ml::SdkError::InvalidArgument("missing DF20 reply".into()))?;
    assert_eq!(reply.parity().kind(), ModeSParityKind::AddressOrDataParity);
    assert!(reply.parity().recovered_address().is_none());
    assert_ne!(reply.parity().overlay_syndrome(), 0);
    Ok(())
}

#[test]
fn df24_family_is_normalized_and_keeps_address_parity_semantics() -> Result<()> {
    let address = 0xABCDEF_u32;
    let mut bytes = vec![0_u8; 14];
    bytes[0] = 0xC8; // First two DF bits are 11; remaining control bits are part of DF24.
    bytes[1..11].copy_from_slice(&[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x55, 0xAA]);
    let crc = mode_s_crc_syndrome(&bytes[..11]);
    let ap = crc ^ address;
    bytes[11] = ((ap >> 16) & 0xFF) as u8;
    bytes[12] = ((ap >> 8) & 0xFF) as u8;
    bytes[13] = (ap & 0xFF) as u8;
    let transmitted_parity =
        (u32::from(bytes[11]) << 16) | (u32::from(bytes[12]) << 8) | u32::from(bytes[13]);
    assert_eq!(crc ^ transmitted_parity, address);
    let mut samples = vec![ComplexSample::default(); (180.0e-6 * FS) as usize];
    add_mode_s(&mut samples, 20.0, &bytes, 1.0, 0.0);
    let result = TransponderDegarbler::default().separate(&capture(samples)?)?;
    let reply = result
        .replies()
        .iter()
        .find_map(|reply| match reply {
            TransponderReply::ModeS(value) if value.downlink_format() == 24 => Some(value),
            _ => None,
        })
        .ok_or_else(|| anderion_cuas_ml::SdkError::InvalidArgument("missing DF24 reply".into()))?;
    assert_eq!(reply.parity().kind(), ModeSParityKind::AddressParity);
    assert_eq!(reply.parity().recovered_address(), Some(address));
    Ok(())
}

#[test]
fn sic_prioritizes_the_stronger_valid_reply_even_when_it_arrives_later() -> Result<()> {
    let bytes = hex_bytes("8D40621D58C382D690C8AC2863A7");
    let mut samples = vec![ComplexSample::default(); (260.0e-6 * FS) as usize];
    add_mode_s(&mut samples, 20.0, &bytes, 0.35, 12_000.0);
    add_mode_s(&mut samples, 82.0, &bytes, 1.0, -18_000.0);
    let result = TransponderDegarbler::default().separate(&capture(samples)?)?;
    let first_mode_s = result
        .replies()
        .iter()
        .find_map(|reply| match reply {
            TransponderReply::ModeS(value) => Some(value),
            _ => None,
        })
        .ok_or_else(|| {
            anderion_cuas_ml::SdkError::InvalidArgument("missing Mode S reply".into())
        })?;
    let expected_start = (82.0e-6 * FS).round() as usize;
    assert!(first_mode_s.start_sample().abs_diff(expected_start) <= 2);
    Ok(())
}
