use std::collections::BTreeSet;

use anderion_cuas_ml::{
    ActiveSensingProvider, ComplexSample, Digest32, OperatorAuthorization, RawIqCapture,
    RawSensingFrame, RecordedSensingProvider, Result, SensingCapabilities, SensingMode,
    SensingRequest,
};

fn session() -> Result<(RecordedSensingProvider, anderion_cuas_ml::SensingSession)> {
    let capabilities = SensingCapabilities::new(
        "ran-site-01",
        BTreeSet::from([SensingMode::ChannelState]),
        16,
    )?;
    let authorization = OperatorAuthorization::new(
        "operator-a",
        "ran-site-01",
        1_000,
        2_000,
        BTreeSet::from([SensingMode::ChannelState]),
        Digest32::from_bytes(b"raw-authorization"),
    )?;
    let request = SensingRequest::new(
        "ran-site-01",
        SensingMode::ChannelState,
        4,
        Digest32::from_bytes(b"raw-config"),
    )?;
    let mut provider = RecordedSensingProvider::new(capabilities, Vec::new())?;
    let session = provider.prepare(&authorization, &request, 1_500)?;
    Ok((provider, session))
}

#[test]
fn raw_sensing_frame_binds_capture_timestamp_and_authorized_session() -> Result<()> {
    let (_provider, session) = session()?;
    let capture = RawIqCapture::new(
        "raw-frame",
        1_500,
        1_000_000.0,
        2_400_000_000.0,
        vec![ComplexSample::new(1.0, 0.0)?; 32],
    )?;
    let frame = RawSensingFrame::new(1_500, "ran-site-01", SensingMode::ChannelState, capture)?;
    frame.validate_against_session(&session)?;
    Ok(())
}

#[test]
fn raw_sensing_frame_rejects_metadata_timestamp_drift() -> Result<()> {
    let capture = RawIqCapture::new(
        "raw-frame-drift",
        1_499,
        1_000_000.0,
        2_400_000_000.0,
        vec![ComplexSample::new(1.0, 0.0)?; 32],
    )?;
    assert!(
        RawSensingFrame::new(1_500, "ran-site-01", SensingMode::ChannelState, capture,).is_err()
    );
    Ok(())
}
