use std::collections::BTreeSet;

use anderion_cuas_ml::{
    ActiveSensingProvider, Digest32, OperatorAuthorization, RecordedSensingProvider, Result,
    SensingCapabilities, SensingFrame, SensingMode, SensingRequest,
};

fn capabilities() -> Result<SensingCapabilities> {
    SensingCapabilities::new(
        "ran-site-01",
        BTreeSet::from([SensingMode::ChannelState, SensingMode::PositioningReference, SensingMode::OperatorManagedActive]),
        16,
    )
}

fn authorization(valid_until_ms: u64) -> Result<OperatorAuthorization> {
    OperatorAuthorization::new(
        "operator-a",
        "ran-site-01",
        1_000,
        valid_until_ms,
        BTreeSet::from([SensingMode::ChannelState, SensingMode::OperatorManagedActive]),
        Digest32::from_bytes(b"signed-operator-authorization-artifact"),
    )
}

#[test]
fn recorded_provider_requires_valid_operator_authorization() -> Result<()> {
    let request = SensingRequest::new(
        "ran-site-01",
        SensingMode::ChannelState,
        4,
        Digest32::from_bytes(b"approved-ran-config"),
    )?;
    let frame = SensingFrame::new(1_500, "ran-site-01", SensingMode::ChannelState, vec![0.1, 0.2, 0.3, 0.4])?;
    let mut provider = RecordedSensingProvider::new(capabilities()?, vec![frame])?;
    let session = provider.prepare(&authorization(2_000)?, &request, 1_500)?;
    let acquired = provider.acquire(&session)?.ok_or_else(|| anderion_cuas_ml::SdkError::InvalidArgument("expected recorded frame".into()))?;
    assert_eq!(acquired.features(), &[0.1, 0.2, 0.3, 0.4]);
    provider.stop(session)?;
    Ok(())
}

#[test]
fn recorded_provider_rejects_expired_wrong_infrastructure_and_unsupported_mode() -> Result<()> {
    let mut provider = RecordedSensingProvider::new(capabilities()?, Vec::new())?;
    let good_request = SensingRequest::new(
        "ran-site-01",
        SensingMode::ChannelState,
        4,
        Digest32::from_bytes(b"cfg"),
    )?;
    assert!(provider.prepare(&authorization(1_100)?, &good_request, 1_500).is_err());

    let wrong_infra = SensingRequest::new(
        "ran-site-02",
        SensingMode::ChannelState,
        4,
        Digest32::from_bytes(b"cfg"),
    )?;
    assert!(provider.prepare(&authorization(2_000)?, &wrong_infra, 1_500).is_err());

    let unsupported = SensingRequest::new(
        "ran-site-01",
        SensingMode::UplinkSounding,
        4,
        Digest32::from_bytes(b"cfg"),
    )?;
    assert!(provider.prepare(&authorization(2_000)?, &unsupported, 1_500).is_err());
    Ok(())
}

#[test]
fn sensing_frame_converts_to_normal_observation_and_binds_verification_context() -> Result<()> {
    let frame = SensingFrame::new(1_500, "ran-site-01", SensingMode::ChannelState, vec![1.0, 2.0])?;
    let observation = frame.to_observation("isac-frame-1")?;
    assert_eq!(observation.sensor_id(), "isac:ran-site-01");

    let authorization = authorization(2_000)?;
    let request = SensingRequest::new(
        "ran-site-01",
        SensingMode::ChannelState,
        2,
        Digest32::from_bytes(b"cfg"),
    )?;
    let context = frame.verification_context(
        Digest32::from_bytes(b"model"),
        &authorization,
        &request,
        "cuas-ontology-v1",
        "perception-v1",
        7,
    )?;
    assert_ne!(context.config_digest(), Digest32::from_bytes(b"cfg"));
    Ok(())
}

#[test]
fn operator_managed_active_mode_requires_explicit_authorization() -> Result<()> {
    let request = SensingRequest::new(
        "ran-site-01",
        SensingMode::OperatorManagedActive,
        2,
        Digest32::from_bytes(b"active-config"),
    )?;
    let frame = SensingFrame::new(1_500, "ran-site-01", SensingMode::OperatorManagedActive, vec![0.2, 0.4])?;
    let mut provider = RecordedSensingProvider::new(capabilities()?, vec![frame])?;
    let session = provider.prepare(&authorization(2_000)?, &request, 1_500)?;
    assert!(provider.acquire(&session)?.is_some());
    provider.stop(session)?;
    Ok(())
}
