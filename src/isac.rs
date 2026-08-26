use std::collections::{BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};

use crate::types::MAX_FEATURES;
use crate::{Digest32, Observation, Result, SdkError, VerificationContext};

const MAX_TEXT_BYTES: usize = 256;
const MAX_RECORDED_FRAMES: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SensingMode {
    ChannelState,
    PositioningReference,
    UplinkSounding,
    OperatorManagedActive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensingCapabilities {
    infrastructure_id: String,
    modes: BTreeSet<SensingMode>,
    max_feature_dimension: usize,
}

impl SensingCapabilities {
    pub fn new(
        infrastructure_id: impl Into<String>,
        modes: BTreeSet<SensingMode>,
        max_feature_dimension: usize,
    ) -> Result<Self> {
        let infrastructure_id = infrastructure_id.into();
        validate_text("infrastructure_id", &infrastructure_id)?;
        if modes.is_empty() {
            return Err(SdkError::InvalidArgument(
                "sensing capabilities must include at least one mode".into(),
            ));
        }
        if max_feature_dimension == 0 || max_feature_dimension > MAX_FEATURES {
            return Err(SdkError::DimensionLimit {
                actual: max_feature_dimension,
                max: MAX_FEATURES,
            });
        }
        Ok(Self {
            infrastructure_id,
            modes,
            max_feature_dimension,
        })
    }

    pub fn infrastructure_id(&self) -> &str {
        &self.infrastructure_id
    }
    pub fn modes(&self) -> &BTreeSet<SensingMode> {
        &self.modes
    }
    pub fn max_feature_dimension(&self) -> usize {
        self.max_feature_dimension
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorAuthorization {
    operator_id: String,
    infrastructure_id: String,
    valid_from_ms: u64,
    valid_until_ms: u64,
    allowed_modes: BTreeSet<SensingMode>,
    authorization_digest: Digest32,
}

impl OperatorAuthorization {
    pub fn new(
        operator_id: impl Into<String>,
        infrastructure_id: impl Into<String>,
        valid_from_ms: u64,
        valid_until_ms: u64,
        allowed_modes: BTreeSet<SensingMode>,
        authorization_digest: Digest32,
    ) -> Result<Self> {
        let operator_id = operator_id.into();
        let infrastructure_id = infrastructure_id.into();
        validate_text("operator_id", &operator_id)?;
        validate_text("infrastructure_id", &infrastructure_id)?;
        if valid_until_ms < valid_from_ms {
            return Err(SdkError::InvalidArgument(
                "authorization validity window is inverted".into(),
            ));
        }
        if allowed_modes.is_empty() {
            return Err(SdkError::InvalidArgument(
                "authorization must allow at least one sensing mode".into(),
            ));
        }
        Ok(Self {
            operator_id,
            infrastructure_id,
            valid_from_ms,
            valid_until_ms,
            allowed_modes,
            authorization_digest,
        })
    }

    pub fn operator_id(&self) -> &str {
        &self.operator_id
    }
    pub fn infrastructure_id(&self) -> &str {
        &self.infrastructure_id
    }
    pub fn authorization_digest(&self) -> Digest32 {
        self.authorization_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensingRequest {
    infrastructure_id: String,
    mode: SensingMode,
    feature_dimension: usize,
    configuration_digest: Digest32,
}

impl SensingRequest {
    pub fn new(
        infrastructure_id: impl Into<String>,
        mode: SensingMode,
        feature_dimension: usize,
        configuration_digest: Digest32,
    ) -> Result<Self> {
        let infrastructure_id = infrastructure_id.into();
        validate_text("infrastructure_id", &infrastructure_id)?;
        if feature_dimension == 0 || feature_dimension > MAX_FEATURES {
            return Err(SdkError::DimensionLimit {
                actual: feature_dimension,
                max: MAX_FEATURES,
            });
        }
        Ok(Self {
            infrastructure_id,
            mode,
            feature_dimension,
            configuration_digest,
        })
    }

    pub fn infrastructure_id(&self) -> &str {
        &self.infrastructure_id
    }
    pub fn mode(&self) -> SensingMode {
        self.mode
    }
    pub fn feature_dimension(&self) -> usize {
        self.feature_dimension
    }
    pub fn configuration_digest(&self) -> Digest32 {
        self.configuration_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensingSession {
    session_id: Digest32,
    infrastructure_id: String,
    mode: SensingMode,
    feature_dimension: usize,
    valid_from_ms: u64,
    valid_until_ms: u64,
    authorization_digest: Digest32,
    configuration_digest: Digest32,
}

impl SensingSession {
    pub fn session_id(&self) -> Digest32 {
        self.session_id
    }
    pub fn infrastructure_id(&self) -> &str {
        &self.infrastructure_id
    }
    pub fn mode(&self) -> SensingMode {
        self.mode
    }
    pub fn feature_dimension(&self) -> usize {
        self.feature_dimension
    }
    pub fn valid_from_ms(&self) -> u64 {
        self.valid_from_ms
    }
    pub fn valid_until_ms(&self) -> u64 {
        self.valid_until_ms
    }
    pub fn authorization_digest(&self) -> Digest32 {
        self.authorization_digest
    }
    pub fn configuration_digest(&self) -> Digest32 {
        self.configuration_digest
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensingFrame {
    timestamp_ms: u64,
    infrastructure_id: String,
    mode: SensingMode,
    features: Vec<f32>,
}

impl SensingFrame {
    pub fn new(
        timestamp_ms: u64,
        infrastructure_id: impl Into<String>,
        mode: SensingMode,
        features: Vec<f32>,
    ) -> Result<Self> {
        let infrastructure_id = infrastructure_id.into();
        validate_text("infrastructure_id", &infrastructure_id)?;
        validate_features(&features)?;
        Ok(Self {
            timestamp_ms,
            infrastructure_id,
            mode,
            features,
        })
    }

    pub fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
    pub fn infrastructure_id(&self) -> &str {
        &self.infrastructure_id
    }
    pub fn mode(&self) -> SensingMode {
        self.mode
    }
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    pub fn to_observation(&self, id: impl Into<String>) -> Result<Observation> {
        self.validate()?;
        Observation::new(
            id,
            format!("isac:{}", self.infrastructure_id),
            self.timestamp_ms,
            self.features.clone(),
        )
    }

    pub fn verification_context(
        &self,
        model_digest: Digest32,
        authorization: &OperatorAuthorization,
        request: &SensingRequest,
        ontology_version: impl Into<String>,
        pipeline_version: impl Into<String>,
        seed: u64,
    ) -> Result<VerificationContext> {
        self.validate()?;
        validate_authorized_request(authorization, request, self.timestamp_ms, None)?;
        if self.infrastructure_id != request.infrastructure_id
            || self.mode != request.mode
            || self.features.len() != request.feature_dimension
        {
            return Err(SdkError::InvalidArgument(
                "sensing frame does not match the authorized request".into(),
            ));
        }
        let config_digest = combined_configuration_digest(authorization, request, self);
        VerificationContext::new(
            model_digest,
            config_digest,
            ontology_version,
            pipeline_version,
            seed,
        )
    }

    fn validate(&self) -> Result<()> {
        validate_text("infrastructure_id", &self.infrastructure_id)?;
        validate_features(&self.features)
    }
}

pub trait ActiveSensingProvider: Send {
    fn capabilities(&self) -> &SensingCapabilities;
    fn prepare(
        &mut self,
        authorization: &OperatorAuthorization,
        request: &SensingRequest,
        now_ms: u64,
    ) -> Result<SensingSession>;
    fn acquire(&mut self, session: &SensingSession) -> Result<Option<SensingFrame>>;
    fn stop(&mut self, session: SensingSession) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct RecordedSensingProvider {
    capabilities: SensingCapabilities,
    frames: VecDeque<SensingFrame>,
    active_session: Option<SensingSession>,
}

impl RecordedSensingProvider {
    pub fn new(capabilities: SensingCapabilities, frames: Vec<SensingFrame>) -> Result<Self> {
        validate_capabilities(&capabilities)?;
        if frames.len() > MAX_RECORDED_FRAMES {
            return Err(SdkError::DimensionLimit {
                actual: frames.len(),
                max: MAX_RECORDED_FRAMES,
            });
        }
        for frame in &frames {
            frame.validate()?;
            if frame.infrastructure_id != capabilities.infrastructure_id {
                return Err(SdkError::InvalidArgument(
                    "recorded frame infrastructure does not match provider capabilities".into(),
                ));
            }
            if !capabilities.modes.contains(&frame.mode) {
                return Err(SdkError::InvalidArgument(
                    "recorded frame mode is not supported by provider capabilities".into(),
                ));
            }
            if frame.features.len() > capabilities.max_feature_dimension {
                return Err(SdkError::DimensionLimit {
                    actual: frame.features.len(),
                    max: capabilities.max_feature_dimension,
                });
            }
        }
        Ok(Self {
            capabilities,
            frames: frames.into(),
            active_session: None,
        })
    }
}

impl ActiveSensingProvider for RecordedSensingProvider {
    fn capabilities(&self) -> &SensingCapabilities {
        &self.capabilities
    }

    fn prepare(
        &mut self,
        authorization: &OperatorAuthorization,
        request: &SensingRequest,
        now_ms: u64,
    ) -> Result<SensingSession> {
        validate_capabilities(&self.capabilities)?;
        validate_authorized_request(authorization, request, now_ms, Some(&self.capabilities))?;
        if self.active_session.is_some() {
            return Err(SdkError::InvalidArgument(
                "a sensing session is already active".into(),
            ));
        }
        let session = SensingSession {
            session_id: session_digest(authorization, request, now_ms),
            infrastructure_id: request.infrastructure_id.clone(),
            mode: request.mode,
            feature_dimension: request.feature_dimension,
            valid_from_ms: authorization.valid_from_ms,
            valid_until_ms: authorization.valid_until_ms,
            authorization_digest: authorization.authorization_digest,
            configuration_digest: request.configuration_digest,
        };
        self.active_session = Some(session.clone());
        Ok(session)
    }

    fn acquire(&mut self, session: &SensingSession) -> Result<Option<SensingFrame>> {
        let active = self
            .active_session
            .as_ref()
            .ok_or_else(|| SdkError::InvalidArgument("no active sensing session".into()))?;
        if active != session {
            return Err(SdkError::InvalidArgument(
                "sensing session does not match active session".into(),
            ));
        }
        let Some(frame) = self.frames.pop_front() else {
            return Ok(None);
        };
        if frame.infrastructure_id != session.infrastructure_id || frame.mode != session.mode {
            return Err(SdkError::InvalidArgument(
                "recorded frame does not match active sensing session".into(),
            ));
        }
        if frame.features.len() != session.feature_dimension {
            return Err(SdkError::DimensionMismatch {
                expected: session.feature_dimension,
                actual: frame.features.len(),
            });
        }
        if frame.timestamp_ms < session.valid_from_ms || frame.timestamp_ms > session.valid_until_ms
        {
            return Err(SdkError::InvalidArgument(
                "recorded frame timestamp is outside the authorized sensing window".into(),
            ));
        }
        Ok(Some(frame))
    }

    fn stop(&mut self, session: SensingSession) -> Result<()> {
        let active = self
            .active_session
            .as_ref()
            .ok_or_else(|| SdkError::InvalidArgument("no active sensing session".into()))?;
        if active != &session {
            return Err(SdkError::InvalidArgument(
                "sensing session does not match active session".into(),
            ));
        }
        self.active_session = None;
        Ok(())
    }
}

fn validate_authorized_request(
    authorization: &OperatorAuthorization,
    request: &SensingRequest,
    now_ms: u64,
    capabilities: Option<&SensingCapabilities>,
) -> Result<()> {
    validate_authorization(authorization)?;
    validate_request(request)?;
    if now_ms < authorization.valid_from_ms || now_ms > authorization.valid_until_ms {
        return Err(SdkError::InvalidArgument(
            "operator authorization is not valid at requested time".into(),
        ));
    }
    if authorization.infrastructure_id != request.infrastructure_id {
        return Err(SdkError::InvalidArgument(
            "operator authorization infrastructure does not match sensing request".into(),
        ));
    }
    if !authorization.allowed_modes.contains(&request.mode) {
        return Err(SdkError::InvalidArgument(
            "operator authorization does not allow requested sensing mode".into(),
        ));
    }
    if let Some(value) = capabilities {
        if value.infrastructure_id != request.infrastructure_id {
            return Err(SdkError::InvalidArgument(
                "provider infrastructure does not match sensing request".into(),
            ));
        }
        if !value.modes.contains(&request.mode) {
            return Err(SdkError::InvalidArgument(
                "provider does not support requested sensing mode".into(),
            ));
        }
        if request.feature_dimension > value.max_feature_dimension {
            return Err(SdkError::DimensionLimit {
                actual: request.feature_dimension,
                max: value.max_feature_dimension,
            });
        }
    }
    Ok(())
}

fn validate_capabilities(capabilities: &SensingCapabilities) -> Result<()> {
    SensingCapabilities::new(
        capabilities.infrastructure_id.clone(),
        capabilities.modes.clone(),
        capabilities.max_feature_dimension,
    )
    .map(|_| ())
}

fn validate_authorization(authorization: &OperatorAuthorization) -> Result<()> {
    OperatorAuthorization::new(
        authorization.operator_id.clone(),
        authorization.infrastructure_id.clone(),
        authorization.valid_from_ms,
        authorization.valid_until_ms,
        authorization.allowed_modes.clone(),
        authorization.authorization_digest,
    )
    .map(|_| ())
}

fn validate_request(request: &SensingRequest) -> Result<()> {
    SensingRequest::new(
        request.infrastructure_id.clone(),
        request.mode,
        request.feature_dimension,
        request.configuration_digest,
    )
    .map(|_| ())
}

fn validate_features(features: &[f32]) -> Result<()> {
    if features.is_empty() {
        return Err(SdkError::EmptyFeatures);
    }
    if features.len() > MAX_FEATURES {
        return Err(SdkError::DimensionLimit {
            actual: features.len(),
            max: MAX_FEATURES,
        });
    }
    if let Some((index, _)) = features
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(SdkError::NonFiniteValue { index });
    }
    Ok(())
}

fn validate_text(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(SdkError::InvalidArgument(format!(
            "{field} must not be empty"
        )));
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(SdkError::DimensionLimit {
            actual: value.len(),
            max: MAX_TEXT_BYTES,
        });
    }
    Ok(())
}

fn session_digest(
    authorization: &OperatorAuthorization,
    request: &SensingRequest,
    now_ms: u64,
) -> Digest32 {
    let mut hasher = Sha256::new();
    hasher.update(b"anderion-cuas-isac-session-v1");
    update_bytes(&mut hasher, authorization.operator_id.as_bytes());
    update_bytes(&mut hasher, authorization.infrastructure_id.as_bytes());
    hasher.update(authorization.authorization_digest.as_bytes());
    hasher.update(authorization.valid_from_ms.to_le_bytes());
    hasher.update(authorization.valid_until_ms.to_le_bytes());
    for mode in &authorization.allowed_modes {
        hasher.update([mode_byte(*mode)]);
    }
    hasher.update(request.configuration_digest.as_bytes());
    hasher.update([mode_byte(request.mode)]);
    hasher.update((request.feature_dimension as u64).to_le_bytes());
    hasher.update(now_ms.to_le_bytes());
    finish_digest(hasher)
}

fn combined_configuration_digest(
    authorization: &OperatorAuthorization,
    request: &SensingRequest,
    frame: &SensingFrame,
) -> Digest32 {
    let mut hasher = Sha256::new();
    hasher.update(b"anderion-cuas-isac-verification-config-v1");
    update_bytes(&mut hasher, authorization.operator_id.as_bytes());
    update_bytes(&mut hasher, authorization.infrastructure_id.as_bytes());
    hasher.update(authorization.authorization_digest.as_bytes());
    hasher.update(authorization.valid_from_ms.to_le_bytes());
    hasher.update(authorization.valid_until_ms.to_le_bytes());
    for mode in &authorization.allowed_modes {
        hasher.update([mode_byte(*mode)]);
    }
    hasher.update(request.configuration_digest.as_bytes());
    update_bytes(&mut hasher, request.infrastructure_id.as_bytes());
    hasher.update([mode_byte(request.mode)]);
    hasher.update((request.feature_dimension as u64).to_le_bytes());
    update_bytes(&mut hasher, frame.infrastructure_id.as_bytes());
    hasher.update([mode_byte(frame.mode)]);
    hasher.update((frame.features.len() as u64).to_le_bytes());
    finish_digest(hasher)
}

fn mode_byte(mode: SensingMode) -> u8 {
    match mode {
        SensingMode::ChannelState => 1,
        SensingMode::PositioningReference => 2,
        SensingMode::UplinkSounding => 3,
        SensingMode::OperatorManagedActive => 4,
    }
}

fn update_bytes(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn finish_digest(hasher: Sha256) -> Digest32 {
    let finalized = hasher.finalize();
    let mut bytes = [0_u8; 32];
    bytes.copy_from_slice(&finalized);
    Digest32::from_digest_bytes(bytes)
}
