use serde::{Deserialize, Serialize};

use crate::isac::{SensingMode, SensingSession};
use crate::raw_iq::RawIqCapture;
use crate::{Result, SdkError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawSensingFrame {
    timestamp_ms: u64,
    infrastructure_id: String,
    mode: SensingMode,
    capture: RawIqCapture,
}

impl RawSensingFrame {
    pub fn new(
        timestamp_ms: u64,
        infrastructure_id: impl Into<String>,
        mode: SensingMode,
        capture: RawIqCapture,
    ) -> Result<Self> {
        let value = Self {
            timestamp_ms,
            infrastructure_id: infrastructure_id.into(),
            mode,
            capture,
        };
        value.validate()?;
        Ok(value)
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

    pub fn capture(&self) -> &RawIqCapture {
        &self.capture
    }

    pub fn validate(&self) -> Result<()> {
        if self.infrastructure_id.trim().is_empty() || self.infrastructure_id.len() > 256 {
            return Err(SdkError::InvalidArgument(
                "raw sensing infrastructure_id must be non-empty and bounded".into(),
            ));
        }
        self.capture.validate()?;
        if self.capture.timestamp_ms() != self.timestamp_ms {
            return Err(SdkError::InvalidArgument(
                "raw sensing frame timestamp must match capture timestamp".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_against_session(&self, session: &SensingSession) -> Result<()> {
        self.validate()?;
        if self.infrastructure_id != session.infrastructure_id() {
            return Err(SdkError::InvalidArgument(
                "raw sensing frame infrastructure does not match active session".into(),
            ));
        }
        if self.mode != session.mode() {
            return Err(SdkError::InvalidArgument(
                "raw sensing frame mode does not match active session".into(),
            ));
        }
        if self.timestamp_ms < session.valid_from_ms()
            || self.timestamp_ms > session.valid_until_ms()
        {
            return Err(SdkError::InvalidArgument(
                "raw sensing frame timestamp is outside the authorized session window".into(),
            ));
        }
        Ok(())
    }
}
