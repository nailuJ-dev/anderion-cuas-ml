use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{
    ActiveSensingProvider, CandidateKinematics, CooperativeTrack, Digest32,
    Observation, OperatorAuthorization, RecordedSensingProvider, Result, SdkError,
    SensingCapabilities, SensingFrame, SensingMode, SensingRequest, VerificationContext,
};

pub const REFERENCE_CUAS_FEATURE_DIM: usize = 13;
const MAX_SCENARIO_TRACKS: usize = 4_096;
const MAX_TEXT_BYTES: usize = 4_096;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceSensorMeasurements {
    range_m: f64,
    radial_velocity_mps: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
    snr_db: f32,
    doppler_spread_hz: f32,
    micro_doppler_index: f32,
    rcs_proxy: f32,
    rf_energy: f32,
    burstiness: f32,
    angular_rate_dps: f32,
}

impl ReferenceSensorMeasurements {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        range_m: f64,
        radial_velocity_mps: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
        snr_db: f32,
        doppler_spread_hz: f32,
        micro_doppler_index: f32,
        rcs_proxy: f32,
        rf_energy: f32,
        burstiness: f32,
        angular_rate_dps: f32,
    ) -> Result<Self> {
        let value = Self {
            range_m,
            radial_velocity_mps,
            azimuth_deg,
            elevation_deg,
            snr_db,
            doppler_spread_hz,
            micro_doppler_index,
            rcs_proxy,
            rf_energy,
            burstiness,
            angular_rate_dps,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        if !self.range_m.is_finite() || self.range_m <= 0.0 || self.range_m > 100_000.0 {
            return Err(SdkError::InvalidArgument("range_m must be finite and in (0,100000]".into()));
        }
        if !self.radial_velocity_mps.is_finite() || self.radial_velocity_mps.abs() > 2_000.0 {
            return Err(SdkError::InvalidArgument("radial_velocity_mps is outside the reference envelope".into()));
        }
        if !self.azimuth_deg.is_finite() || !(-180.0..=180.0).contains(&self.azimuth_deg) {
            return Err(SdkError::InvalidArgument("azimuth_deg must be finite and in -180..=180".into()));
        }
        if !self.elevation_deg.is_finite() || !(-90.0..=90.0).contains(&self.elevation_deg) {
            return Err(SdkError::InvalidArgument("elevation_deg must be finite and in -90..=90".into()));
        }
        for (name, value) in [
            ("snr_db", self.snr_db),
            ("doppler_spread_hz", self.doppler_spread_hz),
            ("micro_doppler_index", self.micro_doppler_index),
            ("rcs_proxy", self.rcs_proxy),
            ("rf_energy", self.rf_energy),
            ("burstiness", self.burstiness),
            ("angular_rate_dps", self.angular_rate_dps),
        ] {
            if !value.is_finite() {
                return Err(SdkError::InvalidArgument(format!("{name} must be finite")));
            }
        }
        if self.doppler_spread_hz < 0.0 || self.rcs_proxy < 0.0 {
            return Err(SdkError::InvalidArgument("doppler_spread_hz and rcs_proxy must be non-negative".into()));
        }
        for (name, value) in [
            ("micro_doppler_index", self.micro_doppler_index),
            ("rf_energy", self.rf_energy),
            ("burstiness", self.burstiness),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(SdkError::InvalidArgument(format!("{name} must be in [0,1]")));
            }
        }
        Ok(())
    }

    pub fn range_m(&self) -> f64 { self.range_m }
    pub fn radial_velocity_mps(&self) -> f64 { self.radial_velocity_mps }
    pub fn azimuth_deg(&self) -> f64 { self.azimuth_deg }
    pub fn elevation_deg(&self) -> f64 { self.elevation_deg }
    pub fn snr_db(&self) -> f32 { self.snr_db }
    pub fn doppler_spread_hz(&self) -> f32 { self.doppler_spread_hz }
    pub fn micro_doppler_index(&self) -> f32 { self.micro_doppler_index }
    pub fn rcs_proxy(&self) -> f32 { self.rcs_proxy }
    pub fn rf_energy(&self) -> f32 { self.rf_energy }
    pub fn burstiness(&self) -> f32 { self.burstiness }
    pub fn angular_rate_dps(&self) -> f32 { self.angular_rate_dps }

    pub fn local_position(&self) -> Result<crate::Position3> {
        let az = self.azimuth_deg.to_radians();
        let el = self.elevation_deg.to_radians();
        let horizontal = self.range_m * el.cos();
        crate::Position3::new(horizontal * az.cos(), horizontal * az.sin(), self.range_m * el.sin())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedSensorFrame {
    id: String,
    sensor_id: String,
    timestamp_ms: u64,
    measurements: ReferenceSensorMeasurements,
}

impl RecordedSensorFrame {
    pub fn new(
        id: impl Into<String>,
        sensor_id: impl Into<String>,
        timestamp_ms: u64,
        measurements: ReferenceSensorMeasurements,
    ) -> Result<Self> {
        let value = Self { id: id.into(), sensor_id: sensor_id.into(), timestamp_ms, measurements };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        validate_text("frame id", &self.id)?;
        validate_text("sensor id", &self.sensor_id)?;
        self.measurements.validate()
    }

    pub fn id(&self) -> &str { &self.id }
    pub fn sensor_id(&self) -> &str { &self.sensor_id }
    pub fn timestamp_ms(&self) -> u64 { self.timestamp_ms }
    pub fn measurements(&self) -> &ReferenceSensorMeasurements { &self.measurements }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceCuasFeatureAdapter;

impl ReferenceCuasFeatureAdapter {
    pub fn feature_dim(&self) -> usize { REFERENCE_CUAS_FEATURE_DIM }

    pub fn to_observation(&self, frame: &RecordedSensorFrame) -> Result<Observation> {
        frame.validate()?;
        let m = &frame.measurements;
        let az = m.azimuth_deg.to_radians();
        let el = m.elevation_deg.to_radians();
        let range_norm = (m.range_m.ln_1p() / 100_000.0_f64.ln_1p()).clamp(0.0, 1.0) as f32;
        let velocity_norm = (m.radial_velocity_mps / 200.0).clamp(-2.0, 2.0) as f32;
        let snr_norm = (m.snr_db / 40.0).clamp(-2.0, 2.0);
        let doppler_norm = (f64::from(m.doppler_spread_hz).ln_1p() / 1_000.0_f64.ln_1p()).clamp(0.0, 1.0) as f32;
        let rcs_norm = (f64::from(m.rcs_proxy).ln_1p() / 100.0_f64.ln_1p()).clamp(0.0, 1.0) as f32;
        let angular_rate_norm = (m.angular_rate_dps / 180.0).clamp(-2.0, 2.0);
        Observation::new(
            frame.id.clone(),
            frame.sensor_id.clone(),
            frame.timestamp_ms,
            vec![
                range_norm,
                velocity_norm,
                az.sin() as f32,
                az.cos() as f32,
                el.sin() as f32,
                el.cos() as f32,
                snr_norm,
                doppler_norm,
                m.micro_doppler_index,
                rcs_norm,
                m.rf_energy,
                m.burstiness,
                angular_rate_norm,
            ],
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenIsacRecordedInput {
    infrastructure_id: String,
    operator_id: String,
    timestamp_ms: u64,
    valid_from_ms: u64,
    valid_until_ms: u64,
    authorization_token: String,
    configuration_token: String,
    features: Vec<f32>,
}

impl GoldenIsacRecordedInput {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        infrastructure_id: impl Into<String>,
        operator_id: impl Into<String>,
        timestamp_ms: u64,
        valid_from_ms: u64,
        valid_until_ms: u64,
        authorization_token: impl Into<String>,
        configuration_token: impl Into<String>,
        features: Vec<f32>,
    ) -> Result<Self> {
        let input = Self {
            infrastructure_id: infrastructure_id.into(),
            operator_id: operator_id.into(),
            timestamp_ms,
            valid_from_ms,
            valid_until_ms,
            authorization_token: authorization_token.into(),
            configuration_token: configuration_token.into(),
            features,
        };
        input.validate()?;
        Ok(input)
    }

    pub fn validate(&self) -> Result<()> {
        validate_text("infrastructure_id", &self.infrastructure_id)?;
        validate_text("operator_id", &self.operator_id)?;
        validate_text("authorization_token", &self.authorization_token)?;
        validate_text("configuration_token", &self.configuration_token)?;
        if self.valid_until_ms < self.valid_from_ms || self.timestamp_ms < self.valid_from_ms || self.timestamp_ms > self.valid_until_ms {
            return Err(SdkError::InvalidArgument("ISAC timestamp must lie inside the authorization window".into()));
        }
        if self.features.len() != REFERENCE_CUAS_FEATURE_DIM {
            return Err(SdkError::DimensionMismatch { expected: REFERENCE_CUAS_FEATURE_DIM, actual: self.features.len() });
        }
        if let Some((index, _)) = self.features.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(SdkError::NonFiniteValue { index });
        }
        Ok(())
    }

    pub(crate) fn materialize(
        &self,
        model_digest: Digest32,
        ontology_version: &str,
        pipeline_version: &str,
        seed: u64,
    ) -> Result<(Observation, VerificationContext)> {
        self.validate()?;
        let mode = SensingMode::OperatorManagedActive;
        let modes = BTreeSet::from([mode]);
        let capabilities = SensingCapabilities::new(
            self.infrastructure_id.clone(),
            modes.clone(),
            REFERENCE_CUAS_FEATURE_DIM,
        )?;
        let authorization = OperatorAuthorization::new(
            self.operator_id.clone(),
            self.infrastructure_id.clone(),
            self.valid_from_ms,
            self.valid_until_ms,
            modes,
            Digest32::from_bytes(self.authorization_token.as_bytes()),
        )?;
        let request = SensingRequest::new(
            self.infrastructure_id.clone(),
            mode,
            REFERENCE_CUAS_FEATURE_DIM,
            Digest32::from_bytes(self.configuration_token.as_bytes()),
        )?;
        let frame = SensingFrame::new(
            self.timestamp_ms,
            self.infrastructure_id.clone(),
            mode,
            self.features.clone(),
        )?;
        let mut provider = RecordedSensingProvider::new(capabilities, vec![frame.clone()])?;
        let session = provider.prepare(&authorization, &request, self.timestamp_ms)?;
        let acquired = provider.acquire(&session)?.ok_or_else(|| SdkError::InvalidArgument("recorded ISAC provider returned no frame".into()))?;
        provider.stop(session)?;
        let observation = acquired.to_observation("golden-isac-frame")?;
        let context = acquired.verification_context(
            model_digest,
            &authorization,
            &request,
            ontology_version,
            pipeline_version,
            seed,
        )?;
        Ok((observation, context))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source_type", rename_all = "snake_case")]
pub enum GoldenCuasSource {
    Recorded { frame: RecordedSensorFrame },
    IsacRecorded { input: GoldenIsacRecordedInput },
}

impl GoldenCuasSource {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Recorded { frame } => frame.validate(),
            Self::IsacRecorded { input } => input.validate(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenCuasScenario {
    name: String,
    source: GoldenCuasSource,
    candidate: Option<CandidateKinematics>,
    cooperative_tracks: Vec<CooperativeTrack>,
    expected_label: Option<String>,
}

impl GoldenCuasScenario {
    pub fn new(
        name: impl Into<String>,
        source: GoldenCuasSource,
        candidate: Option<CandidateKinematics>,
        cooperative_tracks: Vec<CooperativeTrack>,
        expected_label: Option<String>,
    ) -> Result<Self> {
        let value = Self { name: name.into(), source, candidate, cooperative_tracks, expected_label };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        validate_text("scenario name", &self.name)?;
        self.source.validate()?;
        if self.cooperative_tracks.len() > MAX_SCENARIO_TRACKS {
            return Err(SdkError::DimensionLimit { actual: self.cooperative_tracks.len(), max: MAX_SCENARIO_TRACKS });
        }
        if let Some(candidate) = &self.candidate {
            CandidateKinematics::new(candidate.timestamp_ms(), candidate.position(), candidate.velocity())?;
        }
        for track in &self.cooperative_tracks {
            CooperativeTrack::new(
                track.kind(),
                track.identity().to_string(),
                track.timestamp_ms(),
                track.position(),
                track.velocity(),
                track.source_confidence(),
            )?;
        }
        if self.expected_label.as_ref().is_some_and(|value| value.trim().is_empty() || value.len() > MAX_TEXT_BYTES) {
            return Err(SdkError::InvalidArgument("expected_label must be non-empty when provided".into()));
        }
        Ok(())
    }

    pub fn name(&self) -> &str { &self.name }
    pub fn source(&self) -> &GoldenCuasSource { &self.source }
    pub fn candidate(&self) -> Option<&CandidateKinematics> { self.candidate.as_ref() }
    pub fn cooperative_tracks(&self) -> &[CooperativeTrack] { &self.cooperative_tracks }
    pub fn expected_label(&self) -> Option<&str> { self.expected_label.as_deref() }
}

fn validate_text(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > MAX_TEXT_BYTES {
        return Err(SdkError::InvalidArgument(format!("{field} must be non-empty and bounded")));
    }
    Ok(())
}

