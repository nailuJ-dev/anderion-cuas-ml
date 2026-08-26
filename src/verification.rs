use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};

use crate::{ConsistencyReport, Observation, PerceptionResult, Result, SdkError};

const MAX_CONTEXT_TEXT_BYTES: usize = 256;
const MAX_FIXED_POINT_SCALE: u32 = 1_000_000_000;
const CERTIFICATE_ALGORITHM_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Digest32([u8; 32]);

impl Digest32 {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        finish_hash(hasher)
    }

    pub(crate) fn from_digest_bytes(bytes: [u8; 32]) -> Self { Self(bytes) }

    pub fn from_hex(value: &str) -> Result<Self> {
        let bytes = value.as_bytes();
        if bytes.len() != 64 { return Err(SdkError::InvalidArgument("SHA-256 hex digest must contain 64 characters".into())); }
        let mut output = [0_u8; 32];
        for index in 0..32 {
            let high = hex_nibble(bytes[index * 2])?;
            let low = hex_nibble(bytes[index * 2 + 1])?;
            output[index] = (high << 4) | low;
        }
        Ok(Self(output))
    }

    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }

    pub fn to_hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 0x0f) as usize] as char);
        }
        output
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationContext {
    model_digest: Digest32,
    config_digest: Digest32,
    ontology_version: String,
    pipeline_version: String,
    seed: u64,
}

impl VerificationContext {
    pub fn new(
        model_digest: Digest32,
        config_digest: Digest32,
        ontology_version: impl Into<String>,
        pipeline_version: impl Into<String>,
        seed: u64,
    ) -> Result<Self> {
        let ontology_version = ontology_version.into();
        let pipeline_version = pipeline_version.into();
        validate_context_text("ontology_version", &ontology_version)?;
        validate_context_text("pipeline_version", &pipeline_version)?;
        Ok(Self { model_digest, config_digest, ontology_version, pipeline_version, seed })
    }

    pub fn model_digest(&self) -> Digest32 { self.model_digest }
    pub fn config_digest(&self) -> Digest32 { self.config_digest }
    pub fn ontology_version(&self) -> &str { &self.ontology_version }
    pub fn pipeline_version(&self) -> &str { &self.pipeline_version }
    pub fn seed(&self) -> u64 { self.seed }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptionVerificationPolicy {
    min_class_confidence: f32,
    max_classification_uncertainty: f32,
    max_detection_uncertainty: f32,
    min_localization_confidence: f32,
    fixed_point_scale: u32,
    require_ontology_consistency: bool,
}

impl PerceptionVerificationPolicy {
    pub fn new(
        min_class_confidence: f32,
        max_classification_uncertainty: f32,
        max_detection_uncertainty: f32,
        min_localization_confidence: f32,
        fixed_point_scale: u32,
        require_ontology_consistency: bool,
    ) -> Result<Self> {
        validate_probability(min_class_confidence)?;
        validate_probability(max_classification_uncertainty)?;
        validate_probability(max_detection_uncertainty)?;
        validate_probability(min_localization_confidence)?;
        validate_scale(fixed_point_scale)?;
        Ok(Self {
            min_class_confidence,
            max_classification_uncertainty,
            max_detection_uncertainty,
            min_localization_confidence,
            fixed_point_scale,
            require_ontology_consistency,
        })
    }

    pub fn fixed_point_scale(&self) -> u32 { self.fixed_point_scale }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationDecision {
    Accept,
    Abstain,
    Review,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayStatus {
    Exact,
    DecisionEquivalent,
    NonReproducible,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResultCertificate {
    algorithm_version: u32,
    input_digest: Digest32,
    context_digest: Digest32,
    policy_digest: Digest32,
    exact_result_digest: Digest32,
    decision_digest: Digest32,
    decision: VerificationDecision,
    ontology_valid: bool,
    ontology_violation_count: usize,
}

impl ResultCertificate {
    pub fn algorithm_version(&self) -> u32 { self.algorithm_version }
    pub fn input_digest(&self) -> Digest32 { self.input_digest }
    pub fn context_digest(&self) -> Digest32 { self.context_digest }
    pub fn policy_digest(&self) -> Digest32 { self.policy_digest }
    pub fn exact_result_digest(&self) -> Digest32 { self.exact_result_digest }
    pub fn decision_digest(&self) -> Digest32 { self.decision_digest }
    pub fn decision(&self) -> VerificationDecision { self.decision }
    pub fn ontology_valid(&self) -> bool { self.ontology_valid }
    pub fn ontology_violation_count(&self) -> usize { self.ontology_violation_count }
}

pub struct DeterministicVerifier;

impl DeterministicVerifier {
    pub fn verify_perception(
        observation: &Observation,
        result: &PerceptionResult,
        context: &VerificationContext,
        policy: &PerceptionVerificationPolicy,
        consistency: &ConsistencyReport,
    ) -> Result<ResultCertificate> {
        validate_context(context)?;
        validate_policy(policy)?;
        validate_result(result)?;
        let top = result.classification.first().ok_or_else(|| SdkError::InvalidArgument("perception result has no class score".into()))?;
        let decision = if policy.require_ontology_consistency && !consistency.is_valid() {
            VerificationDecision::Review
        } else if result.detection.uncertainty > policy.max_detection_uncertainty {
            VerificationDecision::Abstain
        } else if result.detection.detected
            && (result.unknown
                || result.classification_uncertainty > policy.max_classification_uncertainty
                || top.probability < policy.min_class_confidence)
        {
            VerificationDecision::Abstain
        } else if result.localization.as_ref().is_some_and(|localization| localization.confidence < policy.min_localization_confidence) {
            VerificationDecision::Review
        } else {
            VerificationDecision::Accept
        };
        Ok(ResultCertificate {
            algorithm_version: CERTIFICATE_ALGORITHM_VERSION,
            input_digest: hash_observation(observation),
            context_digest: hash_context(context),
            policy_digest: hash_policy(policy),
            exact_result_digest: hash_result_exact(result),
            decision_digest: hash_result_decision(result, policy.fixed_point_scale)?,
            decision,
            ontology_valid: consistency.is_valid(),
            ontology_violation_count: consistency.violations().len(),
        })
    }

    pub fn compare_replay(original: &ResultCertificate, replayed: &ResultCertificate) -> ReplayStatus {
        if original.algorithm_version != replayed.algorithm_version
            || original.input_digest != replayed.input_digest
            || original.context_digest != replayed.context_digest
            || original.policy_digest != replayed.policy_digest
            || original.decision != replayed.decision
        {
            return ReplayStatus::NonReproducible;
        }
        if original.exact_result_digest == replayed.exact_result_digest
            && original.ontology_valid == replayed.ontology_valid
            && original.ontology_violation_count == replayed.ontology_violation_count
        {
            ReplayStatus::Exact
        } else if original.decision_digest == replayed.decision_digest {
            ReplayStatus::DecisionEquivalent
        } else {
            ReplayStatus::NonReproducible
        }
    }
}

fn validate_result(result: &PerceptionResult) -> Result<()> {
    if result.observation_id.trim().is_empty() || result.sensor_id.trim().is_empty() {
        return Err(SdkError::InvalidArgument("perception identifiers must not be empty".into()));
    }
    validate_probability(result.detection.probability)?;
    validate_probability(result.detection.uncertainty)?;
    if result.classification.is_empty() {
        return Err(SdkError::InvalidArgument("perception result must contain class scores".into()));
    }
    for score in &result.classification {
        if score.label.trim().is_empty() || score.label.len() > 4_096 {
            return Err(SdkError::InvalidArgument("perception class label is invalid".into()));
        }
        validate_probability(score.probability)?;
    }
    validate_probability(result.classification_uncertainty)?;
    if result.embedding.values().is_empty() || result.embedding.values().len() > 8_192 {
        return Err(SdkError::DimensionLimit { actual: result.embedding.values().len(), max: 8_192 });
    }
    if let Some((index, _)) = result.embedding.values().iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(SdkError::NonFiniteValue { index });
    }
    if let Some(localization) = &result.localization {
        if !localization.position.x.is_finite() || !localization.position.y.is_finite() || !localization.position.z.is_finite() {
            return Err(SdkError::InvalidArgument("localization coordinates must be finite".into()));
        }
        if !localization.sigma_m.is_finite() || localization.sigma_m < 0.0 {
            return Err(SdkError::InvalidArgument("localization sigma must be finite and non-negative".into()));
        }
        validate_probability(localization.confidence)?;
    }
    Ok(())
}

fn validate_context(context: &VerificationContext) -> Result<()> {
    validate_context_text("ontology_version", context.ontology_version())?;
    validate_context_text("pipeline_version", context.pipeline_version())?;
    Ok(())
}

fn validate_policy(policy: &PerceptionVerificationPolicy) -> Result<()> {
    validate_probability(policy.min_class_confidence)?;
    validate_probability(policy.max_classification_uncertainty)?;
    validate_probability(policy.max_detection_uncertainty)?;
    validate_probability(policy.min_localization_confidence)?;
    validate_scale(policy.fixed_point_scale)
}

fn hash_policy(policy: &PerceptionVerificationPolicy) -> Digest32 {
    let mut hasher = domain_hasher(b"anderion-cuas-verification-policy-v1");
    update_f32(&mut hasher, policy.min_class_confidence);
    update_f32(&mut hasher, policy.max_classification_uncertainty);
    update_f32(&mut hasher, policy.max_detection_uncertainty);
    update_f32(&mut hasher, policy.min_localization_confidence);
    hasher.update(policy.fixed_point_scale.to_le_bytes());
    hasher.update([u8::from(policy.require_ontology_consistency)]);
    finish_hash(hasher)
}

fn hash_observation(observation: &Observation) -> Digest32 {
    let mut hasher = domain_hasher(b"anderion-cuas-input-v1");
    update_bytes(&mut hasher, observation.id().as_bytes());
    update_bytes(&mut hasher, observation.sensor_id().as_bytes());
    hasher.update(observation.timestamp_ms().to_le_bytes());
    update_f32_slice(&mut hasher, observation.features());
    finish_hash(hasher)
}

fn hash_context(context: &VerificationContext) -> Digest32 {
    let mut hasher = domain_hasher(b"anderion-cuas-context-v1");
    hasher.update(context.model_digest.as_bytes());
    hasher.update(context.config_digest.as_bytes());
    update_bytes(&mut hasher, context.ontology_version.as_bytes());
    update_bytes(&mut hasher, context.pipeline_version.as_bytes());
    hasher.update(context.seed.to_le_bytes());
    finish_hash(hasher)
}

fn hash_result_exact(result: &PerceptionResult) -> Digest32 {
    let mut hasher = domain_hasher(b"anderion-cuas-result-exact-v1");
    update_bytes(&mut hasher, result.observation_id.as_bytes());
    update_bytes(&mut hasher, result.sensor_id.as_bytes());
    update_f32(&mut hasher, result.detection.probability);
    hasher.update([u8::from(result.detection.detected)]);
    update_f32(&mut hasher, result.detection.uncertainty);
    hasher.update((result.classification.len() as u64).to_le_bytes());
    for score in &result.classification {
        update_bytes(&mut hasher, score.label.as_bytes());
        update_f32(&mut hasher, score.probability);
    }
    hasher.update([u8::from(result.unknown)]);
    match &result.localization {
        Some(localization) => {
            hasher.update([1_u8]);
            update_f64(&mut hasher, localization.position.x);
            update_f64(&mut hasher, localization.position.y);
            update_f64(&mut hasher, localization.position.z);
            update_f64(&mut hasher, localization.sigma_m);
            update_f32(&mut hasher, localization.confidence);
        }
        None => hasher.update([0_u8]),
    }
    update_f32_slice(&mut hasher, result.embedding.values());
    update_f32(&mut hasher, result.classification_uncertainty);
    finish_hash(hasher)
}

fn hash_result_decision(result: &PerceptionResult, scale: u32) -> Result<Digest32> {
    let top = result.classification.first().ok_or_else(|| SdkError::InvalidArgument("perception result has no class score".into()))?;
    let mut hasher = domain_hasher(b"anderion-cuas-decision-v1");
    hasher.update(quantize_probability(result.detection.probability, scale)?.to_le_bytes());
    hasher.update([u8::from(result.detection.detected)]);
    hasher.update(quantize_probability(result.detection.uncertainty, scale)?.to_le_bytes());
    update_bytes(&mut hasher, top.label.as_bytes());
    hasher.update(quantize_probability(top.probability, scale)?.to_le_bytes());
    hasher.update([u8::from(result.unknown)]);
    hasher.update(quantize_probability(result.classification_uncertainty, scale)?.to_le_bytes());
    if let Some(localization) = &result.localization {
        hasher.update([1_u8]);
        hasher.update(quantize_probability(localization.confidence, scale)?.to_le_bytes());
    } else {
        hasher.update([0_u8]);
    }
    Ok(finish_hash(hasher))
}

fn domain_hasher(domain: &[u8]) -> Sha256 {
    let mut hasher = Sha256::new();
    update_bytes(&mut hasher, domain);
    hasher
}

fn update_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn update_f32_slice(hasher: &mut Sha256, values: &[f32]) {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values { update_f32(hasher, *value); }
}

fn update_f32(hasher: &mut Sha256, value: f32) {
    let bits = if value == 0.0 { 0_u32 } else { value.to_bits() };
    hasher.update(bits.to_le_bytes());
}

fn update_f64(hasher: &mut Sha256, value: f64) {
    let bits = if value == 0.0 { 0_u64 } else { value.to_bits() };
    hasher.update(bits.to_le_bytes());
}

fn finish_hash(hasher: Sha256) -> Digest32 {
    let finalized = hasher.finalize();
    let mut bytes = [0_u8; 32];
    bytes.copy_from_slice(&finalized);
    Digest32(bytes)
}

fn quantize_probability(value: f32, scale: u32) -> Result<u64> {
    validate_probability(value)?;
    Ok(((value as f64) * (scale as f64)).round() as u64)
}

fn validate_probability(value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(SdkError::InvalidProbability(value));
    }
    Ok(())
}

fn validate_scale(scale: u32) -> Result<()> {
    if scale == 0 || scale > MAX_FIXED_POINT_SCALE {
        return Err(SdkError::InvalidArgument("fixed_point_scale must be in 1..=1_000_000_000".into()));
    }
    Ok(())
}

fn validate_context_text(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() { return Err(SdkError::InvalidArgument(format!("{field} must not be empty"))); }
    if value.len() > MAX_CONTEXT_TEXT_BYTES { return Err(SdkError::DimensionLimit { actual: value.len(), max: MAX_CONTEXT_TEXT_BYTES }); }
    Ok(())
}

fn hex_nibble(value: u8) -> Result<u8> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(SdkError::InvalidArgument("invalid SHA-256 hex digest".into())),
    }
}
