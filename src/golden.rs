use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    BinaryLogisticDetector, CooperativeCorrelation, CooperativeCorrelator, CooperativeDisposition,
    CorrelationPolicy, DegarblingModel, DeterministicVerifier, Digest32, Encoder,
    EnhancedPerceptionPipeline, GoldenCuasScenario, GoldenCuasSource, HashProjectionEncoder,
    LinearLocalizer, Localization, PerceptionPipeline, PerceptionVerificationPolicy, Position3,
    PrototypeClassifier, RecordedSensorFrame, ReferenceCuasFeatureAdapter, ReplayStatus, Result,
    ResultCertificate, SdkError, VerificationContext, VerificationDecision,
    augment_graph_with_cooperative_correlations, semantic_graph_for_perception,
};

const MAX_GOLDEN_FILE_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_EMBEDDING_DIM: usize = 20;
const ONTOLOGY_VERSION: &str = "cuas-reference-ontology-v1";
const PIPELINE_VERSION: &str = "cuas-golden-path-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenCuasTrainingSample {
    frame: RecordedSensorFrame,
    detected: bool,
    class_label: String,
    position: Position3,
}

impl GoldenCuasTrainingSample {
    pub fn new(
        frame: RecordedSensorFrame,
        detected: bool,
        class_label: impl Into<String>,
        position: Position3,
    ) -> Result<Self> {
        let class_label = class_label.into();
        frame.validate()?;
        if class_label.trim().is_empty() || class_label.len() > 4_096 {
            return Err(SdkError::InvalidArgument(
                "golden C-UAS class label must be non-empty and bounded".into(),
            ));
        }
        Position3::new(position.x, position.y, position.z)?;
        Ok(Self {
            frame,
            detected,
            class_label,
            position,
        })
    }

    pub fn frame(&self) -> &RecordedSensorFrame {
        &self.frame
    }
    pub fn detected(&self) -> bool {
        self.detected
    }
    pub fn class_label(&self) -> &str {
        &self.class_label
    }
    pub fn position(&self) -> Position3 {
        self.position
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenCuasTrainingFile {
    pub samples: Vec<GoldenCuasTrainingSample>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenCuasScenarioSet {
    pub scenarios: Vec<GoldenCuasScenario>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenCuasComponentReport {
    source_component_index: usize,
    predicted_label: String,
    probability: f32,
    detected: bool,
    unknown: bool,
    localization: Option<Localization>,
    cooperative_disposition: CooperativeDisposition,
    cooperative_correlations: Vec<CooperativeCorrelation>,
    verification_decision: VerificationDecision,
    ontology_valid: bool,
    certificate: ResultCertificate,
}

impl GoldenCuasComponentReport {
    pub fn source_component_index(&self) -> usize {
        self.source_component_index
    }
    pub fn predicted_label(&self) -> &str {
        &self.predicted_label
    }
    pub fn probability(&self) -> f32 {
        self.probability
    }
    pub fn detected(&self) -> bool {
        self.detected
    }
    pub fn unknown(&self) -> bool {
        self.unknown
    }
    pub fn localization(&self) -> Option<&Localization> {
        self.localization.as_ref()
    }
    pub fn cooperative_disposition(&self) -> CooperativeDisposition {
        self.cooperative_disposition
    }
    pub fn cooperative_correlations(&self) -> &[CooperativeCorrelation] {
        &self.cooperative_correlations
    }
    pub fn verification_decision(&self) -> VerificationDecision {
        self.verification_decision
    }
    pub fn ontology_valid(&self) -> bool {
        self.ontology_valid
    }
    pub fn certificate(&self) -> &ResultCertificate {
        &self.certificate
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenCuasReport {
    scenario_name: String,
    expected_label: Option<String>,
    evidence_digest_sha256: String,
    components: Vec<GoldenCuasComponentReport>,
    fixture_metrics_only: bool,
}

impl GoldenCuasReport {
    pub fn scenario_name(&self) -> &str {
        &self.scenario_name
    }
    pub fn expected_label(&self) -> Option<&str> {
        self.expected_label.as_deref()
    }
    pub fn evidence_digest_sha256(&self) -> &str {
        &self.evidence_digest_sha256
    }
    pub fn components(&self) -> &[GoldenCuasComponentReport] {
        &self.components
    }
    pub fn fixture_metrics_only(&self) -> bool {
        self.fixture_metrics_only
    }
    pub fn correct(&self) -> Option<bool> {
        let predicted = self
            .components
            .first()
            .map(GoldenCuasComponentReport::predicted_label)?;
        self.expected_label
            .as_ref()
            .map(|expected| expected == predicted)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenCuasEvaluation {
    pub scenarios: usize,
    pub correct: usize,
    pub accepted: usize,
    pub exact_replays: usize,
    pub accuracy: f32,
    pub accepted_fraction: f32,
    pub exact_replay_fraction: f32,
    pub fixture_metrics_only: bool,
}

#[derive(Clone)]
pub struct GoldenCuasModel {
    adapter: ReferenceCuasFeatureAdapter,
    pipeline: PerceptionPipeline,
    correlator: CooperativeCorrelator,
    degarbler: Option<Arc<dyn DegarblingModel>>,
    model_digest: Digest32,
    base_context: VerificationContext,
    policy: PerceptionVerificationPolicy,
    seed: u64,
}

impl GoldenCuasModel {
    pub fn fit(samples: &[GoldenCuasTrainingSample], seed: u64) -> Result<Self> {
        if samples.len() < 4 {
            return Err(SdkError::InvalidArgument(
                "golden C-UAS model requires at least four training samples".into(),
            ));
        }
        if !samples.iter().any(|sample| sample.detected)
            || !samples.iter().any(|sample| !sample.detected)
        {
            return Err(SdkError::InvalidArgument(
                "golden C-UAS detector requires positive and negative examples".into(),
            ));
        }
        let adapter = ReferenceCuasFeatureAdapter;
        let encoder =
            HashProjectionEncoder::new(adapter.feature_dim(), DEFAULT_EMBEDDING_DIM, seed)?;
        let mut detector_features = Vec::with_capacity(samples.len());
        let mut detector_labels = Vec::with_capacity(samples.len());
        let mut classifier_samples = Vec::with_capacity(samples.len());
        let mut localizer_samples = Vec::with_capacity(samples.len());
        for sample in samples {
            let observation = adapter.to_observation(&sample.frame)?;
            let embedding = encoder.encode(&observation)?;
            detector_features.push(embedding.values().to_vec());
            detector_labels.push(sample.detected);
            classifier_samples.push((embedding.clone(), sample.class_label.clone()));
            localizer_samples.push((embedding, sample.position));
        }
        let detector =
            BinaryLogisticDetector::fit(&detector_features, &detector_labels, 800, 0.05, 1e-4)?;
        let classifier = PrototypeClassifier::fit(&classifier_samples)?;
        let localizer = LinearLocalizer::fit(&localizer_samples, 600, 0.001, 1e-5)?;
        let pipeline = PerceptionPipeline::new(
            Arc::new(encoder),
            Arc::new(detector),
            Arc::new(classifier),
            0.30,
        )?
        .with_localizer(Arc::new(localizer))?;
        let correlator =
            CooperativeCorrelator::new(CorrelationPolicy::new(300.0, 5_000, 80.0, 0.45)?)?;
        let training_bytes = serde_json::to_vec(samples)?;
        let model_digest = Digest32::from_bytes(&training_bytes);
        let base_context = VerificationContext::new(
            model_digest,
            Digest32::from_bytes(b"golden-cuas-reference-config-v1"),
            ONTOLOGY_VERSION,
            PIPELINE_VERSION,
            seed,
        )?;
        let policy = PerceptionVerificationPolicy::new(0.30, 1.0, 1.0, 0.0, 10_000, true)?;
        Ok(Self {
            adapter,
            pipeline,
            correlator,
            degarbler: None,
            model_digest,
            base_context,
            policy,
            seed,
        })
    }

    pub fn adapter(&self) -> ReferenceCuasFeatureAdapter {
        self.adapter
    }

    pub fn with_degarbler(mut self, degarbler: Arc<dyn DegarblingModel>) -> Self {
        self.degarbler = Some(degarbler);
        self
    }

    pub fn infer(&self, scenario: &GoldenCuasScenario) -> Result<GoldenCuasReport> {
        scenario.validate()?;
        let (observation, context) = match scenario.source() {
            GoldenCuasSource::Recorded { frame } => (
                self.adapter.to_observation(frame)?,
                self.base_context.clone(),
            ),
            GoldenCuasSource::IsacRecorded { input } => input.materialize(
                self.model_digest,
                ONTOLOGY_VERSION,
                PIPELINE_VERSION,
                self.seed,
            )?,
        };
        let mut enhanced_pipeline = EnhancedPerceptionPipeline::new(self.pipeline.clone())
            .with_correlator(self.correlator.clone());
        if let Some(model) = &self.degarbler {
            enhanced_pipeline = enhanced_pipeline.with_degarbler(model.clone());
        }
        let enhanced = enhanced_pipeline.infer(
            &observation,
            scenario.candidate(),
            scenario.cooperative_tracks(),
        )?;
        let mut reports = Vec::with_capacity(enhanced.components().len());
        for component in enhanced.components() {
            let source_observation = enhanced
                .separation()
                .components()
                .get(component.source_component_index())
                .ok_or_else(|| {
                    SdkError::InvalidArgument(
                        "enhanced perception component index is out of range".into(),
                    )
                })?;
            let perception = component.perception();
            let top = perception.classification.first().ok_or_else(|| {
                SdkError::InvalidArgument("golden C-UAS result has no class score".into())
            })?;
            let mut ontology = semantic_graph_for_perception(source_observation, perception)?;
            augment_graph_with_cooperative_correlations(
                &mut ontology,
                component.cooperative_correlations(),
            )?;
            let consistency = ontology.validate_reference_schema();
            let certificate = DeterministicVerifier::verify_perception(
                source_observation,
                perception,
                &context,
                &self.policy,
                &consistency,
            )?;
            reports.push(GoldenCuasComponentReport {
                source_component_index: component.source_component_index(),
                predicted_label: top.label.clone(),
                probability: top.probability,
                detected: perception.detection.detected,
                unknown: perception.unknown,
                localization: perception.localization.clone(),
                cooperative_disposition: component.cooperative_disposition(),
                cooperative_correlations: component.cooperative_correlations().to_vec(),
                verification_decision: certificate.decision(),
                ontology_valid: consistency.is_valid(),
                certificate,
            });
        }
        Ok(GoldenCuasReport {
            scenario_name: scenario.name().to_string(),
            expected_label: scenario.expected_label().map(ToOwned::to_owned),
            evidence_digest_sha256: enhanced.evidence_digest().to_hex(),
            components: reports,
            fixture_metrics_only: true,
        })
    }

    pub fn replay(
        &self,
        scenario: &GoldenCuasScenario,
        original: &GoldenCuasReport,
    ) -> Result<(GoldenCuasReport, ReplayStatus)> {
        let replayed = self.infer(scenario)?;
        if original.components.len() != replayed.components.len()
            || original.evidence_digest_sha256 != replayed.evidence_digest_sha256
        {
            return Ok((replayed, ReplayStatus::NonReproducible));
        }
        let mut overall = ReplayStatus::Exact;
        for (left, right) in original.components.iter().zip(&replayed.components) {
            match DeterministicVerifier::compare_replay(left.certificate(), right.certificate()) {
                ReplayStatus::Exact => {}
                ReplayStatus::DecisionEquivalent => {
                    if overall == ReplayStatus::Exact {
                        overall = ReplayStatus::DecisionEquivalent;
                    }
                }
                ReplayStatus::NonReproducible => {
                    overall = ReplayStatus::NonReproducible;
                    break;
                }
            }
        }
        Ok((replayed, overall))
    }

    pub fn evaluate(&self, scenarios: &[GoldenCuasScenario]) -> Result<GoldenCuasEvaluation> {
        if scenarios.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let mut correct = 0_usize;
        let mut accepted = 0_usize;
        let mut exact_replays = 0_usize;
        for scenario in scenarios {
            let report = self.infer(scenario)?;
            if report.correct() == Some(true) {
                correct = correct.saturating_add(1);
            }
            if report.components.first().is_some_and(|component| {
                component.verification_decision == VerificationDecision::Accept
            }) {
                accepted = accepted.saturating_add(1);
            }
            let (_, replay) = self.replay(scenario, &report)?;
            if replay == ReplayStatus::Exact {
                exact_replays = exact_replays.saturating_add(1);
            }
        }
        let count = scenarios.len() as f32;
        Ok(GoldenCuasEvaluation {
            scenarios: scenarios.len(),
            correct,
            accepted,
            exact_replays,
            accuracy: correct as f32 / count,
            accepted_fraction: accepted as f32 / count,
            exact_replay_fraction: exact_replays as f32 / count,
            fixture_metrics_only: true,
        })
    }
}

pub fn load_golden_cuas_training(path: impl AsRef<Path>) -> Result<GoldenCuasTrainingFile> {
    let bytes = read_bounded(path.as_ref(), MAX_GOLDEN_FILE_BYTES)?;
    let file: GoldenCuasTrainingFile = serde_json::from_slice(&bytes)?;
    if file.samples.len() > 65_536 {
        return Err(SdkError::DimensionLimit {
            actual: file.samples.len(),
            max: 65_536,
        });
    }
    for sample in &file.samples {
        GoldenCuasTrainingSample::new(
            sample.frame.clone(),
            sample.detected,
            sample.class_label.clone(),
            sample.position,
        )?;
    }
    Ok(file)
}

pub fn load_golden_cuas_scenario(path: impl AsRef<Path>) -> Result<GoldenCuasScenario> {
    let bytes = read_bounded(path.as_ref(), MAX_GOLDEN_FILE_BYTES)?;
    let scenario: GoldenCuasScenario = serde_json::from_slice(&bytes)?;
    scenario.validate()?;
    Ok(scenario)
}

pub fn load_golden_cuas_scenarios(path: impl AsRef<Path>) -> Result<GoldenCuasScenarioSet> {
    let bytes = read_bounded(path.as_ref(), MAX_GOLDEN_FILE_BYTES)?;
    let set: GoldenCuasScenarioSet = serde_json::from_slice(&bytes)?;
    if set.scenarios.is_empty() {
        return Err(SdkError::EmptyDataset);
    }
    if set.scenarios.len() > 65_536 {
        return Err(SdkError::DimensionLimit {
            actual: set.scenarios.len(),
            max: 65_536,
        });
    }
    for scenario in &set.scenarios {
        scenario.validate()?;
    }
    Ok(set)
}

fn read_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let limit = u64::try_from(max_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let mut reader = file.take(limit);
    let mut bytes = Vec::with_capacity(max_bytes.min(1024 * 1024));
    reader.read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(SdkError::ArtifactTooLarge {
            actual: bytes.len(),
            max: max_bytes,
        });
    }
    Ok(bytes)
}
