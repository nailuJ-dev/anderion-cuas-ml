#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Anderion C-UAS ML is a sensor-agnostic perception SDK for user-supplied
//! features. The crate is limited to detection, classification, tracking,
//! localization, uncertainty, robustness and evaluation. It contains no
//! effector control, jamming, takeover, targeting or neutralization logic.

mod adaptation;
mod artifact;
mod association;
mod backend;
mod benchmark;
mod calibration;
mod classification;
mod contribution;
mod cooperative;
mod dataset;
mod degarbling;
mod detection;
mod distillation;
mod drift;
mod edge;
mod embedding;
mod enhanced_pipeline;
mod ensemble;
mod error;
mod evaluation;
mod federated;
mod fusion;
mod golden;
mod graph;
mod group_perception;
mod isac;
mod isac_dual_view;
mod linear;
mod localization;
mod micro_doppler;
mod model;
mod multimodal;
mod multimodal_alignment;
mod neural_localization;
mod ontology;
mod open_set;
mod pattern;
mod pipeline;
mod quantization;
mod reacquisition;
mod robustness;
mod scenario;
mod synthetic;
mod temporal;
mod tracking;
mod trajectory;
mod trust;
mod types;
mod uncertainty;
mod uncertainty_fusion;
mod verification;
mod verified_pipeline;
mod weak_supervision;

#[cfg(feature = "server")]
pub mod service;

pub use adaptation::{DomainAdapter, MultiSensorAdapter};
pub use artifact::{ArtifactManifest, ArtifactPolicy, load_verified_payload, verify_payload};
pub use association::LearnedAssociation;
pub use backend::{BackendKind, ComputeBackend, CpuBackend};
pub use benchmark::{BenchmarkConfig, BenchmarkReport, benchmark_pipeline};
pub use calibration::TemperatureScaler;
pub use classification::PrototypeClassifier;
pub use contribution::{ContributionLedger, SensorContribution};
pub use cooperative::{
    CandidateKinematics, CooperativeCorrelation, CooperativeCorrelator, CooperativeDisposition,
    CooperativeIdentityKind, CooperativeTrack, CorrelationPolicy, GeoPosition, VelocityNed,
    augment_graph_with_cooperative_correlations,
};
pub use dataset::{DatasetManifest, DatasetRow, DatasetSplit, grouped_split};
pub use degarbling::{
    DegarblingModel, DegarblingResult, IdentityDegarbler, PrototypeMaskDegarbler,
};
pub use detection::{BinaryLogisticDetector, MeasurementQualityModel};
pub use distillation::{DistilledPrototypeClassifier, SoftLabelSample};
pub use drift::{DriftMonitor, DriftReport};
pub use edge::{EdgeParameterProfile, profile_parameter_matrix};
pub use embedding::HashProjectionEncoder;
pub use enhanced_pipeline::{
    EnhancedPerceptionPipeline, EnhancedPerceptionResult, PerceptionComponentResult,
};
pub use ensemble::WeightedPerceptionEnsemble;
pub use error::{Result, SdkError};
pub use evaluation::{ClassificationMetrics, classification_metrics};
pub use federated::{FederatedAverager, FederatedDelta};
pub use fusion::LearnedSensorFusion;
pub use graph::{GraphEdge, GraphMessagePasser};
pub use group_perception::{FormationType, GroupMember, GroupPerception, perceive_group};
pub use isac::{
    ActiveSensingProvider, OperatorAuthorization, RecordedSensingProvider, SensingCapabilities,
    SensingFrame, SensingMode, SensingRequest, SensingSession,
};
pub use isac_dual_view::{DualViewFusionResult, fuse_isac_dual_view};
pub use localization::LinearLocalizer;
pub use micro_doppler::{MicroDopplerExtractor, MicroDopplerFeatures};
pub use model::{AssociationModel, Classifier, Detector, Encoder, Localizer, OpenSetModel};
pub use multimodal::{MultimodalSelfAttention, MultimodalTransformerEncoder};
pub use multimodal_alignment::PairedModalAligner;
pub use neural_localization::NeuralLocalizer;
pub use open_set::NearestPrototypeOod;
pub use pipeline::{PerceptionPipeline, PerceptionResult};
pub use quantization::{QuantizedEmbedding, SymmetricQuantizer};
pub use reacquisition::{ReacquisitionEnvelope, ReacquisitionScore};
pub use robustness::{AdversarialReport, adversarial_evaluate, bounded_perturbation};
pub use synthetic::{SyntheticTrajectoryConfig, synthetic_feature_trajectory};
pub use temporal::{TemporalClassifier, TemporalSelfAttention};
pub use tracking::{TrackManager, TrackObservation};
pub use trajectory::{AutoregressiveTrajectoryPredictor, TrajectorySample};
pub use trust::{
    CooperativeTrustAssessment, CooperativeTrustPolicy, CooperativeTrustVerdict,
    assess_cooperative_trust,
};
pub use types::{ClassScore, Detection, Embedding, Localization, Observation, Position3, Track};
pub use uncertainty::normalized_entropy;
pub use uncertainty_fusion::UncertaintyWeightedFusion;
pub use weak_supervision::WeakLabelModel;

pub use ontology::{
    ConceptKind, ConsistencyReport, ConsistencyViolation, OntologyGraph, OntologyNode,
    OntologyRelation, RelationKind, semantic_graph_for_perception,
};
pub use pattern::{
    CooccurrencePattern, PatternEngine, PatternEvent, PatternToken, RecurringPattern,
    pattern_event_from_perception,
};
pub use verification::{
    DeterministicVerifier, Digest32, PerceptionVerificationPolicy, ReplayStatus, ResultCertificate,
    VerificationContext, VerificationDecision,
};
pub use verified_pipeline::{VerifiedPerceptionPipeline, VerifiedPerceptionResult};

pub use golden::{
    GoldenCuasComponentReport, GoldenCuasEvaluation, GoldenCuasModel, GoldenCuasReport,
    GoldenCuasScenarioSet, GoldenCuasTrainingFile, GoldenCuasTrainingSample,
    load_golden_cuas_scenario, load_golden_cuas_scenarios, load_golden_cuas_training,
};
pub use scenario::{
    GoldenCuasScenario, GoldenCuasSource, GoldenIsacRecordedInput, REFERENCE_CUAS_FEATURE_DIM,
    RecordedSensorFrame, ReferenceCuasFeatureAdapter, ReferenceSensorMeasurements,
};
