use std::path::Path;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::{
    ArtifactPolicy, BinaryLogisticDetector, HashProjectionEncoder, LinearLocalizer,
    NearestPrototypeOod, Observation, PerceptionPipeline, PerceptionResult, PrototypeClassifier,
    Result, SdkError, TemperatureScaler, load_verified_payload,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerceptionBundle {
    pub encoder: HashProjectionEncoder,
    pub detector: BinaryLogisticDetector,
    pub classifier: PrototypeClassifier,
    pub open_set: Option<NearestPrototypeOod>,
    pub localizer: Option<LinearLocalizer>,
    pub calibrator: Option<TemperatureScaler>,
    pub unknown_confidence_threshold: f32,
}

impl PerceptionBundle {
    pub fn into_pipeline(self) -> Result<PerceptionPipeline> {
        self.encoder.validate()?;
        self.detector.validate()?;
        self.classifier.validate()?;
        if let Some(open_set) = &self.open_set {
            open_set.validate()?;
        }
        if let Some(localizer) = &self.localizer {
            localizer.validate()?;
        }
        let mut pipeline = PerceptionPipeline::new(
            Arc::new(self.encoder),
            Arc::new(self.detector),
            Arc::new(self.classifier),
            self.unknown_confidence_threshold,
        )?;
        if let Some(open_set) = self.open_set {
            pipeline = pipeline.with_open_set(Arc::new(open_set))?;
        }
        if let Some(localizer) = self.localizer {
            pipeline = pipeline.with_localizer(Arc::new(localizer))?;
        }
        if let Some(calibrator) = self.calibrator {
            pipeline = pipeline.with_calibrator(Arc::new(calibrator));
        }
        Ok(pipeline)
    }
}

pub fn load_perception_bundle(
    manifest_path: impl AsRef<Path>,
    payload_path: impl AsRef<Path>,
) -> Result<PerceptionBundle> {
    let policy = ArtifactPolicy::default();
    let (_, payload) = load_verified_payload(manifest_path, payload_path, &policy)?;
    let bundle: PerceptionBundle = serde_json::from_slice(&payload)?;
    if !bundle.unknown_confidence_threshold.is_finite()
        || !(0.0..=1.0).contains(&bundle.unknown_confidence_threshold)
    {
        return Err(SdkError::InvalidProbability(
            bundle.unknown_confidence_threshold,
        ));
    }
    Ok(bundle)
}

#[derive(Clone)]
pub struct ServiceState {
    pipeline: Arc<PerceptionPipeline>,
}

impl ServiceState {
    pub fn new(pipeline: PerceptionPipeline) -> Self {
        Self {
            pipeline: Arc::new(pipeline),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct InferRequest {
    pub id: String,
    pub sensor_id: String,
    pub timestamp_ms: u64,
    pub features: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

pub fn build_router(state: ServiceState) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/perception", post(perception))
        .layer(DefaultBodyLimit::max(256 * 1024))
        .with_state(state)
}

async fn health() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn perception(
    State(state): State<ServiceState>,
    Json(request): Json<InferRequest>,
) -> std::result::Result<Json<PerceptionResult>, (StatusCode, Json<ErrorResponse>)> {
    let observation = Observation::new(
        request.id,
        request.sensor_id,
        request.timestamp_ms,
        request.features,
    )
    .map_err(bad_request)?;
    state
        .pipeline
        .infer(&observation)
        .map(Json)
        .map_err(inference_error)
}

fn bad_request(error: SdkError) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: error.to_string(),
        }),
    )
}

fn inference_error(error: SdkError) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(ErrorResponse {
            error: error.to_string(),
        }),
    )
}
