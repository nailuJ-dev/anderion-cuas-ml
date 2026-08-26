use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    ClassScore, Classifier, Detection, Detector, Embedding, Encoder, Localization,
    Localizer, Observation, OpenSetModel, Result, SdkError, TemperatureScaler, normalized_entropy,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptionResult {
    pub observation_id: String,
    pub sensor_id: String,
    pub detection: Detection,
    pub classification: Vec<ClassScore>,
    pub unknown: bool,
    pub localization: Option<Localization>,
    pub embedding: Embedding,
    pub classification_uncertainty: f32,
}

#[derive(Clone)]
pub struct PerceptionPipeline {
    encoder: Arc<dyn Encoder>,
    detector: Arc<dyn Detector>,
    classifier: Arc<dyn Classifier>,
    open_set: Option<Arc<dyn OpenSetModel>>,
    localizer: Option<Arc<dyn Localizer>>,
    calibrator: Option<Arc<TemperatureScaler>>,
    unknown_confidence_threshold: f32,
}

impl PerceptionPipeline {
    pub fn new(
        encoder: Arc<dyn Encoder>,
        detector: Arc<dyn Detector>,
        classifier: Arc<dyn Classifier>,
        unknown_confidence_threshold: f32,
    ) -> Result<Self> {
        if encoder.output_dim() != detector.input_dim() {
            return Err(SdkError::DimensionMismatch { expected: encoder.output_dim(), actual: detector.input_dim() });
        }
        if encoder.output_dim() != classifier.input_dim() {
            return Err(SdkError::DimensionMismatch { expected: encoder.output_dim(), actual: classifier.input_dim() });
        }
        if !unknown_confidence_threshold.is_finite() || !(0.0..=1.0).contains(&unknown_confidence_threshold) {
            return Err(SdkError::InvalidProbability(unknown_confidence_threshold));
        }
        Ok(Self { encoder, detector, classifier, open_set: None, localizer: None, calibrator: None, unknown_confidence_threshold })
    }

    pub fn with_open_set(mut self, model: Arc<dyn OpenSetModel>) -> Result<Self> {
        if model.input_dim() != self.encoder.output_dim() {
            return Err(SdkError::DimensionMismatch { expected: self.encoder.output_dim(), actual: model.input_dim() });
        }
        self.open_set = Some(model);
        Ok(self)
    }

    pub fn with_calibrator(mut self, calibrator: Arc<TemperatureScaler>) -> Self {
        self.calibrator = Some(calibrator);
        self
    }

    pub fn with_localizer(mut self, model: Arc<dyn Localizer>) -> Result<Self> {
        if model.input_dim() != self.encoder.output_dim() {
            return Err(SdkError::DimensionMismatch { expected: self.encoder.output_dim(), actual: model.input_dim() });
        }
        self.localizer = Some(model);
        Ok(self)
    }

    pub fn infer(&self, observation: &Observation) -> Result<PerceptionResult> {
        if observation.features().len() != self.encoder.input_dim() {
            return Err(SdkError::DimensionMismatch { expected: self.encoder.input_dim(), actual: observation.features().len() });
        }
        let embedding = self.encoder.encode(observation)?;
        let detection = self.detector.detect(&embedding)?;
        let mut classification = self.classifier.classify(&embedding)?;
        if let Some(calibrator) = &self.calibrator { classification = calibrator.calibrate(&classification)?; }
        let classification_uncertainty = normalized_entropy(&classification)?;
        let confidence_unknown = classification.first().map(|score| score.probability < self.unknown_confidence_threshold).unwrap_or(true);
        let model_unknown = self.open_set.as_ref().map(|model| model.is_unknown(&embedding)).transpose()?.unwrap_or(false);
        let localization = self.localizer.as_ref().map(|model| model.localize(&embedding)).transpose()?;
        Ok(PerceptionResult {
            observation_id: observation.id().to_string(),
            sensor_id: observation.sensor_id().to_string(),
            detection,
            classification,
            unknown: confidence_unknown || model_unknown,
            localization,
            embedding,
            classification_uncertainty,
        })
    }

    pub fn infer_batch(&self, observations: &[Observation]) -> Result<Vec<PerceptionResult>> {
        if observations.len() > 65_536 { return Err(SdkError::DimensionLimit { actual: observations.len(), max: 65_536 }); }
        observations.iter().map(|observation| self.infer(observation)).collect()
    }

    pub fn infer_stream<'a, I>(&'a self, observations: I) -> impl Iterator<Item = Result<PerceptionResult>> + 'a
    where
        I: IntoIterator<Item = Observation> + 'a,
    {
        observations.into_iter().map(|observation| self.infer(&observation))
    }
}
