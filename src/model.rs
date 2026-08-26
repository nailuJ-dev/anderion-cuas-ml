use crate::{ClassScore, Detection, Embedding, Localization, Observation, Result};

pub trait Encoder: Send + Sync {
    fn encode(&self, observation: &Observation) -> Result<Embedding>;
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
}

pub trait Detector: Send + Sync {
    fn detect(&self, embedding: &Embedding) -> Result<Detection>;
    fn input_dim(&self) -> usize;
}

pub trait Classifier: Send + Sync {
    fn classify(&self, embedding: &Embedding) -> Result<Vec<ClassScore>>;
    fn input_dim(&self) -> usize;
}

pub trait OpenSetModel: Send + Sync {
    fn is_unknown(&self, embedding: &Embedding) -> Result<bool>;
    fn input_dim(&self) -> usize;
}

pub trait Localizer: Send + Sync {
    fn localize(&self, embedding: &Embedding) -> Result<Localization>;
    fn input_dim(&self) -> usize;
}

pub trait AssociationModel: Send + Sync {
    fn score_features(&self, features: &[f32]) -> Result<f32>;
    fn feature_dim(&self) -> usize;
}
