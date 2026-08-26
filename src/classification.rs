use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{ClassScore, Classifier, Embedding, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypeClassifier {
    dimension: usize,
    prototypes: BTreeMap<String, Vec<f32>>,
    counts: BTreeMap<String, u64>,
    temperature: f32,
}

impl PrototypeClassifier {
    pub fn fit(samples: &[(Embedding, String)]) -> Result<Self> {
        if samples.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let dimension = samples[0].0.dim();
        let mut sums: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let mut counts = BTreeMap::new();
        for (embedding, label) in samples {
            if embedding.dim() != dimension {
                return Err(SdkError::DimensionMismatch {
                    expected: dimension,
                    actual: embedding.dim(),
                });
            }
            if label.trim().is_empty() {
                return Err(SdkError::InvalidArgument(
                    "class label must be non-empty".into(),
                ));
            }
            let sum = sums
                .entry(label.clone())
                .or_insert_with(|| vec![0.0; dimension]);
            for (dst, src) in sum.iter_mut().zip(embedding.values()) {
                *dst += *src;
            }
            *counts.entry(label.clone()).or_insert(0_u64) += 1;
        }
        let mut prototypes = BTreeMap::new();
        for (label, mut sum) in sums {
            let count = counts.get(&label).copied().unwrap_or(1) as f32;
            for value in &mut sum {
                *value /= count;
            }
            prototypes.insert(label, sum);
        }
        Ok(Self {
            dimension,
            prototypes,
            counts,
            temperature: 1.0,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.dimension == 0 || self.dimension > 8_192 || self.prototypes.is_empty() {
            return Err(SdkError::InvalidArgument(
                "invalid prototype classifier dimensions or empty classes".into(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "invalid classifier temperature".into(),
            ));
        }
        for (label, prototype) in &self.prototypes {
            if label.trim().is_empty()
                || prototype.len() != self.dimension
                || prototype.iter().any(|v| !v.is_finite())
            {
                return Err(SdkError::InvalidArgument(
                    "invalid prototype classifier payload".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn prototypes(&self) -> &BTreeMap<String, Vec<f32>> {
        &self.prototypes
    }
}

impl Classifier for PrototypeClassifier {
    fn classify(&self, embedding: &Embedding) -> Result<Vec<ClassScore>> {
        if embedding.dim() != self.dimension {
            return Err(SdkError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.dim(),
            });
        }
        let mut logits = Vec::with_capacity(self.prototypes.len());
        for (label, prototype) in &self.prototypes {
            let distance = squared_distance(embedding.values(), prototype)?;
            logits.push((label.clone(), -distance / self.temperature));
        }
        softmax(&logits)
    }
    fn input_dim(&self) -> usize {
        self.dimension
    }
}

pub(crate) fn squared_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(SdkError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    Ok(a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum())
}

pub(crate) fn softmax(logits: &[(String, f32)]) -> Result<Vec<ClassScore>> {
    if logits.is_empty() {
        return Err(SdkError::EmptyDataset);
    }
    let max = logits
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|(_, v)| (*v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SdkError::InvalidArgument(
            "softmax normalization failed".into(),
        ));
    }
    let mut scores = Vec::with_capacity(logits.len());
    for ((label, _), value) in logits.iter().zip(exps) {
        scores.push(ClassScore::new(label.clone(), value / sum)?);
    }
    scores.sort_by(|a, b| b.probability.total_cmp(&a.probability));
    Ok(scores)
}

impl PrototypeClassifier {
    pub fn add_few_shot_examples(
        &mut self,
        label: impl Into<String>,
        examples: &[Embedding],
    ) -> Result<()> {
        if examples.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let label = label.into();
        if label.trim().is_empty() {
            return Err(SdkError::InvalidArgument(
                "class label must be non-empty".into(),
            ));
        }
        if examples.iter().any(|e| e.dim() != self.dimension) {
            return Err(SdkError::DimensionMismatch {
                expected: self.dimension,
                actual: examples[0].dim(),
            });
        }
        let old_count = self.counts.get(&label).copied().unwrap_or(0);
        let old = self
            .prototypes
            .get(&label)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.dimension]);
        let mut sum: Vec<f32> = old.into_iter().map(|v| v * old_count as f32).collect();
        for example in examples {
            for (dst, src) in sum.iter_mut().zip(example.values()) {
                *dst += *src;
            }
        }
        let new_count = old_count.saturating_add(examples.len() as u64);
        for value in &mut sum {
            *value /= new_count as f32;
        }
        self.prototypes.insert(label.clone(), sum);
        self.counts.insert(label, new_count);
        Ok(())
    }
}
