use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakLabelModel {
    source_weights: Vec<f32>,
}

impl WeakLabelModel {
    pub fn fit(rows: &[Vec<Option<String>>]) -> Result<Self> {
        if rows.is_empty() { return Err(SdkError::EmptyDataset); }
        let sources = rows[0].len();
        if sources == 0 { return Err(SdkError::EmptyFeatures); }
        if rows.iter().any(|row| row.len() != sources) {
            return Err(SdkError::DimensionMismatch { expected: sources, actual: rows.iter().map(Vec::len).max().unwrap_or(0) });
        }
        let pseudo: Vec<Option<String>> = rows.iter().map(|row| majority(row)).collect();
        let mut source_weights = vec![0.5_f32; sources];
        for source in 0..sources {
            let mut correct = 0_u64;
            let mut observed = 0_u64;
            for (row, consensus) in rows.iter().zip(&pseudo) {
                if let (Some(label), Some(target)) = (&row[source], consensus) {
                    observed += 1;
                    if label == target { correct += 1; }
                }
            }
            source_weights[source] = (correct as f32 + 1.0) / (observed as f32 + 2.0);
        }
        Ok(Self { source_weights })
    }

    pub fn predict(&self, labels: &[Option<String>]) -> Result<String> {
        if labels.len() != self.source_weights.len() {
            return Err(SdkError::DimensionMismatch { expected: self.source_weights.len(), actual: labels.len() });
        }
        let mut votes: BTreeMap<String, f32> = BTreeMap::new();
        for (label, weight) in labels.iter().zip(&self.source_weights) {
            if let Some(label) = label { *votes.entry(label.clone()).or_insert(0.0) += *weight; }
        }
        votes.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(label, _)| label)
            .ok_or_else(|| SdkError::InvalidArgument("weak label prediction has no observed labels".into()))
    }

    pub fn source_weights(&self) -> &[f32] { &self.source_weights }
}

fn majority(labels: &[Option<String>]) -> Option<String> {
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for label in labels.iter().flatten() { *counts.entry(label.clone()).or_insert(0) += 1; }
    counts.into_iter().max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0))).map(|(label, _)| label)
}
