use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::temporal::normalize;
use crate::{ClassScore, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedSensorFusion {
    reliability: BTreeMap<String, f32>,
}

impl LearnedSensorFusion {
    pub fn fit(validation: &[(String, bool)]) -> Result<Self> {
        if validation.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let mut stats: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        for (sensor, correct) in validation {
            if sensor.trim().is_empty() {
                return Err(SdkError::InvalidArgument(
                    "sensor id must be non-empty".into(),
                ));
            }
            let entry = stats.entry(sensor.clone()).or_insert((0, 0));
            entry.1 = entry.1.saturating_add(1);
            if *correct {
                entry.0 = entry.0.saturating_add(1);
            }
        }
        let reliability = stats
            .into_iter()
            .map(|(sensor, (correct, total))| {
                let learned = (correct as f32 + 1.0) / (total as f32 + 2.0);
                (sensor, learned)
            })
            .collect();
        Ok(Self { reliability })
    }

    pub fn fuse(&self, predictions: &[(String, Vec<ClassScore>)]) -> Result<Vec<ClassScore>> {
        if predictions.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let mut totals: BTreeMap<String, f32> = BTreeMap::new();
        let mut weight_sum = 0.0_f32;
        for (sensor, scores) in predictions {
            let weight = self.reliability.get(sensor).copied().unwrap_or(0.5);
            weight_sum += weight;
            for score in scores {
                *totals.entry(score.label.clone()).or_insert(0.0) += weight * score.probability;
            }
        }
        if weight_sum <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "fusion weights sum to zero".into(),
            ));
        }
        normalize(
            totals
                .into_iter()
                .map(|(label, value)| (label, value / weight_sum))
                .collect(),
        )
    }

    pub fn reliability(&self, sensor: &str) -> Option<f32> {
        self.reliability.get(sensor).copied()
    }
}
