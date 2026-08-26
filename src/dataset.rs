use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Result, SdkError};
use crate::types::validate_vector;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetRow {
    pub id: String,
    pub group_id: String,
    pub sensor_id: String,
    pub label: String,
    pub features: Vec<f32>,
}

impl DatasetRow {
    pub fn new(
        id: impl Into<String>,
        group_id: impl Into<String>,
        sensor_id: impl Into<String>,
        label: impl Into<String>,
        features: Vec<f32>,
    ) -> Result<Self> {
        validate_vector(&features, 65_536)?;
        let row = Self {
            id: id.into(),
            group_id: group_id.into(),
            sensor_id: sensor_id.into(),
            label: label.into(),
            features,
        };
        if row.id.trim().is_empty() || row.group_id.trim().is_empty() || row.sensor_id.trim().is_empty() || row.label.trim().is_empty() {
            return Err(SdkError::InvalidArgument("dataset identifiers and label must be non-empty".into()));
        }
        Ok(row)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetSplit {
    pub train: Vec<DatasetRow>,
    pub test: Vec<DatasetRow>,
}

pub fn grouped_split(rows: &[DatasetRow], train_fraction: f32, seed: u64) -> Result<DatasetSplit> {
    if rows.is_empty() { return Err(SdkError::EmptyDataset); }
    if !train_fraction.is_finite() || !(0.0..1.0).contains(&train_fraction) {
        return Err(SdkError::InvalidArgument("train_fraction must be in (0,1)".into()));
    }
    let mut groups: BTreeMap<String, Vec<DatasetRow>> = BTreeMap::new();
    for row in rows { groups.entry(row.group_id.clone()).or_default().push(row.clone()); }
    if groups.len() < 2 { return Err(SdkError::InvalidArgument("at least two groups are required".into())); }
    let mut ordered: Vec<(u64, String)> = groups.keys().map(|g| (stable_hash(g.as_bytes(), seed), g.clone())).collect();
    ordered.sort_by_key(|item| item.0);
    let target = (rows.len() as f32 * train_fraction).round() as usize;
    let mut selected = BTreeSet::new();
    let mut count = 0_usize;
    for (_, group) in &ordered {
        if count >= target && !selected.is_empty() { break; }
        if let Some(items) = groups.get(group) {
            selected.insert(group.clone());
            count += items.len();
        }
    }
    if selected.len() == groups.len() {
        if let Some((_, last)) = ordered.last() { selected.remove(last); }
    }
    let mut train = Vec::new();
    let mut test = Vec::new();
    for row in rows {
        if selected.contains(&row.group_id) { train.push(row.clone()); } else { test.push(row.clone()); }
    }
    if train.is_empty() || test.is_empty() { return Err(SdkError::InvalidArgument("split produced empty partition".into())); }
    Ok(DatasetSplit { train, test })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub version: String,
    pub row_count: usize,
    pub sha256: String,
}

impl DatasetManifest {
    pub fn from_rows(version: impl Into<String>, rows: &[DatasetRow]) -> Result<Self> {
        if rows.is_empty() { return Err(SdkError::EmptyDataset); }
        let version = version.into();
        if version.trim().is_empty() { return Err(SdkError::InvalidArgument("dataset version must be non-empty".into())); }
        let bytes = serde_json::to_vec(rows)?;
        let digest = Sha256::digest(&bytes);
        let mut sha256 = String::with_capacity(64);
        for byte in digest { sha256.push_str(&format!("{byte:02x}")); }
        Ok(Self { version, row_count: rows.len(), sha256 })
    }
}

fn stable_hash(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ seed;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}
