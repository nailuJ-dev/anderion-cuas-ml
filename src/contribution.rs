use serde::{Deserialize, Serialize};
use crate::{Result, SdkError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorContribution { pub source:String, pub contribution:f32, pub evidence_quality:f32, pub note:String }
impl SensorContribution {
    pub fn new(source:impl Into<String>, contribution:f32, evidence_quality:f32, note:impl Into<String>)->Result<Self>{
        let source=source.into(); if source.trim().is_empty(){return Err(SdkError::InvalidArgument("contribution source must be non-empty".into()));}
        if !contribution.is_finite()||!(-1.0..=1.0).contains(&contribution){return Err(SdkError::InvalidArgument("contribution must be finite and in [-1,1]".into()));}
        if !evidence_quality.is_finite()||!(0.0..=1.0).contains(&evidence_quality){return Err(SdkError::InvalidProbability(evidence_quality));}
        Ok(Self{source,contribution,evidence_quality,note:note.into()})
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContributionLedger { entries:Vec<SensorContribution> }
impl ContributionLedger {
    pub fn new(mut entries:Vec<SensorContribution>)->Result<Self>{if entries.is_empty(){return Err(SdkError::EmptyDataset);}entries.sort_by(|a,b|b.contribution.abs().total_cmp(&a.contribution.abs()).then_with(||a.source.cmp(&b.source)));Ok(Self{entries})}
    pub fn entries(&self)->&[SensorContribution]{&self.entries}
    pub fn total_contribution(&self)->f32{self.entries.iter().map(|e|e.contribution).sum()}
}
