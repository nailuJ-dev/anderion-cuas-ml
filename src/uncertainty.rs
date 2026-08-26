use crate::{ClassScore, Result, SdkError};

pub fn normalized_entropy(scores: &[ClassScore]) -> Result<f32> {
    if scores.is_empty() { return Err(SdkError::EmptyDataset); }
    if scores.len() == 1 { return Ok(0.0); }
    let sum: f32 = scores.iter().map(|s| s.probability).sum();
    if (sum - 1.0).abs() > 1e-3 { return Err(SdkError::InvalidArgument("probabilities must sum to one".into())); }
    let h = scores.iter().map(|s| {
        let p = s.probability.max(f32::MIN_POSITIVE);
        -p * p.ln()
    }).sum::<f32>();
    Ok((h / (scores.len() as f32).ln()).clamp(0.0, 1.0))
}
