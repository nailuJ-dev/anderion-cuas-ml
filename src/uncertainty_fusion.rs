use crate::{Localization, Position3, Result, SdkError};

#[derive(Debug, Clone, Copy, Default)]
pub struct UncertaintyWeightedFusion;

impl UncertaintyWeightedFusion {
    pub fn fuse(&self, localizations: &[Localization]) -> Result<Localization> {
        if localizations.is_empty() { return Err(SdkError::EmptyDataset); }
        if localizations.len() > 65_536 { return Err(SdkError::DimensionLimit { actual: localizations.len(), max: 65_536 }); }
        let mut weighted = [0.0_f64; 3];
        let mut weight_sum = 0.0_f64;
        for localization in localizations {
            let variance = localization.sigma_m.max(1.0e-6).powi(2);
            let weight = f64::from(localization.confidence.max(1.0e-6)) / variance;
            weighted[0] += weight * localization.position.x;
            weighted[1] += weight * localization.position.y;
            weighted[2] += weight * localization.position.z;
            weight_sum += weight;
        }
        if !weight_sum.is_finite() || weight_sum <= 0.0 { return Err(SdkError::InvalidArgument("localization fusion has invalid total weight".into())); }
        let position = Position3::new(weighted[0] / weight_sum, weighted[1] / weight_sum, weighted[2] / weight_sum)?;
        let sigma_m = (1.0 / weight_sum).sqrt();
        let confidence = (localizations.iter().map(|item| item.confidence).sum::<f32>() / localizations.len() as f32).clamp(0.0, 1.0);
        Localization::new(position, sigma_m, confidence)
    }
}
