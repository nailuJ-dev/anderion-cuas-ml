use serde::{Deserialize, Serialize};
use crate::{MicroDopplerFeatures, Result, SdkError};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DualViewFusionResult { pub micro_doppler_confidence:f32, pub hrrp_concentration:f32, pub hrrp_entropy:f32, pub fused_confidence:f32 }

pub fn fuse_isac_dual_view(micro:&MicroDopplerFeatures,hrrp_power:&[f32])->Result<DualViewFusionResult>{
    if hrrp_power.len()<3{return Err(SdkError::InvalidArgument("HRRP requires at least three range bins".into()));}
    if hrrp_power.iter().any(|v|!v.is_finite()||*v<0.0){return Err(SdkError::InvalidArgument("HRRP power must be finite and non-negative".into()));}
    let total:f64=hrrp_power.iter().map(|v|f64::from(*v)).sum();if total<=f64::EPSILON{return Err(SdkError::EmptyFeatures);}
    let max=f64::from(hrrp_power.iter().copied().fold(0.0_f32,f32::max));
    let concentration=(max/total).clamp(0.0,1.0) as f32;
    let entropy_raw=hrrp_power.iter().filter_map(|v|{let p=f64::from(*v)/total;(p>0.0).then_some(-p*p.ln())}).sum::<f64>();
    let entropy=(entropy_raw/(hrrp_power.len() as f64).ln()).clamp(0.0,1.0) as f32;
    let structural=(0.65*concentration+0.35*(1.0-entropy)).clamp(0.0,1.0);
    let fused=(0.6*micro.confidence+0.4*structural).clamp(0.0,1.0);
    Ok(DualViewFusionResult{micro_doppler_confidence:micro.confidence,hrrp_concentration:concentration,hrrp_entropy:entropy,fused_confidence:fused})
}
