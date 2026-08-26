use serde::{Deserialize, Serialize};

use crate::{Result, SdkError};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeParameterProfile {
    pub parameter_count: usize,
    pub nonzero_parameters: usize,
    pub bytes_f32: usize,
    pub bytes_i8: usize,
}

pub fn profile_parameter_matrix(matrix: &[Vec<f32>]) -> Result<EdgeParameterProfile> {
    if matrix.is_empty() { return Err(SdkError::EmptyDataset); }
    let mut parameter_count = 0_usize;
    let mut nonzero_parameters = 0_usize;
    for row in matrix {
        if row.is_empty() { return Err(SdkError::EmptyFeatures); }
        if row.iter().any(|value| !value.is_finite()) { return Err(SdkError::InvalidArgument("parameters must be finite".into())); }
        parameter_count = parameter_count.checked_add(row.len()).ok_or_else(|| SdkError::InvalidArgument("parameter count overflow".into()))?;
        nonzero_parameters = nonzero_parameters.checked_add(row.iter().filter(|value| **value != 0.0).count()).ok_or_else(|| SdkError::InvalidArgument("nonzero count overflow".into()))?;
    }
    if parameter_count > 16_777_216 { return Err(SdkError::DimensionLimit { actual: parameter_count, max: 16_777_216 }); }
    Ok(EdgeParameterProfile { parameter_count, nonzero_parameters, bytes_f32: parameter_count.saturating_mul(4), bytes_i8: parameter_count })
}
