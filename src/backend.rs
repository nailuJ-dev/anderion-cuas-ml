use crate::{Result, SdkError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    External(String),
}

pub trait ComputeBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn matvec(&self, matrix: &[Vec<f32>], vector: &[f32]) -> Result<Vec<f32>>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }
    fn matvec(&self, matrix: &[Vec<f32>], vector: &[f32]) -> Result<Vec<f32>> {
        if matrix.is_empty() || vector.is_empty() {
            return Err(SdkError::InvalidArgument(
                "matrix and vector must be non-empty".into(),
            ));
        }
        if vector.len() > 1_048_576 || matrix.len() > 1_048_576 {
            return Err(SdkError::DimensionLimit {
                actual: vector.len().max(matrix.len()),
                max: 1_048_576,
            });
        }
        for row in matrix {
            if row.len() != vector.len() {
                return Err(SdkError::DimensionMismatch {
                    expected: vector.len(),
                    actual: row.len(),
                });
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(SdkError::InvalidArgument(
                    "matrix values must be finite".into(),
                ));
            }
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(SdkError::InvalidArgument(
                "vector values must be finite".into(),
            ));
        }
        Ok(matrix
            .iter()
            .map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
            .collect())
    }
}
