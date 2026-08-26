use crate::{Result, SdkError};

pub(crate) fn fit_logistic(
    samples: &[Vec<f32>],
    labels: &[bool],
    epochs: usize,
    learning_rate: f32,
    l2: f32,
) -> Result<(Vec<f32>, f32)> {
    if samples.is_empty() {
        return Err(SdkError::EmptyDataset);
    }
    if samples.len() != labels.len() {
        return Err(SdkError::DimensionMismatch {
            expected: samples.len(),
            actual: labels.len(),
        });
    }
    if epochs == 0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
        || !l2.is_finite()
        || l2 < 0.0
    {
        return Err(SdkError::InvalidArgument(
            "invalid logistic training hyperparameters".into(),
        ));
    }
    let dim = samples[0].len();
    if dim == 0 {
        return Err(SdkError::EmptyFeatures);
    }
    for sample in samples {
        if sample.len() != dim {
            return Err(SdkError::DimensionMismatch {
                expected: dim,
                actual: sample.len(),
            });
        }
        if let Some((index, _)) = sample.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(SdkError::NonFiniteValue { index });
        }
    }
    let mut weights = vec![0.0_f32; dim];
    let mut bias = 0.0_f32;
    let n = samples.len() as f32;
    for _ in 0..epochs {
        let mut grad_w = vec![0.0_f32; dim];
        let mut grad_b = 0.0_f32;
        for (sample, label) in samples.iter().zip(labels) {
            let p = sigmoid(dot(&weights, sample)? + bias);
            let y = if *label { 1.0 } else { 0.0 };
            let error = p - y;
            grad_b += error;
            for (g, x) in grad_w.iter_mut().zip(sample) {
                *g += error * *x;
            }
        }
        bias -= learning_rate * (grad_b / n);
        for (w, g) in weights.iter_mut().zip(grad_w) {
            *w -= learning_rate * (g / n + l2 * *w);
        }
    }
    Ok((weights, bias))
}

pub(crate) fn predict_logistic(weights: &[f32], bias: f32, features: &[f32]) -> Result<f32> {
    if weights.len() != features.len() {
        return Err(SdkError::DimensionMismatch {
            expected: weights.len(),
            actual: features.len(),
        });
    }
    Ok(sigmoid(dot(weights, features)? + bias))
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(SdkError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let value: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    if !value.is_finite() {
        return Err(SdkError::InvalidArgument(
            "dot product became non-finite".into(),
        ));
    }
    Ok(value)
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
