use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphEdge {
    pub from: usize,
    pub to: usize,
    pub weight: f32,
}

impl GraphEdge {
    pub fn new(from: usize, to: usize, weight: f32) -> Result<Self> {
        if !weight.is_finite() || weight < 0.0 {
            return Err(SdkError::InvalidArgument("graph edge weight must be finite and non-negative".into()));
        }
        Ok(Self { from, to, weight })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GraphMessagePasser {
    self_weight: f32,
    neighbor_weight: f32,
}

impl GraphMessagePasser {
    pub fn new(self_weight: f32, neighbor_weight: f32) -> Result<Self> {
        if !self_weight.is_finite() || !neighbor_weight.is_finite() || self_weight < 0.0 || neighbor_weight < 0.0 {
            return Err(SdkError::InvalidArgument("graph weights must be finite and non-negative".into()));
        }
        if self_weight + neighbor_weight <= f32::EPSILON {
            return Err(SdkError::InvalidArgument("graph weights require positive total mass".into()));
        }
        Ok(Self { self_weight, neighbor_weight })
    }

    pub fn propagate(&self, nodes: &[Embedding], edges: &[GraphEdge]) -> Result<Vec<Embedding>> {
        if nodes.is_empty() { return Err(SdkError::EmptyDataset); }
        if nodes.len() > 65_536 { return Err(SdkError::DimensionLimit { actual: nodes.len(), max: 65_536 }); }
        if edges.len() > 1_048_576 { return Err(SdkError::DimensionLimit { actual: edges.len(), max: 1_048_576 }); }
        let dim = nodes[0].dim();
        let elements = nodes.len().checked_mul(dim).ok_or_else(|| SdkError::InvalidArgument("graph element count overflow".into()))?;
        if elements > 4_194_304 {
            return Err(SdkError::DimensionLimit { actual: elements, max: 4_194_304 });
        }
        if nodes.iter().any(|node| node.dim() != dim) {
            return Err(SdkError::InvalidArgument("graph node dimensions must match".into()));
        }
        let mut sums = vec![vec![0.0_f32; dim]; nodes.len()];
        let mut masses = vec![0.0_f32; nodes.len()];
        for edge in edges {
            if edge.from >= nodes.len() || edge.to >= nodes.len() {
                return Err(SdkError::InvalidArgument("graph edge index out of range".into()));
            }
            if edge.weight <= 0.0 { continue; }
            for (dst, value) in sums[edge.to].iter_mut().zip(nodes[edge.from].values()) {
                *dst += edge.weight * *value;
            }
            masses[edge.to] += edge.weight;
        }
        let total = self.self_weight + self.neighbor_weight;
        let mut output = Vec::with_capacity(nodes.len());
        for index in 0..nodes.len() {
            let mut values = vec![0.0_f32; dim];
            for (axis, dst) in values.iter_mut().enumerate() {
                let neighbor = if masses[index] > f32::EPSILON { sums[index][axis] / masses[index] } else { nodes[index].values()[axis] };
                *dst = (self.self_weight * nodes[index].values()[axis] + self.neighbor_weight * neighbor) / total;
            }
            output.push(Embedding::new(values)?);
        }
        Ok(output)
    }
}
