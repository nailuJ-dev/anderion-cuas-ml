use crate::{
    ConsistencyReport, DeterministicVerifier, Observation, OntologyGraph, PerceptionPipeline,
    PerceptionResult, PerceptionVerificationPolicy, ReplayStatus, Result, ResultCertificate,
    VerificationContext, semantic_graph_for_perception,
};

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedPerceptionResult {
    result: PerceptionResult,
    ontology: OntologyGraph,
    consistency: ConsistencyReport,
    certificate: ResultCertificate,
}

impl VerifiedPerceptionResult {
    pub fn result(&self) -> &PerceptionResult {
        &self.result
    }
    pub fn ontology(&self) -> &OntologyGraph {
        &self.ontology
    }
    pub fn consistency(&self) -> &ConsistencyReport {
        &self.consistency
    }
    pub fn certificate(&self) -> &ResultCertificate {
        &self.certificate
    }
}

#[derive(Clone)]
pub struct VerifiedPerceptionPipeline {
    pipeline: PerceptionPipeline,
    context: VerificationContext,
    policy: PerceptionVerificationPolicy,
}

impl VerifiedPerceptionPipeline {
    pub fn new(
        pipeline: PerceptionPipeline,
        context: VerificationContext,
        policy: PerceptionVerificationPolicy,
    ) -> Self {
        Self {
            pipeline,
            context,
            policy,
        }
    }

    pub fn infer(&self, observation: &Observation) -> Result<VerifiedPerceptionResult> {
        let result = self.pipeline.infer(observation)?;
        let ontology = semantic_graph_for_perception(observation, &result)?;
        let consistency = ontology.validate_reference_schema();
        let certificate = DeterministicVerifier::verify_perception(
            observation,
            &result,
            &self.context,
            &self.policy,
            &consistency,
        )?;
        Ok(VerifiedPerceptionResult {
            result,
            ontology,
            consistency,
            certificate,
        })
    }

    pub fn replay(
        &self,
        observation: &Observation,
        original: &ResultCertificate,
    ) -> Result<(VerifiedPerceptionResult, ReplayStatus)> {
        let replayed = self.infer(observation)?;
        let status = DeterministicVerifier::compare_replay(original, replayed.certificate());
        Ok((replayed, status))
    }
}
