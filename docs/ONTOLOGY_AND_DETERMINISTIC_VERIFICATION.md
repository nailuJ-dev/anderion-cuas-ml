# Ontology and Deterministic Verification

## Purpose

This layer adds semantic consistency, recurring-pattern discovery and reproducible verification to the standalone C-UAS perception SDK. It remains perception-only and is opt-in; the existing `PerceptionPipeline` API is unchanged.

## Lightweight ontology

The ontology is a bounded typed Rust graph, not RDF/OWL and not a remote knowledge-graph dependency. The reference schema contains `Observation`, `Sensor`, `Candidate`, `ObjectClass`, `Track`, `LocationEstimate`, `BehaviourPattern`, `HistoricalPattern`, and `Evidence`.

Reference relations cover observation/sensor provenance, candidate classification, track association, localization, behaviour, historical resemblance and evidence. `validate_reference_schema()` validates relation signatures and one-to-one cardinalities with deterministic violation ordering. The graph is limited to 65,536 nodes and 262,144 relations.

`semantic_graph_for_perception()` maps a normal perception result into this schema. Callers can then append track, evidence and behaviour facts without depending on an external ontology service.

## Pattern engine

`PatternEngine` detects repeated temporal sequences and co-occurrence patterns over semantic tokens. Each token can carry an optional embedding-cluster identifier, enabling recurrence checks that combine semantic class and vector-space grouping.

The engine sorts by timestamp and canonical token ordering before analysis and uses ordered maps/sets throughout. Reordering the input collection therefore cannot change the reported patterns.

## Deterministic verification layer

`VerifiedPerceptionPipeline` composes the existing pipeline with semantic validation and certification. A certificate binds:

- the canonical observation and sensor identifiers/features;
- model and configuration SHA-256 digests supplied by the caller;
- ontology and pipeline versions;
- seed;
- exact detection/classification/localization/embedding result;
- a fixed-point decision digest;
- ontology consistency state;
- deterministic verification decision.

The reference policy can abstain on excessive detection or classification uncertainty, abstain on unknown/low-confidence detected objects, and request review for semantic contradictions or insufficient localization confidence.

## Replay

`VerifiedPerceptionPipeline::replay()` returns `Exact`, `DecisionEquivalent`, or `NonReproducible`. Exact mode compares the complete canonical result digest. Decision-equivalent mode compares the deterministic fixed-point decision representation and is intentionally weaker.

Changing the model digest, configuration digest, ontology version, pipeline version or seed breaks context equality even when the resulting class label happens to remain the same.

## Security and guarantees

Critical probabilities, embedding dimensions/finite values, identifiers and localization values are revalidated immediately before certification to defend against malformed values created through deserialization. Ontology and pattern structures are bounded.

The certificate proves deterministic evidence and replayability for the supplied inference context; it does not prove physical ground truth. An external nondeterministic backend can still be used, but replay will expose output divergence.

This layer adds no jamming, takeover, targeting, engagement, neutralization, actuation or effector-control logic. It requires no graph server, private crate, private registry, remote inference service or non-public runtime.
