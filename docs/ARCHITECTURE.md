# Architecture

The SDK is a sensor-agnostic perception pipeline over caller-supplied feature vectors. Validated observations flow through encoding, detection, classification/OOD, optional localization and uncertainty. Learned association and track management consume embeddings and timestamps; no component controls an actuator.

P1 adds sensor-domain adaptation, weak supervision, temporal self-attention, robustness evaluation, drift monitoring and learned reliability fusion.

P2 adds:

1. Spatial ML: `GraphMessagePasser`, `NeuralLocalizer` and `AutoregressiveTrajectoryPredictor`.
2. Multimodal ML: `MultimodalSelfAttention`, `MultimodalTransformerEncoder`, `PairedModalAligner` and uncertainty-weighted localization fusion.
3. Distributed/compressed learning: `FederatedAverager`, soft-label distillation and symmetric quantization.
4. Runtime portability: generic `ComputeBackend`, CPU reference backend, weighted perception ensembles, edge parameter profiling and wasm32 core validation.

Federated aggregation is deliberately transport-free: callers provide bounded deltas in process. External compute backends are explicit trait implementations supplied by the caller; the SDK performs no discovery or network access.

The architecture ends at informational perception and prediction outputs. There is no downstream command path for jamming, engagement, neutralization or effector control.

## Reliability layer (0.3)

The opt-in reliability layer is split into four independent modules: `ontology` for typed semantic consistency, `pattern` for deterministic recurrence analysis, `verification` for canonical digests/certificates/replay, and `verified_pipeline` for composition with the existing inference pipeline. The existing ML pipeline remains unchanged.


## Optional operational enrichment (0.4)

```text
caller observation / operator-authorized sensing frame
        |
        v
optional DegarblingModel
        |
        v
PerceptionPipeline (unchanged P0/P1/P2 ML)
        |
        +--> raw class scores / uncertainty / localization
        |
        v
optional CooperativeCorrelator <--- caller-supplied AIS / ADS-B / Remote ID
        |
        v
EnhancedPerceptionResult
        |
        +--> auditable cooperative disposition
        +--> optional ontology evidence
```

5G-MIMO/ISAC is intentionally separated into an operator integration port. The SDK validates authorization/session/capability contracts and consumes resulting sensing features; it does not issue RAN or waveform commands itself.
