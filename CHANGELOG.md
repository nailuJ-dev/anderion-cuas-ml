
## 0.5.0

- Add a one-command Golden Path from recorded sensor measurements to verified C-UAS perception.
- Add a deterministic 13-feature reference adapter and bundled drone/bird/aircraft/background fixtures.
- Demonstrate AIS comparison, ADS-B/Remote ID correlation and authorized recorded 5G-MIMO/ISAC sensing.
- Add Golden Path model/evaluation APIs, replay proof, CLI and CI smoke gate.
- Fixture metrics are explicitly non-operational and must not be reported as field accuracy.

# Changelog

## 0.5.1

- Add cross-repository integration CI against `spectra-sim`.
- Validate the simulator-to-SDK Golden Path contract on every main-branch change and pull request.


## 0.4.0

- Add deterministic cooperative-track correlation for caller-supplied AIS, ADS-B, Remote ID and custom identities.
- Add optional ontology augmentation for cooperative identity evidence.
- Add `DegarblingModel`, `IdentityDegarbler` and deterministic reference `PrototypeMaskDegarbler`.
- Add `EnhancedPerceptionPipeline` that composes optional degarbling and cooperative evidence without altering raw ML scores.
- Add operator-authorized 5G-MIMO/ISAC contracts, `OperatorManagedActive` sensing mode, recorded provider, bounded sessions and verification-context binding.
- Preserve the perception-only boundary: no jamming, takeover, neutralization, targeting, waveform generation, RAN scheduler control or vendor-specific base-station command implementation.

## 0.3.0 - 2026-08-11

- Added a typed, bounded micro-ontology and deterministic reference-schema consistency checks.
- Added deterministic recurring-sequence and co-occurrence pattern detection.
- Added canonical SHA-256 inference certificates, fixed-point decision digests and replay classification.
- Added opt-in verified pipeline wrappers without changing existing inference APIs.
- Kept the C-UAS SDK fully standalone with no private or remote runtime dependency.

## 0.2.0 - 2026-08-11

- Added P2 graph representation, neural localization and learned trajectory prediction.
- Added multimodal attention/transformer, self-supervised sensor alignment and uncertainty fusion.
- Added transport-free federated aggregation, soft-label distillation, quantization, heterogeneous compute, ensembles and edge profiling.
- Added wasm32 core CI validation and clarified the standalone perception boundary.

## 0.1.0 - 2026-08-11

- Initial P0/P1 reference SDK.
