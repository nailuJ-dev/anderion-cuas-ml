# Anderion C-UAS ML

## Golden Path quickstart

Run an end-to-end reference C-UAS scenario with one command:

```bash
./scripts/run_golden_path.sh
```

The demo covers recorded sensing, detection/classification/localization, AIS/ADS-B/Remote ID correlation, ontology consistency and deterministic replay. A separate bundled scenario exercises the authorized recorded 5G-MIMO/ISAC provider contract. See `docs/GOLDEN_PATH.md`.


A standalone Rust SDK for machine-learning-based drone perception from caller-supplied feature observations. The repository builds and operates without hidden services, non-public crates, remote model registries or closed runtime components.

The system boundary is perception and decision-support output only. It contains no jamming, takeover, effector control, engagement, neutralization or autonomous actuation APIs.

## P0

- validated sensor-agnostic observations and embeddings;
- deterministic reference encoder;
- trainable binary detector and multiclass prototype classifier;
- few-shot class creation and OOD/open-set recognition;
- confidence, calibration and uncertainty;
- supervised 3D localization with residual uncertainty;
- learned measurement-quality and association models;
- bounded track manager, temporal classification and learned sensor reliability fusion;
- SHA-256 verified artifacts, benchmark utilities and optional stateless HTTP service.

## P1

- multi-observation reliability-weighted classification;
- temporal self-attention pooling;
- learned observation-to-track association;
- probabilistic localization;
- sensor-domain adaptation;
- weak-supervision label model;
- incremental few-shot adaptation;
- drift monitoring and bounded adversarial robustness evaluation;
- deterministic synthetic feature trajectories;
- grouped dataset splitting and version manifests.

## P2

- bounded graph message passing / GNN-style representation propagation;
- one-hidden-layer neural 3D localizer with learned residual uncertainty;
- learned autoregressive trajectory predictor producing informational future positions;
- multimodal self-attention and transformer-style residual embedding encoder;
- self-supervised sensor-domain alignment without class labels;
- uncertainty-weighted localization fusion;
- in-process federated model-delta aggregation with L2 clipping and no networking;
- soft-label knowledge distillation;
- symmetric 2–8 bit fake quantization / int8 representation;
- heterogeneous compute backend trait with CPU reference backend;
- weighted classification/detection ensembles;
- edge parameter profiling;
- wasm32 core compile gate.

## Quick start

```bash
cargo run --example basic
cargo test --all-features
```

Optional perception service:

```bash
cargo run --features server --example build_reference_bundle
CUAS_MODEL_MANIFEST=artifacts/perception-model.manifest.json \
CUAS_MODEL_PAYLOAD=artifacts/perception-model.json \
cargo run --features server --bin anderion-cuas-serve
```

## Perception boundary

```text
Caller sensor features
        |
        v
 Representation / embedding
        |
  +-----+------------------+
  |     |                  |
Detect Classify/OOD   Graph/Multimodal ML
  |     |                  |
  +-----+---------+--------+
                  |
          uncertainty models
             /          \
            v            v
      localization      tracks
            |             |
            +------v------+
             trajectory
             prediction
```

Trajectory prediction returns positions only; it does not generate control, targeting, engagement or effector commands.

All algorithms in the default SDK are present in this repository and consume only caller-supplied data. Public extension traits are generic and do not resolve or contact external implementations automatically.

## 0.4 cooperative correlation, degarbling and 5G-MIMO/ISAC

The SDK can enrich ML perception with **optional cooperative identity evidence** from caller-supplied, pre-parsed AIS, ADS-B, Remote ID, or custom tracks. `CooperativeCorrelator` performs bounded spatiotemporal/velocity correlation and returns a deterministic `CooperativeDisposition` without mutating the raw ML class scores. This keeps the ML output auditable while allowing applications to identify likely cooperative vessels, aircraft, or UAS and reduce false-positive escalation.

`EnhancedPerceptionPipeline` can also run an optional `DegarblingModel` before inference. The included `PrototypeMaskDegarbler` is a deterministic reference ML separator driven by non-negative learned/provided prototypes; `IdentityDegarbler` disables separation with zero integration cost.

For 5G-MIMO/ISAC, the crate exposes an `ActiveSensingProvider` contract. Every session requires a bounded `OperatorAuthorization`, a matching `SensingRequest`, supported infrastructure capabilities, and a validity window. The public implementation is `RecordedSensingProvider` for offline testing and replay. Real RAN/vendor adapters are intentionally external to this SDK and must be implemented and authorized by the operator. The SDK ships no waveform generator, base-station scheduler control, private RAN API, or vendor-specific command path.

`SensingMode::OperatorManagedActive` represents operator-controlled active sensing. Validated `SensingFrame` values convert into ordinary `Observation` values and can derive a deterministic verification context that binds the authorization digest and sensing configuration digest.

See `docs/COOPERATIVE_DEGARBLING_ISAC.md` for the complete data flow, trust boundary and integration examples.

## Security

`unsafe` is forbidden. Production source denies `unwrap`, `expect` and `panic` through Clippy. Input and model dimensions are bounded, serialized artifacts are hash/schema/size checked and revalidated, the server is stateless, and federated aggregation performs no networking. CI includes format, Clippy, tests, rustdoc, MSRV, dependency policy and wasm32 core compilation.

See `SECURITY.md`, `docs/ARCHITECTURE.md`, `docs/OPERATIONS.md` and `docs/P0_P1_P2_COVERAGE.md`.

## License

Apache-2.0.

## Ontology, recurring patterns, and deterministic verification

Version 0.3.0 adds an opt-in, self-contained reliability layer for perception results. `OntologyGraph` keeps observations, sensors, candidates, object classes, tracks, location estimates, evidence and behaviour patterns semantically coherent. `PatternEngine` detects deterministic recurring temporal sequences and co-occurrences. `VerifiedPerceptionPipeline` wraps the existing perception pipeline and emits a `ResultCertificate` binding canonical input, model/config digests, ontology/pipeline versions, exact result and deterministic decision digest.

Replay classifies a rerun as `Exact`, `DecisionEquivalent`, or `NonReproducible`. The verifier can abstain on excessive uncertainty and request review on semantic contradictions or weak localization confidence. It does not claim physical ground-truth correctness and does not add any effector-control, jamming, takeover, targeting or neutralization capability.

The entire layer is local and standalone: no graph server, remote model endpoint, private crate, private registry or non-public runtime is required.



## Physics simulator compatibility

This release is continuously tested against the public `spectra-sim` contract. Default compatibility target: `spectra-sim 0.1.1`. The integration is file-based JSON only; there is no Cargo or private-service dependency.

## Compatibility

| SDK | spectra-sim |
|---|---|
| 0.5.1 | 0.1.1 |
