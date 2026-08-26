# P0 / P1 / P2 Coverage

## P0
Validated observations/embeddings; detection; multiclass classification; few-shot; open-set/OOD; uncertainty; supervised localization; measurement quality; learned association/tracking; temporal classification; learned sensor reliability fusion; bounded verified artifacts; benchmarks; optional HTTP serving.

## P1
Multi-observation fusion; temporal self-attention; probabilistic localization; sensor-domain adaptation; weak supervision; incremental few-shot learning; calibration; drift monitoring; adversarial evaluation; synthetic trajectories; dataset manifests and grouped splits.

## P2
Graph message passing; neural localization; learned autoregressive trajectory prediction; multimodal self-attention and transformer-style encoding; self-supervised sensor alignment; uncertainty-weighted localization fusion; transport-free federated delta aggregation; soft-label distillation; symmetric quantization; heterogeneous compute contract with CPU backend; weighted perception ensembles; edge parameter profiling; wasm32 core gate.

The SDK remains perception-only and every default P2 component is implemented inside this repository.

## Cross-cutting reliability (0.3)

Typed bounded micro-ontology; reference-schema relation/cardinality validation; semantic mapping from inference results; deterministic recurring-sequence/co-occurrence detection; SHA-256 canonical input/context/result certificates; fixed-point decision digests; opt-in verified pipeline; exact/decision-equivalent/non-reproducible replay classification.


## 0.4 optional operational extensions

- Cooperative correlation: AIS, ADS-B, Remote ID and custom normalized tracks.
- Deterministic correlation ranking with spatial, temporal, optional velocity and source-confidence evidence.
- Optional ML degarbling with identity/no-op and prototype-mask reference implementations.
- Operator-authorized 5G-MIMO/ISAC provider contract and offline recorded provider.
- `OperatorManagedActive` sensing mode behind explicit operator authorization.
- Enhanced perception output that keeps raw ML predictions separate from cooperative evidence.
