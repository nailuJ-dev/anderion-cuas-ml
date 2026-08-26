# Standalone Boundary

This repository is a complete standalone perception SDK. Its default build has no dependency on non-public crates, remote inference services, remote model registries, hidden datasets, external feature-generation systems or undisclosed model formats.

The caller supplies feature observations, labels, embeddings, model deltas and optional extension-trait implementations directly. The SDK performs no automatic model download or service discovery.

Out of scope by design:

- jamming or protocol takeover;
- effector or weapon control;
- engagement or neutralization logic;
- autonomous actuation commands;
- automatic network egress from the ML runtime.

Detection, classification, tracking, localization, trajectory prediction and uncertainty outputs are informational perception outputs only.

The ontology and deterministic-verification layers are implemented entirely in this repository. They do not resolve schemas, graphs, model metadata or verification evidence through a private or remote service.

