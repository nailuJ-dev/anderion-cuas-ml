# Security policy

## Scope and trust model

Treat every observation, dataset, JSON request, model manifest, and model payload as untrusted input. The library forbids `unsafe` code and validates dimensions, finite numeric values, probabilities, schema versions, model types, artifact sizes, and SHA-256 integrity before inference.

The reference HTTP service is intentionally small and stateless. It defaults to loopback when run directly. The container binds on all interfaces for orchestration, so production deployments must place it behind an authenticated TLS ingress/API gateway with rate limiting, request logging policy, network policy, and deployment-specific authorization.

The SDK does not upload observations or model data and contains no telemetry network client. Application integrators remain responsible for logging/redaction choices around the SDK.

## Artifact handling

Model payloads are read through bounded readers rather than unbounded `read_to_end` calls. A manifest specifies schema version, model type, payload length and SHA-256. Deserialized reference models are revalidated because serialization frameworks can bypass constructors.

Do not load executable model formats or plugins from untrusted paths in privileged processes. External backends should enforce equivalent size, provenance, signature and sandbox policies.

## Supply-chain controls

CI runs formatting, Clippy, unit/integration/property tests, documentation checks, RustSec audit and cargo-deny policy checks. Releases should pin container base images by digest and commit the generated `Cargo.lock` used to build release binaries.

## Reporting vulnerabilities

Use the repository's private vulnerability reporting / GitHub Security Advisory flow. Do not open a public issue containing exploit details for an unpatched vulnerability.

## Cooperative feeds and operator-authorized sensing

AIS, ADS-B, Remote ID and custom cooperative records are untrusted caller inputs. They are normalized and revalidated before correlation; the SDK never fetches cooperative feeds itself. Applications should authenticate and authorize feed providers independently.

`OperatorAuthorization.authorization_digest` binds a sensing session to an external authorization artifact. It is **not** a substitute for validating the operator's legal authority or cryptographic signature. A production operator adapter must validate its own authorization artifact before constructing `OperatorAuthorization`.

The SDK does not contain vendor-specific RAN commands, waveform generation, scheduler manipulation or hidden network egress. `ActiveSensingProvider` is an integration boundary; real active sensing must be implemented by the infrastructure operator under its own access controls, radio policy and regulatory obligations.
