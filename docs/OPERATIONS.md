# Operations manual

## Verification

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

Generate and commit the release `Cargo.lock` before building distributable binaries, then use `--locked` in the release pipeline.

## Model startup

The server requires an immutable manifest/payload pair. Startup fails if byte limits, schema version, allowed model type, payload length, SHA-256, dimensions, finite-value checks or model relationships are invalid.

Environment variables:

- `CUAS_MODEL_MANIFEST` — manifest path;
- `CUAS_MODEL_PAYLOAD` — payload path;
- `BIND_ADDR` — default `127.0.0.1:8080` for direct execution.

The container sets `BIND_ADDR=0.0.0.0:8080`. Mount the two model files read-only. Terminate TLS and authentication at ingress; enforce rate limits and network policy there.

## Endpoints

- `GET /healthz` → `204`;
- `POST /v1/perception` → `PerceptionResult`.

Request bodies are capped at 256 KiB. The service has no endpoint for actuator control and no outbound network dependency.

## Tracking deployment

`TrackManager` is stateful and is not part of the stateless HTTP service. In a distributed system, partition tracking by sensor site or track shard and persist/recover state in an application-owned service if continuity across process restarts is required.
