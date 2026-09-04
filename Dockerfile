# syntax=docker/dockerfile:1.7
FROM rust:1.85.1-bookworm AS builder
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked --features server --bin anderion-cuas-serve \
 && cp /src/target/release/anderion-cuas-serve /usr/local/bin/anderion-cuas-serve

FROM gcr.io/distroless/cc-debian12:nonroot
LABEL org.opencontainers.image.source="https://github.com/nailuJ-dev/anderion-cuas-ml"
COPY --from=builder /usr/local/bin/anderion-cuas-serve /usr/local/bin/anderion-cuas-serve
USER nonroot:nonroot
ENV BIND_ADDR=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/anderion-cuas-serve"]
