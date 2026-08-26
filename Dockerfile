# syntax=docker/dockerfile:1.7
FROM rust:1.85-bookworm AS builder
WORKDIR /src
COPY . .
RUN cargo build --release --features server --bin anderion-cuas-serve

FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=builder /src/target/release/anderion-cuas-serve /usr/local/bin/anderion-cuas-serve
USER nonroot:nonroot
ENV BIND_ADDR=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/anderion-cuas-serve"]
