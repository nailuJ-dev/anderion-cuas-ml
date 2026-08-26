#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/golden-path
docker build -f Dockerfile.golden -t anderion-cuas-golden:0.5.0 .
docker run --rm \
  -v "$PWD/artifacts:/app/artifacts" \
  anderion-cuas-golden:0.5.0 "$@"
