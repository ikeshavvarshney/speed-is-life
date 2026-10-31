#!/usr/bin/env bash
# Counts a track's scored lines the way the harness does: rustfmt with defaults, then scc.
# Needs the image from: docker build -t speeddemon/static-tools:2026-10-28 <harness>/harness/tools
# Usage: tools/count_s.sh a
set -eu
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")/.."
track="$(cygpath -w "$PWD/track-$1" 2>/dev/null || echo "$PWD/track-$1")"
docker run --rm --network=none -v "$track:/src:ro" --entrypoint sh speeddemon/static-tools:2026-10-28 -c '
  cp -r /src /tmp/t && cd /tmp/t
  find . -name "*.rs" -exec rustfmt --edition 2024 {} +
  rm -f README.md PERF.md Cargo.toml Cargo.lock .speed-demon.toml; rm -rf .cargo
  scc --no-cocomo --no-complexity . | grep -E "^(Rust|Dockerfile|Total)"'
