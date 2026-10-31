#!/usr/bin/env bash
# Builds track-b offline, verifies every dataset in the sd-data volume, then times full-seed1.
# Usage: tools/bench_b.sh [timed runs, default 5]
set -eu
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")/.."
runs="${1:-5}"
image=speeddemon-local/track-b
docker build -q --network=none -t "$image" track-b >/dev/null

run() {
  docker run --rm ${CPUS:+--cpuset-cpus=$CPUS} --network=none --read-only --tmpfs /tmp:rw,size=1g,exec \
    --tmpfs /out:rw,size=64m,mode=1777 --memory=8g --memory-swap=8g --pids-limit=512 \
    --cap-drop=ALL --security-opt=no-new-privileges --user 65534:65534 -e TRACK=b \
    --mount "type=volume,src=sd-data,dst=/data,volume-subpath=b/$1,readonly" \
    --mount "type=volume,src=sd-data,dst=/v,readonly" --entrypoint bash "$image" -c '
      TIMEFORMAT=%R
      t=$( { time /app/solution; } 2>&1 )
      read want < /v/expected/b-'"$1"'
      got=$(sha256sum /out/results.txt | cut -d" " -f1)
      [ "$got" = "$want" ] && ok=PASS || ok=FAIL
      echo "'"$1"' $ok ${t}s peak=$(( $(cat /sys/fs/cgroup/memory.peak) / 1000000 ))MB"'
}

for name in worked tiny-seed1 sample-seed1 full-seed1 sample-seed2 full-seed2 tiny-seed3 sample-seed3; do
  run "$name"
done
echo "--- timed"
for _ in $(seq "$runs"); do run full-seed1; done
