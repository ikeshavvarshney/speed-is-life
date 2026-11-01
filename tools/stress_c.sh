#!/usr/bin/env bash
# Repeats track-c with 12 threads forced on, on datasets of several seeds, and fails on any wrong hash.
# Usage: tools/stress_c.sh [runs per dataset, default 10]
set -eu
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")/.."
runs="${1:-10}"
image=speeddemon-local/track-c-forced
ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT
cp -r track-c/. "$ctx"
sed -i 's/threads = .*CPU_COUNT(&cpus).*;/threads = 12;/' "$ctx/main.cpp"
grep -q 'threads = 12;' "$ctx/main.cpp"
docker build -q --network=none -t "$image" "$(cygpath -w "$ctx")" >/dev/null

fail=0
for name in worked tiny-seed1 sample-seed1 sample-seed2 sample-seed3 full-seed2; do
  n=$runs
  case "$name" in full-*) n=$(( (runs + 4) / 5 ));; esac
  result=$(docker run --rm --ulimit core=1 --network=none --read-only --tmpfs /tmp:rw,size=1g,exec \
    --tmpfs /out:rw,size=64m,mode=1777 --memory=8g --memory-swap=8g --pids-limit=512 \
    --cap-drop=ALL --security-opt=no-new-privileges --user 65534:65534 -e TRACK=c \
    --mount "type=volume,src=sd-data,dst=/data,volume-subpath=c/$name,readonly" \
    --mount "type=volume,src=sd-data,dst=/v,readonly" --entrypoint bash "$image" -c '
      read want < /v/expected/c-'"$name"'
      pass=0; bad=0
      for _ in $(seq '"$n"'); do
        rm -f /out/aggregates.txt
        /app/solution || true
        got=$(sha256sum /out/aggregates.txt 2>/dev/null | cut -d" " -f1)
        if [ "$got" = "$want" ]; then pass=$((pass+1)); else bad=$((bad+1)); fi
      done
      echo "'"$name"' pass=$pass bad=$bad peak=$(( $(cat /sys/fs/cgroup/memory.peak) / 1000000 ))MB"')
  echo "$result"
  case "$result" in *bad=0*) ;; *) fail=1;; esac
done
[ "$fail" = 0 ] && echo "ALL PASS" || { echo "FAILED"; exit 1; }
