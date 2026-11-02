#!/usr/bin/env bash
# Pushes track-c only if the bench shows PASS on every dataset, then logs the board result.
# Usage: tools/push_c.sh "commit subject" "note for push-log" [--stress]
set -eu
cd "$(dirname "$0")/.."
subject="$1"; note="$2"; stress="${3:-}"
if git diff --quiet HEAD -- track-c; then echo "NOT PUSHED: track-c has no change"; exit 1; fi
result="$(bash tools/bench_c.sh 2 2>&1)" || { echo "$result"; echo "NOT PUSHED: bench failed"; exit 1; }
echo "$result" | tail -11
pass=$(echo "$result" | grep -c ' PASS ' || true)
fail=$(echo "$result" | grep -c 'FAIL' || true)
if [ "$pass" != 10 ] || [ "$fail" != 0 ]; then echo "NOT PUSHED: pass=$pass fail=$fail"; exit 1; fi
if [ "$stress" = "--stress" ]; then
  bash tools/stress_c.sh 15 2>&1 | tail -7 | tee /dev/stderr | grep -q '^ALL PASS$' || { echo "NOT PUSHED: stress failed"; exit 1; }
fi
git add track-c
# Path-limited, as other sessions may have files of their own tracks staged.
git commit -q -m "$subject" -- track-c
git push -q origin speed-demon 2>&1 | grep -v 'LF will' || true
remote="$(git ls-remote origin speed-demon | cut -c1-40)"
[ "$remote" = "$(git rev-parse HEAD)" ] || { echo "PUSH DID NOT LAND: remote $remote"; exit 1; }
python -I tools/board_log.py --note "$note" 2>&1 | grep -E '^\| C' | cut -c1-125
python -I tools/c_ratio.py 4
