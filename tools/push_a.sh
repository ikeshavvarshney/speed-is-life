#!/usr/bin/env bash
# Copies a candidate into track-a, pushes it only if the bench shows PASS on every dataset, then logs the board result.
# Usage: tools/push_a.sh <candidate dir> "commit subject" "note for push-log"
set -eu
cd "$(dirname "$0")/.."
from="$1"; subject="$2"; note="$3"
cp "$from/main.cpp" "$from/Dockerfile" track-a/
[ -f "$from/p.profdata" ] && cp "$from/p.profdata" track-a/
[ -f "$from/a_ingest.py" ] || rm -f track-a/a_ingest.py
if git diff --quiet HEAD -- track-a; then
  git commit -q --allow-empty -m "$subject"
else
  result="$(bash tools/bench_a.sh 2 2>&1)" || { echo "$result"; echo "NOT PUSHED: bench failed"; exit 1; }
  echo "$result" | tail -14
  pass=$(echo "$result" | grep -c ' PASS ' || true)
  fail=$(echo "$result" | grep -c 'FAIL' || true)
  if [ "$pass" != 13 ] || [ "$fail" != 0 ]; then echo "NOT PUSHED: pass=$pass fail=$fail"; exit 1; fi
  git add track-a
  git commit -q -m "$subject" -- track-a
fi
git push -q origin speed-demon
git log -1 --format='pushed %h %s'
python -I tools/board_log.py --note "$note" | grep -E '^\| A|^Overall'
python -I tools/gaps.py | grep -E '^\| A|^Overall'
