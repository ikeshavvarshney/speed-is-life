#!/usr/bin/env bash
# Pushes a diagnostic build of track-c, waits for the board, prints T and M, then restores the code and commits that.
# Usage: tools/diag_c.sh <python edit script> "commit subject" "note"
# Refuses to run on uncommitted track-c changes. The restore is committed but not pushed: the next push carries it.
set -eu
cd "$(dirname "$0")/.."
git diff --quiet HEAD -- track-c || { echo "track-c has uncommitted changes"; exit 1; }
real="$(git rev-parse HEAD)"
python -I "$1"
docker build -q --network=none -t speeddemon-local/track-c track-c >/dev/null
git add track-c
git commit -q -m "$2"
git push -q origin speed-demon 2>&1 | grep -v 'LF will' || true
python -I tools/board_log.py --note "$3" 2>&1 | grep -E '^\| C' | cut -c1-110
python -I -c "
import json,urllib.request,time
b=json.load(urllib.request.urlopen('https://s3.amazonaws.com/raptors.dev/speeddemon.dev/2026/board/board.json?t=%d'%time.time()))
me=[t for t in b['teams'] if 'another-ello' in t['slug']][0]
x=me['tracks']['c']['latest']
print('T_s',x['T_s'],'M_bytes',x['M_bytes'],'run',x['run_id'],'gates',x['gates'])
"
git checkout -q "$real" -- track-c
git add track-c
git commit -q -m "Track C: restore real code after diagnostic"
git diff --quiet "$real" HEAD -- track-c && echo "restored"
