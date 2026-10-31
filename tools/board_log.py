#!/usr/bin/env python3
"""Waits for the board to score HEAD, then appends the results to push-log.md (gitignored).

    python tools/board_log.py [--note "what we tried and why"] [--timeout 900]
"""
import argparse
import json
import math
import os
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone

FEED = "https://s3.amazonaws.com/raptors.dev/speeddemon.dev/2026/board/board.json"
SLUG_PREFIX = "another-ello"
TRACKS = "abcd"
PENDING = {"queued", "building", "queued_to_run", "running", "dispatched"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "push-log.md")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def fetch():
    request = urllib.request.Request(f"{FEED}?t={int(time.time())}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def our_team(board):
    return next(t for t in board["teams"] if t["slug"].startswith(SLUG_PREFIX))


def finished(team, head):
    runs = [team["tracks"][t].get("latest") for t in TRACKS if t in team["tracks"]]
    return bool(runs) and all(
        r and r["commit"] == head and r["stage"] not in PENDING and (r["ok"] or r["failure"] or r["stage"] == "unchanged")
        for r in runs
    )


def rank_of(board, track, k):
    others = [t["tracks"][track]["best"]["K"] for t in board["teams"]
              if track in t["tracks"] and t["tracks"][track].get("best") and not t["slug"].startswith(SLUG_PREFIX)]
    return sum(1 for other in others if other > k) + 1, len(others) + 1


def track_rows(board, team, head):
    rows = []
    for track in TRACKS:
        info = team["tracks"].get(track)
        run = info and info.get("latest")
        if not run:
            rows.append(f"| {track.upper()} | no run | | | | | | | | |")
        elif run["commit"] != head:
            rows.append(f"| {track.upper()} | not run for this push | {run['run_id']} | | | | | | | |")
        elif run["stage"] != "scored":
            why = (run["failure"] or "").replace("|", "/").replace("\n", " ")
            rows.append(f"| {track.upper()} | {run['stage']} | {run['run_id']} | | | | | | | {why[:120]} |")
        else:
            best = info["best"]
            rank, total = rank_of(board, track, run["K"])
            note = "new best" if best["run_id"] == run["run_id"] else f"best stays {best['run_id']} (K {best['K']:.2f})"
            rows.append(f"| {track.upper()} | scored | {run['run_id']} | {run['T_s']:.4f} | {run['M_bytes'] / 1e6:.1f} "
                        f"| {run['S']} | {run['D']} | {run['K']:.2f} | {run['points']:.2f} | {rank}/{total}, {note} |")
    return rows


def overall(team):
    points = sorted((t["best"]["points"] for t in team["tracks"].values() if t.get("best")), reverse=True)[:3]
    if len(points) < 3:
        return f"needs 3 scored tracks, has {len(points)}"
    mean = math.prod(points) ** (1 / 3) if all(points) else 0
    return f"geometric mean of best three machine scores {mean:.2f} (judges not included)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    head = git("rev-parse", "HEAD")
    deadline = time.time() + args.timeout
    while True:
        board = fetch()
        team = our_team(board)
        if finished(team, head):
            break
        if time.time() > deadline:
            print(f"timed out waiting for {head[:7]}; latest runs are logged as they stand")
            break
        time.sleep(20)

    previous = re.findall(r"^Commit: `(\w+)`", open(LOG, encoding="utf-8").read(), re.M) if os.path.exists(LOG) else []
    span = f"{previous[-1]}..{head}" if previous else head
    changes = git("log", "--reverse", "--format=- %h %s", span)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"\n## {stamp}\n", f"Commit: `{head[:7]}`\n", "Changes:", changes or "- none"]
    if args.note:
        lines += ["", f"Notes: {args.note}"]
    lines += ["", "| Track | Result | Run ID | T (s) | M (MB) | S | D | K | Points | Rank / detail |",
              "|---|---|---|---|---|---|---|---|---|---|", *track_rows(board, team, head), "",
              f"Overall: {overall(team)}\n"]
    entry = "\n".join(lines)
    with open(LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(entry)
    print(entry)


if __name__ == "__main__":
    main()
