#!/usr/bin/env python3
"""Prints our standing on each track from the live board: the gap to first place, or our lead over second.

    python -I tools/gaps.py

Gap and lead are in machine points, with the K ratio behind them. Our latest run is shown too,
as final scoring uses the latest commit rather than the best run.
"""
import json
import math
import time
import urllib.request

FEED = "https://s3.amazonaws.com/raptors.dev/speeddemon.dev/2026/board/board.json"
SLUG_PREFIX = "another-ello"
TRACKS = "abcd"


def fetch():
    # The feed is republished every few seconds, so a read can come back truncated.
    for attempt in range(10):
        try:
            request = urllib.request.Request(f"{FEED}?t={int(time.time())}", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 9:
                raise
            time.sleep(3)


def axes(run):
    return f"{run['T_s']:.4f}, {run['M_bytes'] / 1e6:.1f}, {run['S']}"


def track_row(board, track):
    rows = [(t["tracks"][track]["best"], t) for t in board["teams"] if t["tracks"].get(track, {}).get("best")]
    rows.sort(key=lambda row: -row[0]["K"])
    rank = next((i for i, (_, t) in enumerate(rows) if t["slug"].startswith(SLUG_PREFIX)), None)
    if rank is None:
        return f"| {track.upper()} | no scored run | | | | | | |"
    ours = rows[rank][0]
    rival, team = rows[1] if rank == 0 else rows[0]
    points = ours["points"] - rival["points"]
    if rank == 0:
        standing = f"**leads by {points:.2f}** (K +{(ours['K'] / rival['K'] - 1) * 100:.1f}%) over {team['team']}"
    else:
        standing = f"**{-points:.2f} behind** {team['team']} (needs K +{(rival['K'] / ours['K'] - 1) * 100:.1f}%)"
    latest = rows[rank][1]["tracks"][track].get("latest") or {}
    latest_text = f"{latest['points']:.2f}, K {latest['K']:.2f}" if latest.get("K") else latest.get("stage", "none")
    return (f"| {track.upper()} | {ours['points']:.2f} | {ours['K']:.2f} | {rank + 1}/{len(rows)} | {standing} "
            f"| {rival['K']:.2f}: {axes(rival)} | {axes(ours)}, {ours['run_id']} | {latest_text} |")


def overall_rows(board):
    rows = []
    for team in board["teams"]:
        points = sorted((t["best"]["points"] for t in team["tracks"].values() if t.get("best")), reverse=True)[:3]
        if len(points) == 3 and all(points):
            rows.append((math.prod(points) ** (1 / 3), team))
    rows.sort(key=lambda row: -row[0])
    return rows


def main():
    board = fetch()
    print(f"Board at {board['generated_at']}\n")
    print("| Track | Points | K | Rank | Standing | Rival K: T, M, S | Our T, M, S, best run | Latest run |")
    print("|---|---|---|---|---|---|---|---|")
    for track in TRACKS:
        print(track_row(board, track))
    rows = overall_rows(board)
    rank = next(i for i, (_, t) in enumerate(rows) if t["slug"].startswith(SLUG_PREFIX))
    ours = rows[rank][0]
    rival, team = rows[1] if rank == 0 else rows[0]
    word = f"leads by {ours - rival:.2f} over" if rank == 0 else f"{rival - ours:.2f} behind"
    print(f"\nOverall machine score {ours:.2f}, rank {rank + 1} of {len(rows)}, {word} {team['team']} at {rival:.2f} "
          "(geometric mean of the best three, judges not included).")


if __name__ == "__main__":
    main()
