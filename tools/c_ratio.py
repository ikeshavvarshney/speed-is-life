#!/usr/bin/env python3
"""Lists recent pushes from push-log.md with Track C's T relative to the other tracks in the same push."""
import re, sys
rows, cur = [], {}
for line in open("push-log.md", encoding="utf8"):
    if line.startswith("## "):
        cur = {}
    if line.startswith("Commit:"):
        cur["commit"] = line.split("`")[1]
    m = re.match(r"\| ([ABCD]) \| scored \| (run-\w+) \| ([\d.]+) \| ([\d.]+) \| (\d+) \| \d+ \| ([\d.]+) \| ([\d.]+) \|", line)
    if m:
        cur[m[1]] = m
        if m[1] == "D" and "C" in cur:
            rows.append(cur)
rows = [r for r in rows if "C" in r]
for r in rows[-int(sys.argv[1] if len(sys.argv) > 1 else 12):]:
    c = r["C"]; t = float(c[3])
    print(r.get("commit", "?"), c[2], f"T={t:.4f} M={c[4]} S={c[5]} K={c[6]} C/A={t/float(r['A'][3]):.3f} C/B={t/float(r['B'][3]):.3f}")
