"""Writes distinct names to files for the hash check. usage: names.py"""
import csv
import os

csv.field_size_limit(10_000_000)
os.makedirs("/tmp/names", exist_ok=True)
for track, datasets in (("a", ["full-seed1", "full-seed2", "sample-seed3"]), ("c", ["full-seed1", "full-seed2", "sample-seed3"])):
    for name in datasets:
        seen = set()
        if track == "a":
            with open(f"/v/a/{name}/requests.log", "rb") as f:
                for line in f:
                    parts = line.split(b" ")
                    if len(parts) >= 3:
                        seen.add(parts[2])
        else:
            with open(f"/v/c/{name}/records.csv", "r", encoding="utf-8", newline="", errors="surrogateescape") as f:
                for row in csv.reader(f):
                    if len(row) >= 2 and "\n" not in row[1]:
                        seen.add(row[1].encode("utf-8", "surrogateescape"))
        out = f"/tmp/names/{track}-{name}.txt"
        with open(out, "wb") as f:
            for s in seen:
                f.write(s + b"\n")
        print(out, len(seen), "names, longest", max(map(len, seen)), flush=True)
