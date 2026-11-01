#!/usr/bin/env python3
"""Compares a Track A binary with the official reference on hostile random inputs.

usage: fuzz_a.py <binary> <reference a_ingest.py> <rounds> [first seed]

Each round writes a log, runs the reference for the answer, then the binary with several forced
thread counts (FORCE_THREADS, in dev builds). Lines are arranged to defeat the guess that an hour
ends soon after the next one starts: sorted, reversed, shuffled, with stragglers, whole hours
appended late, far hours interleaved. Names and latencies sit on every boundary the code has.
The binary reads /data/requests.log and writes /out/summary.txt.
"""
import hashlib
import importlib.util
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta

binary, reference, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
first_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
spec = importlib.util.spec_from_file_location("ref", reference)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
data = "/tmp/fuzzdata"
ref_out = "/tmp/fuzzref"
os.makedirs(data, exist_ok=True)
os.makedirs(ref_out, exist_ok=True)
if os.path.islink("/data"):
    os.unlink("/data")
os.symlink(data, "/data")

START = datetime(2026, 10, 30, 18)
SPAN_US = 72 * 3600 * 1000000
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
STATUS = [200, 200, 200, 201, 301, 404, 499, 500, 503, 599, 100, 599]
LATENCY_EDGES = [0, 1, 9, 10, 99, 100, 9999, 99999, 999999, 1048574, 1048575, 1048576, 1048577, 2097151,
                 99999999, 100000000, 100000001, 999999999, 1000000000, 4294967294, 4294967295]
ECHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-_.{}:"
CHUNKS = [1 << 16, 1 << 18]


def stamp(us):
    t = START + timedelta(microseconds=us)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond:06d}Z"


def name_of_length(rng, n):
    return "/" + "".join(rng.choice(ECHARS) for _ in range(n - 1))


def catalogue(rng):
    names = {"/", "/health", "/healthz", "/v2/orders", "/v2/orders-export", "/v2/orders.csv", "/v2/orders/{id}",
             "/v2/orders/{id}/items", "/v2/ordersByDate", "/v2/orders_legacy", "/v2/orderstatus"}
    size = rng.choice([1, 2, 8, 9, 40, 300, 1800])
    lengths = [1, 2, 3, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 79, 80, 81, 95, 96, 97, 111, 112, 113, 124, 125]
    # Names that differ only late, and along one of AES's diagonals, where a weak hash collides.
    for _ in range(size // 4):
        base = name_of_length(rng, rng.choice([64, 80, 96, 112, 125]))
        for _ in range(4):
            chars = list(base)
            at = rng.randrange(1, len(chars))
            chars[at] = rng.choice(ECHARS)
            if len(chars) > 16:
                other = (at + 5) % len(chars) or 1
                chars[other] = rng.choice(ECHARS)
            names.add("".join(chars))
    while len(names) < size:
        names.add(name_of_length(rng, rng.choice(lengths + [rng.randrange(1, 126)])))
    names = sorted(names)
    rng.shuffle(names)
    return names[:1842]


def times(rng, n):
    kind = rng.choice(["sorted", "sorted", "jitter", "shards", "reverse", "shuffled", "stragglers", "late_hour",
                       "far_hours", "one_hour", "two_hours", "first_chunk_late", "last_first", "boundaries"])
    if kind == "one_hour":
        h = rng.randrange(72) * 3600 * 1000000
        base = [h + rng.randrange(3600 * 1000000) for _ in range(n)]
    elif kind == "two_hours":
        a, b = rng.randrange(72), rng.randrange(72)
        base = [rng.choice([a, b]) * 3600 * 1000000 + rng.randrange(3600 * 1000000) for _ in range(n)]
    elif kind == "boundaries":
        base = [rng.randrange(1, 72) * 3600 * 1000000 - rng.choice([0, 1]) for _ in range(n)]
    else:
        base = [rng.randrange(SPAN_US) for _ in range(n)]
    base.sort()
    if kind == "jitter":
        window = rng.choice([100, 2000])
        for i in range(len(base)):
            j = min(len(base) - 1, i + rng.randrange(window))
            base[i], base[j] = base[j], base[i]
    elif kind == "shards":
        # Like the generator: several sorted streams, shipped in chunks by arrival.
        shards = [[] for _ in range(8)]
        for t in base:
            shards[rng.randrange(8)].append(t)
        chunks = []
        for s in shards:
            i = 0
            while i < len(s):
                size = rng.randrange(1, 400)
                chunks.append((s[min(i + size, len(s)) - 1] + rng.randrange(600 * 1000000), s[i:i + size]))
                i += size
        chunks.sort(key=lambda c: c[0])
        base = [t for _, c in chunks for t in c]
    elif kind == "reverse":
        base.reverse()
    elif kind == "shuffled":
        rng.shuffle(base)
    elif kind == "stragglers":
        for _ in range(rng.choice([1, 2, 10, n // 100 + 1])):
            i = rng.randrange(n)
            base.append(base.pop(i))
    elif kind == "late_hour":
        h = rng.randrange(72)
        base = [t for t in base if t // 3600000000 != h] + [t for t in base if t // 3600000000 == h]
    elif kind == "far_hours":
        for i in range(0, len(base), rng.choice([2, 7, 1000])):
            base[i] = rng.randrange(2) * rng.randrange(SPAN_US)
    elif kind == "first_chunk_late":
        # An early chunk starts far ahead, so an hour looks over long before it is.
        for i in range(0, min(len(base), 3000), 1):
            if i % 800 == 0:
                base[i] = SPAN_US - 1 - rng.randrange(1000000)
    elif kind == "last_first":
        base[0], base[-1] = base[-1], base[0]
    return kind, base


def make(seed):
    rng = random.Random(seed)
    names = catalogue(rng)
    n = rng.choice([1, 2, 3, 10, 100, 1000, 3000, 9000, 9000, 30000, 30000, 60000])
    kind, ts = times(rng, n)
    edge_rate = rng.choice([0.0, 0.01, 0.2, 1.0])
    skew = rng.choice([0.5, 1.0, 3.0])
    chunk = rng.choice(CHUNKS)
    align = rng.random() < 0.5
    out = []
    at = 0
    for t in ts:
        name = names[min(int(rng.paretovariate(skew)) - 1, len(names) - 1)]
        latency = rng.choice(LATENCY_EDGES) if rng.random() < edge_rate else rng.randrange(rng.choice([100, 10000, 1 << 20, 1 << 32]))
        status = rng.choice(STATUS)
        method = rng.choice(METHODS)
        line = f"{stamp(t)} {method} {name} {status} {latency}\n"
        if align:
            # Sometimes end a line exactly on a chunk boundary, or one byte to either side.
            room = chunk - (at + len(line)) % chunk + rng.choice([-1, 0, 1])
            if 0 < room < 110:
                longer = name + "x" * room
                if len(longer) <= 125 and (longer in names or len(names) < 1842):
                    if longer not in names:
                        names.append(longer)
                    line = f"{stamp(t)} {method} {longer} {status} {latency}\n"
        out.append(line)
        at += len(line)
    with open(os.path.join(data, "requests.log"), "w", encoding="ascii", newline="\n") as f:
        f.write("".join(out))
    return kind, n


def digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


failures = restarts = runs = 0
kinds = {}
for seed in range(first_seed, first_seed + rounds):
    kind, n = make(seed)
    ref.main(data, ref_out)
    want = digest(os.path.join(ref_out, "summary.txt"))
    for threads in [1, 2, 3, 12]:
        path = "/out/summary.txt"
        if os.path.exists(path):
            os.unlink(path)
        env = dict(os.environ, FORCE_THREADS=str(threads))
        p = subprocess.run([binary], env=env, capture_output=True, timeout=120)
        runs += 1
        said = p.stderr.decode(errors="replace")
        again = "starting again" in said
        restarts += again
        kinds[kind] = kinds.get(kind, 0) + again
        ok = p.returncode == 0 and os.path.exists(path) and digest(path) == want
        if not ok:
            failures += 1
            print(f"FAIL seed {seed} kind {kind} lines {n} threads {threads} exit {p.returncode}", flush=True)
            subprocess.run(["cp", os.path.join(data, "requests.log"), f"/tmp/fail-{seed}.log"])
print(f"fuzz: {rounds} logs, {runs} runs, {failures} failures, {restarts} runs restarted, restarts by kind {kinds}")
sys.exit(1 if failures else 0)
