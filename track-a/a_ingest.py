#!/usr/bin/env python3
"""Speed Demon 2026, Track A (Ingest): dataset generator.

    python3.13 generators/a_ingest.py --seed 1 --scale sample --out .data/a/sample-seed1

Writes one file, requests.log: one HTTP request per line, five fields separated by
single spaces, for example

    2026-10-30T18:00:00.114982Z GET /v2/orders/{id} 200 1841

specs/a-ingest.md is the contract (grammar, task, edge cases). How the data is shaped:

* Catalogue. Each seed has 1,842 endpoint templates: a fixed core (the site's examples,
  health checks, the root path, byte-order traps, the longest template) plus REST-style
  templates composed from a vocabulary. Each template gets a popularity rank (Zipf,
  exponent 1), a method mix, a status profile and a base latency.
* Time. 72 hours from 2026-10-30T18:00:00.000000Z to 2026-11-02T17:59:59.999999Z, with a
  fixed day/night traffic curve.
* Shards. Eight log shards each write non-decreasing timestamps. Shards 1-7 ship a chunk
  every 1 to 120 s of log time, 0 to 30 s late. Shard 0 uploads each hour as one chunk,
  5 to 10 minutes after the hour ends. requests.log is the chunks in arrival order, so
  it is not sorted by time.
* Edge cases. Every case the spec lists is injected on purpose: at a fixed rate in
  sample and full, at least once in tiny.

Determinism: the same seed and scale give the same bytes on any machine. The code uses
only random.Random methods built on the Mersenne Twister (random, getrandbits,
randrange, choice, choices, shuffle), integer arithmetic and exactly rounded float
multiplication. It never calls exp/log/pow, whose last bits differ between C libraries,
and it never iterates a set or relies on hash().
"""
import argparse
import heapq
from itertools import accumulate
import os
import sys
import time
from datetime import datetime, timedelta
from random import Random

FILE_NAME = "requests.log"
LINES = {"full": 3_100_000, "sample": 310_000, "tiny": 1_000}

START = datetime(2026, 10, 30, 18)  # UTC; the first hour bucket
HOURS = 72
SECOND_US = 1_000_000
MINUTE_US = 60 * SECOND_US
HOUR_US = 60 * MINUTE_US

SHARDS = 8                      # shard 0 is the hourly uploader, 1..7 stream
STREAM_CHUNK_MIN_US = 1 * SECOND_US
STREAM_CHUNK_SPAN_US = 119 * SECOND_US   # chunk length is 1 to 120 s of log time
STREAM_DELAY_SPAN_US = 30 * SECOND_US    # and it arrives 0 to 30 s after it closes
UPLOAD_LAG_MIN_US = 5 * MINUTE_US
UPLOAD_LAG_SPAN_US = 5 * MINUTE_US       # hourly uploads land 5 to 10 min after the hour

N_ENDPOINTS = 1842
METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")
LATENCY_MAX = 4_294_967_295

# Relative traffic by UTC hour of day, 00..23.
HOUR_OF_DAY_WEIGHT = (30, 26, 23, 21, 20, 21, 25, 32, 42, 54, 65, 74,
                      81, 87, 92, 96, 100, 99, 95, 89, 80, 68, 54, 40)

LONGEST_ENDPOINT = ("/internal/v1/tenants/{tenant_id}/projects/{project_id}"
                    "/pipelines/{pipeline_id}/runs/{run_id}/artifacts/{artifact_id}:download")
MAX_ENDPOINT_LEN = len(LONGEST_ENDPOINT)  # 125
HEALTH_ENDPOINTS = ("/", "/health", "/healthz")
# Present in every catalogue and every file. The /v2/orders family diverges right after
# a shared prefix on end-of-string, '-', '.', '/', 'B', '_' and 't', so bytewise order
# differs from case-insensitive, locale and per-segment orderings.
CORE_ENDPOINTS = HEALTH_ENDPOINTS + (
    "/v2/checkout",
    "/v2/orders",
    "/v2/orders-export",
    "/v2/orders.csv",
    "/v2/orders/{id}",
    "/v2/orders/{id}/items",
    "/v2/ordersByDate",
    "/v2/orders_legacy",
    "/v2/orderstatus",
    LONGEST_ENDPOINT,
)

PREFIXES = ("/v1", "/v2", "/v3", "/api/v1", "/api/v2", "/internal", "/internal/v1",
            "/admin", "/partner/v1", "/mobile/v4", "/public", "/beta")
RESOURCES = (
    "users", "orders", "carts", "payments", "invoices", "products", "categories",
    "reviews", "sessions", "tokens", "accounts", "addresses", "shipments", "returns",
    "refunds", "coupons", "subscriptions", "plans", "prices", "inventory", "warehouses",
    "stores", "merchants", "customers", "notifications", "messages", "webhooks",
    "events", "reports", "exports", "imports", "files", "images", "uploads",
    "documents", "comments", "followers", "feeds", "recommendations", "wishlists",
    "rewards", "ledgers", "transfers", "balances", "disputes", "payouts", "taxes",
    "currencies", "regions", "settings", "preferences", "roles", "permissions",
    "teams", "organizations", "projects", "tasks", "jobs", "schedules", "metrics",
    "alerts", "incidents", "deployments", "builds", "artifacts", "releases",
    "experiments", "segments", "campaigns", "devices", "apps", "keys", "secrets",
    "certificates", "domains", "quotas", "usage", "billing", "credits",
    "gift-cards", "shipping-rates", "line-items", "api-keys", "payment-methods",
    "price-lists", "audit-logs", "access-tokens", "feature-flags", "tax-rates",
    "return-labels", "order-items", "saved-searches", "push-tokens",
    "feature_toggles", "audit_events", "user_groups", "rate_limits", "cost_centers",
    "login_attempts", "ab_tests",
    "userProfiles", "paymentIntents", "shippingZones", "loyaltyTiers", "priceRules",
    "giftWraps", "storeHours",
)
PARAMS = ("{id}", "{user_id}", "{order_id}", "{account_id}", "{sku}", "{slug}",
          "{uuid}", "{tenant_id}", "{project_id}", "{version}", "{locale}", "{date}",
          "{code}", "{key}", "{name}")
ACTIONS = ("/search", "/export", "/import", "/cancel", "/refund", "/status",
           "/history", "/batch", "/preview", "/validate", "/confirm", "/retry",
           "/summary", "/count", "/stats", "/archive", "/restore", "/approve",
           "/reject", "/sync", "/verify", "/resend", ":batchGet", ":undelete", ":search")
SLOW_WORDS = ("search", "export", "reports", "stats", "summary", "history",
              "recommendations", "batch")

# Method mix per endpoint kind, as integer weights out of 256.
KIND_HEALTH, KIND_ITEM, KIND_COLLECTION, KIND_ACTION = range(4)
METHOD_WEIGHTS = {
    KIND_HEALTH: {"GET": 230, "HEAD": 24, "OPTIONS": 2},
    KIND_ITEM: {"GET": 170, "PUT": 20, "PATCH": 24, "DELETE": 12, "HEAD": 10,
                "OPTIONS": 12, "POST": 8},
    KIND_COLLECTION: {"GET": 160, "POST": 60, "HEAD": 8, "OPTIONS": 20, "PUT": 4,
                      "PATCH": 2, "DELETE": 2},
    KIND_ACTION: {"POST": 220, "OPTIONS": 20, "GET": 12, "PUT": 4},
}
# Base latency in microseconds: lowest value and number of doublings above it.
BASE_LATENCY = {KIND_HEALTH: (15, 3), KIND_ITEM: (250, 4), KIND_COLLECTION: (800, 4),
                KIND_ACTION: (3000, 4)}
SLOW_BASE_LATENCY = (20_000, 4)

# Status mix out of 4096. 200 takes whatever the other codes leave.
COMMON_STATUS = {201: 120, 202: 8, 204: 60, 301: 6, 302: 10, 304: 70, 400: 45,
                 401: 30, 403: 12, 404: 60, 405: 2, 409: 6, 410: 1, 422: 10,
                 429: 8, 499: 3}
ERROR_PROFILES = (  # (share of endpoints out of 100, 5xx weights)
    (60, {500: 2, 501: 1, 502: 1, 503: 1, 504: 1, 599: 1}),
    (28, {500: 10, 501: 1, 502: 4, 503: 6, 504: 3, 599: 1}),
    (10, {500: 60, 501: 2, 502: 30, 503: 60, 504: 20, 599: 2}),
    (2, {500: 500, 501: 4, 502: 200, 503: 300, 504: 100, 599: 4}),
)

# A request's latency is base * 2**k plus a uniform part of the same size; k comes from
# this table (weights out of 1024), which gives the long right tail.
OCTAVE_WEIGHTS = (560, 250, 110, 52, 28, 14, 6, 3, 1)


def expand(weights, size):
    """A list of `size` entries in which each key appears `weight` times."""
    table = []
    for key, weight in weights.items():
        table.extend([key] * weight)
    if len(table) != size:
        raise ValueError(f"weights sum to {len(table)}, expected {size}")
    return table


METHOD_TABLES = {kind: expand(w, 256) for kind, w in METHOD_WEIGHTS.items()}
OCTAVE_TABLE = expand(dict(enumerate(OCTAVE_WEIGHTS)), 1024)


def status_table(errors):
    weights = {200: 4096 - sum(COMMON_STATUS.values()) - sum(errors.values())}
    weights.update(COMMON_STATUS)
    weights.update(errors)
    return expand(weights, 4096)


STATUS_TABLES = [status_table(errors) for _, errors in ERROR_PROFILES]
PROFILE_CUM_WEIGHTS = list(accumulate(share for share, _ in ERROR_PROFILES))


def hour_prefix(hour):
    """'2026-10-30T18:' for hour 0; integer fields only, no locale."""
    t = START + timedelta(hours=hour)
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}T{t.hour:02d}:"


HOUR_PREFIX = [hour_prefix(h) for h in range(HOURS)]
HOUR_WEIGHT = [HOUR_OF_DAY_WEIGHT[(START.hour + h) % 24] for h in range(HOURS)]


def timestamp(hour, offset_us):
    minute, rest = divmod(offset_us, MINUTE_US)
    second, micro = divmod(rest, SECOND_US)
    return f"{HOUR_PREFIX[hour]}{minute:02d}:{second:02d}.{micro:06d}Z"


class Catalogue:
    """The seed's 1,842 endpoint templates and the traffic profile of each one."""

    def __init__(self, seed):
        rng = Random(seed)
        names = list(CORE_ENDPOINTS)
        taken = set(names)
        while len(names) < N_ENDPOINTS:
            name = random_template(rng)
            if name not in taken and len(name) <= MAX_ENDPOINT_LEN:
                taken.add(name)
                names.append(name)
        self.names = names
        self.core = list(range(len(CORE_ENDPOINTS)))
        # Popularity: by_rank[k] is the index of the (k+1)-th most requested endpoint.
        self.by_rank = list(range(N_ENDPOINTS))
        rng.shuffle(self.by_rank)
        # Zipf with exponent 1, in exact integers.
        self.zipf_cum = []
        total = 0
        for rank in range(1, N_ENDPOINTS + 1):
            total += 10**12 // rank
            self.zipf_cum.append(total)
        self.method_table, self.status_table, self.base = [], [], []
        for name in names:
            kind = endpoint_kind(name)
            self.method_table.append(METHOD_TABLES[kind])
            if kind == KIND_HEALTH:
                profile = 0
            else:
                profile = rng.choices(range(len(ERROR_PROFILES)), cum_weights=PROFILE_CUM_WEIGHTS)[0]
            self.status_table.append(STATUS_TABLES[profile])
            low, doublings = BASE_LATENCY[kind]
            if kind != KIND_HEALTH and any(word in name.rsplit("/", 1)[-1] for word in SLOW_WORDS):
                low, doublings = SLOW_BASE_LATENCY
            low <<= rng.randrange(doublings)
            self.base.append(low + rng.randrange(low))

    def draw(self, rng):
        """One endpoint index, drawn by popularity."""
        return rng.choices(self.by_rank, cum_weights=self.zipf_cum)[0]

    def latency(self, rng, e, status):
        low = self.base[e] << OCTAVE_TABLE[rng.getrandbits(10)]
        latency = low + int(rng.random() * low)
        return latency * 3 if status >= 500 else latency

    def payload(self, rng, e, method=None, status=None, latency=None):
        """Everything after the timestamp: ' METHOD endpoint status latency\\n'."""
        if method is None:
            method = self.method_table[e][rng.getrandbits(8)]
        if status is None:
            status = self.status_table[e][rng.getrandbits(12)]
        if latency is None:
            latency = self.latency(rng, e, status)
        return f" {method} {self.names[e]} {status} {latency}\n"


def random_template(rng):
    parts = [rng.choice(PREFIXES)]
    depth = rng.choices((1, 2, 3), cum_weights=(5, 8, 9))[0]
    params = list(PARAMS)   # a template never repeats a parameter name
    for level in range(depth):
        parts.append("/" + rng.choice(RESOURCES))
        if level < depth - 1 or rng.random() < 0.5:
            parts.append("/" + params.pop(rng.randrange(len(params))))
    if rng.random() < 0.3:
        parts.append(rng.choice(ACTIONS))
    return "".join(parts)


def endpoint_kind(name):
    if name in HEALTH_ENDPOINTS:
        return KIND_HEALTH
    if name.endswith("}"):
        return KIND_ITEM
    if any(name.endswith(action) for action in ACTIONS):
        return KIND_ACTION
    return KIND_COLLECTION


class Plan:
    """Injected records, placed by (shard, hour), plus the reserved (hour, endpoint) groups.

    A reserved group holds exactly the records injected for it: random traffic that
    lands on one is redrawn, so a group planned with one request really has one.
    """

    def __init__(self, rng, cat, scale):
        self.rng, self.cat = rng, cat
        self.fixed = {}      # (shard, hour) -> [(offset_us, payload), ...]
        self.reserved = {}   # hour -> set of endpoint indices; membership tests only
        self.count = 0
        n = LINES[scale]
        tiny = scale == "tiny"

        def per_million(rate, tiny_count):
            return tiny_count if tiny else n * rate // 1_000_000

        # Reserved groups come from the less popular half of the catalogue, never the core.
        core = set(cat.core)
        tail = [e for e in cat.by_rank[N_ENDPOINTS // 2:] if e not in core]
        n_single = per_million(100, 3)
        n_extreme = per_million(10, 1)
        n_status = per_million(10, 1)
        groups = []
        while len(groups) < n_single + n_extreme + n_status:
            e, h = rng.choice(tail), rng.randrange(HOURS)
            if e not in self.reserved.get(h, ()):
                self.reserved.setdefault(h, set()).add(e)
                groups.append((h, e))
        self.single_groups = groups[:n_single]
        self.extreme_groups = groups[n_single:n_single + n_extreme]
        self.status_groups = groups[n_single + n_extreme:]
        for h, e in self.single_groups:            # a group with exactly one request
            self.add_random(h, e)
        for h, e in self.extreme_groups:           # latencies 0 and 4294967295 only
            self.add_random(h, e, latency=0)
            self.add_random(h, e, latency=LATENCY_MAX)
        for h, e in self.status_groups:            # 499 and 200 beside 500 and 599
            for status in (200, 499, 500, 599):
                self.add_random(h, e, status=status)

        # Every core endpoint once, all in one hour, so their output lines sit together.
        h = rng.randrange(HOURS)
        for e in cat.core:
            self.add_random(h, e)

        # Hour boundaries. Shard 0 gets HH:59:59.999999 of the earlier hour and a
        # streaming shard gets HH:00:00.000000 of the next. The upload lag puts the
        # earlier-hour line after the later-hour line in the file.
        if tiny:
            boundaries, pairs = (1, 30, HOURS - 1), 1   # 30 is 2026-11-01T00:00Z
        else:
            boundaries, pairs = range(1, HOURS), n // 100_000
        for b in boundaries:
            for _ in range(pairs):
                self.add(0, b - 1, HOUR_US - 1, cat.payload(rng, self.free_endpoint(b - 1)))
                self.add(self.stream_shard(), b, 0, cat.payload(rng, self.free_endpoint(b)))
        self.add(rng.randrange(SHARDS), 0, 0, cat.payload(rng, self.free_endpoint(0)))
        last = HOURS - 1
        self.add(rng.randrange(SHARDS), last, HOUR_US - 1, cat.payload(rng, self.free_endpoint(last)))

        for _ in range(per_million(100, 2)):
            self.add_free(latency=0)
        for _ in range(per_million(100, 2)):
            self.add_free(latency=LATENCY_MAX)
        for _ in range(per_million(100, 2)):       # equal timestamps, one shard
            s, h, off = rng.randrange(SHARDS), rng.randrange(HOURS), self.offset()
            self.add(s, h, off, cat.payload(rng, self.free_endpoint(h)))
            self.add(s, h, off, cat.payload(rng, self.free_endpoint(h)))
        for _ in range(per_million(100, 2)):       # equal timestamps, two shards
            s1 = rng.randrange(SHARDS)
            s2 = (s1 + 1 + rng.randrange(SHARDS - 1)) % SHARDS
            h, off = rng.randrange(HOURS), self.offset()
            self.add(s1, h, off, cat.payload(rng, self.free_endpoint(h)))
            self.add(s2, h, off, cat.payload(rng, self.free_endpoint(h)))
        for _ in range(per_million(50, 1)):        # byte-identical duplicate lines
            s, h, off = rng.randrange(SHARDS), rng.randrange(HOURS), self.offset()
            payload = cat.payload(rng, self.free_endpoint(h))
            self.add(s, h, off, payload)
            self.add(s, h, off, payload)
        for method in METHODS:
            self.add_free(method=method)

        # sample and full: every endpoint of the catalogue at least once.
        if not tiny:
            for e in range(N_ENDPOINTS):
                h = rng.randrange(HOURS)
                while e in self.reserved.get(h, ()):
                    h = rng.randrange(HOURS)
                self.add_random(h, e)

    def offset(self):
        return int(self.rng.random() * HOUR_US)

    def stream_shard(self):
        return 1 + self.rng.randrange(SHARDS - 1)

    def free_endpoint(self, h):
        """A popularity-weighted endpoint that is not reserved in hour h."""
        banned = self.reserved.get(h, ())
        e = self.cat.draw(self.rng)
        while e in banned:
            e = self.cat.draw(self.rng)
        return e

    def add(self, shard, hour, offset, payload):
        self.fixed.setdefault((shard, hour), []).append((offset, payload))
        self.count += 1

    def add_random(self, hour, e, **fields):
        """Endpoint e in the given hour, at a random shard and moment."""
        self.add(self.rng.randrange(SHARDS), hour, self.offset(),
                 self.cat.payload(self.rng, e, **fields))

    def add_free(self, **fields):
        h = self.rng.randrange(HOURS)
        self.add(self.rng.randrange(SHARDS), h, self.offset(),
                 self.cat.payload(self.rng, self.free_endpoint(h), **fields))


def cell_lines(rng, cat, hour, n, fixed, banned):
    """n random lines plus the injected ones for one shard and hour, in time order.

    Returns [(microseconds since START, line), ...].
    """
    rand, bits = rng.random, rng.getrandbits
    offsets = sorted([int(rand() * HOUR_US) for _ in range(n)])
    picks = rng.choices(cat.by_rank, cum_weights=cat.zipf_cum, k=n)
    names, method_table, status_table = cat.names, cat.method_table, cat.status_table
    base = cat.base
    records = []
    for off, e in zip(offsets, picks):
        while e in banned:
            e = cat.draw(rng)
        status = status_table[e][bits(12)]
        low = base[e] << OCTAVE_TABLE[bits(10)]
        latency = low + int(rand() * low)
        if status >= 500:
            latency *= 3
        records.append((off, f" {method_table[e][bits(8)]} {names[e]} {status} {latency}\n"))
    if fixed:
        records.extend(fixed)
        records.sort(key=lambda r: r[0])  # stable, so the result does not depend on payloads
    start = hour * HOUR_US
    return [(start + off, timestamp(hour, off) + payload) for off, payload in records]


def shard_chunks(shard, rng, cat, counts, plan):
    """Yield (arrival_us, shard, seq, text) for each chunk one shard ships, in arrival order."""
    seq = 0
    cells = ((h, cell_lines(rng, cat, h, counts[shard * HOURS + h],
                            plan.fixed.get((shard, h)), plan.reserved.get(h, ())))
             for h in range(HOURS))
    if shard == 0:
        for h, lines in cells:
            arrival = (h + 1) * HOUR_US + UPLOAD_LAG_MIN_US + int(rng.random() * UPLOAD_LAG_SPAN_US)
            if lines:
                yield arrival, shard, seq, "".join(line for _, line in lines)
                seq += 1
        return
    close = STREAM_CHUNK_MIN_US + int(rng.random() * STREAM_CHUNK_SPAN_US)
    arrival = 0
    buf = []
    for _, lines in cells:
        for t, line in lines:
            while t >= close:
                if buf:
                    arrival = max(arrival, close + int(rng.random() * STREAM_DELAY_SPAN_US))
                    yield arrival, shard, seq, "".join(buf)
                    seq += 1
                    buf = []
                close += STREAM_CHUNK_MIN_US + int(rng.random() * STREAM_CHUNK_SPAN_US)
            buf.append(line)
    if buf:
        arrival = max(arrival, close + int(rng.random() * STREAM_DELAY_SPAN_US))
        yield arrival, shard, seq, "".join(buf)


def build(seed, scale):
    """Everything needed to write the file: catalogue, plan and per-shard chunk streams."""
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    master = Random(seed)
    catalogue_seed = master.getrandbits(64)
    scale_seeds = {name: master.getrandbits(64) for name in ("full", "sample", "tiny")}
    cat = Catalogue(catalogue_seed)
    rng = Random(scale_seeds[scale])
    shard_seeds = [rng.getrandbits(64) for _ in range(SHARDS)]
    plan = Plan(rng, cat, scale)
    # Random lines fill the rest, spread over (shard, hour) cells by the traffic curve.
    n_random = LINES[scale] - plan.count
    cells = SHARDS * HOURS
    cum, total = [], 0
    for c in range(cells):
        total += HOUR_WEIGHT[c % HOURS]
        cum.append(total)
    counts = [0] * cells
    left = n_random
    while left:
        batch = min(left, 1_000_000)
        for c in rng.choices(range(cells), cum_weights=cum, k=batch):
            counts[c] += 1
        left -= batch
    streams = [shard_chunks(s, Random(shard_seeds[s]), cat, counts, plan) for s in range(SHARDS)]
    return cat, plan, streams


def generate(seed, scale, out_dir):
    """Write out_dir/requests.log and return its path."""
    _, _, streams = build(seed, scale)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FILE_NAME)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        for _, _, _, text in heapq.merge(*streams):
            f.write(text)
    return path


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Generate the Track A (Ingest) dataset.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scale", choices=sorted(LINES), required=True)
    parser.add_argument("--out", help="output directory (default: .data/a/<scale>-seed<N>)")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be a non-negative integer")
    out = args.out or os.path.join(here, "..", ".data", "a", f"{args.scale}-seed{args.seed}")
    began = time.perf_counter()
    path = generate(args.seed, args.scale, out)
    took = time.perf_counter() - began
    sys.stderr.write(f"{path}: {LINES[args.scale]} lines, {os.path.getsize(path)} bytes, {took:.1f} s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
