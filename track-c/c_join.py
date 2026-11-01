#!/usr/bin/env python3.13
"""Speed Demon 2026, Track C (Join): dataset generator.

    python3.13 generators/c_join.py --seed 1 --scale sample --out .data/c/sample-seed1

Writes records.csv (the hostile file) and regions.csv (the plain lookup table)
exactly as specs/c-join.md describes them.

The output is a pure function of (seed, scale). Every choice comes from one
random.Random(seed) instance and becomes bytes through integer arithmetic only
(no float maths, no hash(), no set iteration), so the files are byte-identical
on any machine and any run.
"""

import argparse
import bisect
import os
import random

RECORDS_FILE = "records.csv"
REGIONS_FILE = "regions.csv"

BOM = b"\xef\xbb\xbf"
CRLF = b"\r\n"
LF = b"\n"
HEADER = b"id,name,note,region_id,amount"

# full and sample share every parameter except the record count, so sample is
# the same distribution at a tenth of the size (and not a prefix of full).
# tiny keeps the record-level rates but shrinks the lookup and the pools so a
# test, or a person, can read the whole thing.
_SHARED = {"region_space": 999_999, "regions": 200_000, "countries": 240,
           "names": 100_000, "notes": 60_000}
SCALES = {
    "full": {"records": 3_320_000, **_SHARED},
    "sample": {"records": 332_000, **_SHARED},
    "tiny": {"records": 1_000, "region_space": 9_999, "regions": 2_000,
             "countries": 24, "names": 400, "notes": 300},
}

BLOCK_EVERY = 25_000    # one edge-case block per this many records ...
MIN_BLOCKS = 4          # ... and never fewer than this, so tiny has every case
BLOCK_RECORDS = 12      # records per block (the prefix record plus R1..R11)
BOM_ONE_IN = 5_000      # an ordinary record boundary carries a BOM 1 time in this
UNKNOWN_ONE_IN = 250    # an ordinary record's region_id is unknown 1 time in this
ZIPF_SCALE = 10 ** 12   # integer Zipf weights: rank k weighs ZIPF_SCALE // k

CONSONANTS = "bcdfghjklmnprstvwz"
VOWELS = "aeiou"
CODAS = "nrlsmk"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
WORDS = (
    "account", "added", "after", "again", "agreed", "amount", "and", "asked",
    "back", "balance", "bank", "before", "billing", "box", "branch", "call",
    "called", "card", "cash", "check", "claim", "client", "closed", "credit",
    "customer", "date", "day", "debit", "delayed", "delivery", "desk", "did",
    "disputed", "done", "double", "due", "email", "error", "fee", "fixed",
    "for", "form", "from", "fund", "goods", "hold", "home", "invoice", "issue",
    "item", "late", "left", "letter", "line", "manager", "march", "message",
    "missing", "month", "moved", "new", "next", "no", "not", "note", "now",
    "office", "old", "on", "open", "order", "paid", "parcel", "partial",
    "payment", "pending", "phone", "post", "price", "receipt", "refund",
    "rejected", "repeat", "reply", "return", "said", "sent", "service",
    "shipped", "signed", "split", "store", "swap", "the", "then", "ticket",
    "to", "today", "transfer", "twice", "until", "update", "urgent", "via",
    "wait", "week", "with", "wrong", "yes",
)


def make_rng(seed, scale):
    """One stream per (seed, scale), so full and sample share no records and
    no lookup. A str seed goes through sha512, not hash(), so it is stable
    across processes, machines and PYTHONHASHSEED."""
    return random.Random(f"speed-demon-2026/c-join/{scale}/{seed}")


def quoted(text):
    """CSV-encode a non-empty value: wrap in quotes, double every quote."""
    return b'"' + text.replace('"', '""').encode("ascii") + b'"'


def money(cents):
    """Integer cents as -?[0-9]+.[0-9]{2}, never -0.00."""
    if cents < 0:
        return b"-%d.%02d" % (-cents // 100, -cents % 100)
    return b"%d.%02d" % (cents // 100, cents % 100)


class Generator:
    def __init__(self, seed, scale):
        self.rng = make_rng(seed, scale)
        self.p = SCALES[scale]
        self.next_id = 0
        self._build_lookup()
        self._build_pools()

    # ----------------------------------------------------------------- lookup
    def _build_lookup(self):
        rng, p = self.rng, self.p
        codes, seen = [], set()
        while len(codes) < p["countries"]:
            code = "".join(rng.choice(UPPER) for _ in range(3))
            if code not in seen:
                seen.add(code)
                codes.append(code)
        self.codes = codes
        # The last two countries are special: one only ever receives amounts
        # in -0.99..-0.01, the other is in regions.csv but never referenced.
        self.negative_country = len(codes) - 2
        self.unreferenced_country = len(codes) - 1

        region_ids = sorted(rng.sample(range(1, p["region_space"] + 1), p["regions"]))
        country_of = [rng.randrange(len(codes)) for _ in region_ids]
        # Guarantee both special countries own at least one region.
        a, b = rng.sample(range(len(region_ids)), 2)
        country_of[a] = self.negative_country
        country_of[b] = self.unreferenced_country
        self.region_rows = list(zip(region_ids, country_of))
        self.region_set = set(region_ids)          # membership tests only
        self.max_region = region_ids[-1]

        popular = [r for r, c in self.region_rows if c < self.negative_country]
        rng.shuffle(popular)                       # popularity rank independent of id
        self.popular = popular
        cum, total = [], 0
        for rank in range(1, len(popular) + 1):
            total += ZIPF_SCALE // rank
            cum.append(total)
        self.zipf_cum, self.zipf_total = cum, total
        self.negative_regions = [r for r, c in self.region_rows if c == self.negative_country]

    # ------------------------------------------------------------------ pools
    def _word(self, syllables):
        rng = self.rng
        parts = []
        for _ in range(syllables):
            parts.append(rng.choice(CONSONANTS) + rng.choice(VOWELS))
            if rng.randrange(4) == 0:
                parts.append(rng.choice(CODAS))
        return "".join(parts).capitalize()

    def _name(self):
        """A non-empty name with no leading space and no trailing space."""
        rng = self.rng
        surname = self._word(rng.randrange(2, 4))
        form = rng.randrange(100)
        if form < 35:
            return surname
        given = self._word(2)
        if form < 65:
            return f"{given} {surname}"
        if form < 88:
            return f"{surname}, {given}"              # comma inside quotes
        return f'{given} "{self._word(1)}" {surname}'  # escaped quotes

    def _note(self):
        """A non-empty note: words, maybe a comma, quotes and LFs."""
        rng = self.rng
        count = rng.randrange(1, 8)
        words = [rng.choice(WORDS) for _ in range(count)]
        roll = rng.randrange(100)
        if roll < 8:
            i = rng.randrange(count)
            words[i] = f'"{words[i]}"'                 # said "no" twice
        elif roll < 12 and count >= 2:
            words.insert(rng.randrange(1, count), '"')  # a lone quote: quote " and
            count += 1
        if count >= 2 and rng.randrange(100) < 20:
            words[rng.randrange(count - 1)] += ","
        seps = [" "] * (count - 1)
        if count >= 2 and rng.randrange(100) < 10:
            lfs = 1 if rng.randrange(100) < 60 else rng.randrange(2, 4)
            for i in rng.sample(range(count - 1), min(lfs, count - 1)):
                seps[i] = "\n"                          # embedded bare LF
        note = words[0] + "".join(s + w for s, w in zip(seps, words[1:]))
        if rng.randrange(100) < 3:
            note += " "
        return note

    def _build_pools(self):
        rng, p = self.rng, self.p
        names, seen = [], set()
        while len(names) < p["names"]:
            if names and rng.randrange(100) < 5:
                # differs from an existing name only by trailing spaces
                name = names[rng.randrange(len(names))] + " " * rng.randrange(1, 3)
            else:
                name = self._name()
            if name not in seen:
                seen.add(name)
                names.append(name)
        self.names = [quoted(n) for n in names]
        self.notes = [quoted(self._note()) for _ in range(p["notes"])]

    # ---------------------------------------------------------------- records
    def _take_id(self):
        roll = self.rng.randrange(100)
        self.next_id += 1 if roll < 90 else roll - 88   # gaps of 1, sometimes 2..11
        return self.next_id

    def _known_region(self):
        rng = self.rng
        if rng.getrandbits(1):
            return self.popular[rng.randrange(len(self.popular))]
        return self.popular[bisect.bisect_right(self.zipf_cum, rng.randrange(self.zipf_total))]

    def _hole_region(self):
        rng = self.rng
        while True:
            region = rng.randrange(1, self.max_region)
            if region not in self.region_set:
                return region

    def _high_region(self):
        space = self.p["region_space"] + 1
        return self.rng.randrange(space, 10 * space)

    def _cents(self):
        rng = self.rng
        roll = rng.randrange(100)
        if roll < 2:
            return 0
        if roll < 6:
            digits = rng.randrange(1, 6)
            return -rng.randrange(10 ** (digits - 1), 10 ** digits)
        digits = rng.randrange(1, 8)
        return rng.randrange(10 ** (digits - 1), 10 ** digits)

    def _empty(self):
        return b'""' if self.rng.getrandbits(1) else b""

    def _record(self, name, note, region, cents, term):
        return b"%d,%b,%b,%d,%b%b" % (self._take_id(), name, note, region, money(cents), term)

    def _ordinary(self, term=None):
        rng = self.rng
        if rng.randrange(100) == 0:
            name = self._empty()
        else:
            name = self.names[rng.randrange(len(self.names))]
        if rng.randrange(100) < 3:
            note = self._empty()
        else:
            note = self.notes[rng.randrange(len(self.notes))]
        if rng.randrange(UNKNOWN_ONE_IN) == 0:
            region = self._hole_region() if rng.getrandbits(1) else self._high_region()
        else:
            region = self._known_region()
        if term is None:
            term = CRLF if rng.getrandbits(1) else LF
        return self._record(name, note, region, self._cents(), term)

    def _words(self, k):
        return [self.rng.choice(WORDS) for _ in range(k)]

    def _block(self, b):
        """Every edge case in the spec, in BLOCK_RECORDS records."""
        rng = self.rng
        before_bom = CRLF if b % 2 == 0 else LF           # cycles through all four
        after_bom = CRLF if (b // 2) % 2 == 0 else LF     # BOM neighbour pairs
        w = self._words(8)
        same_region = self._known_region()                # both empty-name forms
        twin_region = self._known_region()                # trailing-space twins
        twin = self._name()
        out = [
            self._ordinary(term=before_bom),
            BOM,
            # R1, right after the mid-file BOM: comma in the name, "" in the note
            self._record(quoted(f"{self._word(2)}, {self._word(2)}"),
                         quoted(f'{w[0]} "{w[1]}" {w[2]}'),
                         self._known_region(), self._cents(), after_bom),
            # R2: a note that spans two lines
            self._record(self.names[rng.randrange(len(self.names))],
                         quoted(f"{w[3]} {w[4]}\n{w[5]} {w[6]}"),
                         self._known_region(), self._cents(), LF),
            # R3: empty unquoted name, empty quoted note, zero amount
            self._record(b"", b'""', same_region, 0, CRLF),
            # R4: trailing space in the name; comma, quote and LF in one note;
            # a negative amount
            self._record(quoted(self._name() + " "),
                         quoted(f'{w[7]}, {w[0]} " {w[1]}\n{w[2]}'),
                         self._known_region(), -rng.randrange(1, 100_000), LF),
            # R5: a note that spans several lines
            self._record(self.names[rng.randrange(len(self.names))],
                         quoted("\n".join(" ".join(self._words(2)) for _ in range(rng.randrange(3, 5)))),
                         self._known_region(), self._cents(), CRLF),
            # R6, R7: same region, names that differ only by a trailing space
            self._record(quoted(twin), self.notes[rng.randrange(len(self.notes))],
                         twin_region, self._cents(), LF),
            self._record(quoted(twin + " "), self.notes[rng.randrange(len(self.notes))],
                         twin_region, self._cents(), CRLF),
            # R8: empty quoted name, empty unquoted note, same region as R3
            self._record(b'""', b"", same_region, self._cents(), LF),
            # R9: unknown region_id inside the lookup's id range
            self._record(self.names[rng.randrange(len(self.names))],
                         self.notes[rng.randrange(len(self.notes))],
                         self._hole_region(), self._cents(), CRLF),
            # R10: unknown region_id above the lookup's id range
            self._record(self.names[rng.randrange(len(self.names))],
                         self.notes[rng.randrange(len(self.notes))],
                         self._high_region(), self._cents(), LF),
            # R11: the country whose amounts are all in -0.99..-0.01
            self._record(self.names[rng.randrange(len(self.names))],
                         self.notes[rng.randrange(len(self.notes))],
                         self.negative_regions[rng.randrange(len(self.negative_regions))],
                         -rng.randrange(1, 100), CRLF),
        ]
        assert len(out) == BLOCK_RECORDS + 1    # records plus the BOM
        return out

    def write_records(self, path):
        rng = self.rng
        total = self.p["records"]
        blocks = max(MIN_BLOCKS, total // BLOCK_EVERY)
        ordinary = total - blocks * BLOCK_RECORDS
        span = ordinary // blocks
        # Block b goes before ordinary record starts[b], inside its own span,
        # never before the first record and never after the last.
        starts = [b * span + rng.randrange(1, span) for b in range(blocks)]
        with open(path, "wb") as f:
            buf = [BOM + HEADER + CRLF]
            b = 0
            for i in range(ordinary):
                if b < blocks and starts[b] == i:
                    buf.extend(self._block(b))
                    b += 1
                elif i > 0 and rng.randrange(BOM_ONE_IN) == 0:
                    buf.append(BOM)
                # The last record of the file is always terminated, by CRLF.
                buf.append(self._ordinary(term=CRLF if i == ordinary - 1 else None))
                if len(buf) >= 16_384:
                    f.write(b"".join(buf))
                    buf.clear()
            f.write(b"".join(buf))
        assert b == blocks

    def write_regions(self, path):
        with open(path, "wb") as f:
            f.write(b"region_id,country\n")
            f.write(b"".join(b"%d,%s\n" % (r, self.codes[c].encode("ascii"))
                             for r, c in self.region_rows))


def generate(seed, scale, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gen = Generator(seed, scale)
    gen.write_regions(os.path.join(out_dir, REGIONS_FILE))
    gen.write_records(os.path.join(out_dir, RECORDS_FILE))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scale", choices=sorted(SCALES), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    generate(args.seed, args.scale, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
