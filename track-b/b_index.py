#!/usr/bin/env python3.13
"""Dataset generator for Speed Demon 2026, Track B (Index).

    python3.13 generators/b_index.py --seed 1 --scale sample --out .data/b/sample-seed1

Writes docs.txt and queries.txt into --out. The formats, the distribution and
every edge case are described in specs/b-index.md. Standard library only.

Determinism: every random draw comes from a random.Random instance seeded with
a string built from (seed, scale, purpose). The weights are integers and the
only float arithmetic is inside the random module itself, so no platform libm
call can change a byte of output. Sets are used for membership only, never
iterated to produce output.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import itertools
import operator
import os
import random
import re
import sys
import time

# The tokenizer, duplicated from reference/b_index.py on purpose (the test suite
# checks that both agree): ASCII-lowercase the bytes, then every maximal run of
# [a-z0-9] is a token. Every other byte, including all bytes >= 0x80, separates.
TOKEN_RE = re.compile(rb"[a-z0-9]+")


def tokenize(text: bytes) -> list[bytes]:
    return TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------
# Scales: (documents, queries). Queries are 62% BOOL and 38% PHRASE.
# --------------------------------------------------------------------------
SCALES = {
    "full": (250_000, 50_000),
    "sample": (25_000, 5_000),
    "tiny": (1_000, 1_000),
}
BOOL_SHARE_PCT = 62

# --------------------------------------------------------------------------
# Vocabulary. Rank 1 is the most frequent word. Stopwords take the top ranks
# in this fixed order, legal words are spread over ranks 123 to 7,082 in a
# seed-dependent order, and seed-dependent synthetic words fill the rest.
# --------------------------------------------------------------------------
VOCAB_SIZE = 600_000

STOPWORDS = list(dict.fromkeys("""
the of and to in a that is for by be or as on not it with this any shall such
was which are from at his an have no been its under he has all may upon if had
were their other there would so but we they one who than these them only our
when also can into more must does did those each being what where after before
between without within whether against because nothing herein thereof hereby
therein whereas wherein hereto thereto same very her him she own both either
neither nor then thus here now out up down over off again further once most
some few many much every another about above below during through until while
""".split()))

LEGAL = [w for w in dict.fromkeys("""
indemnity indemnify indemnitor indemnitee surety sureties suretyship writ writs
habeas corpus bail bailable bond bonds bondsman recognizance forfeiture forfeit
forfeited matter matters court courts plaintiff plaintiffs defendant defendants
appellant appellee petitioner respondent judge judges jury juror verdict judgment
appeal appeals motion order orders statute statutes section subsection contract
contracts breach damages liability liable negligence negligent tort torts estate
estates trust trustee executor executrix probate deed title lease lessee lessor
tenant landlord mortgage mortgagee mortgagor lien easement covenant warranty
guaranty guarantor subrogation assignment assignee assignor consideration offer
acceptance rescission restitution injunction equity equitable remedy relief claim
claims counterclaim complaint answer pleading pleadings discovery deposition
subpoena summons service jurisdiction venue standing evidence testimony witness
witnesses hearsay objection sustained overruled counsel attorney attorneys
prosecutor prosecution defense accused indictment arraignment plea guilty
acquittal conviction sentence sentencing probation parole custody detention
arrest warrant search seizure suppression felony misdemeanor offense crime
criminal civil limitations laches estoppel waiver ratification agency principal
agent partnership corporation shareholder director fiduciary duty loyalty fraud
misrepresentation duress undue influence mistake capacity minor guardian ward
conservator insolvency bankruptcy creditor debtor garnishment attachment levy
execution foreclosure replevin ejectment trespass nuisance conversion bailment
bailee bailor chattel chattels property possession adverse tenancy fee remainder
reversion devise bequest legacy heir heirs intestate testator testatrix codicil
affidavit oath perjury contempt sanction sanctions mandamus prohibition
certiorari quo warranto coram nobis amicus curiae pro se bono forma pauperis
prima facie res judicata ipsa loquitur stare decisis mens rea actus reus voir
dire ultra vires bona fide quantum meruit nunc tunc ex parte post facto sua
sponte de novo inter alia et al seq ibid supra infra versus circuit district
supreme superior appellate tribunal magistrate clerk bailiff sheriff marshal
docket record transcript exhibit brief memorandum opinion dissent concurrence
holding dictum precedent reversed affirmed remanded vacated dismissed denied
granted moot ripe justiciable constitutional amendment clause due equal
protection commerce privilege immunity sovereign qualified culpable reasonable
doubt preponderance burden proof presumption inference finding conclusion fact
law question issue cause action proximate causation foreseeable standard conduct
intent malice recklessness gross comparative contributory assumption risk strict
products defect merchantability fitness purpose express implied condition
subsequent performance discharge frustration impossibility impracticability
force majeure liquidated penalty specific nominal punitive compensatory
consequential incidental mitigation interest costs fees award arbitration
arbitrator mediation settlement release accord satisfaction novation surrender
party parties hereinafter aforesaid notwithstanding pursuant thereunder whereof
herewith covenantor covenantee grantor grantee vendor vendee purchaser lender
borrower premises habendum tenement hereditament demise mesne profits rents
escrow closing default cure notice demand tender payment installment note holder
endorsement negotiable instrument draft check bill lading carrier shipper
consignee freight charter vessel salvage admiralty maritime collision insurance
insurer insured policy premium coverage exclusion rider beneficiary annuity
claimant adjuster subrogee statutory regulation regulatory rulemaking
administrative hearing examiner commissioner board license permit zoning
variance ordinance municipal county township state federal territory nation
citizen alien naturalization deportation asylum visa extradition treaty
convention tariff customs excise tax taxation assessment exemption deduction
credit refund audit abatement harmless foregoing reasons affirm reverse remand
discretion erroneous review conditions released prejudice evidence convincing
trial essence witness agree terms ordered finds set doctrine
""".split()) if w not in STOPWORDS]

# Multi-word legal phrases inserted whole into the running text. Their weights
# are Zipf-like over a seed-dependent order.
TEMPLATES = [
    "writ of habeas corpus", "habeas corpus", "in the matter of", "court of appeals",
    "summary judgment", "motion to dismiss", "motion for summary judgment",
    "beyond a reasonable doubt", "due process of law", "equal protection of the laws",
    "statute of limitations", "breach of contract", "breach of fiduciary duty",
    "power of attorney", "res judicata", "prima facie", "pro se", "pro bono",
    "amicus curiae", "voir dire", "quantum meruit", "bona fide purchaser", "force majeure",
    "burden of proof", "in good faith", "null and void", "cease and desist", "free and clear",
    "the court finds that", "it is hereby ordered", "it is so ordered", "on or about",
    "and/or", "bail bond", "surety bond", "forfeiture of the bond", "release on recognizance",
    "released on his own recognizance", "indemnity agreement", "hold harmless", "mens rea",
    "actus reus", "stare decisis", "nunc pro tunc", "ex parte", "ex post facto", "sua sponte",
    "de novo review", "inter alia", "et al.", "plaintiff-appellant", "defendant-appellee",
    "cross-examination", "pre-trial", "post-conviction relief", "attorney-client privilege",
    "work-product doctrine", "the court's", "the party's", "in forma pauperis",
    "res ipsa loquitur", "quo warranto", "writ of mandamus", "writ of certiorari",
    "petition for a writ of certiorari", "the defendant's motion", "the plaintiff's complaint",
    "for the foregoing reasons", "we affirm", "we reverse and remand", "reversed and remanded",
    "affirmed in part and reversed in part", "abuse of discretion", "clearly erroneous",
    "standard of review", "conditions of release", "bail was set at", "the surety shall",
    "jointly and severally", "to have and to hold", "time is of the essence",
    "in witness whereof", "the parties agree that", "notwithstanding the foregoing",
    "subject to the terms and conditions", "terms and conditions", "without prejudice",
    "with prejudice", "a preponderance of the evidence", "clear and convincing evidence",
    "the trial court", "the supreme court", "the district court",
]

# Synthetic words: 1 to 3 syllables of onset + nucleus + coda, 3 to 12 letters.
# Repeated entries weight the plain consonants and vowels.
ONSETS = ["", "", "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t",
          "v", "w", "z", "br", "cr", "dr", "fr", "gr", "pr", "tr", "bl", "cl", "fl", "pl", "sl",
          "st", "sp", "ch", "sh", "th"]
NUCLEI = ["a", "e", "i", "o", "u", "a", "e", "i", "o", "u", "y", "ai", "ea", "ee", "ou", "io"]
CODAS = ["", "", "", "", "", "n", "r", "s", "l", "m", "t", "x", "nd", "st", "ck", "ng"]
SYLLABLES = [o + n + c for o in ONSETS for n in NUCLEI for c in CODAS]
SYLLABLE_COUNTS = (1, 2, 3)
SYLLABLE_CUM = (25, 80, 100)

# --------------------------------------------------------------------------
# Text rendering.
# --------------------------------------------------------------------------
# Document length in word units: (low, high) buckets with integer weights.
LENGTH_BUCKETS = [(1, 9), (10, 79), (80, 199), (200, 399), (400, 999)]
LENGTH_CUM = list(itertools.accumulate([5, 35, 42, 16, 2]))

# Separator placed between two units, with integer weights.
SEPARATORS = [
    (" ", 8600), (", ", 550), (". ", 450), ("; ", 60), (": ", 40), (" (", 30), (") ", 30),
    ("-", 80), ("/", 15), ("'s ", 40), (" — ", 10), ("—", 10), (' "', 20),
    ('" ', 20), (" & ", 10), ("... ", 10), (" - ", 15), (" ", 5), ("’s ", 5),
]
SEP_STRINGS = [s for s, _ in SEPARATORS]
SEP_CUM = list(itertools.accumulate(w for _, w in SEPARATORS))
ENDINGS = [".", "", ";", ":", "?"]
ENDING_CUM = list(itertools.accumulate([70, 15, 5, 5, 5]))

TEMPLATE_PPM = 20_000   # share of word units replaced by a legal phrase
NUMBER_PPM = 15_000     # share of word units replaced by a number or citation

# --------------------------------------------------------------------------
# Document edge cases, injected at a fixed rate with a minimum count.
# (kind, parts per million of documents, minimum)
# --------------------------------------------------------------------------
DOC_EDGES = [
    ("empty", 500, 2),       # text is the empty string
    ("seponly", 500, 2),     # text has bytes but no token
    ("utf8", 20_000, 12),    # 1 to 3 items from UTF8_ITEMS
    ("upper", 10_000, 6),    # whole text in ASCII upper case
    ("title", 10_000, 6),    # whole text in ASCII title case
    ("repeat", 5_000, 6),    # a run "w w", "w w w" or "w w w w" of one content word
    ("digitpad", 5_000, 6),  # a case number with a zero-padded token, e.g. 00123
    ("punct", 10_000, 6),    # a punctuation-joined unit from PUNCT_UNITS
]

SEPONLY_TEXTS = ["—", "§", "§§ ¶", "...", "* * *", "“”",
                 "½", "- - -", "()", "--", "/", "№", "[ ]", "· · ·"]

# Non-ASCII words. Under the byte rule each one splits at its non-ASCII bytes:
# "Zürich" gives the tokens "z" and "rich", "273K" (Kelvin sign) gives "273".
UTF8_ITEMS = [
    "café", "naïve", "Zürich", "résumé", "façade", "São Paulo",
    "Straße", "İstanbul", "273K", "ﬁduciary", "€500", "½",
    "Ⅳ", "“bail”", "bail—bond", "bail bond", "\U0001d504ct",
    "Œuvre", "Ångström", "Ωmega", "déjà vu", "Müller",
    "Gödel", "Søren", "Łódź", "coöperate", "naïveté",
    "¶ 12", "§§", "ﬂoor", "\U0001f3db", "Ñandú",
]

# Punctuation-joined units: every one of them is several tokens.
PUNCT_UNITS = [
    "habeas-corpus", "Habeas-Corpus", "and/or", "plaintiff-appellant", "defendant-appellee",
    "cross-examination", "court's", "pre-trial", "non-compete", "co-defendant", "U.S.C.",
    "e-mail", "bail/bond", "surety-bond", "attorney-client", "self-incrimination",
    "re-sentencing", "O'Brien", "out-of-state", "follow-up", "court-appointed",
    "first-degree", "bail-bond", "writ-of-habeas-corpus", "in.the.matter.of",
    "forfeiture/recognizance", "plaintiff(s)", "defendant(s)", "a.k.a.", "d/b/a",
]

# Query terms that must never match: what a Unicode-normalising or
# punctuation-dropping tokenizer would produce from the items above.
TRAP_TERMS = [
    "zurich", "cafe", "naive", "resume", "facade", "strasse", "istanbul", "273k",
    "oeuvre", "angstrom", "omega", "muller", "godel", "soren", "lodz", "cooperate",
    "habeascorpus", "andor", "crossexamination", "pretrial", "usc", "obrien", "aka", "dba",
    "nandu", "floor", "fiduciary12", "dejavu",
]

# --------------------------------------------------------------------------
# Queries.
# --------------------------------------------------------------------------
# Document-frequency bands, as a fraction of the document count N:
#   0 VC: df > N/10    1 C: N/100 < df <= N/10    2 M: N/1000 < df <= N/100
#   3 U: N/10000 < df <= N/1000                   4 R: 1 <= df <= N/10000
BAND_NAMES = ("VC", "C", "M", "U", "R")
ANCHOR_DIV = 1000   # a phrase anchor is a token with df <= N/1000 (bands U and R)

BOOL_TERMS = [(2, 38), (3, 32), (4, 18), (5, 8), (6, 4)]
BOOL_OPS = [("AND", 50), ("OR", 30), ("NOT", 20)]
BOOL_BANDS = [(0, 3), (1, 12), (2, 30), (3, 30), (4, 25)]
FROM_SOURCE_PCT = 75   # chance a BOOL term comes from the query's source document
PHRASE_LENGTHS = [(2, 35), (3, 30), (4, 20), (5, 9), (6, 4), (7, 1), (8, 1)]

# (kind, parts per million of all queries, minimum count)
BOOL_KINDS = [
    ("single", 20_000, 3),      # one term, no operator
    ("absent", 10_000, 3),      # a term that is not in the corpus
    ("not_empty", 5_000, 3),    # NOT empties a non-empty set
    ("or_chain", 1_000, 3),     # 3 to 6 common terms joined by OR
    ("op_words", 5_000, 3),     # the lowercase words and / or / not as terms
    ("left_assoc", 10_000, 3),  # mixed operators where precedence would matter
    ("digits", 5_000, 3),       # zero-padded digit tokens
    ("utf8", 5_000, 3),         # a token cut out of a non-ASCII word
    ("upper", 5_000, 3),        # terms from an all-caps document
    ("repeat", 5_000, 3),       # a term repeated inside one document
    ("punct", 5_000, 3),        # the parts of a punctuation-joined word
]
PHRASE_KINDS = [
    ("scrambled", 40_000, 3),   # a real window, reversed or with two tokens swapped
    ("absent", 5_000, 3),       # a real window with one term replaced by an absent one
    ("repeat", 10_000, 3),      # repeated tokens: "w w", "w w w", one longer than the run
    ("utf8", 5_000, 3),         # a window around a token cut out of a non-ASCII word
    ("upper", 5_000, 3),        # a window from an all-caps document
    ("digits", 5_000, 3),       # a window around a zero-padded digit token
    ("punct", 5_000, 3),        # a window across a punctuation-joined word
    ("cross", 5_000, 3),        # a window across a sentence boundary ". "
]


def weighted(pairs):
    """[(value, weight), ...] -> (values, cumulative integer weights)."""
    return [v for v, _ in pairs], list(itertools.accumulate(w for _, w in pairs))


def draw(rng: random.Random, table) -> object:
    values, cum = table
    return rng.choices(values, cum_weights=cum)[0]


def per_million(rng: random.Random, amount_ppm: int) -> int:
    """Randomised rounding of amount_ppm / 1e6 to an integer."""
    return (amount_ppm + rng.randrange(1_000_000)) // 1_000_000


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
class Vocab:
    def __init__(self, seed: int):
        rng = random.Random(f"b-index:{seed}:vocab")
        legal = list(LEGAL)
        rng.shuffle(legal)
        reserved = set(STOPWORDS) | set(LEGAL) | set(TRAP_TERMS)
        for phrase in TEMPLATES + PUNCT_UNITS:
            reserved.update(t.decode() for t in tokenize(phrase.encode()))
        n_synth = VOCAB_SIZE - len(STOPWORDS) - len(legal)
        synth = []
        while len(synth) < n_synth:
            k = rng.choices(SYLLABLE_COUNTS, cum_weights=SYLLABLE_CUM)[0]
            word = "".join(rng.choices(SYLLABLES, k=k))
            if 3 <= len(word) <= 12 and word not in reserved:
                reserved.add(word)
                synth.append(word)
        # Legal word j sits at 0-based rank position S + j + j*j//40.
        start = len(STOPWORDS)
        legal_at = {start + j + j * j // 40: w for j, w in enumerate(legal)}
        synth_iter = iter(synth)
        self.words = list(STOPWORDS)
        for pos in range(start, VOCAB_SIZE):
            self.words.append(legal_at[pos] if pos in legal_at else next(synth_iter))
        # Zipf-Mandelbrot weight of rank r is proportional to 1 / (r + 1/2).
        self.cum = list(itertools.accumulate(10**15 // (2 * r + 1) for r in range(1, VOCAB_SIZE + 1)))
        templates = list(TEMPLATES)
        rng.shuffle(templates)
        self.templates = (templates, list(itertools.accumulate(10**9 // (i + 2) for i in range(len(templates)))))
        # Synthetic words of rank 2,001 to 50,000, used for repeated runs and case captions.
        legal_set = set(LEGAL)
        self.mid_words = [w for w in self.words[2_000:50_000] if w not in legal_set]


def ordinal(n: int) -> str:
    suffix = "th"
    if not 10 <= n % 100 <= 20:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def number_unit(rng: random.Random, vocab: Vocab) -> str:
    k = rng.randrange(100)
    if k < 22:
        return str(rng.randint(1, 99))
    if k < 40:
        return str(rng.randint(1790, 2026))
    if k < 48:
        return f"{rng.randint(1, 50)} U.S.C. § {rng.randint(1, 9999)}"
    if k < 55:
        return f"§ {rng.randint(1, 999)}"
    if k < 62:
        return f"No. {rng.randint(1, 26):02d}-{rng.randint(1, 9999)}"
    if k < 68:
        return f"{rng.randint(1, 999)} F.{rng.choice(('2d', '3d', '4th'))} {rng.randint(1, 1500)}"
    if k < 72:
        return f"{rng.randint(1, 600)} U.S. {rng.randint(1, 1200)}"
    if k < 76:
        return f"Fed. R. Civ. P. {rng.randint(1, 86)}({rng.choice('abcdefgh')})({rng.randint(1, 9)})"
    if k < 81:
        return f"${rng.randint(1, 999)},{rng.randint(0, 999):03d}.{rng.randint(0, 99):02d}"
    if k < 86:
        return ordinal(rng.randint(1, 99))
    if k < 89:
        return f"{rng.randint(1, 100)}%"
    a, b = rng.choice(vocab.mid_words), rng.choice(vocab.mid_words)
    return f"{a.capitalize()} v. {b.capitalize()}"


def capitalise(sentence: str) -> str:
    if sentence and "a" <= sentence[0] <= "z":
        return sentence[0].upper() + sentence[1:]
    return sentence


def draw_units(rng: random.Random, vocab: Vocab) -> list[str]:
    low, high = rng.choices(LENGTH_BUCKETS, cum_weights=LENGTH_CUM)[0]
    n = rng.randint(low, high)
    units = rng.choices(vocab.words, cum_weights=vocab.cum, k=n)
    templates, template_cum = vocab.templates
    for _ in range(per_million(rng, n * TEMPLATE_PPM)):
        units[rng.randrange(n)] = rng.choices(templates, cum_weights=template_cum)[0]
    for _ in range(per_million(rng, n * NUMBER_PPM)):
        units[rng.randrange(n)] = number_unit(rng, vocab)
    return units


def render(rng: random.Random, units: list[str]) -> str:
    seps = rng.choices(SEP_STRINGS, cum_weights=SEP_CUM, k=len(units) - 1)
    text = units[0] + "".join(map(operator.add, seps, units[1:]))
    text = ". ".join([capitalise(s) for s in text.split(". ")])
    return text + rng.choices(ENDINGS, cum_weights=ENDING_CUM)[0]


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
def generate_docs(vocab: Vocab, seed: int, scale: str, n_docs: int):
    """Return (texts, df, specials).

    texts[i] is the UTF-8 text of document i, df maps token -> number of
    documents containing it, specials lists the injected edge-case documents.
    """
    rng = random.Random(f"b-index:{seed}:{scale}:docs")
    counts = [(kind, max(minimum, n_docs * ppm // 1_000_000)) for kind, ppm, minimum in DOC_EDGES]
    chosen = rng.sample(range(n_docs), sum(c for _, c in counts))
    doc_kind = {}
    pos = 0
    for kind, c in counts:
        for doc_id in chosen[pos:pos + c]:
            doc_kind[doc_id] = kind
        pos += c

    specials = {kind: [] for kind, _, _ in DOC_EDGES}
    texts = []
    df = collections.Counter()
    for doc_id in range(n_docs):
        kind = doc_kind.get(doc_id)
        if kind == "empty":
            text = b""
        elif kind == "seponly":
            text = rng.choice(SEPONLY_TEXTS).encode()
        else:
            units = draw_units(rng, vocab)
            extra = None
            if kind == "utf8":
                extra = [rng.choice(UTF8_ITEMS) for _ in range(rng.randint(1, 3))]
                for item in extra:
                    units.insert(rng.randrange(len(units) + 1), item)
            elif kind == "repeat":
                word = rng.choice(vocab.mid_words)
                run = rng.choices((2, 3, 4), cum_weights=(50, 85, 100))[0]
                units.insert(rng.randrange(len(units) + 1), " ".join([word] * run))
                extra = (word, run)
            elif kind == "digitpad":
                number = rng.randint(1, 9999)
                year = rng.randint(1990, 2026)
                court = rng.choice(("cv", "cr", "mc", "ap"))
                units.insert(rng.randrange(len(units) + 1), f"Case No. {year}-{court}-{number:05d}")
                extra = f"{number:05d}"
            elif kind == "punct":
                extra = rng.choice(PUNCT_UNITS)
                units.insert(rng.randrange(len(units) + 1), extra)
            text = render(rng, units).encode()
            if kind == "upper":
                text = text.upper()     # bytes.upper(): ASCII letters only
            elif kind == "title":
                text = text.title()     # bytes.title(): ASCII letters only
            if kind is not None:
                specials[kind].append((doc_id, extra))
        if kind in ("empty", "seponly"):
            specials[kind].append((doc_id, None))
        texts.append(text)
        df.update(set(tokenize(text)))
    return texts, df, specials


def write_docs(path: str, texts: list[bytes]) -> int:
    total = 0
    with open(path, "wb") as f:
        for start in range(0, len(texts), 10_000):
            chunk = b"".join(b"%d\t%s\n" % (start + i, t) for i, t in enumerate(texts[start:start + 10_000]))
            f.write(chunk)
            total += len(chunk)
    return total


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------
def plan_kinds(n_queries: int) -> list[tuple[str, str]]:
    n_bool = n_queries * BOOL_SHARE_PCT // 100
    n_phrase = n_queries - n_bool
    plan = []
    for prefix, kinds, total in (("BOOL", BOOL_KINDS, n_bool), ("PHRASE", PHRASE_KINDS, n_phrase)):
        used = 0
        for kind, ppm, minimum in kinds:
            c = max(minimum, n_queries * ppm // 1_000_000)
            plan.extend([(prefix, kind)] * c)
            used += c
        plan.extend([(prefix, "normal")] * (total - used))
    return plan


class QueryMaker:
    def __init__(self, rng: random.Random, seed: int, scale: str, texts, df, specials):
        self.rng = rng
        self.texts = texts
        self.df = df
        self.specials = specials
        self.n = len(texts)
        bands = [[] for _ in BAND_NAMES]
        for token, count in df.items():
            bands[self.band(count)].append(token)
        self.bands = [sorted(b) for b in bands]
        self.absent = self.absent_pool(seed, scale)
        self.traps = [t for t in self.absent if t.decode() in TRAP_TERMS]
        self.absent_draws = 0
        self.bool_terms = weighted(BOOL_TERMS)
        self.bool_ops = weighted(BOOL_OPS)
        self.bool_bands = weighted(BOOL_BANDS)
        self.phrase_lengths = weighted(PHRASE_LENGTHS)

    # ---- helpers -------------------------------------------------------
    def band(self, count: int) -> int:
        n = self.n
        if count * 10 > n:
            return 0
        if count * 100 > n:
            return 1
        if count * 1000 > n:
            return 2
        if count * 10000 > n:
            return 3
        return 4

    def global_band(self, b: int) -> list[bytes]:
        """The band's terms, or the nearest non-empty band (rarer side first)."""
        for step in range(len(BAND_NAMES)):
            for cand in (b + step, b - step):
                if 0 <= cand < len(BAND_NAMES) and self.bands[cand]:
                    return self.bands[cand]
        raise ValueError("corpus has no tokens")

    def is_anchor(self, token: bytes) -> bool:
        return self.df[token] * ANCHOR_DIV <= self.n

    def absent_pool(self, seed: int, scale: str) -> list[bytes]:
        pool = [t.encode() for t in TRAP_TERMS if t.encode() not in self.df]
        rng = random.Random(f"b-index:{seed}:{scale}:absent")
        while len(pool) < 2 * len(TRAP_TERMS):
            word = "".join(rng.choices(SYLLABLES, k=rng.randint(2, 3))).encode()
            if 3 <= len(word) <= 12 and word not in self.df and word not in pool:
                pool.append(word)
        return pool

    def absent_term(self) -> bytes:
        """Every other absent term is a normalisation trap, so each scale has some."""
        self.absent_draws += 1
        if self.absent_draws % 2 == 1 and self.traps:
            return self.rng.choice(self.traps)
        return self.rng.choice(self.absent)

    def tokens(self, doc_id: int) -> list[bytes]:
        return tokenize(self.texts[doc_id])

    def random_doc(self, min_tokens: int) -> tuple[int, list[bytes]]:
        while True:
            doc_id = self.rng.randrange(self.n)
            toks = self.tokens(doc_id)
            if len(toks) >= min_tokens:
                return doc_id, toks

    def window_around(self, toks: list[bytes], start: int, length: int, k: int,
                      need_anchor: bool = True) -> list[bytes] | None:
        """A random window containing toks[start:start+length].

        It has k tokens, or more (up to 8) if no k-token window qualifies.
        With need_anchor, the window must also contain an anchor token.
        Returns None if no window qualifies.
        """
        for size in range(max(k, length, 2), 9):
            if size > len(toks):
                return None
            lo, hi = max(0, start + length - size), min(start, len(toks) - size)
            starts = [s for s in range(lo, hi + 1)
                      if not need_anchor or any(self.is_anchor(t) for t in toks[s:s + size])]
            if starts:
                s = self.rng.choice(starts)
                return toks[s:s + size]
        return None

    def anchored_window(self, toks: list[bytes], k: int):
        if len(toks) < k:
            return None
        anchors = [i for i, t in enumerate(toks) if self.is_anchor(t)]
        if not anchors:
            return None
        return self.window_around(toks, self.rng.choice(anchors), 1, k, need_anchor=False)

    def normal_window(self, k: int) -> list[bytes]:
        while True:
            _, toks = self.random_doc(k)
            window = self.anchored_window(toks, k)
            if window is not None:
                return window

    def special(self, kind: str):
        return self.rng.choice(self.specials[kind])

    def source_bands(self, toks: list[bytes]) -> list[list[bytes]]:
        by_band = [[] for _ in BAND_NAMES]
        for t in sorted(set(toks)):
            by_band[self.band(self.df[t])].append(t)
        return by_band

    def pick_term(self, source: list[list[bytes]], band: int | None = None) -> bytes:
        if band is None:
            band = draw(self.rng, self.bool_bands)
        if self.rng.randrange(100) < FROM_SOURCE_PCT and source[band]:
            return self.rng.choice(source[band])
        return self.rng.choice(self.global_band(band))

    def mid_term(self, source) -> bytes:
        """A term from band M, U or R (cheap to evaluate)."""
        return self.pick_term(source, self.rng.choice((2, 3, 4)))

    def doc_term(self, source) -> bytes:
        """A token of the source document itself, from band M, U or R if it has one."""
        pool = source[2] + source[3] + source[4]
        if not pool:
            pool = source[0] + source[1]
        return self.rng.choice(pool)

    def ops(self, m: int) -> list[str]:
        return [draw(self.rng, self.bool_ops) for _ in range(m - 1)]

    @staticmethod
    def bool_line(terms, ops) -> str:
        parts = [terms[0].decode()]
        for op, term in zip(ops, terms[1:]):
            parts += [op, term.decode()]
        return "BOOL " + " ".join(parts)

    @staticmethod
    def phrase_line(terms) -> str:
        return 'PHRASE "' + " ".join(t.decode() for t in terms) + '"'

    def find(self, toks: list[bytes], sub: list[bytes]) -> int:
        k = len(sub)
        for i in range(len(toks) - k + 1):
            if toks[i:i + k] == sub:
                return i
        raise ValueError("sub-sequence not found")

    # ---- BOOL ------------------------------------------------------------
    def bool_normal(self) -> str:
        m = draw(self.rng, self.bool_terms)
        _, toks = self.random_doc(1)
        source = self.source_bands(toks)
        return self.bool_line([self.pick_term(source) for _ in range(m)], self.ops(m))

    def bool_single(self) -> str:
        _, toks = self.random_doc(1)
        return self.bool_line([self.pick_term(self.source_bands(toks))], [])

    def bool_absent(self) -> str:
        _, toks = self.random_doc(1)
        source = self.source_bands(toks)
        absent = self.absent_term()
        x, y = self.mid_term(source), self.mid_term(source)
        pattern = self.rng.randrange(6)
        if pattern == 0:
            return self.bool_line([absent], [])
        if pattern == 1:
            return self.bool_line([absent, x], ["OR"])
        if pattern == 2:
            return self.bool_line([x, absent], ["AND"])
        if pattern == 3:
            return self.bool_line([x, absent], ["OR"])
        if pattern == 4:
            return self.bool_line([x, absent], ["NOT"])
        return self.bool_line([x, y, absent], ["AND", "OR"])

    def bool_not_empty(self) -> str:
        _, toks = self.random_doc(1)
        source = self.source_bands(toks)
        x, y = self.mid_term(source), self.mid_term(source)
        pattern = self.rng.randrange(3)
        if pattern == 0:
            return self.bool_line([x, x], ["NOT"])
        if pattern == 1:
            return self.bool_line([x, y, x], ["AND", "NOT"])
        return self.bool_line([x, y, x, y], ["OR", "NOT", "NOT"])

    def bool_or_chain(self) -> str:
        m = self.rng.randint(3, 6)
        terms = [self.rng.choice(self.global_band(0 if self.rng.randrange(10) < 7 else 1)) for _ in range(m)]
        return self.bool_line(terms, ["OR"] * (m - 1))

    def bool_op_words(self) -> str:
        _, toks = self.random_doc(1)
        source = self.source_bands(toks)
        word = self.rng.choice((b"and", b"or", b"not"))
        x = self.mid_term(source)
        pattern = self.rng.randrange(4)
        if pattern == 0:
            return self.bool_line([x, word], ["AND"])
        if pattern == 1:
            return self.bool_line([word, x], ["AND"])
        if pattern == 2:
            return self.bool_line([x, word], ["NOT"])
        return self.bool_line([x, self.mid_term(source), word], ["OR", "AND"])

    def bool_left_assoc(self) -> str:
        _, toks = self.random_doc(1)
        source = self.source_bands(toks)
        a, b, c = self.mid_term(source), self.mid_term(source), self.mid_term(source)
        ops = self.rng.choice((["OR", "AND"], ["AND", "OR"], ["OR", "NOT"], ["NOT", "OR"]))
        return self.bool_line([a, b, c], ops)

    def bool_digits(self) -> str:
        doc_id, padded = self.special("digitpad")
        padded = padded.encode()
        plain = padded.lstrip(b"0")
        toks = self.tokens(doc_id)
        other = self.rng.choice(toks)
        pattern = self.rng.randrange(3)
        if pattern == 0:
            return self.bool_line([padded], [])
        if pattern == 1:
            return self.bool_line([padded, plain], ["OR"])
        return self.bool_line([padded, other], ["AND"])

    def utf8_doc(self):
        """An injected non-ASCII item that yields tokens, its doc tokens and its own tokens."""
        while True:
            doc_id, items = self.special("utf8")
            item_tokens = [tokenize(item.encode()) for item in items]
            usable = [t for t in item_tokens if t]
            if usable:
                return self.tokens(doc_id), self.rng.choice(usable)

    def bool_utf8(self) -> str:
        toks, frag = self.utf8_doc()
        term = self.rng.choice(frag)
        anchors = [t for t in toks if self.is_anchor(t)]
        if anchors:
            return self.bool_line([term, self.rng.choice(anchors)], ["AND"])
        return self.bool_line([term], [])

    def bool_upper(self) -> str:
        doc_id, _ = self.special("upper")
        source = self.source_bands(self.tokens(doc_id))
        return self.bool_line([self.doc_term(source), self.doc_term(source)], ["AND"])

    def bool_repeat(self) -> str:
        doc_id, (word, _) = self.special("repeat")
        word = word.encode()
        if self.rng.randrange(2) == 0:
            return self.bool_line([word], [])
        source = self.source_bands(self.tokens(doc_id))
        return self.bool_line([word, self.doc_term(source)], ["AND"])

    def bool_punct(self) -> str:
        _, unit = self.special("punct")
        parts = tokenize(unit.encode())
        return self.bool_line(parts, ["AND"] * (len(parts) - 1))

    # ---- PHRASE ----------------------------------------------------------
    def phrase_normal(self) -> str:
        return self.phrase_line(self.normal_window(draw(self.rng, self.phrase_lengths)))

    def phrase_scrambled(self) -> str:
        while True:
            window = self.normal_window(draw(self.rng, self.phrase_lengths))
            if self.rng.randrange(2) == 0:
                scrambled = window[::-1]
            else:
                i = self.rng.randrange(len(window) - 1)
                scrambled = window[:i] + [window[i + 1], window[i]] + window[i + 2:]
            if scrambled != window:
                return self.phrase_line(scrambled)

    def phrase_absent(self) -> str:
        window = self.normal_window(draw(self.rng, self.phrase_lengths))
        window[self.rng.randrange(len(window))] = self.absent_term()
        return self.phrase_line(window)

    def phrase_repeat(self) -> str:
        doc_id, (word, run) = self.special("repeat")
        word = word.encode()
        pattern = self.rng.randrange(4)
        if pattern == 0:
            return self.phrase_line([word, word])
        if pattern == 1:
            return self.phrase_line([word] * run)
        if pattern == 2:
            return self.phrase_line([word] * (run + 1))
        toks = self.tokens(doc_id)
        start = self.find(toks, [word] * run)
        window = self.window_around(toks, start, run, run + self.rng.randint(1, 2), need_anchor=False)
        return self.phrase_line(window if window is not None else [word] * run)

    def phrase_utf8(self) -> str:
        while True:
            toks, frag = self.utf8_doc()
            start = self.find(toks, frag)
            window = self.window_around(toks, start, len(frag), draw(self.rng, self.phrase_lengths))
            if window is not None:
                return self.phrase_line(window)

    def phrase_upper(self) -> str:
        while True:
            doc_id, _ = self.special("upper")
            window = self.anchored_window(self.tokens(doc_id), draw(self.rng, self.phrase_lengths))
            if window is not None:
                return self.phrase_line(window)

    def phrase_digits(self) -> str:
        while True:
            doc_id, padded = self.special("digitpad")
            toks = self.tokens(doc_id)
            start = self.find(toks, [padded.encode()])
            window = self.window_around(toks, start, 1, draw(self.rng, self.phrase_lengths))
            if window is not None:
                return self.phrase_line(window)

    def phrase_punct(self) -> str:
        while True:
            doc_id, unit = self.special("punct")
            toks = self.tokens(doc_id)
            parts = tokenize(unit.encode())
            start = self.find(toks, parts)
            window = self.window_around(toks, start, len(parts), len(parts) + 1)
            if window is not None:
                return self.phrase_line(window)

    def phrase_cross(self) -> str:
        """A window that spans ". " and contains an anchor."""
        while True:
            doc_id, toks = self.random_doc(2)
            lowered = self.texts[doc_id].lower()
            starts = [m.start() for m in TOKEN_RE.finditer(lowered)]
            # Only a bare ". " between two tokens counts as a boundary here.
            boundaries = [bisect.bisect_left(starts, m.start())
                          for m in re.finditer(rb"(?<=[a-z0-9])\. (?=[a-z0-9])", lowered)]
            boundaries = [b for b in boundaries if 0 < b < len(toks)]
            if not boundaries:
                continue
            b = self.rng.choice(boundaries)
            k = self.rng.randint(2, 4)
            lo, hi = max(0, b - k + 1), min(b - 1, len(toks) - k)
            starts_ok = [s for s in range(lo, hi + 1) if any(self.is_anchor(t) for t in toks[s:s + k])]
            if starts_ok:
                s = self.rng.choice(starts_ok)
                return self.phrase_line(toks[s:s + k])

    def make(self, prefix: str, kind: str) -> str:
        return getattr(self, f"{prefix.lower()}_{kind}")()


def generate_queries(seed: int, scale: str, n_queries: int, texts, df, specials) -> list[str]:
    rng = random.Random(f"b-index:{seed}:{scale}:queries")
    maker = QueryMaker(rng, seed, scale, texts, df, specials)
    plan = plan_kinds(n_queries)
    rng.shuffle(plan)
    return [maker.make(prefix, kind) for prefix, kind in plan]


def write_queries(path: str, lines: list[str]) -> int:
    data = "".join(line + "\n" for line in lines).encode("ascii")
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def generate(seed: int, scale: str, out_dir: str) -> dict:
    n_docs, n_queries = SCALES[scale]
    t0 = time.monotonic()
    vocab = Vocab(seed)
    texts, df, specials = generate_docs(vocab, seed, scale, n_docs)
    os.makedirs(out_dir, exist_ok=True)
    docs_bytes = write_docs(os.path.join(out_dir, "docs.txt"), texts)
    lines = generate_queries(seed, scale, n_queries, texts, df, specials)
    query_bytes = write_queries(os.path.join(out_dir, "queries.txt"), lines)
    return {"docs": n_docs, "docs_bytes": docs_bytes, "queries": len(lines),
            "queries_bytes": query_bytes, "distinct_tokens": len(df),
            "seconds": round(time.monotonic() - t0, 1)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Track B (Index) dataset.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scale", choices=sorted(SCALES), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    stats = generate(args.seed, args.scale, args.out)
    print(" ".join(f"{k}={v}" for k, v in stats.items()), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
