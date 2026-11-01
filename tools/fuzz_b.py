#!/usr/bin/env python3
"""Compares a Track B binary with the official reference on hostile random inputs.

usage: fuzz.py <binary> <reference b_index.py> <rounds> [first seed]

Each round writes a small corpus and query file, runs the reference module's logic for the answer,
then the binary with several thread counts. Inputs are built to reach every path: tokens of 1 to 80
bytes, mixed case, bytes above 0x7F, terms common enough for bitsets, chains of every operator shape,
empty documents and repeated phrase terms. The binary reads /data and writes /out/results.txt.
"""
import hashlib
import importlib.util
import os
import random
import subprocess
import sys

binary, reference, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
first_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
spec = importlib.util.spec_from_file_location("ref", reference)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
data = "/tmp/fuzzdata"
os.makedirs(data, exist_ok=True)
if os.path.islink("/data"):
    os.unlink("/data")
os.symlink(data, "/data")
ALPHA = "abcdefghijklmnopqrstuvwxyz0123456789"


def word(rng, longest):
    n = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18, 24, 31, 32, 33, 40, 63, 64, 65, 80])
    n = min(n, longest)
    return "".join(rng.choice(ALPHA) for _ in range(n))


def make(seed):
    rng = random.Random(seed)
    # A skewed vocabulary, so some terms land in bitsets and others in short lists.
    vocab_size = rng.choice([30, 200, 2000, 20000])
    vocab = [word(rng, rng.choice([8, 16, 32, 80])) for _ in range(vocab_size)]
    doc_count = rng.choice([1, 2, 63, 64, 65, 511, 512, 513, 1024, 3000, 8000])
    seps = [" ", "  ", ", ", ". ", "-", "/", "'s ", " - ", "\u2014", "\u00a0", "\u2019s ", "\u00fc", "\u212a", "\t".replace("\t", ";")]
    docs = []
    for i in range(doc_count):
        kind = rng.random()
        if kind < 0.02:
            text = ""
        elif kind < 0.04:
            text = "\u00a7\u00a7 * * *"
        else:
            parts = []
            for _ in range(rng.choice([1, 2, 5, 20, 80, 300])):
                w = vocab[min(int(rng.paretovariate(0.7)) - 1, vocab_size - 1)]
                r = rng.random()
                if r < 0.05:
                    w = w.upper()
                elif r < 0.1:
                    w = w.capitalize()
                parts.append(w)
                parts.append(rng.choice(seps))
            text = "".join(parts)
        docs.append(f"{i}\t{text}\n")
    queries = []
    tokenized = [ref.tokenize(d.split("\t", 1)[1].rstrip("\n").encode()) for d in docs]
    for _ in range(rng.choice([1, 20, 300, 3000])):
        if rng.random() < 0.6:
            terms = []
            for _ in range(rng.randint(1, 6)):
                r = rng.random()
                if r < 0.05:
                    terms.append("absentterm" + str(rng.randint(0, 3)))
                elif r < 0.1:
                    terms.append(rng.choice(["and", "or", "not"]))
                else:
                    terms.append(vocab[min(int(rng.paretovariate(0.5)) - 1, vocab_size - 1)][:32])
            line = "BOOL " + terms[0]
            for t in terms[1:]:
                line += " " + rng.choice(["AND", "OR", "NOT"]) + " " + t
        else:
            d = rng.choice(tokenized)
            if len(d) >= 2 and rng.random() < 0.8:
                n = rng.randint(2, 8)
                at = rng.randint(0, max(0, len(d) - n))
                terms = [t.decode() for t in d[at:at + n]]
                if rng.random() < 0.2:
                    rng.shuffle(terms)
                terms = [t[:32] for t in terms]
                if len(terms) < 2:
                    terms = terms * 2
            else:
                w = vocab[rng.randint(0, min(5, vocab_size - 1))][:32]
                terms = [w] * rng.randint(2, 4)
            line = 'PHRASE "' + " ".join(terms) + '"'
        queries.append(line + "\n")
    return docs, queries


failures = 0
runs = 0
for seed in range(first_seed, first_seed + rounds):
    docs, queries = make(seed)
    with open(data + "/docs.txt", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(docs)
    with open(data + "/queries.txt", "w", encoding="ascii", newline="\n") as f:
        f.writelines(queries)
    ref.main(data, "/tmp/fuzzref")
    want = hashlib.sha256(open("/tmp/fuzzref/results.txt", "rb").read()).hexdigest()
    for threads in (1, 2, 5, 12):
        env = dict(os.environ, FORCE_THREADS=str(threads))
        if os.path.exists("/out/results.txt"):
            os.unlink("/out/results.txt")
        r = subprocess.run([binary], env=env, stderr=subprocess.DEVNULL, timeout=60)
        runs += 1
        got = hashlib.sha256(open("/out/results.txt", "rb").read()).hexdigest() if os.path.exists("/out/results.txt") else "none"
        if r.returncode != 0 or got != want:
            failures += 1
            print(f"FAIL seed {seed} threads {threads} rc {r.returncode} docs {len(docs)} queries {len(queries)}")
            os.system(f"cp {data}/docs.txt /tmp/fail_docs_{seed}.txt; cp {data}/queries.txt /tmp/fail_queries_{seed}.txt; cp /out/results.txt /tmp/fail_out_{seed}.txt; cp /tmp/fuzzref/results.txt /tmp/fail_ref_{seed}.txt")
            break
print(f"fuzz: {runs} runs over {rounds} inputs, {failures} failures")
if os.path.islink("/data"):
    os.unlink("/data")
os.symlink("/v/b/full-seed1", "/data")
