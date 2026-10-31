# Track B: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 56.35 | 2,200 | 0 |
| 1 | Rust rewrite: index only query terms, bitsets for common terms, no positions, phrases verified by re-reading candidates | run-0136d17c3355 | 0.554 | 162 | 54.5 |
| 2 | Match phrases live during tokenizing, triggered by each phrase's rarest term, so no document is re-read. One 16-byte table entry per term, so a token costs one cache miss. Remove duplicates when lists are sorted | run-c4b57ba6ab55 | 0.335 | 152 | 59.8 |
| 3 | Mask-based tokenizer, two passes per document with prefetch, 16-bit hit log and lists built without sorting, pair bitset for anchors, parallel query reading into one table, boolean queries evaluated from text | run-9d8f92934f66 | 0.157 | 58 | 71.5 |

Experiments measured locally on one core, not pushed on their own:

- Buffering common-term bits for 64 documents instead of writing them directly: no measurable gain, kept because it removes an atomic per token.
- A bigram filter as the phrase trigger: answering fell from 2.4 s to 0.5 s, but tokenizing grew by about 1.4 s. Replaced by the rare-term trigger.
- `mimalloc`: about 3% faster but 220 MB against 134 MB, net -4.1 points after the dependency cost. Rejected.

How step 3 was built, measured locally on a laptop (full seed 1, 1 core, then 4 threads):

| Change | 1 core | 4 threads | Memory | Note |
|---|---|---|---|---|
| Starting point | 3.8 s | 2.05 s | 134 MB | |
| Mask-based tokenizer, 16-bit hit log and lists, list build without sorting | 2.45 s | 1.09 s | 46 MB | Tokenizing alone fell from 0.85 s to 0.2 s |
| Hash each window first, then prefetch and look up | 2.22 s | 1.04 s | 46 MB | Replaced two steps later |
| Branch-free token actions | 2.17 s | 1.01 s | 46 MB | Small gain |
| Pair bitset to reject anchor tokens | 2.15 s | 0.96 s | 46 MB | Reading neighbours from bytes was itself slow |
| Tokenize and look up in separate passes, prefetch 12 tokens ahead | 1.96 s | 0.78 s | 46 MB | Also removed code |
| Read queries in parallel, evaluate boolean queries from text | 1.9 s | 0.85 s | 46 MB | Aimed at the 12-thread box, where serial setup matters most |

Other experiments: prefetch distances 4, 12 and 24 were within noise. The same hot loop in C++ with gcc 16 took 0.76 to 0.81 s against 0.60 to 0.79 s in Rust, with identical checksums, so no language switch.

Wrong guesses worth recording: buffering common-term bit writes gave nothing, and removing per-token phrase bookkeeping gave far less than a simple test had suggested. Removing one layer at a time showed the real cost was a mispredicted branch on a load still missing cache.
