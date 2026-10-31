# Track B: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 56.35 | 2,200 | 0 |
| 1 | Rust rewrite: index only query terms, bitsets for common terms, no positions, phrases verified by re-reading candidates | run-0136d17c3355 | 0.554 | 162 | 54.5 |
| 2 | Match phrases live during tokenizing, triggered by each phrase's rarest term, so no document is re-read. One 16-byte table entry per term, so a token costs one cache miss. Remove duplicates when lists are sorted | run-c4b57ba6ab55 | 0.335 | 152 | 59.8 |

Experiments measured locally on one core, not pushed on their own:

- Buffering common-term bits for 64 documents instead of writing them directly: no measurable gain, kept because it removes an atomic per token.
- A bigram filter as the phrase trigger: answering fell from 2.4 s to 0.5 s, but tokenizing grew by about 1.4 s. Replaced by the rare-term trigger.
- `mimalloc`: about 3% faster but 220 MB against 134 MB, net -4.1 points after the dependency cost. Rejected.
