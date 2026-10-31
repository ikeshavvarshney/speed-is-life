# Track A: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 32.05 | 228 | 0 |
| 1 | Rust rewrite: per-hour logs with no atomics, SWAR and AVX2 parsing, selection for percentiles, mmap input | run-f84389cf2c3a | 0.043 | 29 | 74.0 |

Experiments that did not pay:

| Change | Run ID | T (s) | M (MB) | S | K | Result |
|---|---|---|---|---|---|---|
| Inline key compare, simpler bucket push, faster number output, sort for small groups | run-83a72f8bdff1 | 0.0405 | 29.9 | 538 | 30.06 | 6% faster but 82 more scored lines, so K fell from 30.16. Time alone is the wrong measure |
