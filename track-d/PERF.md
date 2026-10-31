# Track D: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 72.69 | 893 | 0 |
| 1 | Rust rewrite: parallel parse, bucketed graph build, bidirectional Dijkstra with small hash tables. Tried first and rejected locally: forward-only search, depth-first renumbering, identity hash | run-89b55425b612 | 0.342 | 138 | 58.3 |
