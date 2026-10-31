# Track D: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 72.69 | 893 | 0 |
| 1 | Rust rewrite: parallel parse, bucketed graph build, bidirectional Dijkstra with small hash tables. Tried first and rejected locally: forward-only search, depth-first renumbering, identity hash | run-89b55425b612 | 0.342 | 138 | 58.3 |

Experiments measured locally, not pushed on their own (full seed 1, 4 threads on a laptop):

- Forward-only search with a check for targets that have no incoming edges: about 6 s and 77 MB. Bidirectional search took about 2.8 s and 109 MB, worth far more than the memory it costs.
- Depth-first renumbering of nodes: queries got faster, but the ordering and rebuild added about 1.1 s. Dropped.
- Identity hash on renumbered ids: slower, from clustering. Dropped.
- Releasing large search tables after each query: slower, as searches kept regrowing them. Tables now keep their capacity and clear only what was used.
- `mimalloc`: 19% faster but 205 MB against 110 MB, net -2.7 points. `radix-heap`: about 2% faster, net -2.1 points. Both rejected.

Unchanged code has scored from 0.332 s to 0.410 s on the board (run-3ef6e7cc130a, run-3e28bed889bd), so single runs on this track need care.
