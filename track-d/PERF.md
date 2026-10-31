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

Redesign measured locally until the next board run (full seed 1, laptop, 1 core and 4 threads):

| Change | 1 core | 4 threads | Memory, 4 threads | Note |
|---|---|---|---|---|
| Starting point | 4.2 s | 2.33 s | 110 MB | 140 MB with 12 threads forced, as on the board |
| Fixed 4-slot adjacency placed during parsing, one label table, packed heap, huge pages | 2.13 s | 1.44 s | 88 MB | Removes the build phase |
| Bucket queue | 2.05 s | 1.3 s | 88 MB | Small gain, as the study below predicted |
| Prefetch next node's neighbours' labels | 1.8 s | 1.12 s | 88 MB | Cycle counts showed a label lookup at about 105 cycles was the largest cost |
| Shared pool of large label tables, 85% load | 1.9 s | 1.07 s | 77 MB | 95 MB with 12 threads forced |

Experiments that failed:

- Skipping the edge back to the parent removed a third of relaxations and changed nothing, as those lookups were already cache hits.
- `mallopt` to return freed memory: 140 MB to 94 MB, but kernel time went from 0.6 s to 3 s, from mapping and unmapping.
- Node renumbering: queries 25 to 30% faster, for 0.65 to 0.85 s of serial work.
- Choosing the search side by distance rather than frontier size: 2.5x slower.
- Pruning nodes the far side has settled: slightly slower.
- Peeking two or three nodes ahead: no gain over one.
- A plain vector per bucket: 14% slower on 4 threads.
