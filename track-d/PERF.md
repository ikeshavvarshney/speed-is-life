# Track D: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 72.69 | 893 | 0 |
| 1 | Rust rewrite: parallel parse, bucketed graph build, bidirectional Dijkstra with small hash tables. Tried first and rejected locally: forward-only search, depth-first renumbering, identity hash | run-89b55425b612 | 0.342 | 138 | 58.3 |
| 2 | Redesign: fixed 4-slot adjacency placed during parsing, bucket queue, two-stage prefetch, one label table, shared pool of large tables. Details below | run-dcf64c4d623d | 0.189 | 92 | 66.0 |

Experiments measured locally, not pushed on their own (full seed 1, 4 threads on a laptop):

- Forward-only search with a check for targets that have no incoming edges: about 6 s and 77 MB. Bidirectional search took about 2.8 s and 109 MB, worth far more than the memory it costs.
- Depth-first renumbering of nodes: queries got faster, but the ordering and rebuild added about 1.1 s. Dropped.
- Identity hash on renumbered ids: slower, from clustering. Dropped.
- Releasing large search tables after each query: slower, as searches kept regrowing them. Tables now keep their capacity and clear only what was used.
- `mimalloc`: 19% faster but 205 MB against 110 MB, net -2.7 points. `radix-heap`: about 2% faster, net -2.1 points. Both rejected.

Unchanged code has scored from 0.332 s to 0.410 s on the board (run-3ef6e7cc130a, run-3e28bed889bd), so single runs on this track need care.

Redesign, measured locally before step 2 (full seed 1, laptop, 1 core and 4 threads):

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

## Port to C++

The first port looked level with Rust on the 2-core laptop, but was slower on the 12-thread box. Three build-level fixes followed, the same on all tracks: writable globals moved off shared cache lines, `u64` given a different type from `size_t` so writes to it do not force sizes and masks to be reloaded, and `-mtune=alderlake` to match the box's P-cores. Tuning for the core appeared to be the largest of these, in the one push that changed it across all tracks. `prefetchw` was tried on tracks C and D and removed in the same push as the first two fixes, so its own cost was not isolated.

| Step | Run ID | T (s) | M (MB) | S | K | Points |
|---|---|---|---|---|---|---|
| Last Rust best | run-68fc0ccbbb86 | 0.1827 | 92.6 | 584 | 21.22 | 66.34 |
| First port, with prefetchw | run-c35aabfc65fd | 0.2342 | 104.6 | 455 | 18.99 | 63.93 |
| Globals separated, distinct type, no prefetchw | run-f5af21ff8c7b | 0.2000 | 104.6 | 455 | 20.55 | 65.64 |
| Tuned for the core | run-70420ed269bc | 0.1967 | 102.7 | 462 | 20.75 | 65.85 |
| Label tables mapped directly | run-a5d65991e14d | 0.1996 | 101.9 | 463 | 20.63 | 65.72 |
| Regrow copy sized exactly | run-950a97e36a10 | 0.1971 | 97.2 | 456 | 21.00 | 66.11 |

Rust scored K 20.6 to 21.2 over recent runs, so C++ is level on K. Two memory guesses failed. Freeing edge lists while merging did nothing, as there are only about 59,000 loose edges, 0.7 MB. Mapping label tables directly did nothing either. Measuring the cgroup peak after each phase then showed the peak falls during search, where a temporary copy made while a label table grows was held twice by doubling. Sizing it exactly cut the peak from 102 to 97 MB. Both failed changes were reverted.

## Final day

| Step | Run ID | T (s) | S | K | Result |
|---|---|---|---|---|---|
| Thread stacks capped below 2 MB, so huge pages do not charge them (c8c82af) | not run alone | | | | Reverted: the next push replaced it before it ran |
| Built with clang 17 from the swift:6.2 image instead of gcc 16 (932f8ed) | run-5162d57c3ae8 | | 456 | 20.68 | Rejected: inside gcc's range of K 20.1 to 21.3, so no gain |
| Back to the gcc 16 build of the best run, run-e62eb6af6e64 (2d0d9ef) | run-2f197a870ae4 | 0.1966 | 456 | 21.11 | Kept, final version |

Over many runs the gcc build scored K 20.1 to 21.3, and the board varies about 2% between runs. One clang run inside that range proved nothing, so the final version is the gcc build that every earlier measurement used.
