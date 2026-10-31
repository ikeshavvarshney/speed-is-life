# Track D: Traverse

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/d-traverse.md).

## Approach

Rust, standard library only, one thread per available CPU.

The graph has four 4-byte slots per node at a computed address, one array per direction. Over 98% of nodes have at most four edges, so finding a node's edges needs no lookup, and they can be prefetched from the node id alone. A slot holds the head above 11 bits of weight, as only 0.07% of weights need more. Edges that do not fit go to a small side list that the node's last slot points to.

Edges are placed directly as they are parsed, with no log and no build phase. Each placement is prefetched and made 16 edges later, so the cache misses of these random writes overlap. Huge pages are requested for both arrays.

Each query runs bidirectional Dijkstra, and threads claim queries one at a time because search sizes have a long tail. The side with the smaller frontier expands next, and the search stops when the two frontiers together can no longer beat the best meeting point. When either queue empties, that side has reached all it can, so unreachable targets end immediately.

Distances for both directions share one table, so a relaxation touches one slot. A bucket queue replaces the heap: one bucket per distance over a window that an edge cannot jump past, with a bitmap to find the next used bucket.

Random label lookups are the largest cost, because node ids are random. So prefetching happens in two stages: pushing a node prefetches its edges, and popping a node reads the next node's edges and prefetches its neighbours' labels, so they are ready when that node is processed.

Each thread owns a small label table. Searches that outgrow it borrow a large one from a shared pool, so no search pays to map or unmap memory, which stalls every thread.

## Dependencies

None. `mimalloc` and `radix-heap` were tried and rejected: see `PERF.md`.

## With another 24 hours

Reduce line count, and find a cheap way to give node ids locality. Renumbering made queries 25 to 30% faster, but cost more than it saved.
