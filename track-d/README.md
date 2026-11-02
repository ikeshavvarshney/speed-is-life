# Track D: Traverse

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/d-traverse.md).

## Approach

C++20 with gcc 16, standard library only, one thread per available CPU. The first versions were Rust; the same algorithms moved to C++ for 22% fewer scored lines at about the same K.

The graph has four 4-byte slots per node at a computed address, one array per direction. Over 98% of nodes have at most four edges, so finding a node's edges needs no lookup, and they can be prefetched from the node id alone. A slot holds the head above 11 bits of weight, as only 0.07% of weights need more. Edges that do not fit go to a small side list that the node's last slot points to.

Edges are placed directly as they are parsed, with no log and no build phase. Each placement is prefetched and made 16 edges later, so the cache misses of these random writes overlap. Huge pages are requested for both arrays.

Each query runs bidirectional Dijkstra, and threads claim queries one at a time because search sizes have a long tail. The side with the smaller frontier expands next, and the search stops when the two frontiers together can no longer beat the best meeting point. When either queue empties, that side has reached all it can, so unreachable targets end immediately.

Distances for both directions share one table, so a relaxation touches one slot. A bucket queue replaces the heap: one bucket per distance over a window that an edge cannot jump past, with a bitmap to find the next used bucket.

Random label lookups are the largest cost, because node ids are random. So prefetching happens in two stages: pushing a node prefetches its edges, and popping a node reads the next node's edges and prefetches its neighbours' labels, so they are ready when that node is processed.

Each thread owns a small label table. Searches that outgrow it borrow a large one from a shared pool, so no search pays to map or unmap memory, which stalls every thread. A table grows when it is 7/8 full.

## Dependencies

None. `mimalloc` and `radix-heap` were tried and rejected: see `PERF.md`.

## With another 24 hours

- Give node ids locality cheaply. Renumbering made queries 25 to 30% faster but cost 0.65 to 0.85 s of serial work, so the next attempt is an order built in parallel during parsing, judged by label misses per popped node.
- Measure the phase split on the box, as Tracks B and C did, since the laptop's small L3 exaggerates the query share.
- Try a few landmarks only for the long searches that outgrow the per-thread table, where the tail of the time is.

## Limits

The spec rules these cases out, and the code relies on that without checking at run time:

- Node ids fit in 21 bits (n below 2,097,152), and every query names a node below n.
- Distances stay below 2^43. A longer path would corrupt a label rather than fail.
- Failures of `open`, `fstat` and `mmap` are not checked. Only the output `write` is.
