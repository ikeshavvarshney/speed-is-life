# Track D: Traverse

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/d-traverse.md).

## Approach

Rust, standard library only, one thread per available CPU.

The graph is mapped with `mmap` and parsed in parallel. Each worker sorts edges into 64 node ranges as it parses, so each range can then be built into compressed adjacency arrays by one thread, inside its cache, with no shared counters. The reverse graph is built the same way.

Each query runs bidirectional Dijkstra, and threads claim queries one at a time because the search sizes have a long tail. The side with the smaller heap expands next, and the search stops when the two frontiers together can no longer beat the best meeting point. When either heap empties, that side has reached all it can, so unreachable targets end immediately. Distances use 64 bits.

Distances live in small open-addressing tables rather than arrays of one entry per node, which would cost about 28 MB per thread. Each search clears only the part of its table it used.

## Dependencies

None.

## With another 24 hours

Lower the cost per settled node, and find a cheaper node ordering. Depth-first renumbering cut query time, but cost more to build than it saved.
