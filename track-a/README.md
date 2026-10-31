# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

Rust, standard library only, one thread per available CPU.

The input is mapped with `mmap` and workers claim 256 KB ranges from an atomic counter. A line belongs to the range its first byte falls in, so no line is parsed twice or skipped.

Each line is parsed without scanning most of it. The timestamp has a fixed width, so the hour comes directly from four digit positions. The method is skipped with a word-at-a-time search, the endpoint end is found with AVX2, and the latency is decoded with SWAR so there is no branch per digit.

Each worker interns endpoints in its own small hash table and appends `(latency, endpoint id, error flag)` to its own bucket for that hour, in 6 bytes per request. Nothing is shared during parsing, so there are no atomics on the hot path.

After parsing, endpoint names are merged and sorted once. A group's position in the output is then just its hour and its endpoint's rank. Each hour is an independent task: count per endpoint, place latencies contiguously, take percentiles with selection (or a sort for groups of 32 or fewer), and format. An hour's blocks are freed once it has been placed, which keeps peak memory near the size of the parsed log.

The spec guarantees the 72-hour span and the 1,842-endpoint catalogue for every seed, and the hour mapping and 15-bit endpoint ids rely on those guarantees.

## Dependencies

None.

## With another 24 hours

Write finished hours straight to the output in order, so formatted text does not sit in memory. Try profile-guided optimisation, and tune chunk size and thread count on the scoring box.
