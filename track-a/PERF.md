# Track A: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 32.05 | 228 | 0 |
| 1 | Rust rewrite: per-hour logs with no atomics, SWAR and AVX2 parsing, selection for percentiles, mmap input | run-f84389cf2c3a | 0.043 | 29 | 74.0 |
| 2 | Second rewrite: 64-byte space mask, shared tag table, 4-byte log entries, AVX2 selection, branch-free digits, logs given back after use, writing during render, static link, input page tables cleared in parallel. Details below | run-2f0cf497a665 | 0.0179 | 23 | 83.5 |

The second rewrite, measured locally before step 2 (full seed 1, laptop, 2 cores and 4 threads, best of several alternating runs). The board beat the local estimate of 0.022 s, and memory came in above the local 21 MB at 12 threads:

| Change | 1 core | 4 threads | Memory, 4 threads | Note |
|---|---|---|---|---|
| Starting point | 351 ms | 273 ms | 23 MB | Parsing was 70% of the time, about 90 ns per line |
| New parser: 64-byte space mask, shared tag table, 4-byte log entries, one set of threads, streaming output | 200 ms | 122 ms | 30 MB | Memory rose only because nothing was freed yet |
| AVX2 selection, sorting network, branch-free digits, compact name block | 173 ms | 111 ms | 34 MB | Render fell from about 70 to 38 cycles per line |
| Give logs back after use, reuse them for the next hour | 188 ms | 111 ms | 19 MB | 26 MB at 12 threads, from text buffers that built up |
| Cap the pool of text buffers at 12 | | | 19 MB | 20 MB at 12 threads. A cap of 2 caused map and fault churn |
| Static linking, compact first-seen ids | | | 19 MB | About 0.6 to 1 ms per process, less memory |
| Clear the input's page tables in parallel after parsing | 172 ms | about 100 ms | 18 MB | Unmapping is per page, and the harness clock includes exit |

Bugs the tests caught, each of which would have failed the correctness gate on some seed or run:

- A single AES round after the last input merged names differing only in byte 125, as it spreads a byte over only four bytes. Two rounds fixed it. Checked on 300 seed catalogues and 3.1M near-identical pairs, with no collisions.
- Scratch space overlapped the values when an hour had big latencies.
- `madvise` failed silently on unaligned regions, and on one that happened to be aligned, the rounded length zeroed the start of the next region while another thread used it. Only some runs failed. Regions are now whole pages, and the check repeats runs at several thread counts.

Experiments measured locally that did not pay:

- `pread` per chunk instead of `mmap`: 30 ms more CPU, same wall time. To retry on the box, where exit costs weigh more.
- `MADV_POPULATE_READ` per chunk, 1024 buckets, a single malloc arena, PGO: no clear gain.
- Waiting for a turn to write instead of handing hours over: 30 fewer lines, but simulated 1 to 2 ms slower with the real hour sizes.
- Rendering hours in other orders: in order won or nearly won in all scenarios simulated.

Experiments that did not pay on the board:

| Change | Run ID | T (s) | M (MB) | S | K | Result |
|---|---|---|---|---|---|---|
| Inline key compare, simpler bucket push, faster number output, sort for small groups | run-83a72f8bdff1 | 0.0405 | 29.9 | 538 | 30.06 | 6% faster but 82 more scored lines, so K fell from 30.16. Time alone is the wrong measure |
