# Track C: performance log

Every step from the reference to the final version, each with the live-board run ID that measured it. Rejected experiments are listed too. Local timings on the 2-core laptop proved unreliable here, so decisions were made on board runs. One board run varies by about 2%.

| Step | Change | Run ID | T (s) | M (MB) | S | Points | Outcome |
|---|---|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 14.44 | 367 | 108 | 0 | |
| 1 | Rust rewrite: in-place parsing, shared name table with per-country bitsets, prefetch pipeline, huge pages | run-00ffb948bcd6 | 0.0504 | 9.7 | 485 | 72.34 | Kept |
| 2 | CLMUL structural masks, numbers read from known field ends, group prefetching | run-c264e27b27d8 | 0.0494 | 9.7 | 499 | 72.46 | Kept. 16% less CPU on one local core, but no faster on the box, so parsing was not the limit |
| 3 | Private per-thread name logs instead of shared bitsets, testing cache-line bouncing | run-7860a0462204 | 0.0475 | 39.0 | 523 | 66.69 | Barely faster. Reversed in step 14 |
| 4 | Regions load and input unmap made parallel, hot counters on their own cache lines, tables touched before workers start, small pages for logs | run-62c87ca33f7f | 0.0307 | 13.7 | 563 | 75.72 | Kept, the largest gain |
| 5 | Name table with 8-way buckets compared by vector, AVX2 decode of three numbers | run-4ddeb89b73ae | 0.0288 | 13.2 | 592 | 76.43 | Kept |
| 6 | Prefetch input 1024 bytes ahead | run-8515688437b5 | 0.0288 | 13.2 | 593 | 76.42 | Rejected, no change |
| 7 | `pread` of 64 KB chunks instead of `mmap` | run-7ad6cec8487e | 0.0303 | 13.6 | 580 | 75.81 | Rejected, 5% slower |
| 8 | Profile-guided build | run-bea72f0cd5bd | 0.0288 | 13.1 | 592 | 76.45 | Dropped, no time change. Steps 6 to 8 suggested a throughput limit |
| 9 | Fewer micro-ops per record: pairs recorded only when present, AES hash, one vector add for totals | run-9da35aaada88 | 0.0264 | 12.9 | 574 | 77.55 | Kept |
| 10 | One pass, without prefetch or record buffer | run-12d129ab1dcd | 0.0294 | 13.2 | 545 | 76.44 | Rejected, 11% slower |
| 11 | Batch of 8 blocks, then 32 (run-3ed9a4780bde, 0.0265 s, 13.8 MB) | run-3316be06e9d1 | 0.0265 | 13.0 | 574 | 77.48 | Within noise, stayed at 16 |
| 12 | Port to C++ with gcc 16 | run-c722cf94e3ca | 0.0261 | 13.2 | 414 | 78.63 | Kept. Same speed, 28% fewer lines |
| 13 | First two `""` pairs counted without a branch, shared name counter removed | run-c04ebff90857 | 0.0245 | 13.1 | 412 | 79.40 | Kept. Measured: 25.2% of blocks hold pairs, 1.7% more than two |
| 14 | Shared bitsets again, in place of the logs | run-bdcf951e9670 | 0.0243 | 9.6 | 394 | 80.95 | Kept. Step 3's conclusion was wrong, as the log version had a false-sharing bug |

## What was learned

- Step 2 was the turning point: a real CPU saving that did not move the board meant the bottleneck was elsewhere.
- Step 3 was tested too early. Its 39 MB is probably threads racing to fault the same huge page, each charged 2 MB until the race resolves. Step 4's fix removed it, but the cause was not measured directly.
- The cost of bitset writes was never isolated on the box. Step 14 shows only that the logs did not beat them.
- By a hand count of micro-ops, parsing should now take less time than the board shows, so part of the remaining time is unexplained.
