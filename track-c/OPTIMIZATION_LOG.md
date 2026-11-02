# Track C: optimization log, 2026-11-01 to 2026-11-02

Every experiment of this session, in order. All numbers come from the live board: one run on full seed 1, on the Intel i5-13500 box at a fixed 2.5 GHz with 12 threads. Build: gcc 16, `-O3 -flto -march=x86-64-v3 -mtune=alderlake -mpclmul -maes -mprfchw -static`, unless the change says otherwise. Earlier steps are in `PERF.md`.

K = (T_ref/T)^0.50 x (M_ref/M)^0.20 x (S_ref/S)^0.15 x (D_ref/D)^0.15, and points = 50 x log10(K). D is 1 throughout. C/A and C/B divide T by Track A's and Track B's T in the same push, as both were unchanged, which shows when the box itself was slow.

Correctness "Pass" means PASS on all 8 local datasets before the push, and the board's hash gate after it. `+stress` adds repeated runs with 12 threads forced on 4 CPUs.

| Experiment | Commit | Run ID | Change | Correctness | T (s) | M (MB) | S | K | Points | C/B | Decision |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|
| Baseline | 8c8f827 | run-33e3561d13f4 | Starting point | Pass | 0.0194 | 10.4 | 378 | 46.17 | 83.22 | | |
| E01 | f0e5f77 | run-94013ec2579c | clang 17 with PGO | Pass | 0.0210 | 10.4 | 381 | 44.27 | 82.31 | 0.320 | Rejected |
| E02 | 8b2a4b5 | run-2db4c74624e2 | Raw threads, spin waits, parallel merge and format, groups and stacks in the huge pages, `madvise` clamped | Pass, stress | 0.0190 | 9.1 | 419 | 47.06 | 83.63 | 0.295 | Kept |
| E02 again | a1b55ac | run-7ab08c974e8f | Same code. The commit subject wrongly says prefetch test | Pass | 0.0194 | 10.0 | 419 | 45.81 | 83.05 | 0.302 | Sample |
| Diagnostic | 227e3aa | run-6339eae738c1 | Phase times through T and M | Pass | | | | | | | Measurement |
| E03 | b685814 | run-240aea77489e | Prefetch 6 lines of the next page at each page crossing | Pass | 0.0185 | 8.1 | 424 | 48.83 | 84.44 | 0.285 | Kept |
| E04 | 8428f6b | run-d6efa91552fd | Trim 424 to 362 lines | Pass, stress | 0.0187 | 9.8 | 362 | 47.85 | 83.99 | 0.286 | Kept |
| E05 | 744ee07 | run-16b7fc1fb9e4 | No worker writes first to a page, maps on 2 MB boundaries, one huge page cleared after spawning | Pass, stress | 0.0186 | 8.1 | 371 | 49.68 | 84.81 | 0.287 | Kept |
| E06 | ee1f470 | run-a6a838c7585a | gcc PGO | Pass | 0.0189 | 8.4 | 375 | 48.93 | 84.48 | 0.291 | Rejected |
| **E05 again** | **db57537** | **run-edadce99ac41** | **Same code as E05. Final version** | **Pass** | **0.0186** | **7.9** | **371** | **49.97** | **84.94** | **0.283** | **Best** |
| E07 | 778aebe | run-4a7b4de6f0a3 | `-fno-tree-loop-distribute-patterns`, no `memset` call per batch | Pass | 0.0187 | 8.1 | 371 | 49.59 | 84.77 | 0.276 | No clear gain |
| Diagnostic | 111390a | run-9609e00b8f24 | Cycles per pass through T and M | Pass | | | | | | | Measurement |
| M1 | fb54d83 | run-924c80cb7770 | Bit writes without a lock. Loses updates, never a candidate | Hash mismatch, as expected | 0.0172 | 8.2 | | | | | Measurement |
| E08 | dd95fcf | run-d1d9c920ac4e | Bits buffered in 16 partitions, one lock per 64 | Pass, stress | 0.0213 | 8.5 | 391 | 45.53 | 82.91 | 0.330 | Rejected |
| E09 | dcb6c7d | run-c2a01a0b5d9b | The same with 64 partitions of 32 | Pass | 0.0186 | 8.4 | 391 | 48.95 | | | Rejected |
| E10 | b740d59 | run-0922446af5f8 | One lock and 16 names per country, lines fetched before locking | Pass, stress | 0.0188 | 8.4 | 389 | 48.79 | 84.41 | 0.290 | Rejected |
| E11 | 7a5aa19 | run-58dff57b5021 | Plain clang 17 | Pass | 0.0196 | 12.5 | 371 | 44.33 | 82.33 | 0.303 | Rejected |
| E12 | 15e7581 | run-b1b4e00da5ec | Faults grouped at each chunk's start | Pass | 0.0185 | 10.6 | 373 | 47.20 | 83.70 | 0.284 | See E13 |
| E13 | 7b6e38f | run-d38b4f2244b1 | E12 with page tables for the first 16 MB before the workers start | Pass | 0.0185 | 8.1 | 373 | 49.76 | | 0.282 | Candidate |
| E13 again | 5bf48ce | run-62ad7e8e1992 | Same code, as a control | Pass | 0.0185 | 8.6 | 373 | 49.22 | 84.61 | 0.288 | Sample |
| E14 | 2acca3b | run-5b51879173c4 | Chunks taken far apart, all page tables made first | Pass, stress | 0.0189 | 7.9 | 374 | 49.49 | 84.73 | 0.291 | Rejected |
| E15 | ff661bb | run-8236bc76c666 | `MADV_SEQUENTIAL` on the input | Pass | 0.0198 | 8.3 | 374 | 47.85 | | 0.302 | Rejected |
| E16 | d3b5c50 | run-d26e88d8ea2d | Page crossing burst of 12 lines | Pass | 0.0225 | 8.2 | 373 | 44.98 | | 0.349 | Rejected |
| E17 | 1671e65 | run-0e3aeaace879 | Burst of 4 lines | Pass | 0.0185 | 8.2 | 373 | 49.65 | | 0.289 | Flat |
| E18 | da25e82 | run-7526874645fa | Burst of 6 lines into L2 only | Pass | 0.0186 | 8.2 | 373 | 49.49 | | 0.289 | Flat |
| E19 | 8f73ad8 | run-def04b3c7790 | `pread` of 32 KB chunks | Pass, stress | 0.0203 | 10.0 | 371 | 45.54 | 82.92 | 0.312 | Rejected |
| E20 | 56875f3, 0f8ab90 | run-43ae6454571a, run-743efdfe642e | Stacks on different L1 sets | Pass, stress | 0.0204, 0.0187 | 8.2, 8.7 | 373 | 47.28, 48.86 | | 0.289, 0.290 | Rejected. Track B was 10% slow in the first run |
| E21 | a6b18c9 | run-3a891db55e9c | 32-bit name tags | Pass, stress | 0.0192 | 8.9 | 373 | 47.86 | | 0.296 | Rejected |
| E22 | f13588d, cde3e24, 018db09 | run-feef3161d32a, run-945d1621fd65, run-49fbebca614e | Workers wait for thread 0's page table frontier | Pass, stress | 0.0185, 0.0186, 0.0185 | 8.1, 8.0, 8.1 | 378 | 49.61, 49.70, 49.70 | 84.82 | 0.285, 0.275, 0.281 | Candidate |
| E23 | bbf965b, 1c3444e | run-05ce611b36f3, run-d7791bb6838d | Trim to 368, frontier store as seq_cst | Pass, stress | 0.0187, 0.0185 | 10.0, 10.4 | 368 | 47.60, 47.46 | | 0.289, 0.275 | Rejected |
| E24 | d7103d4 | run-0f1f036b3924 | E23 with the relaxed frontier store | Pass | 0.0187 | 8.1 | 370 | 49.58 | | 0.286 | Candidate |
| E25 | c1d010c | run-053c715a50aa | `-O2` | Pass | 0.0192 | 8.2 | 370 | 48.82 | | 0.293 | Rejected |
| E26 | 343be2b | run-75269556bcf2 | Batches of 12 blocks | Pass | 0.0186 | 8.2 | 370 | 49.50 | | 0.286 | Flat |
| E27 | 1797358 | run-b566e5fa1028 | Chunks of 128 KB | Pass | 0.0187 | 11.1 | 370 | 46.57 | | 0.288 | Rejected, led to E28 |
| E28 | 626c40e | run-ae455436b511 | E27 with the frontier wait before the first read | Pass, stress | 0.0187 | 7.9 | 370 | 49.76 | | 0.288 | Proved the cause |
| E29 | 85fe188, df17fea | run-9cd5c0047aba, run-6883587abee6 | 256 KB chunks, seq_cst frontier store | Pass, stress | 0.0207, 0.0185 | 8.4, 8.7 | 368 | 46.80, 49.12 | | 0.318, 0.284 | Rejected |
| E30 | 404b6fb | run-cf935e8a4c22 | 256 KB chunks, relaxed frontier store, fixed wait | Pass | 0.0186 | 8.1 | 370 | 49.76 | | 0.285 | Level with the final version |

Blank points cells were not recorded for that run, and K is exact from the board.

## Reading the results

- T settled at 0.0185 to 0.0187 from E03 onward. Only M and S moved K after that.
- One sample does not separate versions within about 1% of K. E05's two samples are K 49.68 and 49.97, and E30's one is 49.76, so these are level within noise. The final version is the highest measured, not a proven winner.
- Single T outliers happen on unchanged code, as E29 shows with 0.0207 then 0.0185. Final scoring averages the middle three of five runs, which removes them.

## Correctness of the final version

Run before the final push: all 8 local datasets, the forced 12-thread stress set, and a comparison with the official Python reference on datasets the code had never seen (seed 7 sample, seed 11 sample, seed 7 full), 15, 15 and 5 runs, at the normal thread count and at 12 forced. 70 of 70 runs matched the reference's SHA-256.
