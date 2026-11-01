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

## Trim and build flags

The first port looked level with Rust on the 2-core laptop, but was slower on the 12-thread box. Three build-level fixes followed, the same on all tracks: writable globals moved off shared cache lines, `u64` given a different type from `size_t` so writes to it do not force sizes and masks to be reloaded, and `-mtune=alderlake` to match the box's P-cores. Tuning for the core appeared to be the largest of these, in the one push that changed it across all tracks. `prefetchw` was tried on tracks C and D and removed in the same push as the first two fixes, so its own cost was not isolated.

| Step | Run ID | T (s) | M (MB) | S | K | Points |
|---|---|---|---|---|---|---|
| Before | run-bdcf951e9670 | 0.0243 | 9.6 | 394 | 41.59 | 80.95 |
| With prefetchw, mask table in writable memory | run-f3c50fedccfc | 0.0263 | 10.5 | 390 | 39.35 | 79.74 |
| Table on its own lines, no prefetchw | run-cfa65c3d21bb | 0.0248 | 9.9 | 390 | 41.03 | 80.65 |
| Tuned for the core | run-cdbde97725e8 | 0.0246 | 9.9 | 390 | 41.21 | 80.75 |
| Same code, later run (best so far) | run-3e3bdb5d3832 | 0.0238 | 9.3 | 390 | 42.33 | 81.33 |
| Same code, later run | run-b8a1c133b819 | 0.0245 | 10.2 | 390 | 41.02 | 80.65 |

The last rows show run-to-run noise of about 3% in K on unchanged code.

## Instruction throughput (2026-11-01)

Plain T was too noisy for small changes, so these decisions use C/B: Track C's T divided by Track B's in the same push. Track B did not change, so this cancels the box's load. On unchanged code it was 0.313 to 0.317.

**Measure first.** Diagnostic builds report through T and M: the output stays byte-exact, then the program touches memory and sleeps so that M and T encode its counters. They score about zero and cannot become a best run.

| Run ID | Measured on the box |
|---|---|
| run-ebc8576769e1 | Thread cycles per record: scan 56, decode 72, group 38, bit write 32. Total 198, matching T |
| run-7e3069987c61 | About 4,000 page faults and 60 ms of kernel time out of about 260 ms of thread time. Setup 1.0 ms, after parsing 1.4 ms, exec and exit 0.8 ms |
| run-bc1f3109f6f0 | After the changes below: scan 54, decode 50, group 38, bit write 28. Total 170 |

My guess was the locked bitset write, since 89.8% of records set a new bit. It was 16% of the time. Scan and decode were 64%.

**The kernel's share did not lead anywhere.** It looked like the largest target, but direct attacks on it failed. Copying costs about what faulting costs, so removing faults alone wins nothing.

| Change | Run ID | T (s) | C/B | Outcome |
|---|---|---|---|---|
| `MADV_POPULATE_READ` per chunk | run-f4c28358d0cf | 0.0263 | 0.350 | Rejected |
| `pread` of 64 KB with 1 KB overlap | run-37035719b347 | 0.0254 | 0.320 | Rejected, and 11.0 MB |
| 1 MB chunks, fewer threads per page table | run-bfd94851ee0f | 0.0245 | 0.326 | Rejected |
| Pin each worker to a CPU | run-1f229baacca1, run-6c1185b0f876 | 0.0234, 0.0237 | 0.309, 0.316 | Rejected. About 1%, inside the noise, for 8 lines. My first reading of plain T claimed 4% |
| Count distinct names when setting a bit, unconditional `fetch_or` | run-f3eb8e90d8d9 | 0.0243 | 0.325 | Rejected |
| The same, with the read test kept | run-d0fb2d56e6dd | 0.0245 | 0.325 | Rejected |

**The cause.** Running the decode loop twice per batch and timing the second run took 69 ticks warm against 93 cold, on one local core, so only a quarter was memory. The disassembly had about 150 instructions with heavy stack spilling and no hidden branches. Two hyperthreads share a 6-wide core, so each gets about 3 micro-ops a cycle, and 150 instructions take about 50 cycles. The loop was limited by how many instructions it had.

| Step | Change | Run ID | T (s) | M (MB) | S | K | C/B | Outcome |
|---|---|---|---|---|---|---|---|---|
| 15 | Decode rewritten: lengths from one vector subtract, dot removed by shuffle so cents decode directly, BOM and sign by branch, no length in the hash, sums widened in the next pass | run-36eeb18fe4c3 | 0.0219 | 9.5 | 405 | 43.79 | 0.290 | Kept. Local decode 85 to 48 ticks |
| 16 | Min and max by predicted branch, as a new extreme is rare | run-cbcac110af1f | 0.0216 | 9.6 | 407 | 43.90 | 0.289 | Kept |
| 17 | Pair counts in straight lines. gcc had kept the "unconditional" loop, at 14 instructions per pass | run-797ad36975f6 | 0.0213 | 9.6 | 411 | 44.16 | 0.283 | Kept |
| 18 | Region table sized by spec 2.3, output order from the country code, no sort or strings, detached threads | run-93587ef0f4f2 | 0.0214 | 10.3 | 395 | 43.71 | 0.283 | Kept. M rose about 1 MB here and the cause is not known |
| 19 | No huge pages once the tables are faulted, table pointers in registers | run-d44bc75eaac2 | 0.0201 | 10.5 | 395 | 44.91 | 0.267 | Kept, the largest step after 15. Attributed to workers no longer clearing a 2 MB stack page, not measured separately |
| 20 | Six unconditional marks per block, not eight | run-a53889219e88 | 0.0208 | 10.6 | 395 | 44.05 | 0.277 | Rejected. Mispredictions cost more than the 8 instructions |
| 21 | Fixed moves between batches, not three library calls | run-e8c3608f3bd4 | 0.0201 | 10.6 | 401 | 44.78 | 0.254 | Kept, though Track B was unusually slow in this push, so its C/B flatters it |
| 22 | `-mprfchw`, so the bitset prefetch fetches the line writable | run-fdb4a0f94bf8, run-51f14a1529b6 | 0.0196, 0.0197 | 10.6, 10.7 | 401 | 45.30, 45.11 | 0.260, 0.264 | Kept |
| 23 | Line trim | run-10ff06d92ce1 | 0.0196 | 10.5 | 391 | 45.60 | 0.256 | Kept, best so far |
| 24 | Prefetch input 512 bytes ahead, retest | run-5cac8a1a30e0 | 0.0206 | 10.4 | 392 | 44.54 | 0.271 | Rejected |
| 25 | No prefetch of the country table | run-ce93221e9697 | 0.0202 | 10.6 | 390 | 44.79 | 0.261 | Rejected |
| 26 | `join` instead of detach, testing step 18's memory rise | run-c5d30816fae8 | 0.0203 | 11.9 | 393 | 43.63 | 0.270 | Rejected. My explanation was wrong |
| 27 | Batches of 8 blocks | run-2dbd013a3a56 | 0.0201 | 10.4 | 391 | 45.05 | 0.268 | Rejected |

**Correctness.** The hash gained a third AES round after the last input, as two rounds left crafted names colliding. Decode now relies on more of spec section 2.2, so the final code was also compared with the official reference on seeds the code had never seen: sample seeds 7 and 11 and full seed 7, 15 runs each, plus 25 runs of the worked example, all with 12 threads forced. Every run matched.

**Still unexplained.** Scan stayed at 54 to 56 cycles on the box after losing instructions, so it looks limited by memory or faults, but input prefetch made it worse. Local memory at 12 threads is 8.4 MB against 10.5 MB on the box. Per-CPU charge batching in the kernel's memory accounting would fit, but this was not measured.

The board's 9.08 MB and 0.0236 s (run-ed8076394be0, K 42.79) became 10.5 MB and 0.0196 s (run-10ff06d92ce1, K 45.60).
