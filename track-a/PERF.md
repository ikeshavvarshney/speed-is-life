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

## Port to C++

The first port looked level with Rust on the 2-core laptop, but was slower on the 12-thread box. Three build-level fixes followed, the same on all tracks: writable globals moved off shared cache lines, `u64` given a different type from `size_t` so writes to it do not force sizes and masks to be reloaded, and `-mtune=alderlake` to match the box's P-cores. Tuning for the core appeared to be the largest of these, in the one push that changed it across all tracks. `prefetchw` was tried on tracks C and D and removed in the same push as the first two fixes, so its own cost was not isolated.

| Step | Run ID | T (s) | M (MB) | S | K | Points |
|---|---|---|---|---|---|---|
| Last Rust best | run-bac4a108f903 | 0.0176 | 23.1 | 600 | 47.29 | 83.74 |
| First port | run-98b8e52b93fd | 0.0231 | 23.5 | 474 | 42.57 | 81.46 |
| Globals separated, distinct type | run-cbf55bf8bb63 | 0.0225 | 23.5 | 475 | 43.08 | 81.72 |
| Tuned for the core | run-06a97ad2fd18 | 0.0196 | 23.8 | 475 | 46.12 | 83.19 |
| Same code, later run | run-a2b887667b80 | 0.0193 | 23.0 | 475 | 46.84 | 83.53 |

Rust scored K 45.4 to 47.3 over recent runs, so C++ is level on K rather than ahead. T is about 10% worse than the best Rust run, and 21% fewer lines make up for it. The cause of the remaining time difference was not found.

## C++ steps after the port (2026-11-01)

Each kept step, in order. "A/B" is Track A's time over Track B's in the same push: B's code did not change, so the ratio separates a change from the box's run-to-run swings of up to 8%.

| Step | Change | Run ID | T (s) | M (MB) | S | K |
|---|---|---|---|---|---|---|
| 3 | Render each hour as soon as it is over, restart on a late line, regions reused | run-0c0c1f149f2b | 0.0200 | 15.0 | 459 | 50.31 |
| 4 | Two walks per chunk, so a core works on two lines at once (one walk retested: 0.0167 against 0.0162) | run-c2e90f05568c | 0.0196 | 15.7 | 481 | 50.01 |
| 5 | Same code built by clang 17 from the allowlisted `swift:6.2` image with PGO: A/B 0.262 to 0.245, the largest build change | run-8ce8725b186a | 0.0185 | 15.1 | 485 | 51.76 |
| 6 | Input prefetched into the first cache level | run-9f00557627dd | 0.0181 | 15.6 | 486 | 52.12 |
| 7 | Free regions trimmed only over the last 12 hours: fewer faults, more memory | run-c49ca13a0e9e | 0.0174 | 17.2 | 486 | 52.06 |
| 8 | One shared `mmap` of the input, cleared in parallel at the end, prefetch 1024 bytes ahead: A/B 0.231 to 0.215 | run-78f3f98ed06f | 0.0162 | 16.9 | 493 | 54.04 |
| 9 | Compact rewrite: vector latency parse, one permutation per partition, bounded scratch | run-33e4dbcf61ae | 0.0161 | 17.4 | 454 | 54.51 |
| 10 | PGO trained on full-scale generated data | run-e6a411d5724b | 0.0160 | 17.4 | 450 | 54.79 |
| 11 | The main thread alone writes the output, between chunks of parsing | run-5d11c39ef9f9 | 0.0160 | 16.8 | 448 | 55.15 |
| 12 | Line trim with no change in behaviour | run-95dd669ce84c | 0.0162 | 16.8 | 417 | 55.39 |
| 13 | The log prefetched in the count pass, as those reads come from other cores | run-6cb9c06217bc | 0.0158 | 17.3 | 418 | 55.73 |
| 14 | Names in one word, a stall limit of 48 chunks past the frontier, trimming from hour 60 (best single run) | run-8579db581edf | 0.0157 | 17.0 | 408 | 56.42 |

Step 14's code ran unchanged in 49 pushes made for other tracks: T median 0.0161 (0.0157 to 0.0174), M median 17.2 (16.5 to 20.7), A/B median 0.248. Those numbers are the baseline below.

Tried after the port and rejected:

| Change | Run ID | Result |
|---|---|---|
| No position-independent code | run-5bf9464c9fa3 | No change in T. The flag stayed, as it costs nothing |
| `-march=native` | run-08ea6d663463 | A/B 0.219 against 0.210 |
| Context-sensitive PGO | run-4452aece2aea | No change in T, more build steps |
| `munmap` of the input after parsing | run-86cf466a4167 | T 0.0214, only 0.2 MB saved |
| 1 MB chunks | run-7b47901e6784 | T 0.0157 but M 18.6, K 54.53 |
| Free regions keep 256 KB warm | run-28261bb4ca30 | T 0.0158 but M 19.0, K 54.13 |
| Render each hour an hour later | run-591ebd369029 | Same T, M 17.7 |
| Non-temporal prefetch | run-e8b980023d4f | T 0.0171 |
| Prefetch 512 and 2048 bytes ahead | run-120399faccba, run-fbea911ec097 | T 0.0164 both |
| Forked workers, each mapping one chunk | run-a4b88ab43be5 | T 0.0180, M 20.4, S 538 |
| One mapping populated ahead of the parsers | run-3ead9dcb37eb | T 0.0252 |
| Each thread maps the input | run-19905ebfb7db | M up 3 MB |
| `pread` of 64 KB chunks | run-c49ca13a0e9e | T 0.0174, 7% slower than `mmap` under clang |

## Final day (2026-11-02): where the time and memory go

**Measured first, on one laptop core** (full seed 1, TSC ticks per line, best of several runs):

| Phase | Ticks per line |
|---|---|
| Parse, warm input | 50 |
| Parse, one 256 KB chunk repeated in cache | 44 |
| Render: count 10, place 9, select 10, format 5 | 35 |

Parsing is mostly instruction-bound even here: memory adds only 6 ticks. The hot loop is about 83 instructions per line.

**Facts about the data:**

- Hours hold 14,484 to 73,244 lines.
- The last six hours are among the busiest (58,940 to 73,244), so the memory peak falls at the end of the file, when the output is nearly complete.
- All 1,842 names appear within the first 100,000 lines, and only the first two hours have new names.
- 1.2% of latencies need more than 20 bits.

| Step | Change | Run ID | T (s) | A/B | M (MB) | S | K | Verdict |
|---|---|---|---|---|---|---|---|---|
| E0 | LSD radix sort of each hour, in place of count, place and select (40 fewer lines) | lab only | | | | | | Rejected: 43 to 54 ticks per line against 33 to 39. Selection with a vector partition is cheaper than three radix passes |
| E1 | Raw `clone` on preset stacks, spin locks in place of mutexes, `pause` in idle loops | run-0af85f7eb1e9, run-d59fd349b8fa | 0.0414, 0.0203 | 0.623, 0.317 | 16.3, 20.2 | 421 | 34.83, 47.63 | Rejected. With 4 workers on the laptop it was 20% faster, so spinning, not the work, is what hurt at 12 threads |
| | Step 14 restored, as a reference sample | run-0c380088b249 | 0.0162 | 0.251 | 17.5 | 408 | 55.09 | |
| E2 | Workers never charge memory: the main thread faults their stacks, the globals, a pool of 10 regions and every input page table | run-70e804ece3b6, run-03f01acc2761 | 0.0149, 0.0161 | 0.232, 0.242 | 20.1, 23.8 | 439 | 55.35, 51.55 | Rejected for memory. Faster, but regions beyond the pool were faulted anyway and nothing was trimmed until the end |
| F6 | E2 with the text written in the log's region, spare regions trimmed from hour 60, line trims | run-333463c45d55 | 0.0172 | 0.267 | 17.9 | 422 | 53.06 | Rejected |
| F9 | F6 with trimmed regions dropped, the main thread keeping one faulted spare | run-deb83b396eea | 0.0176 | 0.272 | 18.0 | 433 | 52.20 | Rejected |
| G1 | Step 14 plus only the parts of E2 that cost no memory (below), line trims, one-load digit mask | run-12af58a94720, run-95f8666c90d9 | 0.0153, 0.0154 | 0.230, 0.237 | 17.9, 17.0 | 413 | 56.42, 56.70 | Kept. The second run is the best K so far |
| G2 | G1 with the PGO profile trained on the laptop and committed as data, so the Dockerfile has no training stage (4 lines fewer) | run-ecc9403f442a, run-f4a113312595, run-9ef3361a61e9 | 0.0159, 0.0159, 0.0155 | 0.245, 0.240, 0.235 | 17.7, 17.2, 17.6 | 409 | 55.59, 55.86, 56.37 | Rejected. About 2.7% slower than G1 on average, which outweighs 4 lines. G1's profile is trained during the build, so in final scoring it is trained on the box with 12 real threads |
| G3 | G2 with regions trimmed only once every hour has started, as the peak is at the end | run-a2375667e3e7 | 0.0188 | 0.290 | 20.0 | 411 | 49.79 | Rejected. Regions faulted in mid-run bursts stayed charged, and all the releases landed on the final tail |
| G4 | G1 with one name shortened so a line no longer wraps (402 lines) | see push log | | | | 412 | | Final version, same code as G1 |

**What G1 changed, and why:**

1. **Page tables.** The main thread touches one byte of every 2 MB of the input, far ahead of the parsers, so no worker waits to allocate a page table or is charged for one.
2. **Stacks.** Thread stacks are our own memory, faulted by the main thread, so they need no `mmap`, and no unmapping or `madvise` when a thread ends.
3. **Scratch.** Selection's scratch sits right after the hour's values instead of at the top of the region, so it lands in pages already touched.
4. **Digit mask.** The latency's digits are masked by one table load rather than a computed shift, an `andnot` and an `and`.

G1's two runs average A/B 0.2335 against the baseline's 0.248, about 6% less time. The same source with the committed profile (G2) averaged 0.240 over three runs.

**Why memory stays near 17 to 18 MB:**

- The output is 10.1 MB on tmpfs.
- The harness's own processes, the input's page tables (0.47 MB) and the globals add about 2 MB.
- About five hours are being rendered at the end of the file, each holding its values and its log or text, about 0.65 MB per hour.
- Each CPU that charges memory keeps a reserve of up to 256 KB, as the kernel charges in batches of 64 pages per CPU, and `memory.peak` counts the reserve.

E2 removed that reserve, but a pool big enough for the busiest moment costs more than the reserve it saves. The leader's 14.8 MB shows that a lower floor exists. Reaching it here would need a render that holds less than a whole hour's values at once.

**Checks for every pushed version:** all 11 local datasets, including full seed 2 and sample seed 3, matched the reference hashes. Repeated runs (6 to 25 per dataset, at 4 and 2 CPUs with 12 threads) all matched. The final binary also matched the reference on 120 generated hostile logs (480 runs), with lines sorted, reversed, shuffled, with stragglers, late hours and boundary latencies.
