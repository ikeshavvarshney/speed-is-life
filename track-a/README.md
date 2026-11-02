# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

C++20, standard library only, statically linked, built by clang 17 from the allowlisted `swift:6.2` image.

**The committed profile.** `p.profdata` was trained on full-scale data from the published generator (seed 2). Clang names a static function in a profile by its source path, and the training compiled a different path from the build. So the profile only covers `main` and the library's sort routines, and the hot functions are built without one. This was found late and kept on purpose. Profiles that did cover the hot functions measured slower, or varied from build to build by up to 15% in run time, while this build is the same binary every time (see `PERF.md`). Twelve threads, one per hardware thread of the box, are started once and both parse and answer. The first versions were Rust, and the same algorithms moved to C++ for 21% fewer scored lines at about the same K.

**Parsing.** The input is mapped once with `mmap`, and workers claim 256 KB ranges from an atomic counter. A line belongs to the range its first byte falls in, so no line is parsed twice or skipped. Each range is read by two walks that take turns, so a core works on two independent lines at once. Nearly every line has the hour of the one before it, so the hour is only worked out when eight bytes of the timestamp change. Two AVX2 loads give a mask of the spaces in the next 64 bytes, which finds the method and the endpoint with no branch on their length. The latency's digits are masked by one table load and merged in pairs, fours and eights with vector multiply-adds, with no branch on their count.

**Names.** The whole endpoint is hashed at a fixed cost with AES rounds, three after the last byte so that every byte reaches every bit kept. The hash is looked up in one shared table of 8,192 words, each holding 48 bits of the hash and the name's id. Together with the slot that is 60 bits, which over 1,842 names collide with odds of about 2e-12. After the first few thousand lines the table is only read, so sharing it costs nothing and nothing needs merging. Ids are given out in first-seen order.

**Storage.** A line becomes one 4-byte entry: 11 bits of id, an error bit and 20 bits of latency. A latency that does not fit, on about 1% of lines, takes the next slot whole. Each hour has its own sparse region, and threads take 256-entry blocks from it with one atomic, so appending shares nothing.

**Answering.** The file is the shards' chunks in arrival order, so each hour is answered as soon as parsing is far enough past it, which keeps memory low. The spec promises no bound on lateness, so this is checked: a line older than an hour already taken restarts the program, which then holds every hour until the end. An hour's entries are counted per name, then placed so each name's values are contiguous. Percentiles come from an AVX2 selection: a partition that moves lanes with a permutation built by `pdep` and `pext`, down to eight values, which a 19-comparator sorting network sorts in one register. Numbers are written with no branch on their length, by computing all eight digits at once and shifting off the leading zeros. The main thread alone writes the hours, in order, between chunks of parsing.

**Memory and the kernel.**

- The output is on tmpfs on the box, so it counts toward peak memory while it grows. An hour's log is reused once read, and from hour 60 a region given back is also returned to the system.
- The main thread makes the input's page tables far ahead of the parsers, and faults the threads' stacks, which are our own memory. So workers never wait on a page table, and a finished thread needs no unmapping.
- When parsing ends, all threads clear the input's mapping in 8 MB pieces, as exit would otherwise unmap it page by page on one thread with the clock still running.
- The thread that writes the last hour ends the process.

**Limits.** The spec guarantees the 72-hour span, well-formed lines and exactly 1,842 endpoints. The hour mapping and the 11-bit ids rely on that: more than 2,047 distinct names would not fit, and this is not checked at run time.

## Dependencies

None.

## With another 24 hours

- Render one hour on two threads, splitting the count and place passes. The last hour's render is a serial tail of about 0.9 ms, and fewer hours in flight would also lower the memory peak.
- Hold less than a whole hour's values per render, to approach the leader's 14.8 MB.
- Measure raw `clone` again with futex waits instead of spinning, to see whether the clone itself helps.
