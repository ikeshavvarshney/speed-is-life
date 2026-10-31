# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

C++20 with gcc 16, standard library only, statically linked, one thread per available CPU. Threads are started once and do both parsing and output. The first versions were Rust; the same algorithms moved to C++ for 21% fewer scored lines at about the same K.

**Parsing.** The input is mapped with `mmap` and workers claim 256 KB ranges from an atomic counter. A line belongs to the range its first byte falls in, so no line is parsed twice or skipped. Nearly every line has the hour of the one before it, so the hour is only worked out when eight bytes of the timestamp change. Two AVX2 loads give a mask of the spaces in the next 64 bytes, which finds the method and the endpoint with no branch on their length. The latency is decoded with SWAR, so there is no branch per digit either.

**Names.** The whole endpoint is hashed at a fixed cost with AES rounds. The hash is looked up in one shared table with eight ways per bucket, compared at once by a vector load. After the first few lines the table is only read, so sharing it costs nothing and no merging is needed. Ids are given out in first-seen order, and popular names come early, which packs their counters into few cache lines.

**Storage.** A line becomes one 4-byte entry: 11 bits of id, an error bit and 20 bits of latency. Latencies that do not fit, about 1% of lines, go to a side list. Each hour has its own sparse region, and threads take 256-entry blocks from it with one atomic, so appending shares nothing. Unused parts of blocks read as zero, which is id 0, a group that is never printed.

**Answering.** Hours are independent tasks. Entries are counted per name, then placed so each name's values are contiguous. Percentiles come from an AVX2 selection: a partition that moves lanes with a permutation built by `pdep` and `pext`, down to eight values, which a 19-comparator sorting network sorts in one register. Numbers are written with no branch on their length, by computing all eight digits at once and shifting off the leading zeros.

**Memory and the kernel.** The output is on tmpfs on the scoring box, so it counts toward peak memory while it grows. To keep the peak down, an hour's log is reused for the next hour's values once read, and the region used before it goes back to the system. Hours are written in order while other hours are still being answered, by whichever thread finds the next hour ready. When parsing ends, all threads clear the input's page tables in pieces, as exit would otherwise do that page by page on one thread with the clock still running. The thread that writes the last hour ends the process.

The spec guarantees the 72-hour span, well-formed lines and exactly 1,842 endpoints. The hour mapping and the 11-bit ids rely on that, and the program stops with an error if more than 2,047 names appear. Names are identified by 40 bits of hash, with odds of about 2e-6 of a collision over 1,842 names. This was tested on the catalogues of 300 seeds and on 3.1 million pairs of nearly identical names.

## Dependencies

None.

## With another 24 hours

Measure thread count and `pread` on the scoring box, parse several streams per thread, and trim the source.
