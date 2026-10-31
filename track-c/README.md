# Track C: Join

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/c-join.md).

## Approach

Rust, standard library only, one thread per available CPU.

The input is mapped with `mmap` and workers claim 256 KB ranges. Splitting relies on spec section 2.2: a newline inside a note is always followed by a lowercase word or a quote, so a newline followed by a digit or a BOM is always a record boundary. No quote-parity pass is needed.

Fields are parsed in place with no copying or unescaping. Quoted fields are scanned with AVX2, and each `""` pair found reduces the value length by one, which is all the byte totals need. Numbers use SWAR. Money stays in integer cents throughout.

The join is a flat `u8` array indexed by `region_id`. Each worker keeps its own running totals per country, merged at the end.

`names_distinct` is the interesting part. A hash set per country would hold about three million entries. Instead, each name's 64-bit hash goes into one shared lock-free table, and its slot number becomes a small id. Each country then needs only a bitset over those ids, about 4 MB in total. A given name value always has the same raw spelling, so hashing the raw bytes is safe. A hash collision would merge two names, at a chance of about 3e-10 for 100,000 names. If a future pool outgrew the table, the run restarts with a larger one.

Those two tables are the main cost, as both are random memory accesses on nearly every record. Each access is prefetched and then performed 16 records later, so the misses overlap instead of stalling the parser, and the tables use huge pages to reduce TLB misses.

## Dependencies

None.

## With another 24 hours

Tune the prefetch distance, chunk size and thread count on the scoring box, and try profile-guided optimisation.
