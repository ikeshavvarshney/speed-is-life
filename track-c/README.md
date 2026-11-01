# Track C: Join

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/c-join.md).

## Approach

C++20 with gcc 16, standard library only, one thread per available CPU. The first version was Rust. The same algorithm in C++ ran at the same speed in 28% fewer scored lines, so it moved.

The input is mapped with `mmap` and workers claim 256 KB ranges. Splitting relies on spec section 2.2: a newline inside a note is always followed by a lowercase word or a quote, so a newline followed by a digit or a BOM is always a record boundary. No quote-parity pass over the file is needed.

Each 64-byte block becomes bitmasks with AVX2. A carry-less multiply of the quote bits by all ones gives a running XOR, which marks what lies inside quotes, as in simdjson. Commas and LFs outside quotes become a flat list of positions, and every record is exactly five of them. Field lengths are then known without a branch on their contents.

Numbers are read backwards from their known ends. Three numbers are converted together in one vector, and a shuffle removes the amount's dot so that it decodes straight to cents. Money stays in integer cents throughout. The decode loop is limited by instruction throughput, as two hyperthreads share each core, so it avoids work rather than branches where the branch is predictable: BOMs and signs take a rare branch, and one vector subtraction of neighbouring marks gives every field length.

A `""` pair is an opening quote directly after a closing quote, because fields always have a comma between them. Each pair reduces its field's value length by one, which is all the byte totals need, so nothing is unescaped or copied.

The join is a flat `u8` array indexed by `region_id`. Each worker keeps its own totals per country on one cache line, added to as one vector, and they are merged at the end.

`names_distinct` is the interesting part. A hash set per country would hold about three million entries. Instead, each name's 64-bit hash goes into one shared lock-free table, whose buckets are one cache line of eight hashes compared at once. The slot number becomes a small id, and each country needs only a bitset over those ids, about 4 MB in total. A given name value always has the same raw spelling, so hashing raw bytes is safe. A collision would merge two names, at a chance of about 3e-10 for 100,000 names.

The name table, region table and bitsets are random accesses on nearly every record. Group prefetching makes their misses overlap: one pass parses a batch and prefetches, the next looks up, and the last writes the bits.

Output order comes from indexing countries by their base-26 code, so nothing is sorted. Huge pages are disabled once the tables are faulted, so a worker's first touch of its stack costs 4 KB, not a cleared 2 MB page.

Setup and teardown are parallel as well. Workers load `regions.csv` together and clear the input's page tables together, as process exit would otherwise do that on one thread while the clock runs.

## Dependencies

None.

## With another 24 hours

Find out why scan stays at 54 cycles per record on the box after losing instructions, and why memory rose about 1 MB at the commit that simplified setup. Try a profile-guided build again, now that the loops have less spilling. Trim output formatting and setup code.
