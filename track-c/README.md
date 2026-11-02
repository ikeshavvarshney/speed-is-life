# Track C: Join

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/c-join.md).

## Approach

C++20 with gcc 16, standard library only, one thread per available CPU up to 12. The first version was Rust. The same algorithm in C++ ran at the same speed in 28% fewer scored lines, so it moved.

The input is mapped with `mmap` and workers claim 256 KB ranges. Splitting relies on spec section 2.2: a newline inside a note is always followed by a lowercase word or a quote, so a newline followed by a digit or a BOM is always a record boundary. No quote-parity pass over the file is needed.

Each 64-byte block becomes bitmasks with AVX2. A carry-less multiply of the quote bits by all ones gives a running XOR, which marks what lies inside quotes, as in simdjson. Commas and LFs outside quotes become a flat list of positions, and every record is exactly five of them. Field lengths are then known without a branch on their contents.

The hardware prefetcher stops at each 4 KB page, so the first lines of every page would be demand misses to memory. Just before crossing a page, the loop prefetches six lines of the next one. This helped, and 12 lines cost 20%, so the number is not arbitrary.

Numbers are read backwards from their known ends. Three numbers are converted together in one vector, and a shuffle removes the amount's dot so that it decodes straight to cents. Money stays in integer cents throughout. The decode loop is limited by instruction throughput, as two hyperthreads share each core, so it avoids work rather than branches where the branch is predictable: BOMs and signs take a rare branch, and one vector subtraction of neighbouring marks gives every field length.

A `""` pair is an opening quote directly after a closing quote, because fields always have a comma between them. Each pair reduces its field's value length by one, which is all the byte totals need, so nothing is unescaped or copied.

The join is a flat `u8` array indexed by `region_id`. Each worker keeps its own totals per country on one cache line, added to as one vector, and they are merged at the end.

`names_distinct` is the interesting part. A hash set per country would hold about three million entries. Instead, each name's 64-bit hash goes into one shared lock-free table, whose buckets are one cache line of eight hashes compared at once. The slot number becomes a small id, and each country needs only a bitset over those ids, about 4 MB in total. A given name value always has the same raw spelling, so hashing raw bytes is safe. A collision would merge two names, at a chance of about 3e-10 for 100,000 names.

The name table, region table and bitsets are random accesses on nearly every record. Group prefetching makes their misses overlap: one pass parses a batch and prefetches, the next looks up, and the last writes the bits.

## Memory

All writable memory is one mapping at a fixed address, so its addresses are constants in instructions. Its first 6 MB take three huge pages, which hold the tables and bitsets, and what those leave free holds each worker's stack and totals, at no extra cost.

The first memory charge on a CPU reserves 256 KB for it, and peak memory counts the reserve. So the main thread faults what workers would otherwise be charged for: the huge pages, the small pages they write, and page tables. Thread 0 keeps making page tables ahead of the others while they parse.

## Setup and teardown

Threads are raw `clone` calls on preset stacks, which avoids mapping and protecting a stack per thread, and unmapping it on exit. They wait by spinning, as waits last microseconds. A worker may call only what keeps no thread state in libc.

Work is split into jobs with a claim counter and a done counter: regions, chunks, then countries. The last job merges totals, counts bits and formats a line on whichever thread claims it, so nothing runs serially until the final write. Output order comes from the base-26 country code, so nothing is sorted. The same job clears the input's page tables, as process exit would otherwise do that on one thread while the clock runs.

## Dependencies

None.

## With another 24 hours

Removing the bitset's locked instruction gained 8% in a test that lost updates, and three correct designs did not reach it, so a fourth is the largest known target. Add the page table fix that `PERF.md` describes, which makes peak memory steadier, and sample it enough to compare with the final version. Find out why a page crossing burst of 12 lines costs 20% when 4 and 6 are level.
