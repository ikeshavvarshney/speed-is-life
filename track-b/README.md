# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

C++20 with gcc 16, standard library only, one thread per available CPU. The first versions were Rust; the same algorithms moved to C++ for 30% fewer scored lines and a better K.

Queries are read first, on all threads, and only the terms they use are indexed, which the spec allows. Terms go into one table of 8-byte slots holding part of the term's hash, its id and an anchor flag, about 1 MB in all. The hash stands in for the bytes: with about 600,000 distinct tokens and 76,000 terms, the chance of any collision is about 1 in 10 million.

Each document takes two passes. The first tokenizes: AVX2 nibble lookups classify 64 bytes at once, tokens come straight from the resulting bit masks, and each is hashed where it lies. The second walks those hashes doing lookups, prefetching 12 tokens ahead so that memory arrives in time. A mispredicted branch on a load that is still missing cache costs about 100 ns, so this split matters more than any single instruction count. Token actions are branch-free, as a token's kind is unpredictable.

A small sample of documents picks the common terms. Those get a bitset, gathered 64 documents at a time in a local buffer. The others go to a temporary log of 16 bits per hit: a term within a range of 512, and a document within a group of 64. Each range of terms is then built by one thread, visiting runs of hits by group, so lists come out sorted without sorting. Lists hold 16-bit ids in blocks of 65,536 documents. A wrong guess by the sample only costs space, because both forms hold the exact answer.

Boolean queries are not stored. They are read again from the query text when answered, in parallel, strictly left to right as the spec requires.

Phrases are matched live during tokenizing, as all of them are known in advance. Any document matching a phrase must contain the phrase's rarest term, so nothing happens until that anchor token appears. Nearly all anchors match nothing, so a small bitset of (term, neighbour) pairs rejects them first, and the rest compare against the document's hashes. No positions are stored, no documents are re-read, and terms used only in phrases need no lists at all.

## Dependencies

None. `mimalloc` was tried and rejected: see `PERF.md`.

## With another 24 hours

Search lists directly when answering instead of decoding them whole, and simplify the code, which is longer than the leaders'.
