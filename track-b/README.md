# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

Rust, standard library only, one thread per available CPU.

Queries are read first, and only the terms they use are indexed, which the spec allows. Documents are tokenized in parallel by classifying bytes with AVX2. Each token becomes a zero-padded 32-byte key, the longest a term can be, whose 64-bit hash is looked up in one table holding the term's id and role. A lookup therefore touches one cache line. The hash stands in for the bytes: with about 600,000 distinct tokens and 76,000 terms, the chance of any collision is about 2 in a billion.

A sample of the first 1% of documents picks the common terms. Those get a bitset, gathered 64 documents at a time in a small local buffer, and the others get a sorted list of document ids. A wrong guess only costs space, because both forms hold the exact answer.

Boolean queries run strictly left to right as the spec requires, using whichever operation fits each pair of forms.

Phrases are matched live during tokenizing, as all of them are known in advance. Any document matching a phrase must contain the phrase's rarest term, so nothing happens until that anchor token appears. The phrase is then checked against the document's latest tokens once its remaining tokens have been read. No positions are stored, no documents are re-read, and terms used only in phrases need no lists at all.

## Dependencies

None. `mimalloc` was tried and rejected: see `PERF.md`.

## With another 24 hours

Tokenizing and term lookup dominate the run. Shrink the temporary list of hits, which sets peak memory.
