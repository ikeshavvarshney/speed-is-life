# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

Rust, standard library only, one thread per available CPU.

Queries are read first, and only the terms they use are indexed, which the spec allows. Each term becomes a zero-padded 32-byte key, the longest a term can be. Documents are tokenized in parallel by classifying bytes with AVX2, so each token also becomes one 32-byte key that is hashed and compared in fixed time.

A sample of the first 1% of documents picks the common terms. Those get a bitset, and the others get a sorted list of document ids. A wrong guess only costs space, because both forms hold the exact answer.

Boolean queries run strictly left to right as the spec requires, using whichever operation fits each pair of forms.

No positions are stored. A phrase intersects its terms from rarest to most common, and then re-reads only the candidate documents, comparing tokens with the phrase's own terms. Nearly every phrase contains a rare term, so about 11 candidates per phrase remain on average.

## Dependencies

None.

## With another 24 hours

Tokenizing and term lookup dominate the run, so prefetch the term table. Reorder runs of AND by size, and answer queries from the source document's chunk where possible.
