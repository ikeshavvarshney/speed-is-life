# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

C++20, standard library only, built by clang 17 from the allowlisted `swift:6.2` image with a profile trained during the build on generator output (seed 2; the generator is copied in byte for byte). One thread per available CPU. One pool of threads goes through the steps below, with a barrier between steps, and any serial work runs on one thread at the barrier.

**Queries first.** Queries are read on all threads, and only the terms they use are indexed, which the spec allows. Terms go into a lock-free draft table, which also counts how often each is used. Each boolean query is kept as term ids and operators, so its text is never read again.

**Plan.** How often a term occurs in the queries decides how it is stored. Phrases are windows of documents and boolean terms are drawn by band, so the queries are themselves a sample of the corpus, and no document needs sampling. Ids are given out in one short serial pass. Then, on all threads again, the rarest term of each phrase becomes its anchor and the final lookup table is filled, 8 bytes per slot, holding part of the term's hash, its id and two flags. It is sparse, so nearly every term sits in its home slot, and it sits on one huge page. The hash stands in for the bytes: with about 600,000 distinct tokens and 76,000 terms, the chance of any collision is about 1 in 100 million.

**Documents.** Each thread claims a range of the mapped file, and owns the blocks of 512 documents that start in it. A document takes four passes, each a function of its own so its loop keeps its constants in registers:

1. AVX2 classifies 64 bytes at once. The bits where tokens start and end go a byte at a time through a table of bit positions, with no branch, as a branch per token would be mispredicted once per window. Each token is then hashed where it lies with three AES rounds, in one long loop, and prefetched in the table.
2. A vector gather looks up four tokens at a time. The hit or miss is unpredictable, so no branch depends on it.
3. A vector step sorts the resulting ids into common terms and list terms, as a token's kind is unpredictable too. For a list term it also makes the whole log entry, with its partition above it.
4. Each kind then runs a tight loop. Common terms set a bit in a private cache line per term, written to the shared bitset once per 512 documents. List terms append a 16-bit entry to their partition's log: a term within a range of 512, and a document within a group of 64.

**Phrases are matched live.** All phrases are known before any document is read. Any document matching a phrase must contain its anchor, so nothing happens until an anchor token appears. Nearly all anchors match nothing, so a small bitset of (term, neighbour) pairs rejects them first, and the rest compare against the document's hashes. No positions are stored, no documents are re-read, and terms used only in phrases need no lists at all.

**Lists.** All hits of a partition lie in one sparse region, in small blocks claimed by atomic counter. One thread builds each partition: it visits runs of hits by group, so lists come out sorted without sorting, writes them to a private image and copies that over the log. Log and lists never both take memory, and nothing is mapped or unmapped per list. Lists hold 16-bit ids within blocks of 65,536 documents. A wrong guess about which terms are common only costs space, as both forms hold the exact answer.

**Answers.** Filters between two ORs commute, so a run starts from its smallest list, with NOT last, and the result is the same as strictly left to right, as the spec requires. Lists of similar sizes compare 8 ids against 8 in one vector step, and a list over 4 times longer than the ids looked up in it is searched where it lies by 16-bit vectors, never decoded. The last OR or NOT needs no result list: the ids shared with the last list give the result's size and sum, and only its first ten ids are built. Unions over 1,024 ids move to a bitset, whose count and sum come from an AVX2 positional popcount.

**Memory and the kernel.** The box has huge pages set to `always`, which charges 2 MB for a first touch of a sparse region or a thread stack, so they are off for the process except to fault the lookup table. All threads clear the input's page tables after parsing and give back all memory before the output is written, as exit would otherwise do that on one thread with the clock running. Query reading does no malloc, and every step reuses the same per-thread pages. Everything a thread writes often, such as a chunk's output, sits on a cache line of its own: sharing one with a neighbouring chunk on another thread cost 4% of the run.

## Dependencies

None. `mimalloc` was tried and rejected: see `PERF.md`.

## With another 24 hours

Bring memory under the 44 MB where it stops paying, which mostly means shrinking the hit log. Read each chunk of the file as one stream rather than document by document, which would remove about three mispredicted loop exits per document. Run every candidate at least three times on the board: single runs of identical code vary by up to 9%.
