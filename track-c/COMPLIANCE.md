# Track C: compliance

Checked on 2026-11-01 against the frozen `specs/c-join.md`, `harness/` and the site's full context file (speeddemon.dev/speed-demon-context.md). The harness is the referee, and its code was read for how T, M and S are measured.

## A note on sources

Neither the context file nor the board page has a list called a blacklist of techniques. The rules below are the prohibitions the official sources do state. If a separate list has been announced elsewhere, such as on Discord, it was not available for this check.

## Rules and how this entry meets them

| Rule (source) | This entry |
|---|---|
| Output byte-identical to the reference (spec, correctness gate) | Compared by SHA-256 on 8 local datasets and on seeds the code had never seen |
| Banned: dataframes, SQL or query engines, libraries that do the join or group-by (spec section 6) | No libraries. The join is an array lookup and the aggregation is hand-written |
| Dependencies priced on D (site rule 02) | None, D = 1 |
| No precomputed answers, no recognising the input (site rule 04) | Nothing in the image comes from a dataset. The build has no data step. No code tests for a seed or a hash |
| Solution code written in the 72-hour window (site rule 05) | All commits fall inside the window |
| No network, read-only `/data` and root, one file in `/out` (sandbox gate) | Opens two input files read-only and writes `/out/aggregates.txt`. All board runs passed the sandbox gate |
| No tampering with the dataset or mount, no attacking the harness (site rules) | `/data` is only mapped read-only. T and M are not touched: the program does its work and exits |
| Dockerfile rules (allowlisted images pinned by digest, no refused instructions) | `gcc:16` and `debian:13-slim`, both pinned. Passed the board's check on every push |
| No more than 12 workers (run contract) | Thread count is the available CPUs, capped at 12 |
| `io_uring` is blocked by seccomp (run contract) | Not used. `clone`, `mmap`, `madvise` are ordinary calls that worked on the box |

## Reliance on the spec

Spec section 2.2 says "you may rely on both" of its parsing sections, and these hold for every seed. The code relies on these points, each noted beside the code that uses it:

| Fact | Used for |
|---|---|
| A newline inside a note is followed by a lowercase word or a quote | Finding a record start from any chunk offset |
| `id`, `region_id` and `amount` fit eight characters, and amounts have the form `-?d+.dd` | Decoding numbers by fixed vector steps |
| Region ids stay below 1,000,000, with at most 240 countries | Sizing the region table and the country arrays |
| The name pool holds about 100,000 values | Sizing the name table at 131,072 slots. This one is stated as a planning figure, not a guarantee. The generator is fixed, so it holds at every seed |

## Known risks, stated plainly

- Names are compared by a 64-bit hash, not by bytes. A collision would merge two names and change a count, at a chance of about 3e-10 for 100,000 names. This is not zero.
- The board's M varies between runs on the final version (7.9 and 8.1 on the two samples), so the final score's worst-of-five M may be higher than the best sample.

## Measurement builds

Diagnostic builds reported timings through T and M, which scored about zero and cannot become a best run. One measurement (run-924c80cb7770) removed a lock on purpose and failed the hash, to bound the lock's cost. It was restored at once. None of these is the recorded commit, and none alters what the final code does.
