#include <algorithm>
#include <atomic>
#include <barrier>
#include <bit>
#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <malloc.h>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <thread>
#include <unistd.h>
#include <vector>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
// Not size_t's type, so writing one never forces sizes and masks to be reloaded.
using u64 = unsigned long long;
using v4 = __m256i;
using v2 = __m128i;
using std::vector;
constexpr auto relaxed = std::memory_order_relaxed;
#define INLINE [[gnu::always_inline]] inline

const size_t CHUNK = 1 << 19, QUERY_CHUNK = 1 << 14;
const size_t PAGE = 1 << 12, HUGE_PAGE = 1 << 21, RELEASE_STEP = 1 << 22;
// Decoding a list costs about as much per id as a miss in a search of it costs per 32 ids.
const size_t IN_PLACE_RATIO = 4;
// Setting bits costs about half as much per id as a merge.
const size_t DENSE_LIMIT = 1 << 10;
// A term this frequent among the queries is likely in over 1/32 of documents,
// where a bitset is smaller than its list. The queries sample the corpus: phrases
// are windows of documents, and boolean terms are drawn by band.
const u32 HIGH_COUNT = 25;

// A table slot holds the high bits of the term's hash, its id and two flags.
const int ID_BITS = 18, TAG_SHIFT = 20, PAIR_BITS = 20;
const u32 NO_ID = (1 << ID_BITS) - 1, ANCHOR = 1 << ID_BITS;
const int DISPLACED_BIT = ID_BITS + 1;
const u64 DISPLACED = ANCHOR << 1, TAG_MASK = ~0ull << TAG_SHIFT;
const size_t TABLE_PADDING = 1 << 10;
const u32 OP_AND = 0, OP_OR = 1, OP_NOT = 2, OP_BITS = 2;
const u64 SEEN_IN_BOOL = 1ull << 32 | 1;

// A logged hit is 16 bits: a term within its partition, a document within its group
// of 64, and a clear low bit. A worker writes a marker when it starts a group: the
// group's number and a set low bit. So a term's id shifted left by HIT_TERM_SHIFT is
// a hit with its partition just above it, and a vector step can make it. Workers claim
// small blocks of a partition's region, so all hits of a partition lie together with
// no copying as a log grows, and few pages stay partly used.
const int GROUP_BITS = 6, PARTITION_BITS = 9, HIT_TERM_SHIFT = GROUP_BITS + 1;
const u32 GROUP_DOCS = 1 << GROUP_BITS, PARTITION_TERMS = 1 << PARTITION_BITS;
const u16 LOG_END = UINT16_MAX;
const size_t LOG_BLOCK = 1 << 9;

// Common terms gather one cache line per term, 512 documents, before it is written
// to the shared bitset, and each thread owns whole blocks of 512 documents.
const u32 LINE_DOCS = 512, LINE_WORDS = 8;

// Document ids in lists are 16 bits within a block of 65,536 documents.
const int BLOCK_BITS = 16, GROUPS_PER_BLOCK_BITS = BLOCK_BITS - GROUP_BITS;

INLINE v4 load(const void *p) { return _mm256_loadu_si256((const v4 *)p); }
INLINE void store(void *p, v4 x) { _mm256_storeu_si256((v4 *)p, x); }
INLINE v4 ones(int x) { return _mm256_set1_epi8(x); }
INLINE v4 add8(v4 a, v4 b) { return _mm256_add_epi8(a, b); }
INLINE v4 add64(v4 a, v4 b) { return _mm256_add_epi64(a, b); }
INLINE v4 shuffle(v4 a, v4 b) { return _mm256_shuffle_epi8(a, b); }
INLINE u32 lanes(v4 x) { return _mm256_movemask_ps(_mm256_castsi256_ps(x)); }
alignas(64) static u64 lane_orders[256];

// Moves the lanes set in `mask` to the front, in order.
INLINE v4 gather(v4 x, u32 mask) {
  v2 order = _mm_loadl_epi64((const v2 *)(lane_orders + mask));
  return _mm256_permutevar8x32_epi32(x, _mm256_cvtepu8_epi32(order));
}

// Mapped over a larger anonymous mapping so vector loads near the end of the file
// stay readable.
static const u8 *map_input(const char *path, size_t &size) {
  int fd = open(path, O_RDONLY);
  size = lseek(fd, 0, SEEK_END);
  void *base =
      mmap(0, size + PAGE, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  // An empty file fails here, and its reads see the anonymous mapping.
  mmap(base, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
  return (const u8 *)base;
}

// Address space only: pages cost memory once touched, and never need growing.
static u8 *reserve(size_t bytes) {
  return (u8 *)mmap(0, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
}

// For what is used in full and touched all over, for the sake of the TLB. Huge pages
// are off for the process, as a first touch elsewhere would charge 2 MB, so they are
// turned on just to fault these, on one thread so no two threads race for a page.
static u8 *huge_pages(size_t bytes) {
  auto raw = (size_t)reserve(bytes + HUGE_PAGE);
  auto aligned = (u8 *)((raw + HUGE_PAGE - 1) & -HUGE_PAGE);
  madvise(aligned, bytes, MADV_HUGEPAGE);
  prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0);
  for (size_t at = 0; at + HUGE_PAGE <= bytes; at += HUGE_PAGE)
    aligned[at] = 0;
  prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
  return aligned;
}

alignas(64) static u8 low_bytes[64];

INLINE u64 mix_rounds(v2 state) {
  state = _mm_aesenc_si128(state, _mm_set_epi64x(0x2D358DCCAA6C78A5, 0x8BB84B93));
  state = _mm_aesenc_si128(state, _mm_set_epi64x(0x4B33A62ED433D4A3, 0xC2B2AE3D));
  return _mm_cvtsi128_si64(
      _mm_aesenc_si128(state, _mm_set_epi64x(0x9E3779B97F4A7C15, 0xBF58476D)));
}

// Up to 32 bytes are all covered by the first and last 16, and longer tokens differ
// from any term by their length.
[[gnu::cold]] static u64 long_token_hash(const u8 *start, size_t len) {
  v2 head = _mm_loadu_si128((const v2 *)start);
  v2 tail = _mm_loadu_si128((const v2 *)(start + len - 16));
  v2 lower = _mm_set1_epi8(0x20);
  head = _mm_aesenc_si128(_mm_or_si128(head, lower), _mm_or_si128(tail, lower));
  return mix_rounds(_mm_xor_si128(head, _mm_set1_epi64x(len)));
}

// Hashes a token where it lies. Token bytes are letters and digits, so OR-ing 0x20
// lowercases them, and masking removes whatever follows. AES spreads every byte in far
// fewer micro-ops than a wide multiply, but it takes three rounds. After two, half of
// the state still collides for tokens differing only along one of AES's diagonals:
// `wodeand` and `woweandvi` both occur in the seed 2 corpus, and had the same hash. The
// hash stands in for the bytes: with about 600,000 distinct tokens and 76,000 terms,
// the chance of any collision is about 1 in 100 million. `back` is minus the length.
INLINE u64 token_hash(const u8 *start, long back) {
  if (back < -16)
    return long_token_hash(start, -back);
  v2 bytes = _mm_or_si128(_mm_loadu_si128((const v2 *)start), _mm_set1_epi8(0x20));
  return mix_rounds(
      _mm_and_si128(bytes, _mm_loadu_si128((const v2 *)(low_bytes + 32 + back))));
}

// A constant the compiler keeps in a register: left to itself, it rebuilds some of them
// in every iteration of the loop using them, at three micro-ops each.
INLINE v4 held(int byte) {
  v4 x = ones(byte);
  asm("" : "+x"(x));
  return x;
}

// Calls `visit` with each token of the line at `p`, and returns what follows its
// newline. One classification covers 64 bytes, and the bit mask of where tokens start
// and end goes through a table a byte at a time, with no branch. A branch per token
// would be mispredicted once per window, as their number varies, which costs more than
// this does. Tokens are then read in one long loop. `edges` needs room for the line's
// edges, and 8 more. Documents and query lines are both read this way.
template <class F>
INLINE const u8 *scan_line(const u8 *p, u32 *edges, F &&visit) {
  const v4 lower = held(0x20), letter_add = held(127 - 'z'), digit_add = held(127 - '9');
  const v4 letter_top = held(127 - 26), digit_top = held(127 - 10), newline = held('\n');
  // Adding moves each range to the top of the signed bytes, where one compare finds
  // it. OR-ing 0x20 maps only upper case into the range of lower case.
  auto token_mask = [&](v4 bytes) -> u32 {
    v4 letter = add8(_mm256_or_si256(bytes, lower), letter_add);
    v4 digit = add8(bytes, digit_add);
    return _mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpgt_epi8(letter, letter_top),
                                                _mm256_cmpgt_epi8(digit, digit_top)));
  };
  const v4 eight = _mm256_set1_epi32(8);
  u32 *edge = edges;
  u64 newlines = 0, carry = 0;
  size_t at = 0;
  for (;; at += 64) {
    v4 first = load(p + at), second = load(p + at + 32);
    u64 tokens = token_mask(first) | (u64)token_mask(second) << 32;
    v4 low = _mm256_cmpeq_epi8(first, newline);
    v4 high = _mm256_cmpeq_epi8(second, newline);
    v4 any = _mm256_or_si256(low, high);
    // Most windows hold no newline, and those need no bit mask of them.
    if (!_mm256_testz_si256(any, any)) {
      newlines = (u32)_mm256_movemask_epi8(low) |
                 (u64)_mm256_movemask_epi8(high) << 32;
      tokens &= newlines - 1;
    }
    u64 changes = tokens ^ (tokens << 1 | carry);
    carry = tokens >> 63;
    v4 base = _mm256_set1_epi32(at);
    for (int byte = 0; byte < 8; byte++, changes >>= 8) {
      u32 bits = changes & 255;
      v4 order = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const v2 *)(lane_orders + bits)));
      store(edge, _mm256_add_epi32(order, base));
      edge += __builtin_popcount(bits);
      base = _mm256_add_epi32(base, eight);
    }
    if (newlines)
      break;
  }
  for (const u32 *e = edges; e < edge; e += 2) {
    const u8 *start = p + e[0];
    visit(start, token_hash(start, (long)e[0] - e[1]));
  }
  return p + at + _tzcnt_u64(newlines) + 1;
}

INLINE u32 parse_id(const u8 *&p) {
  u32 id = 0;
  for (; *p != '\t'; p++)
    id = id * 10 + *p - '0';
  p++;
  return id;
}

// Counters that every thread writes, away from what the hot loops read.
struct alignas(64) Counter {
  std::atomic<size_t> value{0};
  size_t take(size_t n = 1) { return value.fetch_add(n, relaxed); }
};

// One chunk of the query file, in slices of address space sized by its share of the
// file, as a term takes at least 2 bytes of it. Each line starts with how many boolean
// terms it has, 0 for a phrase, and each term holds its operator in the low bits. It
// has a cache line of its own, as neighbouring chunks are on different threads, and
// every term read and every piece of output written changes it.
struct alignas(64) QueryChunk {
  u32 *lines, *slots, *lengths;
  u64 *hashes;
  size_t line_words, slot_count, phrase_count;
  std::string out;
};

// Terms as they are found in the queries, by all threads at once. Its low half counts
// a term's uses. Above that, it counts uses in boolean queries, and later holds its id.
struct DraftSlot {
  std::atomic<u64> hash, seen;
};

struct Draft {
  DraftSlot *slots;
  size_t mask, len;

  u32 intern(u64 hash, u64 one) {
    for (size_t at = hash & mask;; at++) {
      u64 value = slots[at].hash.load(relaxed);
      if (value == hash ||
          (!value && (slots[at].hash.compare_exchange_strong(value, hash, relaxed) ||
                      value == hash))) {
        slots[at].seen.fetch_add(one, relaxed);
        return at;
      }
    }
  }
};

// The lookup table. Nearly every token needs just one load: at this low load most
// terms sit in their home slot. A home slot with no match answers for all other
// tokens, unless its flag says that a term with this home went elsewhere. The hit or
// miss itself is unpredictable, so no branch depends on it.
struct Table {
  u64 *slots;
  size_t mask;

  // By all threads at once, and any number of times for one term.
  void add(u64 hash, u64 payload) {
    for (size_t at = hash & mask;; at++) {
      u64 value = 0;
      if (__atomic_compare_exchange_n(slots + at, &value, (hash & TAG_MASK) | payload,
                                      false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
        return;
      if (!((value ^ hash) >> TAG_SHIFT)) {
        __atomic_fetch_or(slots + at, payload & ANCHOR, __ATOMIC_RELAXED);
        return;
      }
      if (at == (hash & mask))
        __atomic_fetch_or(slots + at, DISPLACED, __ATOMIC_RELAXED);
    }
  }

  [[gnu::cold]] u32 find_displaced(u64 hash) const {
    for (size_t at = (hash & mask) + 1; slots[at]; at++)
      if (!((slots[at] ^ hash) >> TAG_SHIFT))
        return slots[at] & (NO_ID | ANCHOR);
    return NO_ID;
  }
};

struct Hit {
  u32 phrase, doc;
};

INLINE size_t pair_bit(u64 first, u64 second) {
  return (first + second * 3) >> (64 - PAIR_BITS);
}

// All phrases are known before any document is read, so they are matched live
// during tokenizing. Any document matching a phrase must contain its rarest term,
// its anchor, so nothing happens until an anchor token appears. No positions are
// stored, no documents are re-read, and phrase-only terms need no lists at all.
// Nearly all anchor tokens match nothing, so a small bitset of (term, neighbour)
// pairs rejects them before the phrase tables are touched. Phrases sharing an
// anchor are linked, numbered from 1.
struct Phrase {
  u32 start, next;
  u8 anchor_at, length;
};

struct Phrases {
  vector<Phrase> list;
  vector<u32> first_of_anchor;
  vector<u64> pairs;
  const u64 *terms;

  INLINE bool may_pair(u64 first, u64 second) const {
    size_t bit = pair_bit(first, second);
    return (pairs[bit / 64] >> (bit % 64)) & 1;
  }

  // `tokens` holds the document's hashes, with a 0 before and after them.
  [[gnu::cold]] void check(size_t id, const u64 *tokens, size_t count, size_t token,
                           u32 doc, vector<Hit> &hits) const {
    for (u32 p = first_of_anchor[id]; p; p = list[p].next) {
      const u64 *first = terms + list[p].start;
      size_t len = list[p].length, at = list[p].anchor_at;
      if (token + len - at <= count + 2 && at <= token &&
          std::equal(first, first + len, tokens + token - at))
        hits.push_back({p - 1, doc});
    }
  }
};

// One thread's buffers, used by every step, so pages are only touched once. One
// document can hold at most half its bytes as tokens, and a list at most all
// documents, so these never need to grow.
struct alignas(64) Scratch {
  u64 *tokens;
  u32 *ids, *high_ids, *low_ids;
  size_t highs, lows;
  vector<Hit> hits;
};

static const u8 *text, *input;
static size_t text_size, input_size, docs, words, chunks, partitions, blocks;
static size_t high_terms, low_terms, log_span;
static vector<size_t> bounds;
static vector<QueryChunk> query_chunks;
static u32 *query_lines, *query_slots, *query_lengths;
static u64 *query_hashes;
static vector<u32> phrase_bases, matched, phrase_docs;
static vector<Scratch> scratches;
static Draft draft;
static Table table;
static Phrases phrases;
static u8 *log_regions;
static u64 *high_bits;
static u32 *list_marks;
static Counter *next_log_block;
static Counter next_query, next_plan, next_chunk, next_unmap, next_partition;
static Counter next_answer, next_release;

// Query lines all end in a newline.
static void find_query_chunks() {
  bounds = {0};
  while (size_t at = text_size - bounds.back()) {
    at = bounds.back() + std::min(QUERY_CHUNK, at - 1);
    bounds.push_back((const u8 *)memchr(text + at, '\n', text_size - at) - text + 1);
  }
  chunks = bounds.size() - 1;
}

static void read_chunk(size_t c, u32 *edges) {
  size_t from = bounds[c] / 2;
  QueryChunk &chunk = query_chunks[c];
  chunk = {query_lines + from, query_slots + from, query_lengths + from,
           query_hashes + from};
  for (const u8 *p = text + bounds[c]; p < text + bounds[c + 1];) {
    bool is_bool = *p == 'B';
    size_t index = 0, before = chunk.slot_count, header = chunk.line_words;
    u32 op = OP_AND;
    if (is_bool)
      chunk.line_words++;
    p = scan_line(p, edges, [&](const u8 *word, u64 hash) {
      if (is_bool) {
        if (index % 2)
          chunk.lines[chunk.line_words++] = draft.intern(hash, SEEN_IN_BOOL) << OP_BITS | op;
        else if (index)
          op = *word == 'A' ? OP_AND : *word == 'O' ? OP_OR : OP_NOT;
      } else if (index) {
        chunk.slots[chunk.slot_count] = draft.intern(hash, 1);
        chunk.hashes[chunk.slot_count++] = hash;
      }
      index++;
    });
    if (is_bool) {
      chunk.lines[header] = index / 2;
    } else {
      chunk.lines[chunk.line_words++] = 0;
      chunk.lengths[chunk.phrase_count++] = chunk.slot_count - before;
    }
  }
}

// Final ids run through the terms with lists, then the common terms, then those used
// only in phrases, so an id alone says how a token is handled.
static void plan_terms() {
  size_t phrase_count = 0;
  for (auto &chunk : query_chunks) {
    phrase_bases.push_back(phrase_count);
    phrase_count += chunk.phrase_count;
  }
  auto is_high = [](u64 seen) { return (u32)seen > HIGH_COUNT; };
  for (size_t at = 0; at < draft.len; at++) {
    u64 seen = draft.slots[at].seen.load(relaxed);
    if (seen >> 32)
      (is_high(seen) ? high_terms : low_terms)++;
  }
  size_t next_low = 0, next_high = low_terms, next_other = low_terms + high_terms;
  for (size_t at = 0; at < draft.len; at++) {
    u64 seen = draft.slots[at].seen.load(relaxed);
    u64 id = !(seen >> 32) ? next_other : is_high(seen) ? next_high : next_low;
    next_other += !(seen >> 32) && seen;
    next_high += (seen >> 32) && is_high(seen);
    next_low += (seen >> 32) && !is_high(seen);
    draft.slots[at].seen.store(id << 32 | (u32)seen, relaxed);
  }
  phrases.terms = query_hashes;
  phrases.list.resize(phrase_count + 1);
  phrases.first_of_anchor.resize(next_other);
  phrases.pairs.resize(1 << (PAIR_BITS - 6));
  // Four times the size of the draft, as a lookup that misses its home slot is costly:
  // about 1% do at this size, 3% at half of it.
  size_t table_size = (draft.mask + 1) * 4;
  table = {(u64 *)huge_pages((table_size + TABLE_PADDING) * 8), table_size - 1};
}

// Picks each phrase's anchor and fills the lookup table, on all threads. The table only
// holds terms with lists or bitsets and anchors, as phrases compare hashes.
static void plan_chunk(size_t c) {
  QueryChunk &chunk = query_chunks[c];
  u32 phrase = phrase_bases[c];
  const u32 *list = chunk.slots;
  const u64 *hashes = chunk.hashes;
  auto seen = [](u32 slot) { return draft.slots[slot].seen.load(relaxed); };
  for (size_t p = 0; p < chunk.phrase_count; p++) {
    size_t length = chunk.lengths[p];
    size_t at = std::min_element(list, list + length, [&](u32 a, u32 b) {
                  return (u32)seen(a) < (u32)seen(b);
                }) - list;
    u32 id = seen(list[at]) >> 32;
    phrase++;
    table.add(hashes[at], id | ANCHOR);
    u32 next = __atomic_exchange_n(&phrases.first_of_anchor[id], phrase, __ATOMIC_RELAXED);
    phrases.list[phrase] = {(u32)(hashes - query_hashes), next, (u8)at, (u8)length};
    size_t one = at + 1 < length ? at : at - 1;
    size_t bit = pair_bit(hashes[one], hashes[one + 1]);
    __atomic_fetch_or(&phrases.pairs[bit / 64], 1ull << (bit % 64), __ATOMIC_RELAXED);
    list += length;
    hashes += length;
  }
  for (size_t i = 0; i < chunk.line_words; i++)
    for (u32 left = chunk.lines[i]; left; left--) {
      u32 &term = chunk.lines[++i];
      u32 slot = term >> OP_BITS, id = seen(slot) >> 32;
      table.add(draft.slots[slot].hash.load(relaxed), id);
      term = id << OP_BITS | (term & ((1 << OP_BITS) - 1));
    }
}

static u16 *claim_log_block(size_t partition) {
  return (u16 *)(log_regions + partition * log_span +
                 next_log_block[partition].take() * LOG_BLOCK);
}

// The passes over a document are separate functions, so each loop has the registers for
// its own constants: inlined together, the compiler rebuilds them in every iteration.
// Each lookup is prefetched as soon as its hash is known, so memory arrives in time.
[[gnu::noinline]] static const u8 *hash_tokens(const u8 *p, u32 *edges, u64 *tokens,
                                               size_t &count) {
  const u64 *slots = table.slots;
  const size_t mask = table.mask;
  u64 *token = tokens;
  p = scan_line(p, edges, [&](const u8 *, u64 hash) {
    _mm_prefetch(slots + (hash & mask), _MM_HINT_T0);
    *token++ = hash;
  });
  count = token - tokens;
  return p;
}

// Four lookups in one vector step: an id with its anchor flag, or an id no list or
// bitset uses. A displaced term, or a token sharing its home slot with one, takes
// another look, about 1% of tokens.
[[gnu::noinline]] static void look_up(const u64 *tokens, u32 *ids, size_t count) {
  const u64 *slots = table.slots;
  const v4 slot_mask = _mm256_set1_epi64x(table.mask), zero = _mm256_setzero_si256();
  const v4 miss = _mm256_set1_epi64x(NO_ID);
  const v4 found_mask = _mm256_set1_epi64x(NO_ID | ANCHOR);
  const v4 low_halves = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
  for (size_t at = 1; at <= count; at += 4) {
    v4 hash = load(tokens + at);
    v4 value = _mm256_i64gather_epi64((const long long *)slots,
                                      _mm256_and_si256(hash, slot_mask), 8);
    v4 same = _mm256_cmpeq_epi64(
        _mm256_srli_epi64(_mm256_xor_si256(value, hash), TAG_SHIFT), zero);
    u32 is_same = _mm256_movemask_pd(_mm256_castsi256_pd(same));
    u32 displaced = _mm256_movemask_pd(
        _mm256_castsi256_pd(_mm256_slli_epi64(value, 63 - DISPLACED_BIT)));
    v4 found = _mm256_blendv_epi8(miss, _mm256_and_si256(value, found_mask), same);
    _mm_storeu_si128((v2 *)(ids + at), _mm256_castsi256_si128(
        _mm256_permutevar8x32_epi32(found, low_halves)));
    for (u32 again = displaced & ~is_same; again; again &= again - 1) {
      size_t i = at + __builtin_ctz(again);
      ids[i] = table.find_displaced(tokens[i]);
    }
  }
  for (size_t i = 1; i <= 8; i++)
    ids[count + i] = NO_ID;
}

// Sorts ids by kind in a vector step, as a token's kind is unpredictable and a branch on
// it would be mispredicted constantly. A common term leaves its line's byte offset, and
// a list term leaves its hit with the partition above it.
[[gnu::noinline]] static void sort_by_kind(Scratch &scratch, size_t count, u32 doc) {
  const u64 *tokens = scratch.tokens;
  u32 *ids = scratch.ids, *high_ids = scratch.high_ids, *low_ids = scratch.low_ids;
  const v4 id_mask = _mm256_set1_epi32(NO_ID);
  const v4 low_limit = _mm256_set1_epi32(low_terms);
  const v4 high_limit = _mm256_set1_epi32(low_terms + high_terms);
  const v4 position = _mm256_set1_epi32((doc & (GROUP_DOCS - 1)) << 1);
  size_t highs = 0, lows = 0;
  for (size_t at = 1; at <= count; at += 8) {
    v4 x = load(ids + at), id = _mm256_and_si256(x, id_mask);
    u32 is_low = lanes(_mm256_cmpgt_epi32(low_limit, id));
    u32 is_high = is_low ^ lanes(_mm256_cmpgt_epi32(high_limit, id));
    v4 offset = _mm256_slli_epi32(id, HIT_TERM_SHIFT - 1);
    store(high_ids + highs, gather(offset, is_high));
    store(low_ids + lows,
          gather(_mm256_or_si256(_mm256_add_epi32(offset, offset), position), is_low));
    highs += __builtin_popcount(is_high);
    lows += __builtin_popcount(is_low);
    u32 anchors = lanes(_mm256_slli_epi32(x, 31 - ID_BITS));
    for (; anchors; anchors &= anchors - 1) {
      size_t i = at + __builtin_ctz(anchors);
      if (phrases.may_pair(tokens[i], tokens[i + 1]) ||
          phrases.may_pair(tokens[i - 1], tokens[i]))
        phrases.check(ids[i] & NO_ID, tokens, count, i, doc, scratch.hits);
    }
  }
  scratch.highs = highs;
  scratch.lows = lows;
}

static void parse_docs(Scratch &scratch) {
  vector<u16 *> heads(partitions);
  for (size_t part = 0; part < partitions; part++)
    heads[part] = claim_log_block(part);
  u64 *tokens = scratch.tokens;
  const u32 *high_ids = scratch.high_ids;
  const auto low_hits = (const u16 *)scratch.low_ids;
  const u32 high = high_terms;
  const size_t row_words = words, line_bytes = LINE_WORDS * 8;
  auto lines = (u64 *)aligned_alloc(64, (high + 1) * line_bytes);
  memset(lines, 0, (high + 1) * line_bytes);
  u32 group = UINT32_MAX;
  auto flush = [&](u32 doc) {
    size_t at = doc / LINE_DOCS * LINE_WORDS;
    for (size_t row = 0; row < high; row++) {
      u64 *from = lines + row * LINE_WORDS, *to = high_bits + row * row_words + at;
      v4 a = load(from), b = load(from + 4), any = _mm256_or_si256(a, b);
      // Terms that barely occur keep their pages untouched.
      if (!_mm256_testz_si256(any, any)) {
        store(to, a);
        store(to + 4, b);
        store(from, _mm256_setzero_si256());
        store(from + 4, _mm256_setzero_si256());
      }
    }
  };
  auto push = [&](size_t part, u16 entry) {
    u16 *head = heads[part];
    *head++ = entry;
    // Every block starts with a marker.
    if (!((size_t)head & (LOG_BLOCK - 1))) {
      head = claim_log_block(part);
      *head++ = group << 1 | 1;
    }
    heads[part] = head;
  };
  const u8 *file_end = input + input_size;
  for (size_t off; (off = next_chunk.take(CHUNK)) < input_size;) {
    const u8 *p = input + off;
    const u8 *stop = input + std::min(off + CHUNK, input_size);
    // A claim holds the blocks starting in its range, so blocks have one owner.
    if (off) {
      p = (const u8 *)memchr(p - 1, '\n', file_end - p + 1) + 1;
      if (p < file_end) {
        const u8 *q = p;
        for (u32 skip = -parse_id(q) & (LINE_DOCS - 1); skip && p < file_end; skip--)
          p = (const u8 *)memchr(p, '\n', file_end - p) + 1;
      }
    }
    while (p < file_end) {
      const u8 *line_start = p;
      u32 doc = parse_id(p);
      if (line_start >= stop && doc % LINE_DOCS == 0)
        break;
      if (doc >> GROUP_BITS != group) {
        group = doc >> GROUP_BITS;
        for (size_t part = 0; part < partitions; part++)
          push(part, group << 1 | 1);
      }
      size_t count;
      tokens[0] = 0;
      p = hash_tokens(p, scratch.ids, tokens + 1, count);
      tokens[count + 1] = 0;
      look_up(tokens, scratch.ids, count);
      sort_by_kind(scratch, count, doc);
      u64 bit = 1ull << (doc & (GROUP_DOCS - 1));
      // Offsets count from the first id of a common term.
      size_t base = (size_t)(lines + (doc >> GROUP_BITS) % LINE_WORDS) -
                    low_terms * line_bytes;
      for (size_t i = 0; i < scratch.highs; i++)
        *(u64 *)(base + high_ids[i]) |= bit;
      for (size_t i = 0; i < scratch.lows * 2; i += 2)
        push(low_hits[i + 1], low_hits[i]);
      if (doc % LINE_DOCS == LINE_DOCS - 1 || p == file_end)
        flush(doc);
    }
  }
  for (u16 *head : heads)
    *head = LOG_END;
  free(lines);
}

// Turns a partition's log into its lists. Runs of hits are visited by group, so
// every list comes out sorted without sorting. The lists are written to a private
// image, then copied over the log they came from, so log and lists never both take
// memory.
static void build_lists(size_t p, Scratch &scratch) {
  auto log = (u16 *)(log_regions + p * log_span);
  auto image = (u16 *)scratch.tokens;
  u32 *runs = scratch.ids, *order = scratch.high_ids;
  u32 *group_starts = scratch.low_ids;
  size_t used = next_log_block[p].value.load(relaxed) * LOG_BLOCK;
  size_t groups = docs / GROUP_DOCS + 1, run_count = 0;
  u32 counts[PARTITION_TERMS] = {};
  memset(group_starts, 0, (groups + 2) * 4);
  for (size_t i = 0; i < used / 2; i++) {
    u16 entry = log[i];
    if (!(entry & 1)) {
      counts[entry >> HIT_TERM_SHIFT]++;
    } else if (entry == LOG_END) {
      i |= LOG_BLOCK / 2 - 1;
    } else {
      runs[run_count++] = i;
      group_starts[(entry >> 1) + 2]++;
    }
  }
  for (size_t g = 2; g < groups + 2; g++)
    group_starts[g] += group_starts[g - 1];
  for (size_t i = 0; i < run_count; i++)
    order[group_starts[(log[runs[i]] >> 1) + 1]++] = runs[i];

  u32 cursors[PARTITION_TERMS], sum = 0;
  for (size_t term = 0; term < PARTITION_TERMS; term++) {
    cursors[term] = sum;
    sum += counts[term];
  }
  const size_t first_term = p << PARTITION_BITS;
  const size_t last_term = std::min<size_t>(first_term + PARTITION_TERMS, low_terms);
  auto mark = [&](size_t block) {
    for (size_t term = first_term; term < last_term; term++)
      list_marks[term * (blocks + 1) + block] = cursors[term - first_term];
  };
  mark(0);
  size_t block = 0;
  u32 last[PARTITION_TERMS];
  memset(last, 255, sizeof last);
  for (size_t r = 0; r < run_count; r++) {
    u32 run = order[r], group = log[run] >> 1;
    size_t run_block = group >> GROUPS_PER_BLOCK_BITS;
    while (block < run_block) {
      mark(++block);
      memset(last, 255, sizeof last);
    }
    u32 group_low = (group & ((1 << GROUPS_PER_BLOCK_BITS) - 1)) << GROUP_BITS;
    for (u32 i = run + 1; !(log[i] & 1); i++) {
      size_t term = log[i] >> HIT_TERM_SHIFT;
      u32 doc_low = group_low | (log[i] >> 1 & (GROUP_DOCS - 1));
      // A term repeated within a document appears once per token.
      image[cursors[term]] = doc_low;
      cursors[term] += last[term] != doc_low;
      last[term] = doc_low;
    }
  }
  while (block < blocks)
    mark(++block);
  memcpy(log, image, sum * 2);
  size_t kept = (sum * 2 + PAGE - 1) & -PAGE;
  if (used > kept)
    madvise((u8 *)log + kept, used - kept, MADV_DONTNEED);
}

static size_t list_size(size_t term) {
  const u32 *m = list_marks + term * (blocks + 1);
  return m[blocks] - m[0];
}

// Whole vectors are written and read past the end, so buffers have room for them. The
// compiler cannot vectorize this itself, as `out` might overlap the marks.
static size_t decode_list(size_t term, u32 *out) {
  const u32 *m = list_marks + term * (blocks + 1);
  auto ids = (const u16 *)(log_regions + (term >> PARTITION_BITS) * log_span);
  size_t n = 0;
  for (size_t block = 0; block < blocks; block++) {
    const v4 top = _mm256_set1_epi32(block << BLOCK_BITS);
    size_t count = m[block + 1] - m[block];
    const u16 *from = ids + m[block];
    for (size_t i = 0; i < count; i += 8)
      store(out + n + i, _mm256_or_si256(top, _mm256_cvtepu16_epi32(
                                                  _mm_loadu_si128((const v2 *)(from + i)))));
    n += count;
  }
  return n;
}

static void write_line(std::string &out, u64 count, u64 sum, const u32 *first) {
  char line[256], *end = line;
  auto put = [&](u64 value, char after) {
    end = std::to_chars(end, end + 20, value).ptr;
    *end++ = after;
  };
  put(count, ' ');
  put(sum, ' ');
  for (u64 i = 0; i < std::min<u64>(count, 10); i++)
    put(first[i], ' ');
  end[-1] = '\n';
  out.append(line, end);
}

// Counts a bitset's set bits and sums their positions. Nibble lookups give each
// byte's count and the sum of positions inside it, and byte and word weights
// turn those into positions in the whole set: about 5 micro-ops per word, against
// 20 for six masked population counts. `n` is a multiple of 4 words.
static void count_bits(const u64 *bits, size_t n, u64 &count, u64 &sum) {
  const v4 pops = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4));
  const v4 sums = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6));
  const v4 weights = _mm256_set1_epi64x(0x3830282018100800);
  const v4 zero = _mm256_setzero_si256();
  v4 running = zero, weighted = zero, inside = zero;
  for (size_t i = 0; i < n; i += 4) {
    v4 x = load(bits + i), lo = _mm256_and_si256(x, ones(0x0F));
    v4 hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), ones(0x0F));
    v4 pop_hi = shuffle(pops, hi), pop = add8(shuffle(pops, lo), pop_hi);
    v4 pos = add8(add8(shuffle(sums, lo), shuffle(sums, hi)),
                  _mm256_slli_epi16(pop_hi, 2));
    inside = add64(inside, _mm256_sad_epu8(pos, zero));
    v4 byte_part = _mm256_maddubs_epi16(weights, pop);
    inside = _mm256_add_epi32(
        inside, _mm256_madd_epi16(byte_part, _mm256_set1_epi16(1)));
    running = add64(running, _mm256_sad_epu8(pop, zero));
    weighted = add64(weighted, running);
  }
  u64 per_lane[4], later[4], rest[4];
  store(per_lane, running);
  store(later, weighted);
  store(rest, inside);
  count = sum = 0;
  // `weighted` adds each vector's counts once per vector from it to the end.
  for (u64 lane = 0; lane < 4; lane++) {
    u64 before = n / 4 * per_lane[lane] - later[lane];
    count += per_lane[lane];
    sum += (rest[lane] & UINT32_MAX) + (rest[lane] >> 32) +
           64 * (4 * before + lane * per_lane[lane]);
  }
}

// Merges two ascending lists without branching on their contents, as those
// comparisons are unpredictable.
template <u32 OP>
static size_t merge(const u32 *a, size_t na, const u32 *b, size_t nb, u32 *out) {
  size_t i = 0, j = 0, n = 0;
  while (i < na && j < nb) {
    u32 x = a[i], y = b[j];
    out[n] = std::min(x, y);
    n += OP == OP_OR || x < y;
    i += x <= y;
    j += y <= x;
  }
  n += std::copy(a + i, a + na, out + n) - (out + n);
  if (OP == OP_OR)
    n += std::copy(b + j, b + nb, out + n) - (out + n);
  return n;
}

// For lists of similar sizes: compares 8 ids of each list, all against all, then
// moves on from whichever block ends first. Both lists are followed by 8 ids above
// any document, and different ones.
static size_t intersect(const u32 *a, size_t na, const u32 *b, size_t nb, u32 *out) {
  const v4 rotate = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
  size_t i = 0, j = 0, n = 0;
  while (i < na && j < nb) {
    v4 x = load(a + i), y = load(b + j), equal = _mm256_cmpeq_epi32(x, y);
    for (int k = 1; k < 8; k++) {
      y = _mm256_permutevar8x32_epi32(y, rotate);
      equal = _mm256_or_si256(equal, _mm256_cmpeq_epi32(x, y));
    }
    u32 found = lanes(equal);
    store(out + n, gather(x, found));
    n += __builtin_popcount(found);
    u32 a_last = a[i + 7], b_last = b[j + 7];
    i += (a_last <= b_last) * 8;
    j += (b_last <= a_last) * 8;
  }
  return n;
}

// For a list far longer than the ids looked up in it: searched where it lies, by whole
// vectors, so the list is never decoded.
static size_t probe_list(const u32 *a, size_t na, size_t term, u32 *out) {
  const u32 *m = list_marks + term * (blocks + 1);
  auto ids = (const u16 *)(log_regions + (term >> PARTITION_BITS) * log_span);
  size_t n = 0, block = 0, j = m[0];
  for (size_t i = 0; i < na; i++) {
    size_t at = a[i] >> BLOCK_BITS;
    if (at != block) {
      block = at;
      j = m[at];
    }
    const size_t end = m[at + 1];
    const u16 low = a[i];
    while (j + 128 <= end && ids[j + 127] < low)
      j += 128;
    while (j + 16 <= end && ids[j + 15] < low)
      j += 16;
    u32 equal = _mm256_movemask_epi8(_mm256_cmpeq_epi16(load(ids + j), _mm256_set1_epi16(low)));
    out[n] = a[i];
    n += (equal & _bzhi_u32(~0u, 2 * std::min<size_t>(end - j, 16))) != 0;
  }
  return n;
}

static u64 sum_of(const u32 *ids, size_t n) {
  u64 sum = 0;
  for (size_t i = 0; i < n; i++)
    sum += ids[i];
  return sum;
}

// Answers one boolean query. Filters between two ORs commute, so within such a run
// the smallest list goes first and NOT comes last, which leaves little to compare.
struct Evaluator {
  u32 *r, *tmp, *l;
  u64 *bits;
  size_t n = 0;
  bool dense = false;

  void to_dense(const u64 *start) {
    memcpy(bits, start, words * 8);
    for (size_t i = 0; i < n; i++)
      bits[r[i] / 64] |= 1ull << (r[i] % 64);
    dense = true;
  }

  void filter_bits(const u64 *row, bool keep) {
    size_t kept = 0;
    for (size_t i = 0; i < n; i++) {
      r[kept] = r[i];
      kept += ((row[r[i] / 64] >> (r[i] % 64)) & 1) == keep;
    }
    n = kept;
  }

  void apply(u32 op, u32 id) {
    if (!dense && !n && op != OP_OR)
      return;
    if (id >= low_terms) {
      const u64 *row = high_bits + (id - low_terms) * words;
      if (dense) {
        for (size_t i = 0; i < words; i += 4) {
          v4 x = load(bits + i), y = load(row + i);
          store(bits + i, op == OP_AND  ? _mm256_and_si256(x, y)
                          : op == OP_OR ? _mm256_or_si256(x, y)
                                        : _mm256_andnot_si256(y, x));
        }
      } else if (op == OP_OR) {
        to_dense(row);
      } else {
        filter_bits(row, op == OP_AND);
      }
      return;
    }
    if (!dense && op != OP_OR) {
      size_t shared = find_shared(id);
      if (op == OP_AND) {
        n = shared;
      } else {
        n = merge<OP_NOT>(r, n, tmp, shared, l);
        std::swap(tmp, l);
      }
      std::swap(r, tmp);
      return;
    }
    size_t nl = decode_list(id, l);
    if (dense) {
      if (op == OP_AND) {
        std::swap(r, l);
        n = nl;
        filter_bits(bits, true);
        dense = false;
      } else {
        for (size_t i = 0; i < nl; i++) {
          u64 bit = 1ull << (l[i] % 64);
          u64 &word = bits[l[i] / 64];
          word = op == OP_OR ? word | bit : word & ~bit;
        }
      }
      return;
    }
    if (n + nl > DENSE_LIMIT) {
      to_dense(high_bits + high_terms * words);
      apply(op, id);
      return;
    }
    n = merge<OP_OR>(r, n, l, nl, tmp);
    std::swap(r, tmp);
  }

  // Leaves in `tmp` the ids that `r` and the term's list share.
  size_t find_shared(u32 id) {
    if (n * IN_PLACE_RATIO < list_size(id))
      return probe_list(r, n, id, tmp);
    return shared_with_decoded(decode_list(id, l));
  }

  size_t shared_with_decoded(size_t nl) {
    for (int i = 0; i < 8; i++) {
      r[n + i] = UINT32_MAX;
      l[nl + i] = UINT32_MAX - 1;
    }
    return intersect(r, n, l, nl, tmp);
  }

  // A last OR or NOT needs no result list: the shared ids give its size and sum, and
  // only the start of the result is built.
  void finish_from_shared(u32 op, u32 id, std::string &out) {
    if (op == OP_OR) {
      size_t nl = decode_list(id, l), shared = shared_with_decoded(nl);
      u64 sum = sum_of(r, n) - sum_of(tmp, shared);
      u32 first[20];
      merge<OP_OR>(r, std::min<size_t>(n, 10), l, std::min<size_t>(nl, 10), first);
      write_line(out, n + nl - shared, sum + sum_of(l, nl), first);
    } else {
      size_t shared = find_shared(id);
      merge<OP_NOT>(r, std::min(n, shared + 10), tmp, shared, l);
      write_line(out, n - shared, sum_of(r, n) - sum_of(tmp, shared), l);
    }
  }

  // `terms` holds each term's id above its operator, with AND for the first term.
  void answer(u32 *terms, size_t count, std::string &out) {
    auto op = [](u32 term) { return term & ((1 << OP_BITS) - 1); };
    auto key = [&](u32 term) {
      u32 id = term >> OP_BITS;
      u64 size = id >= low_terms ? UINT32_MAX : list_size(id);
      return size | (u64)(op(term) == OP_NOT) << 32;
    };
    for (size_t i = 0; i < count;) {
      size_t end = i + 1;
      while (end < count && op(terms[end]) != OP_OR)
        end++;
      std::sort(terms + i + (i > 0), terms + end,
                [&](u32 a, u32 b) { return key(a) < key(b); });
      i = end;
    }
    u32 base = terms[0] >> OP_BITS;
    n = 0;
    dense = false;
    if (base >= low_terms)
      to_dense(high_bits + (base - low_terms) * words);
    else
      n = decode_list(base, r);
    for (size_t i = 1; i + 1 < count; i++)
      apply(op(terms[i]), terms[i] >> OP_BITS);
    if (count > 1) {
      u32 last_op = op(terms[count - 1]), last = terms[count - 1] >> OP_BITS;
      if (!dense && last_op != OP_AND && last < low_terms &&
          n + list_size(last) <= words * 2) {
        finish_from_shared(last_op, last, out);
        return;
      }
      apply(last_op, last);
    }
    if (dense) {
      u32 first[10], m = 0;
      u64 total, sum;
      count_bits(bits, words, total, sum);
      for (size_t w = 0; w < words && m < 10; w++)
        for (u64 rest = bits[w]; rest && m < 10; rest &= rest - 1)
          first[m++] = w * 64 + __builtin_ctzll(rest);
      write_line(out, total, sum, first);
    } else {
      write_line(out, n, sum_of(r, n), r);
    }
  }
};

static void answer_queries(Scratch &scratch) {
  Evaluator evaluator{scratch.ids, scratch.high_ids, scratch.low_ids, scratch.tokens};
  for (size_t c; (c = next_answer.take()) < chunks;) {
    QueryChunk &chunk = query_chunks[c];
    std::string &out = chunk.out;
    out.reserve(bounds[c + 1] - bounds[c]);
    size_t phrase = phrase_bases[c];
    for (size_t i = 0; i < chunk.line_words; i++) {
      u32 size = chunk.lines[i];
      if (size) {
        evaluator.answer(chunk.lines + i + 1, size, out);
        i += size;
      } else {
        // A document can match a phrase more than once, but it counts once.
        u32 *first = phrase_docs.data() + matched[phrase];
        u32 *end = phrase_docs.data() + matched[phrase + 1];
        std::sort(first, end);
        end = std::unique(first, end);
        write_line(out, end - first, sum_of(first, end - first), first);
        phrase++;
      }
    }
  }
}

// Each run on one thread once all threads have arrived at the barrier.
static void after_plan() {
  free(draft.slots);
  munmap(query_slots, text_size * 2);
  munmap(query_lengths, text_size * 2);
  // One more row than terms: nothing writes it, so it reads as zeros.
  high_bits = (u64 *)reserve((high_terms + 1) * words * 8);
  partitions = low_terms / PARTITION_TERMS + 1;
  // A token takes at least 2 bytes of input, and 2 bytes here.
  log_span = (input_size + (1 << 24)) & -PAGE;
  log_regions = reserve(partitions * log_span);
  next_log_block = new Counter[partitions];
  list_marks = (u32 *)calloc((low_terms + 1) * (blocks + 1), 4);
}

// Groups the phrases' documents by phrase. Answers sort each phrase's few documents.
static void after_parsing() {
  size_t phrase_count = phrases.list.size() - 1, total = 0;
  munmap(query_hashes, text_size * 4);
  matched.resize(phrase_count + 1);
  for (auto &scratch : scratches)
    for (const Hit &hit : scratch.hits)
      matched[hit.phrase + 1]++, total++;
  for (size_t i = 1; i <= phrase_count; i++)
    matched[i] += matched[i - 1];
  phrase_docs.resize(total);
  vector<u32> cursors(matched);
  for (auto &scratch : scratches)
    for (const Hit &hit : scratch.hits)
      phrase_docs[cursors[hit.phrase]++] = hit.doc;
  phrases = {};
  munmap(table.slots, (table.mask + 1 + TABLE_PADDING) * 8);
  munmap((void *)input, input_size + PAGE);
}

static void nop() {}

static void write_output() {
  int fd = open("/out/results.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  for (auto &chunk : query_chunks)
    if (write(fd, chunk.out.data(), chunk.out.size()) != (ssize_t)chunk.out.size())
      abort();
  _exit(0);
}

static void between_steps() noexcept {
  static void (*const steps[])() = {plan_terms, after_plan, after_parsing,
                                    nop,        nop,        write_output};
  static int step = 0;
  steps[step++]();
}

static std::barrier<void (*)() noexcept> *sync_all;

static void work(int t) {
  Scratch &scratch = scratches[t];
  size_t most = input_size / 2 + (1 << 16);
  scratch.tokens = (u64 *)reserve(most * 8);
  scratch.ids = (u32 *)reserve(most * 8);
  scratch.high_ids = (u32 *)reserve(most * 4);
  scratch.low_ids = (u32 *)reserve(most * 4);
  for (size_t c; (c = next_query.take()) < chunks;)
    read_chunk(c, scratch.ids);
  sync_all->arrive_and_wait();
  for (size_t c; (c = next_plan.take()) < chunks;)
    plan_chunk(c);
  sync_all->arrive_and_wait();
  parse_docs(scratch);
  // Exit would otherwise clear the input's page tables page by page on one
  // thread, with the clock still running.
  for (size_t off; (off = next_unmap.take(RELEASE_STEP)) < input_size;)
    madvise((void *)(input + off), std::min(RELEASE_STEP, input_size - off),
            MADV_DONTNEED);
  sync_all->arrive_and_wait();
  for (size_t p; (p = next_partition.take()) < partitions;)
    build_lists(p, scratch);
  // Bitsets take over these pages while answering.
  madvise(scratch.tokens, most * 8, MADV_DONTNEED);
  sync_all->arrive_and_wait();
  answer_queries(scratch);
  sync_all->arrive_and_wait();
  // Nothing but the answers is needed now. Giving it all back here, on all threads,
  // lowers the peak as the output grows, and leaves exit almost nothing to do.
  for (size_t p; (p = next_release.take()) <= partitions;)
    if (p == partitions)
      madvise(high_bits, high_terms * words * 8, MADV_DONTNEED);
    else
      madvise(log_regions + p * log_span,
              next_log_block[p].value.load(relaxed) * LOG_BLOCK, MADV_DONTNEED);
  sync_all->arrive_and_wait();
}

int main() {
  // The box has huge pages set to always, which charges 2 MB for a first touch
  // of a sparse region or a thread stack, and more when threads race to fault one.
  prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
  mallopt(M_MMAP_THRESHOLD, 1 << 16);
  memset(low_bytes, 255, 32);
  for (u32 mask = 0; mask < 256; mask++)
    for (u64 lane = 0, used = 0; lane < 8; lane++)
      if (mask >> lane & 1)
        lane_orders[mask] |= lane << (8 * used++);
  cpu_set_t cpus;
  sched_getaffinity(0, sizeof cpus, &cpus);
  const int threads = CPU_COUNT(&cpus);
  text = map_input("/data/queries.txt", text_size);
  find_query_chunks();
  size_t size = std::max<size_t>(std::bit_ceil(text_size / 16 + 16), 1 << 10);
  draft.len = size + TABLE_PADDING;
  draft.mask = size - 1;
  draft.slots = (DraftSlot *)calloc(draft.len, sizeof(DraftSlot));
  query_chunks.resize(chunks);
  // A line has a size as well as its terms, and a line takes at least 7 bytes.
  query_lines = (u32 *)reserve(text_size * 3);
  query_slots = (u32 *)reserve(text_size * 2);
  query_lengths = (u32 *)reserve(text_size * 2);
  query_hashes = (u64 *)reserve(text_size * 4);
  input = map_input("/data/docs.txt", input_size);
  if (input_size) {
    const u8 *last = input + input_size - 1;
    while (last > input && last[-1] != '\n')
      last--;
    docs = parse_id(last) + 1;
  }
  blocks = (docs >> BLOCK_BITS) + 1;
  words = (docs / 64 | (LINE_WORDS - 1)) + 1;
  scratches.resize(threads);
  std::barrier<void (*)() noexcept> barrier(threads, between_steps);
  sync_all = &barrier;
  vector<std::thread> pool;
  for (int t = 1; t < threads; t++)
    pool.emplace_back(work, t);
  work(0);
}
