#include <algorithm>
#include <atomic>
#include <barrier>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <malloc.h>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
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
using std::atomic;
using std::vector;
constexpr auto relaxed = std::memory_order_relaxed;
#define INLINE [[gnu::always_inline]] inline

const size_t CHUNK = 1 << 19, QUERY_CHUNK = 1 << 14, PREFETCH_AHEAD = 12;
// A term this frequent among the queries is likely in over 1/32 of documents,
// where a bitset is smaller than the hit log and its list. The queries sample the
// corpus: phrases are windows of documents, and boolean terms are drawn by band.
const u32 HIGH_COUNT = 25;
// Terms this frequent among the queries are placed in the table first.
const u32 HOT_COUNT = 4;
const size_t UNMAP_STEP = 1 << 22;

// A table slot holds the high bits of the term's hash, the term's id and two flags.
const int ID_BITS = 18, TAG_SHIFT = 20;
const u64 ID_MASK = (1 << ID_BITS) - 1, ANCHOR_FLAG = 1 << ID_BITS;
const u64 DISPLACED = ANCHOR_FLAG << 1, FOUND_MASK = ID_MASK | ANCHOR_FLAG;
const u64 TAG_MASK = ~0ull << TAG_SHIFT;
const size_t TABLE_PADDING = 1 << 10;
const u32 OP_AND = 0, OP_OR = 1, OP_NOT = 2, OP_BITS = 2;

// Common terms gather one cache line per term, 512 documents, before it is written
// to the shared bitset. Each thread owns whole blocks of 512 documents.
const u32 LINE_DOCS = 512, LINE_WORDS = 8;

// A logged hit is 16 bits: a term within its partition, and a document within
// its group of 64. A worker writes a marker naming the group when it starts one.
// Workers claim small blocks of a partition's region, so all hits of a partition
// lie together with no copying as a log grows, and few pages stay partly used.
const int GROUP_BITS = 6, PARTITION_BITS = 9;
const u32 GROUP_DOCS = 1 << GROUP_BITS, NO_GROUP = UINT32_MAX;
const size_t PARTITION_TERMS = 1 << PARTITION_BITS, LOG_BLOCK = 1 << 9;
const size_t PAGE = 1 << 12;
const u16 MARKER = 1 << 15, LOG_END = UINT16_MAX;

// Document ids in lists are stored as 16 bits within a block of 65,536 documents.
const int BLOCK_BITS = 16, GROUPS_PER_BLOCK_BITS = BLOCK_BITS - GROUP_BITS;

const u64 HASH_0 = 0x2D358DCCAA6C78A5, HASH_1 = 0x8BB84B93962EACC9;
const u64 HASH_2 = 0x4B33A62ED433D4A3, HASH_3 = 0xC2B2AE3D27D4EB4F;
const int PAIR_BITS = 20;
const size_t PROBE_RATIO = 6;

struct Hit {
  u32 phrase, doc;
  auto operator<=>(const Hit &) const = default;
};
static int threads;
alignas(64) static u8 low_bytes[64];

// Mapped over a larger anonymous mapping so vector loads near the end of the file
// stay readable.
static const u8 *map_input(const char *path, size_t &size) {
  int fd = open(path, O_RDONLY);
  struct stat info;
  fstat(fd, &info);
  size = info.st_size;
  void *base =
      mmap(0, size + 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (size)
    mmap(base, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
  return (const u8 *)base;
}

// Address space only: pages cost memory once touched, and never need growing.
static u8 *reserve(size_t bytes) {
  return (u8 *)mmap(0, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
}

INLINE u64 mum(u64 a, u64 b) {
  auto product = (unsigned __int128)a * b;
  return (u64)product ^ (u64)(product >> 64);
}

[[gnu::cold]] static u64 long_token_hash(const u8 *start, size_t len) {
  if (len > 32)
    return HASH_0;
  v4 bytes = _mm256_or_si256(_mm256_loadu_si256((const v4 *)start),
                             _mm256_set1_epi8(0x20));
  v4 kept = _mm256_and_si256(
      bytes, _mm256_loadu_si256((const v4 *)(low_bytes + 32 - len)));
  u64 w[4];
  _mm256_storeu_si256((v4 *)w, kept);
  return mum(w[0] ^ HASH_0, w[1] ^ HASH_1) ^ mum(w[2] ^ HASH_2, w[3] ^ HASH_3);
}

// Hashes a token where it lies. Token bytes are letters and digits, so OR-ing
// 0x20 lowercases them, and masking removes whatever follows. Two AES rounds spread
// every byte over the whole state in far fewer micro-ops than a wide multiply.
// Terms are at most 32 bytes, so a longer token hashes to a value no term has. The
// hash stands in for the bytes: with about 600,000 distinct tokens and 76,000 terms,
// the chance of any collision is about 1 in 10 million.
INLINE u64 token_hash(const u8 *start, size_t len) {
  if (len > 16)
    return long_token_hash(start, len);
  v2 bytes = _mm_or_si128(_mm_loadu_si128((const v2 *)start),
                          _mm_set1_epi8(0x20));
  v2 kept = _mm_and_si128(
      bytes, _mm_loadu_si128((const v2 *)(low_bytes + 32 - len)));
  v2 mixed = _mm_aesenc_si128(kept, _mm_set_epi64x(HASH_0, HASH_1));
  return _mm_cvtsi128_si64(
      _mm_aesenc_si128(mixed, _mm_set_epi64x(HASH_2, HASH_3)));
}

// Classifies by nibble lookups: the tables only share a set bit for letters and
// digits.
INLINE u32 token_mask(v4 bytes) {
  const v4 low = _mm256_setr_epi8(
      5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 2, 2, 2, 2, 2, //
      5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 2, 2, 2, 2, 2);
  const v4 high = _mm256_setr_epi8(
      0, 0, 0, 1, 2, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, //
      0, 0, 0, 1, 2, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0);
  v4 nibbles =
      _mm256_and_si256(_mm256_srli_epi16(bytes, 4), _mm256_set1_epi8(0x0F));
  v4 both = _mm256_and_si256(_mm256_shuffle_epi8(low, bytes),
                             _mm256_shuffle_epi8(high, nibbles));
  return ~_mm256_movemask_epi8(
      _mm256_cmpeq_epi8(both, _mm256_setzero_si256()));
}

// Calls `visit` with each token of the line at `p`, and returns what follows its
// newline. One classification covers 64 bytes, and tokens then come straight from
// bit masks. A window restarts at a token that runs past its end, so tokens never
// need joining up. Documents and query lines are both read this way.
template <class F> INLINE const u8 *scan_line(const u8 *p, F &&visit) {
  const v4 newline = _mm256_set1_epi8('\n');
  for (;;) {
    v4 first = _mm256_loadu_si256((const v4 *)p);
    v4 second = _mm256_loadu_si256((const v4 *)(p + 32));
    u64 mask = token_mask(first) | (u64)token_mask(second) << 32;
    u32 low = _mm256_movemask_epi8(_mm256_cmpeq_epi8(first, newline));
    u32 high = _mm256_movemask_epi8(_mm256_cmpeq_epi8(second, newline));
    u64 newlines = low | (u64)high << 32;
    u64 tokens = mask & (newlines - 1) & ~newlines;
    u64 starts = tokens & ~(tokens << 1), ends = ~tokens & (tokens << 1);
    for (; ends; starts &= starts - 1, ends &= ends - 1) {
      int start = __builtin_ctzll(starts);
      visit(p + start, token_hash(p + start, __builtin_ctzll(ends) - start));
    }
    if (newlines)
      return p + __builtin_ctzll(newlines) + 1;
    if (starts == 1) {
      p += 64;
      while (isalnum(*p))
        p++;
    } else {
      p += starts ? __builtin_ctzll(starts) : 64;
    }
  }
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
  atomic<size_t> value{0};
  size_t take(size_t n = 1) { return value.fetch_add(n, relaxed); }
};

// Terms as they are found in the queries, by all threads at once.
struct Draft {
  atomic<u64> *hashes;
  size_t mask, len;

  explicit Draft(size_t query_bytes) {
    size_t size = std::max<size_t>(std::bit_ceil(query_bytes / 16 + 16), 1 << 10);
    len = size + TABLE_PADDING;
    hashes = (atomic<u64> *)calloc(len, 8);
    mask = size - 1;
  }

  u32 intern(u64 hash) {
    for (size_t at = hash & mask;; at++) {
      u64 value = hashes[at].load(relaxed);
      if (value == hash ||
          (!value && (hashes[at].compare_exchange_strong(value, hash, relaxed) ||
                      value == hash)))
        return at;
    }
  }
};

// The lookup table, about 1 MB. Nearly every token needs just one load: frequent
// terms are placed first, so they sit in their home slot. A home slot with no
// match answers for all other tokens, unless its flag says that a term with this
// home went elsewhere. The hit or miss itself is unpredictable, so no branch
// depends on it.
struct Table {
  u64 *slots = nullptr;
  size_t mask = 0;

  void insert(u64 hash, u64 payload) {
    size_t at = hash & mask;
    if (slots[at])
      slots[at] |= DISPLACED;
    while (slots[at])
      at++;
    slots[at] |= (hash & TAG_MASK) | payload;
  }

  INLINE void prefetch(u64 hash) const {
    _mm_prefetch((const char *)(slots + (hash & mask)), _MM_HINT_T0);
  }

  [[gnu::cold]] u32 find_displaced(u64 hash) const {
    for (size_t at = (hash & mask) + 1; slots[at]; at++)
      if (!((slots[at] ^ hash) >> TAG_SHIFT))
        return slots[at] & FOUND_MASK;
    return ID_MASK;
  }

  // The term's id with its anchor flag, or an id no list or bitset uses.
  INLINE u32 find(u64 hash) const {
    u64 value = slots[hash & mask];
    bool same = !((value ^ hash) >> TAG_SHIFT);
    if (__builtin_expect(!same && (value & DISPLACED), 0))
      return find_displaced(hash);
    return same ? value & FOUND_MASK : ID_MASK;
  }
};

INLINE size_t pair_bit(u64 first, u64 second) {
  return mum(first ^ HASH_2, second ^ HASH_3) >> (64 - PAIR_BITS);
}

// All phrases are known before any document is read, so they are matched live
// during tokenizing. Any document matching a phrase must contain its rarest term,
// its anchor, so nothing happens until an anchor token appears. No positions are
// stored, no documents are re-read, and phrase-only terms need no lists at all.
// Nearly all anchor tokens match nothing, so a small bitset of (term, neighbour)
// pairs rejects them before the phrase tables are touched.
struct Phrases {
  vector<u32> starts, anchor_starts;
  vector<u64> terms, pairs;
  vector<std::pair<u32, u32>> anchored;

  INLINE bool may_pair(u64 first, u64 second) const {
    size_t bit = pair_bit(first, second);
    return (pairs[bit / 64] >> (bit % 64)) & 1;
  }

  // `tokens` holds the document's hashes, with a 0 before and after them.
  [[gnu::cold]] void check(size_t id, const u64 *tokens, size_t count,
                           size_t at_token, u32 doc, vector<Hit> &hits) const {
    for (size_t a = anchor_starts[id]; a < anchor_starts[id + 1]; a++) {
      auto [phrase, at] = anchored[a];
      const u64 *first = terms.data() + starts[phrase];
      size_t len = starts[phrase + 1] - starts[phrase], after = len - at;
      if (at_token + after <= count + 2 && at <= at_token &&
          std::equal(first + at, first + len, tokens + at_token) &&
          std::equal(first, first + at, tokens + at_token - at))
        hits.push_back({phrase, doc});
    }
  }
};

// One chunk of the query file. Each line's size says how many boolean terms it has,
// with 0 for a phrase, and each boolean term holds its operator in the low bits.
struct ChunkPhrases {
  vector<u8> sizes;
  vector<u32> terms;
  vector<u32> lengths, slots;
  vector<u64> hashes;
};

static vector<size_t> query_chunks(const u8 *text, size_t size) {
  vector<size_t> bounds{0};
  for (size_t at = QUERY_CHUNK; at < size; at += QUERY_CHUNK) {
    auto newline = (const u8 *)memchr(text + at, '\n', size - at);
    at = newline ? newline - text + 1 : size;
    bounds.push_back(at);
  }
  if (bounds.back() < size)
    bounds.push_back(size);
  return bounds;
}

static ChunkPhrases read_chunk(const u8 *text, size_t from, size_t to,
                               Draft &table) {
  ChunkPhrases chunk;
  for (const u8 *p = text + from; p < text + to;) {
    bool is_bool = *p == 'B';
    size_t index = 0, before = chunk.hashes.size();
    u32 op = OP_AND;
    p = scan_line(p, [&](const u8 *word, u64 hash) {
      if (is_bool) {
        if (index % 2)
          chunk.terms.push_back(table.intern(hash) << OP_BITS | op);
        else if (index)
          op = *word == 'A' ? OP_AND : *word == 'O' ? OP_OR : OP_NOT;
      } else if (index) {
        chunk.slots.push_back(table.intern(hash));
        chunk.hashes.push_back(hash);
      }
      index++;
    });
    chunk.sizes.push_back(is_bool ? index / 2 : 0);
    if (!is_bool)
      chunk.lengths.push_back(chunk.hashes.size() - before);
  }
  return chunk;
}

struct Plan {
  Phrases phrases;
  vector<u32> phrase_bases;
  size_t high = 0, low = 0;
};

// Final ids run through the common terms, then those with lists, then anchors used
// only in phrases, so an id alone says how a token is handled. Other terms used
// only in phrases need no table entry, as phrases compare hashes.
static Plan plan_index(const Draft &draft, vector<ChunkPhrases> &chunks,
                       Table &table) {
  Plan plan;
  vector<u16> phrase_seen(draft.len), bool_seen(draft.len);
  size_t phrase_count = 0, term_count = 0;
  for (auto &chunk : chunks) {
    for (u32 at : chunk.slots)
      phrase_seen[at]++;
    for (u32 term : chunk.terms)
      bool_seen[term >> OP_BITS]++;
    plan.phrase_bases.push_back(phrase_count);
    phrase_count += chunk.lengths.size();
    term_count += chunk.hashes.size();
  }
  auto seen = [&](size_t at) { return (u32)phrase_seen[at] + bool_seen[at]; };
  vector<u32> ids(draft.len, ID_MASK), hot, cold;
  for (size_t at = 0; at < draft.len; at++)
    if (bool_seen[at]) {
      bool is_high = seen(at) > HIGH_COUNT;
      (is_high ? plan.high : plan.low)++;
      ids[at] = is_high;
    }
  size_t next_high = 0, next_low = plan.high;
  for (size_t at = 0; at < draft.len; at++)
    if (bool_seen[at]) {
      ids[at] = ids[at] ? next_high++ : next_low++;
      (seen(at) >= HOT_COUNT ? hot : cold).push_back(at);
    }
  size_t next_other = next_low;

  Phrases &phrases = plan.phrases;
  phrases.starts.reserve(phrase_count + 1);
  phrases.terms.reserve(term_count);
  phrases.starts.push_back(0);
  phrases.anchored.resize(phrase_count);
  phrases.pairs.resize(1 << (PAIR_BITS - 6));
  vector<bool> is_anchor(draft.len);
  vector<std::pair<u32, u32>> anchors;
  anchors.reserve(phrase_count);
  for (auto &chunk : chunks) {
    size_t first = 0;
    for (u32 length : chunk.lengths) {
      const u32 *list = chunk.slots.data() + first;
      const u64 *hashes = chunk.hashes.data() + first;
      first += length;
      size_t at =
          std::min_element(list, list + length,
                           [&](u32 a, u32 b) { return seen(a) < seen(b); }) -
          list;
      if (ids[list[at]] == ID_MASK) {
        ids[list[at]] = next_other++;
        cold.push_back(list[at]);
      }
      is_anchor[list[at]] = true;
      anchors.push_back({ids[list[at]], (u32)at});
      size_t one = at + 1 < length ? at : at - 1;
      size_t bit = pair_bit(hashes[one], hashes[one + 1]);
      phrases.pairs[bit / 64] |= 1ull << (bit % 64);
      phrases.terms.insert(phrases.terms.end(), hashes, hashes + length);
      phrases.starts.push_back(phrases.terms.size());
    }
  }
  phrases.anchor_starts.resize(next_other + 2);
  for (auto [id, at] : anchors)
    phrases.anchor_starts[id + 2]++;
  for (size_t i = 2; i < next_other + 2; i++)
    phrases.anchor_starts[i] += phrases.anchor_starts[i - 1];
  for (size_t phrase = 0; phrase < anchors.size(); phrase++) {
    auto [id, at] = anchors[phrase];
    phrases.anchored[phrases.anchor_starts[id + 1]++] = {(u32)phrase, at};
  }

  for (auto &chunk : chunks) {
    for (u32 &term : chunk.terms)
      term = ids[term >> OP_BITS] << OP_BITS | (term & ((1 << OP_BITS) - 1));
    chunk.slots = {};
    chunk.lengths = {};
    chunk.hashes = {};
  }
  table.slots = (u64 *)calloc(draft.len, 8);
  table.mask = draft.mask;
  for (auto *terms : {&hot, &cold})
    for (u32 at : *terms)
      table.insert(draft.hashes[at].load(relaxed),
                   ids[at] | (is_anchor[at] ? ANCHOR_FLAG : 0));
  return plan;
}

alignas(64) static u64 lane_orders[256];

// The permutation moving the lanes set in `mask` to the front, in order.
INLINE v4 gather_lanes(u32 mask) {
  return _mm256_cvtepu8_epi32(_mm_loadl_epi64((const v2 *)(lane_orders + mask)));
}

static void fill_lane_orders() {
  for (u32 mask = 0; mask < 256; mask++) {
    u64 order = 0;
    int used = 0;
    for (u64 lane = 0; lane < 8; lane++)
      if (mask >> lane & 1)
        order |= lane << (8 * used++);
    lane_orders[mask] = order;
  }
}

// One thread's buffers, used by every step, so pages are only touched once. One
// document can hold at most half its bytes as tokens, and a list at most all
// documents, so these never need to grow. Only the pages used take memory.
struct Scratch {
  u64 *tokens;
  u32 *ids, *high_ids, *low_ids;
  vector<Hit> hits;
};

static const u8 *input;
static size_t input_size;

static u8 *log_regions;
static size_t log_span;
static Counter *next_log_block;

static u16 *claim_log_block(size_t partition, u32 group) {
  auto block = (u16 *)(log_regions + partition * log_span +
                       next_log_block[partition].take() * LOG_BLOCK);
  *block = MARKER | group;
  return block + 1;
}

static void parse_docs(const Table &table, const Plan &plan, u64 *bits,
                       size_t words, Scratch &out, Counter &next) {
  const u32 high = plan.high, low = plan.low;
  const size_t partitions = plan.low / PARTITION_TERMS + 1;
  vector<u16 *> heads(partitions, nullptr);
  vector<Hit> &hits = out.hits;
  u64 *tokens = out.tokens;
  u32 *ids = out.ids, *high_ids = out.high_ids, *low_ids = out.low_ids;
  auto lines = (u64 *)aligned_alloc(64, (plan.high + 1) * LINE_WORDS * 8);
  memset(lines, 0, (plan.high + 1) * LINE_WORDS * 8);
  u32 group = NO_GROUP;
  auto flush = [&](u32 doc) {
    size_t at = doc / 64 / LINE_WORDS * LINE_WORDS;
    const v4 zero = _mm256_setzero_si256();
    for (size_t row = 0; row < plan.high; row++) {
      u64 *from = lines + row * LINE_WORDS, *to = bits + row * words + at;
      v4 a = _mm256_load_si256((const v4 *)from);
      v4 b = _mm256_load_si256((const v4 *)(from + 4));
      v4 any = _mm256_or_si256(a, b);
      // Terms that barely occur keep their pages untouched.
      if (!_mm256_testz_si256(any, any)) {
        _mm256_storeu_si256((v4 *)to, a);
        _mm256_storeu_si256((v4 *)(to + 4), b);
        _mm256_store_si256((v4 *)from, zero);
        _mm256_store_si256((v4 *)(from + 4), zero);
      }
    }
  };
  const u8 *file_end = input + input_size;
  for (;;) {
    size_t off = next.take(CHUNK);
    if (off >= input_size)
      break;
    const u8 *p = input + off;
    const u8 *stop = input + std::min(off + CHUNK, input_size);
    // Claims start at a document that starts a block, so blocks have one owner.
    if (off) {
      p = (const u8 *)memchr(p - 1, '\n', file_end - p + 1) + 1;
      if (p < file_end) {
        const u8 *q = p;
        for (u32 skip = -parse_id(q) & (LINE_DOCS - 1); skip && p < file_end; skip--)
          p = (const u8 *)memchr(p, '\n', file_end - p) + 1;
      }
    }
    while (p < file_end) {
      u32 doc = parse_id(p);
      if (p - 1 >= stop && doc % LINE_DOCS == 0)
        break;
      auto push = [&](size_t partition, u16 entry) {
        u16 *head = heads[partition];
        *head++ = entry;
        if (!((size_t)head & (LOG_BLOCK - 1)))
          head = claim_log_block(partition, group);
        heads[partition] = head;
      };
      if (doc >> GROUP_BITS != group) {
        group = doc >> GROUP_BITS;
        for (size_t partition = 0; partition < partitions; partition++)
          if (heads[partition])
            push(partition, MARKER | group);
          else
            heads[partition] = claim_log_block(partition, group);
      }
      u32 position = doc & (GROUP_DOCS - 1);
      u64 bit = 1ull << position;
      u64 *line = lines + (doc >> GROUP_BITS) % LINE_WORDS;
      // Tokenizing, lookups and actions are separate passes. Lookups are prefetched
      // far enough ahead for memory to arrive in time. Actions run per kind of
      // token after a vector step sorts ids by kind, as a token's kind is
      // unpredictable and a branch on it would be mispredicted constantly.
      tokens[0] = 0;
      u64 *token = tokens + 1;
      const u8 *end = scan_line(p, [&](const u8 *, u64 hash) { *token++ = hash; });
      size_t count = token - tokens - 1;
      for (size_t i = 0; i <= PREFETCH_AHEAD; i++)
        token[i] = 0;
      for (size_t at = 1; at <= count; at++) {
        table.prefetch(tokens[at + PREFETCH_AHEAD]);
        ids[at] = table.find(tokens[at]);
      }
      for (size_t i = 1; i <= 8; i++)
        ids[count + i] = ID_MASK;
      const v4 id_mask = _mm256_set1_epi32(ID_MASK);
      const v4 high_limit = _mm256_set1_epi32(high);
      const v4 list_limit = _mm256_set1_epi32(high + low);
      size_t highs = 0, lows = 0;
      for (size_t at = 1; at <= count; at += 8) {
        v4 x = _mm256_loadu_si256((const v4 *)(ids + at));
        v4 id = _mm256_and_si256(x, id_mask);
        u32 is_high = _mm256_movemask_ps(
            _mm256_castsi256_ps(_mm256_cmpgt_epi32(high_limit, id)));
        u32 is_low = is_high ^ _mm256_movemask_ps(_mm256_castsi256_ps(
                                   _mm256_cmpgt_epi32(list_limit, id)));
        _mm256_storeu_si256((v4 *)(high_ids + highs),
                            _mm256_permutevar8x32_epi32(id, gather_lanes(is_high)));
        _mm256_storeu_si256((v4 *)(low_ids + lows),
                            _mm256_permutevar8x32_epi32(id, gather_lanes(is_low)));
        highs += __builtin_popcount(is_high);
        lows += __builtin_popcount(is_low);
        u32 anchors = _mm256_movemask_ps(_mm256_castsi256_ps(
            _mm256_slli_epi32(x, 31 - ID_BITS)));
        for (; anchors; anchors &= anchors - 1) {
          size_t i = at + __builtin_ctz(anchors);
          u64 hash = tokens[i];
          if (plan.phrases.may_pair(hash, tokens[i + 1]) ||
              plan.phrases.may_pair(tokens[i - 1], hash))
            plan.phrases.check(ids[i] & ID_MASK, tokens, count, i, doc, hits);
        }
      }
      for (size_t i = 0; i < highs; i++)
        line[high_ids[i] * LINE_WORDS] |= bit;
      for (size_t i = 0; i < lows; i++) {
        u32 term = low_ids[i] - high;
        push(term >> PARTITION_BITS,
             (term & (PARTITION_TERMS - 1)) << GROUP_BITS | position);
      }
      p = end;
      if (doc % LINE_DOCS == LINE_DOCS - 1)
        flush(doc);
      else if (p == file_end)
        flush(doc);
    }
  }
  for (u16 *head : heads)
    if (head)
      *head = LOG_END;
  free(lines);
}

struct Lists {
  const u32 *marks;
  size_t blocks;

  size_t size(size_t term) const {
    const u32 *m = marks + term * (blocks + 1);
    return m[blocks] - m[0];
  }

  size_t decode(size_t term, u32 *out) const {
    const u32 *m = marks + term * (blocks + 1);
    auto ids = (const u16 *)(log_regions + (term >> PARTITION_BITS) * log_span);
    size_t n = 0;
    for (size_t block = 0; block < blocks; block++) {
      u32 top = (u32)block << BLOCK_BITS;
      for (u32 i = m[block]; i < m[block + 1]; i++)
        out[n++] = top | ids[i];
    }
    return n;
  }
};

// One thread turns each partition's log into its lists. Runs of hits are visited
// by group, so every list comes out sorted without sorting. The lists are written
// to a private image, then copied over the log they came from, so log and lists
// never both take memory.
struct ListBuilder {
  u32 *marks;
  size_t terms, blocks, groups;
  u16 *image;
  vector<u32> runs, order, group_starts;

  void build(size_t p) {
    auto log = (u16 *)(log_regions + p * log_span);
    size_t used = next_log_block[p].value.load(relaxed) * LOG_BLOCK;
    u32 counts[PARTITION_TERMS] = {};
    runs.clear();
    group_starts.assign(groups + 2, 0);
    for (size_t i = 0; i < used / 2;) {
      u16 entry = log[i];
      if (entry < MARKER) {
        counts[entry >> GROUP_BITS]++;
      } else if (entry == LOG_END) {
        i = (i | (LOG_BLOCK / 2 - 1)) + 1;
        continue;
      } else {
        runs.push_back(i);
        group_starts[(entry & ~MARKER) + 2]++;
      }
      i++;
    }
    for (size_t g = 2; g < groups + 2; g++)
      group_starts[g] += group_starts[g - 1];
    order.resize(runs.size());
    for (u32 run : runs)
      order[group_starts[(log[run] & ~MARKER) + 1]++] = run;

    u32 cursors[PARTITION_TERMS], sum = 0;
    for (size_t term = 0; term < PARTITION_TERMS; term++) {
      cursors[term] = sum;
      sum += counts[term];
    }
    const size_t first_term = p << PARTITION_BITS;
    const size_t last_term = std::min(first_term + PARTITION_TERMS, terms);
    auto mark = [&](size_t block) {
      for (size_t term = first_term; term < last_term; term++)
        marks[term * (blocks + 1) + block] = cursors[term - first_term];
    };
    mark(0);
    size_t block = 0;
    u32 last[PARTITION_TERMS];
    memset(last, 255, sizeof last);
    for (u32 run : order) {
      u32 group = log[run] & ~MARKER;
      size_t run_block = group >> GROUPS_PER_BLOCK_BITS;
      while (block < run_block) {
        mark(++block);
        memset(last, 255, sizeof last);
      }
      u32 group_low = (group & ((1 << GROUPS_PER_BLOCK_BITS) - 1)) << GROUP_BITS;
      for (u32 i = run + 1; log[i] < MARKER; i++) {
        size_t term = log[i] >> GROUP_BITS;
        u32 low = group_low | (log[i] & (GROUP_DOCS - 1));
        // A term repeated within a document appears once per token.
        image[cursors[term]] = low;
        cursors[term] += last[term] != low;
        last[term] = low;
      }
    }
    while (block < blocks)
      mark(++block);
    memcpy(log, image, sum * 2);
    size_t kept = (sum * 2 + PAGE - 1) & -PAGE;
    if (used > kept)
      madvise((u8 *)log + kept, used - kept, MADV_DONTNEED);
  }
};

struct Index {
  size_t high, words;
  const u64 *bits;
  Lists lists;
};

static void push_int(std::string &out, u64 value) {
  char digits[20];
  out.append(digits, std::to_chars(digits, digits + 20, value).ptr);
}

static void write_line(std::string &out, u64 count, u64 sum, const u32 *first) {
  push_int(out, count);
  out += ' ';
  push_int(out, sum);
  for (u64 i = 0; i < std::min<u64>(count, 10); i++) {
    out += ' ';
    push_int(out, first[i]);
  }
  out += '\n';
}

// Counts a bitset's set bits and sums their positions. Nibble lookups give each
// byte's count and the sum of positions inside it, and byte and word weights
// turn those into positions in the whole set: about 5 micro-ops per word, against
// 20 for six masked population counts. `n` is a multiple of 4 words.
static void count_bits(const u64 *words, size_t n, u64 &count, u64 &sum) {
  const v4 pops = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, //
                                   0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  const v4 sums = _mm256_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, //
                                   0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6);
  const v4 weights = _mm256_set1_epi64x(0x3830282018100800);
  const v4 nibble = _mm256_set1_epi8(0x0F), ones = _mm256_set1_epi16(1);
  const v4 zero = _mm256_setzero_si256();
  v4 running = zero, weighted = zero, inside = zero;
  for (size_t i = 0; i < n; i += 4) {
    v4 x = _mm256_loadu_si256((const v4 *)(words + i));
    v4 lo = _mm256_and_si256(x, nibble);
    v4 hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), nibble);
    v4 pop_hi = _mm256_shuffle_epi8(pops, hi);
    v4 pop = _mm256_add_epi8(_mm256_shuffle_epi8(pops, lo), pop_hi);
    v4 pos = _mm256_add_epi8(
        _mm256_add_epi8(_mm256_shuffle_epi8(sums, lo),
                        _mm256_shuffle_epi8(sums, hi)),
        _mm256_slli_epi16(pop_hi, 2));
    inside = _mm256_add_epi64(inside, _mm256_sad_epu8(pos, zero));
    inside = _mm256_add_epi32(
        inside, _mm256_madd_epi16(_mm256_maddubs_epi16(weights, pop), ones));
    running = _mm256_add_epi64(running, _mm256_sad_epu8(pop, zero));
    weighted = _mm256_add_epi64(weighted, running);
  }
  alignas(32) u64 per_lane[4], later[4], rest[4];
  _mm256_store_si256((v4 *)per_lane, running);
  _mm256_store_si256((v4 *)later, weighted);
  _mm256_store_si256((v4 *)rest, inside);
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
template <u8 OP>
static size_t merge(const u32 *a, size_t na, const u32 *b, size_t nb, u32 *out) {
  size_t i = 0, j = 0, n = 0;
  while (i < na && j < nb) {
    u32 x = a[i], y = b[j];
    out[n] = std::min(x, y);
    n += OP == 'O' || (OP == 'A' ? x == y : x < y);
    i += x <= y;
    j += y <= x;
  }
  if (OP != 'A')
    while (i < na)
      out[n++] = a[i++];
  if (OP == 'O')
    while (j < nb)
      out[n++] = b[j++];
  return n;
}

// For lists of very different sizes. Each id of the short list skips ahead in the
// long one by whole vectors and compares 8 at once. `b` is followed by 8 ids above
// any document.
template <bool KEEP>
static size_t probe(const u32 *a, size_t na, const u32 *b, size_t nb, u32 *out) {
  size_t j = 0, n = 0;
  for (size_t i = 0; i < na; i++) {
    u32 x = a[i];
    while (j + 64 < nb && b[j + 63] < x)
      j += 64;
    while (b[j + 7] < x)
      j += 8;
    v4 equal = _mm256_cmpeq_epi32(_mm256_loadu_si256((const v4 *)(b + j)),
                                  _mm256_set1_epi32(x));
    out[n] = x;
    n += (_mm256_movemask_epi8(equal) != 0) == KEEP;
  }
  return n;
}

// Answers one boolean query. Filters between two ORs commute, so within such a run
// the smallest list goes first and NOT comes last, which leaves little to compare.
struct Evaluator {
  const Index &index;
  u32 *r, *tmp, *l;
  u64 *bits;
  size_t n = 0;
  bool dense = false;

  u64 term_size(u32 id) const {
    return id < index.high ? UINT32_MAX : index.lists.size(id - index.high);
  }

  void to_dense(const u64 *start) {
    if (start)
      memcpy(bits, start, index.words * 8);
    else
      memset(bits, 0, index.words * 8);
    for (size_t i = 0; i < n; i++)
      bits[r[i] / 64] |= 1ull << (r[i] % 64);
    dense = true;
  }

  void start(u32 id) {
    n = 0;
    dense = false;
    if (id < index.high)
      to_dense(index.bits + id * index.words);
    else
      n = index.lists.decode(id - index.high, r);
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
    if (id < index.high) {
      const u64 *row = index.bits + id * index.words;
      if (dense) {
        for (size_t i = 0; i < index.words; i += 4) {
          v4 x = _mm256_loadu_si256((const v4 *)(bits + i));
          v4 y = _mm256_loadu_si256((const v4 *)(row + i));
          _mm256_storeu_si256((v4 *)(bits + i),
                              op == OP_AND  ? _mm256_and_si256(x, y)
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
    size_t nl = index.lists.decode(id - index.high, l);
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
    if (op == OP_OR) {
      // A long list of ids takes more room than a bitset.
      if (n + nl > index.words * 2) {
        to_dense(nullptr);
        apply(op, id);
        return;
      }
      n = merge<'O'>(r, n, l, nl, tmp);
    } else {
      for (int i = 0; i < 8; i++)
        l[nl + i] = r[n + i] = UINT32_MAX;
      if (n * PROBE_RATIO < nl)
        n = op == OP_AND ? probe<true>(r, n, l, nl, tmp)
                         : probe<false>(r, n, l, nl, tmp);
      else if (op == OP_AND && nl * PROBE_RATIO < n)
        n = probe<true>(l, nl, r, n, tmp);
      else
        n = op == OP_AND ? merge<'A'>(r, n, l, nl, tmp)
                         : merge<'N'>(r, n, l, nl, tmp);
    }
    std::swap(r, tmp);
  }

  // `terms` holds each term's id above its operator, with AND for the first term.
  void answer(u32 *terms, size_t count, std::string &out) {
    auto op = [](u32 term) { return term & ((1 << OP_BITS) - 1); };
    auto key = [&](u32 term) {
      return term_size(term >> OP_BITS) | (u64)(op(term) == OP_NOT) << 32;
    };
    for (size_t i = 0; i < count;) {
      size_t end = i + 1;
      while (end < count && op(terms[end]) != OP_OR)
        end++;
      std::sort(terms + i + (i > 0), terms + end,
                [&](u32 a, u32 b) { return key(a) < key(b); });
      i = end;
    }
    start(terms[0] >> OP_BITS);
    for (size_t i = 1; i < count; i++)
      apply(op(terms[i]), terms[i] >> OP_BITS);
    u32 first[10];
    u64 total = n, sum = 0;
    if (dense) {
      count_bits(bits, index.words, total, sum);
      u32 m = 0;
      for (size_t w = 0; w < index.words && m < 10; w++)
        for (u64 rest = bits[w]; rest && m < 10; rest &= rest - 1)
          first[m++] = w * 64 + __builtin_ctzll(rest);
    } else {
      for (size_t i = 0; i < n; i++)
        sum += r[i];
      std::copy_n(r, std::min<size_t>(n, 10), first);
    }
    write_line(out, total, sum, first);
  }
};

static const u8 *text;
static size_t text_size, docs, words, chunks;
static vector<size_t> bounds;
static u32 *list_marks;
static vector<ChunkPhrases> phrase_chunks;
static Plan plan;
static Draft *draft;
static Table table;
static u64 *bits;
static vector<Scratch> scratches;
static Index index_all;
static vector<u32> matched, phrase_docs;
static vector<std::string> outputs;
static Counter next_query, next_chunk, next_unmap, next_partition, next_answer;
static Counter next_release;

// Runs once on one thread each time all threads have arrived at the barrier.
static void between_steps() noexcept {
  static int step = 0;
  if (step == 0) {
    plan = plan_index(*draft, phrase_chunks, table);
    free(draft->hashes);
    bits = (u64 *)reserve(plan.high * words * 8 + 64);
    size_t partitions = plan.low / PARTITION_TERMS + 1;
    // Each token takes at least 2 bytes of input and 2 bytes here.
    log_span = (input_size + (1 << 24)) & -PAGE;
    log_regions = reserve(partitions * log_span);
    next_log_block = new Counter[partitions];
    list_marks = (u32 *)calloc((plan.low + 1) * ((docs >> BLOCK_BITS) + 2), 4);
  } else if (step == 1) {
    index_all = {plan.high, words, bits, {list_marks, (docs >> BLOCK_BITS) + 1}};
    // A document can match a phrase more than once, but it counts once. Answers
    // sort each phrase's few documents.
    size_t phrase_count = plan.phrases.starts.size() - 1, total = 0;
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
    plan.phrases = {};
    free(table.slots);
    munmap((void *)input, input_size + 4096);
  } else if (step == 2) {
  } else if (step == 3) {
  } else {
    int fd = open("/out/results.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    for (auto &out : outputs)
      if (write(fd, out.data(), out.size()) != (ssize_t)out.size())
        abort();
    _exit(0);
  }
  step++;
}

static std::barrier<void (*)() noexcept> *sync_all;

static void answer_queries(Scratch &scratch) {
  Evaluator evaluator{index_all, scratch.ids, scratch.high_ids, scratch.low_ids,
                      scratch.tokens};
  for (size_t c; (c = next_answer.take()) < chunks;) {
    std::string &out = outputs[c];
    out.reserve(bounds[c + 1] - bounds[c]);
    size_t phrase = plan.phrase_bases[c];
    u32 *terms = phrase_chunks[c].terms.data();
    for (u8 size : phrase_chunks[c].sizes) {
      if (size) {
        evaluator.answer(terms, size, out);
        terms += size;
      } else {
        u32 *first = phrase_docs.data() + matched[phrase];
        u32 *end = phrase_docs.data() + matched[phrase + 1];
        std::sort(first, end);
        end = std::unique(first, end);
        u64 sum = 0;
        for (u32 *at = first; at < end; at++)
          sum += *at;
        write_line(out, end - first, sum, first);
        phrase++;
      }
    }
  }
}

static void work(int t) {
  Scratch &scratch = scratches[t];
  size_t most = input_size / 2 + (1 << 16);
  scratch.tokens = (u64 *)reserve(most * 8);
  scratch.ids = (u32 *)reserve(most * 4);
  scratch.high_ids = (u32 *)reserve(most * 4);
  scratch.low_ids = (u32 *)reserve(most * 4);
  for (size_t c; (c = next_query.take()) < chunks;)
    phrase_chunks[c] = read_chunk(text, bounds[c], bounds[c + 1], *draft);
  sync_all->arrive_and_wait();
  parse_docs(table, plan, bits, words, scratch, next_chunk);
  // Exit would otherwise clear the input's page tables page by page on one
  // thread, with the clock still running.
  for (size_t off; (off = next_unmap.take(UNMAP_STEP)) < input_size;)
    madvise((void *)(input + off), std::min(UNMAP_STEP, input_size - off),
            MADV_DONTNEED);
  sync_all->arrive_and_wait();
  const size_t partitions = plan.low / PARTITION_TERMS + 1;
  {
    ListBuilder builder{list_marks, plan.low, (docs >> BLOCK_BITS) + 1,
                        docs / GROUP_DOCS + 1, (u16 *)scratch.tokens};
    for (size_t p; (p = next_partition.take()) < partitions;)
      builder.build(p);
    // Bitsets take over these pages while answering.
    madvise(scratch.tokens, most * 8, MADV_DONTNEED);
  }
  sync_all->arrive_and_wait();
  answer_queries(scratch);
  sync_all->arrive_and_wait();
  // Nothing but the answers is needed now. Giving it all back here, on all threads,
  // lowers the peak as the output grows, and leaves exit almost nothing to do.
  for (size_t p; (p = next_release.take()) <= partitions;)
    if (p == partitions)
      madvise(bits, plan.high * words * 8, MADV_DONTNEED);
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
  mallopt(M_ARENA_MAX, 1);
  memset(low_bytes, 255, 32);
  cpu_set_t cpus;
  sched_getaffinity(0, sizeof cpus, &cpus);
  threads = CPU_COUNT(&cpus);
  text = map_input("/data/queries.txt", text_size);
  bounds = query_chunks(text, text_size);
  chunks = bounds.size() - 1;
  draft = new Draft(text_size);
  fill_lane_orders();
  phrase_chunks.resize(chunks);
  outputs.resize(chunks);
  input = map_input("/data/docs.txt", input_size);
  if (input_size) {
    const u8 *last = input + input_size - 1;
    while (last > input && last[-1] != '\n')
      last--;
    docs = parse_id(last) + 1;
  }
  words = (docs / 64 | (LINE_WORDS - 1)) + 1;
  scratches.resize(threads);
  std::barrier<void (*)() noexcept> barrier(threads, between_steps);
  sync_all = &barrier;
  vector<std::thread> pool;
  for (int t = 1; t < threads; t++)
    pool.emplace_back(work, t);
  work(0);
}
