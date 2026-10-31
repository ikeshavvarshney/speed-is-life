#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <malloc.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using v4 = __m256i;
using v2 = __m128i;
using std::atomic;
using std::vector;
constexpr auto relaxed = std::memory_order_relaxed;
#define INLINE [[gnu::always_inline]] inline

const size_t CHUNK = 1 << 18, QUERY_CHUNK = 1 << 14, PREFETCH_AHEAD = 12;
const size_t SAMPLE_SHARE = 512;
// A term in over 1/32 of documents is smaller as a bitset than in the hit log.
const size_t HIGH_SHARE = 32;

// A table slot holds the high bits of the term's hash, an anchor flag and the
// term's id. While queries are read, the lowest bit instead records that a
// boolean query uses the term.
const int ID_BITS = 18;
const u64 ID_MASK = (1 << ID_BITS) - 1, IN_BOOL = 1, ANCHOR_FLAG = 1 << ID_BITS;
const u64 TAG_ONE = ANCHOR_FLAG << 1, TAG_MASK = ~(TAG_ONE - 1);
const size_t TABLE_PADDING = 1 << 10;

// A logged hit is 16 bits: a term within its partition, and a document within
// its group of 64. A worker writes a marker naming the group when it starts one.
const int GROUP_BITS = 6, PARTITION_BITS = 9;
const u32 GROUP_DOCS = 1 << GROUP_BITS, NO_GROUP = UINT32_MAX;
const size_t PARTITION_TERMS = 1 << PARTITION_BITS, BUCKET_START = 1 << 15;
const u16 MARKER = 1 << 15;

// Document ids in lists are stored as 16 bits within a block of 65,536 documents.
const int BLOCK_BITS = 16, GROUPS_PER_BLOCK_BITS = BLOCK_BITS - GROUP_BITS;

const u64 HASH_0 = 0x2D358DCCAA6C78A5, HASH_1 = 0x8BB84B93962EACC9;
const u64 HASH_2 = 0x4B33A62ED433D4A3, HASH_3 = 0xC2B2AE3D27D4EB4F;
const int PAIR_BITS = 20;

struct Hit {
  u32 phrase, doc;
  auto operator<=>(const Hit &) const = default;
};
struct Log {
  u16 *data;
  size_t len;
};

static int threads;
static u8 low_bytes[64];

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

INLINE u64 mum(u64 a, u64 b) {
  auto product = (unsigned __int128)a * b;
  return (u64)product ^ (u64)(product >> 64);
}

[[gnu::cold]] static u64 long_token_hash(const u8 *start, size_t len) {
  if (len > 32)
    return 0;
  v4 bytes = _mm256_or_si256(_mm256_loadu_si256((const v4 *)start),
                             _mm256_set1_epi8(0x20));
  v4 kept = _mm256_and_si256(
      bytes, _mm256_loadu_si256((const v4 *)(low_bytes + 32 - len)));
  u64 w[4];
  _mm256_storeu_si256((v4 *)w, kept);
  return mum(w[0] ^ HASH_0, w[1] ^ HASH_1) ^ mum(w[2] ^ HASH_2, w[3] ^ HASH_3);
}

// Hashes a token where it lies. Token bytes are letters and digits, so OR-ing
// 0x20 lowercases them, and masking removes whatever follows. Terms are at most
// 32 bytes, so a longer token hashes to a value no term has. The hash stands in
// for the bytes: with about 600,000 distinct tokens and 76,000 terms, the chance
// of any collision is about 1 in 10 million.
INLINE u64 token_hash(const u8 *start, size_t len) {
  if (len > 16)
    return long_token_hash(start, len);
  v2 bytes = _mm_or_si128(_mm_loadu_si128((const v2 *)start),
                          _mm_set1_epi8(0x20));
  v2 kept = _mm_and_si128(
      bytes, _mm_loadu_si128((const v2 *)(low_bytes + 32 - len)));
  return mum(_mm_cvtsi128_si64(kept) ^ HASH_0,
             _mm_extract_epi64(kept, 1) ^ HASH_1);
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

// Runs `work` on all threads, which claim numbers below `count` until none are
// left.
template <class F> static void on_all_threads(size_t count, F &&work) {
  atomic<size_t> next{0};
  auto run = [&] {
    for (size_t i; (i = next.fetch_add(1, relaxed)) < count;)
      work(i);
  };
  vector<std::thread> pool;
  for (int t = 1; t < threads; t++)
    pool.emplace_back(run);
  run();
  for (auto &thread : pool)
    thread.join();
}

// The one lookup table, about 1 MB, with everything in an 8-byte slot. A slot is
// found by the hash's low bits and confirmed by its high bits. Query reading fills
// it from all threads.
struct Table {
  atomic<u64> *slots;
  size_t mask, len;

  explicit Table(size_t query_bytes) {
    size_t size = std::max<size_t>(std::bit_ceil(query_bytes / 16 + 16), 1 << 10);
    len = size + TABLE_PADDING;
    slots = (atomic<u64> *)calloc(len, 8);
    mask = size - 1;
  }

  u32 intern(u64 hash, u64 flag) {
    u64 tag = (hash | TAG_ONE) & TAG_MASK;
    for (size_t at = hash & mask;; at++) {
      u64 value = 0;
      if (slots[at].compare_exchange_strong(value, tag | flag, relaxed))
        return at;
      if ((value & TAG_MASK) == tag) {
        if ((value & flag) != flag)
          slots[at].fetch_or(flag, relaxed);
        return at;
      }
    }
  }

  INLINE void prefetch(u64 hash) const {
    _mm_prefetch((const char *)(slots + (hash & mask)), _MM_HINT_T0);
  }

  // The hash's slot, or -1 if it has none.
  INLINE long position(u64 hash) const {
    u64 tag = (hash | TAG_ONE) & TAG_MASK;
    for (size_t at = hash & mask;; at++) {
      u64 value = slots[at].load(relaxed);
      if ((value & TAG_MASK) == tag)
        return at;
      if (!value)
        return -1;
    }
  }

  // Compares four slots at once. Tokens missing from the table are unpredictable,
  // so no branch decides the result, which is then an id no list or bitset uses.
  INLINE u64 find(u64 hash) const {
    u64 tag = (hash | TAG_ONE) & TAG_MASK;
    const v4 tags = _mm256_set1_epi64x(tag);
    const v4 tag_mask = _mm256_set1_epi64x(TAG_MASK);
    for (size_t at = hash & mask;; at += 4) {
      auto group_at = (const u64 *)(slots + at);
      v4 group = _mm256_loadu_si256((const v4 *)group_at);
      v4 equal = _mm256_cmpeq_epi64(_mm256_and_si256(group, tag_mask), tags);
      v4 empty = _mm256_cmpeq_epi64(group, _mm256_setzero_si256());
      u32 hits = _mm256_movemask_pd(_mm256_castsi256_pd(equal));
      if (hits | _mm256_movemask_pd(_mm256_castsi256_pd(empty))) {
        u64 found = -(u64)(hits != 0);
        return (group_at[__builtin_ctz(hits | 16)] & found) |
               (ID_MASK & ~found);
      }
    }
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
  [[gnu::cold]] void check(size_t id, const vector<u64> &tokens, size_t at_token,
                           u32 doc, vector<Hit> &hits) const {
    for (size_t a = anchor_starts[id]; a < anchor_starts[id + 1]; a++) {
      auto [phrase, at] = anchored[a];
      const u64 *first = terms.data() + starts[phrase];
      size_t len = starts[phrase + 1] - starts[phrase], after = len - at;
      if (at_token + after <= tokens.size() && at <= at_token &&
          std::equal(first + at, first + len, tokens.begin() + at_token) &&
          std::equal(first, first + at, tokens.begin() + at_token - at))
        hits.push_back({phrase, doc});
    }
  }
};

struct ChunkPhrases {
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

// Records every term in the table. Boolean queries need nothing else now, as they
// are read again from the text when they are answered.
static ChunkPhrases read_chunk(const u8 *text, size_t from, size_t to,
                               Table &table) {
  ChunkPhrases chunk;
  for (const u8 *p = text + from; p < text + to;) {
    bool is_bool = *p == 'B';
    size_t index = 0, before = chunk.hashes.size();
    p = scan_line(p, [&](const u8 *, u64 hash) {
      if (is_bool) {
        if (index % 2)
          table.intern(hash, IN_BOOL);
      } else if (index) {
        chunk.slots.push_back(table.intern(hash, 0));
        chunk.hashes.push_back(hash);
      }
      index++;
    });
    if (!is_bool)
      chunk.lengths.push_back(chunk.hashes.size() - before);
  }
  return chunk;
}

// Counts tokens rather than documents, which is close enough to decide how to
// store a term.
static vector<u16> sample_counts(const u8 *input, const Table &table,
                                 size_t docs) {
  vector<u16> seen(table.len);
  const u8 *p = input;
  for (size_t i = 0; i < docs / SAMPLE_SHARE; i++) {
    parse_id(p);
    p = scan_line(p, [&](const u8 *, u64 hash) {
      long at = table.position(hash);
      if (at >= 0 && seen[at] < UINT16_MAX)
        seen[at]++;
    });
  }
  return seen;
}

struct Plan {
  Phrases phrases;
  vector<u32> phrase_bases;
  size_t high = 0, low = 0;
};

// Final ids run through the common terms, then those with lists, then phrase-only
// terms, so an id alone says how a token is handled.
static Plan plan_index(Table &table, const vector<ChunkPhrases> &chunks,
                       const vector<u16> &seen, size_t sample) {
  Plan plan;
  auto threshold = (u16)std::max<size_t>(sample / HIGH_SHARE, 1);
  auto slots = (u64 *)table.slots;
  size_t terms = 0;
  for (size_t at = 0; at < table.len; at++)
    if (slots[at]) {
      terms++;
      if (slots[at] & IN_BOOL)
        (seen[at] > threshold ? plan.high : plan.low)++;
    }
  size_t next_high = 0, next_low = plan.high, next_other = plan.high + plan.low;
  for (size_t at = 0; at < table.len; at++)
    if (slots[at]) {
      size_t &next = !(slots[at] & IN_BOOL)  ? next_other
                     : seen[at] > threshold ? next_high
                                            : next_low;
      slots[at] = (slots[at] & TAG_MASK) | next++;
    }

  Phrases &phrases = plan.phrases;
  size_t phrase_count = 0;
  for (auto &chunk : chunks) {
    plan.phrase_bases.push_back(phrase_count);
    phrase_count += chunk.lengths.size();
  }
  phrases.starts.push_back(0);
  phrases.anchor_starts.resize(terms + 2);
  phrases.anchored.resize(phrase_count);
  phrases.pairs.resize(1 << (PAIR_BITS - 6));
  vector<std::pair<u32, u32>> anchors;
  for (auto &chunk : chunks) {
    size_t first = 0;
    for (u32 length : chunk.lengths) {
      const u32 *list = chunk.slots.data() + first;
      const u64 *hashes = chunk.hashes.data() + first;
      first += length;
      size_t at =
          std::min_element(list, list + length,
                           [&](u32 a, u32 b) { return seen[a] < seen[b]; }) -
          list;
      slots[list[at]] |= ANCHOR_FLAG;
      u32 id = slots[list[at]] & ID_MASK;
      phrases.anchor_starts[id + 2]++;
      anchors.push_back({id, (u32)at});
      size_t one = at + 1 < length ? at : at - 1;
      size_t bit = pair_bit(hashes[one], hashes[one + 1]);
      phrases.pairs[bit / 64] |= 1ull << (bit % 64);
      phrases.terms.insert(phrases.terms.end(), hashes, hashes + length);
      phrases.starts.push_back(phrases.terms.size());
    }
  }
  for (size_t i = 2; i < terms + 2; i++)
    phrases.anchor_starts[i] += phrases.anchor_starts[i - 1];
  for (size_t phrase = 0; phrase < anchors.size(); phrase++) {
    auto [id, at] = anchors[phrase];
    phrases.anchored[phrases.anchor_starts[id + 1]++] = {(u32)phrase, at};
  }
  return plan;
}

// A growable log written through raw pointers, so a write costs one store and one
// compare.
struct Bucket {
  u16 *log, *head, *end;

  explicit Bucket(size_t capacity) {
    head = log = (u16 *)malloc(capacity * 2);
    end = log + capacity;
  }

  INLINE void push(u16 entry) {
    *head++ = entry;
    if (head == end)
      grow();
  }

  [[gnu::cold]] void grow() {
    size_t len = head - log;
    log = (u16 *)realloc(log, len * 4);
    head = log + len;
    end = log + len * 2;
  }
};

struct Worker {
  vector<Log> buckets;
  vector<Hit> hits;
};

static const u8 *input;
static size_t input_size;

static void parse_docs(const Table &table, const Plan &plan, u64 *bits,
                       size_t words, Worker &out, atomic<size_t> &next) {
  const u64 high = plan.high, low = plan.low;
  const size_t partitions = plan.low / PARTITION_TERMS + 1, sink = partitions;
  // One extra bucket takes the writes of tokens that need no list. It is never
  // read and costs only one page, as its head is reset often.
  vector<Bucket> buckets(partitions + 1, Bucket(0));
  for (auto &bucket : buckets)
    bucket = Bucket(BUCKET_START);
  vector<Hit> &hits = out.hits;
  vector<u64> tokens;
  tokens.reserve(1 << 12);
  // Common terms set one bit per token, so 64 documents are gathered locally.
  vector<u64> columns(plan.high + 1);
  u32 group = NO_GROUP;
  auto flush = [&] {
    for (size_t row = 0; row < plan.high; row++)
      if (columns[row]) {
        std::atomic_ref(bits[row * words + group])
            .fetch_or(columns[row], relaxed);
        columns[row] = 0;
      }
  };
  for (;;) {
    size_t off = next.fetch_add(CHUNK, relaxed);
    if (off >= input_size)
      break;
    size_t i = off;
    if (off)
      while (input[i - 1] != '\n')
        i++;
    size_t stop = std::min(off + CHUNK, input_size);
    while (i < stop) {
      const u8 *p = input + i;
      u32 doc = parse_id(p);
      if (doc >> GROUP_BITS != group) {
        if (group != NO_GROUP)
          flush();
        group = doc >> GROUP_BITS;
        for (auto &bucket : buckets)
          bucket.push(MARKER | group);
        buckets[sink].head = buckets[sink].log;
      }
      u32 position = doc & (GROUP_DOCS - 1);
      u64 bit = 1ull << position;
      // Tokenizing and lookups are separate passes, so each lookup can be
      // prefetched far enough ahead for memory to arrive in time.
      tokens.clear();
      tokens.push_back(0);
      const u8 *end =
          scan_line(p, [&](const u8 *, u64 hash) { tokens.push_back(hash); });
      size_t count = tokens.size() - 1;
      tokens.resize(count + 2 + PREFETCH_AHEAD);
      for (size_t at = 1; at <= count; at++) {
        table.prefetch(tokens[at + PREFETCH_AHEAD]);
        u64 hash = tokens[at], slot = table.find(hash), id = slot & ID_MASK;
        if ((slot & ANCHOR_FLAG) && (plan.phrases.may_pair(hash, tokens[at + 1]) ||
                                     plan.phrases.may_pair(tokens[at - 1], hash)))
          plan.phrases.check(id, tokens, at, doc, hits);
        // A token's kind is unpredictable, so every token does both writes, and
        // tokens of the other kinds land in a slot nothing reads.
        columns[std::min(id, high)] |= bit;
        u64 term = id - high;
        size_t target = term < low ? term >> PARTITION_BITS : sink;
        buckets[target].push((term & (PARTITION_TERMS - 1)) << GROUP_BITS |
                             position);
      }
      i = end - input;
    }
  }
  if (group != NO_GROUP)
    flush();
  free(buckets[sink].log);
  buckets.pop_back();
  for (auto &bucket : buckets)
    out.buckets.push_back({bucket.log, (size_t)(bucket.head - bucket.log)});
}

struct Lists {
  u16 *ids;
  u32 *marks;
  size_t blocks;

  void decode(size_t term, vector<u32> &out) const {
    out.clear();
    for (size_t block = 0; block < blocks; block++) {
      size_t at = term * (blocks + 1) + block;
      u32 top = (u32)block << BLOCK_BITS;
      for (u32 i = marks[at]; i < marks[at + 1]; i++)
        out.push_back(top | ids[i]);
    }
  }
};

struct Segment {
  u32 group;
  u16 first;
  u32 bucket, start, end;
};

// One thread builds each partition of terms inside its cache, and frees its logs
// as it goes. Runs of hits are visited by group, so every list comes out sorted
// without sorting.
static Lists build_lists(const vector<Worker> &workers, size_t terms,
                         size_t docs) {
  const size_t partitions = terms / PARTITION_TERMS + 1;
  vector<size_t> bases(partitions + 1);
  for (size_t p = 0; p < partitions; p++) {
    bases[p + 1] = bases[p];
    for (auto &worker : workers)
      bases[p + 1] += worker.buckets[p].len;
  }
  const size_t blocks = (docs >> BLOCK_BITS) + 1;
  auto ids = (u16 *)calloc(bases[partitions] + 1, 2);
  auto marks = (u32 *)calloc((terms + 1) * (blocks + 1), 4);
  on_all_threads(partitions, [&](size_t p) {
    vector<Segment> segments;
    u32 counts[PARTITION_TERMS] = {};
    for (size_t b = 0; b < workers.size(); b++) {
      const Log log = workers[b].buckets[p];
      u32 start = 0, group = 0;
      bool open = false;
      auto close = [&](size_t end) {
        if (open && start != end)
          segments.push_back({group, (u16)(log.data[start] & (GROUP_DOCS - 1)),
                              (u32)b, start, (u32)end});
      };
      for (size_t i = 0; i < log.len; i++) {
        u16 entry = log.data[i];
        if (entry & MARKER) {
          close(i);
          open = true;
          group = entry & ~MARKER;
          start = i + 1;
        } else {
          counts[entry >> GROUP_BITS]++;
        }
      }
      close(log.len);
    }
    std::sort(segments.begin(), segments.end(), [](auto &a, auto &b) {
      return a.group != b.group ? a.group < b.group : a.first < b.first;
    });

    u32 cursors[PARTITION_TERMS], sum = bases[p];
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
    for (const Segment &segment : segments) {
      size_t segment_block = segment.group >> GROUPS_PER_BLOCK_BITS;
      while (block < segment_block) {
        mark(++block);
        memset(last, 255, sizeof last);
      }
      u32 group_low = (segment.group & ((1 << GROUPS_PER_BLOCK_BITS) - 1))
                      << GROUP_BITS;
      const u16 *log = workers[segment.bucket].buckets[p].data;
      for (u32 i = segment.start; i < segment.end; i++) {
        size_t term = log[i] >> GROUP_BITS;
        u32 low = group_low | (log[i] & (GROUP_DOCS - 1));
        // A term repeated within a document appears once per token.
        if (last[term] != low) {
          last[term] = low;
          ids[cursors[term]++] = low;
        }
      }
    }
    while (block < blocks)
      mark(++block);
    for (auto &worker : workers)
      free(worker.buckets[p].data);
  });
  return {ids, marks, blocks};
}

struct Index {
  size_t high, words;
  const u64 *bits;
  Lists lists;
};

INLINE bool has_bit(const u64 *bits, u32 id) {
  return (bits[id / 64] >> (id % 64)) & 1;
}

static void intersect(const vector<u32> &a, const vector<u32> &b,
                      vector<u32> &out) {
  out.clear();
  auto &small = a.size() <= b.size() ? a : b;
  auto &large = a.size() <= b.size() ? b : a;
  if (large.size() > small.size() * 16) {
    auto rest = large.begin();
    for (u32 id : small) {
      rest = std::lower_bound(rest, large.end(), id);
      if (rest != large.end() && *rest == id)
        out.push_back(id);
    }
  } else {
    for (size_t i = 0, j = 0; i < a.size() && j < b.size();) {
      u32 x = a[i], y = b[j];
      if (x == y)
        out.push_back(x);
      i += x <= y;
      j += y <= x;
    }
  }
}

static void merge_union(const vector<u32> &a, const vector<u32> &b,
                        vector<u32> &out) {
  out.clear();
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    u32 x = a[i], y = b[j];
    out.push_back(std::min(x, y));
    i += x <= y;
    j += y <= x;
  }
  out.insert(out.end(), a.begin() + i, a.end());
  out.insert(out.end(), b.begin() + j, b.end());
}

static void difference(const vector<u32> &a, const vector<u32> &b,
                       vector<u32> &out) {
  out.clear();
  auto rest = b.begin();
  for (u32 id : a) {
    rest = std::lower_bound(rest, b.end(), id);
    if (rest == b.end() || *rest != id)
      out.push_back(id);
  }
}

static void push_int(std::string &out, u64 value) {
  char digits[20];
  out.append(digits, std::to_chars(digits, digits + 20, value).ptr);
}

// The sum of a word's set bit positions is a sum of six masked population counts.
static void write_bits(std::string &out, const vector<u64> &bits) {
  static const u64 POSITION_BITS[6] = {
      0xAAAAAAAAAAAAAAAA, 0xCCCCCCCCCCCCCCCC, 0xF0F0F0F0F0F0F0F0,
      0xFF00FF00FF00FF00, 0xFFFF0000FFFF0000, 0xFFFFFFFF00000000};
  u64 count = 0, sum = 0;
  for (size_t i = 0; i < bits.size(); i++) {
    u64 ones = __builtin_popcountll(bits[i]);
    count += ones;
    sum += ones * (i * 64);
    for (int k = 0; k < 6; k++)
      sum += (u64)__builtin_popcountll(bits[i] & POSITION_BITS[k]) << k;
  }
  push_int(out, count);
  out += ' ';
  push_int(out, sum);
  int left = 10;
  for (size_t i = 0; i < bits.size() && left; i++)
    for (u64 rest = bits[i]; rest && left; rest &= rest - 1, left--) {
      out += ' ';
      push_int(out, i * 64 + __builtin_ctzll(rest));
    }
  out += '\n';
}

// Writes the count, the sum and the ten smallest of `n` ascending ids.
template <class F> static void write_ids(std::string &out, size_t n, F &&id_at) {
  u64 sum = 0;
  for (size_t i = 0; i < n; i++)
    sum += id_at(i);
  push_int(out, n);
  out += ' ';
  push_int(out, sum);
  for (size_t i = 0; i < std::min<size_t>(n, 10); i++) {
    out += ' ';
    push_int(out, id_at(i));
  }
  out += '\n';
}

struct Evaluator {
  bool as_bits = false;
  vector<u32> ids, term, merged;
  vector<u64> bits;

  void start(const Index &index, size_t id) {
    as_bits = id < index.high;
    if (as_bits)
      bits.assign(index.bits + id * index.words,
                  index.bits + (id + 1) * index.words);
    else
      index.lists.decode(id - index.high, ids);
  }

  // Applies one operator, strictly left to right. `op` is its word's first byte.
  void apply(const Index &index, u8 op, size_t id) {
    if (!as_bits && ids.empty() && op != 'O')
      return;
    if (id < index.high) {
      const u64 *other = index.bits + id * index.words;
      if (as_bits) {
        for (size_t i = 0; i < bits.size(); i++)
          bits[i] = op == 'A'   ? bits[i] & other[i]
                    : op == 'O' ? bits[i] | other[i]
                                : bits[i] & ~other[i];
      } else if (op == 'O') {
        bits.assign(other, other + index.words);
        for (u32 doc : ids)
          bits[doc / 64] |= 1ull << (doc % 64);
        as_bits = true;
      } else {
        bool keep = op == 'A';
        std::erase_if(ids, [&](u32 doc) { return has_bit(other, doc) != keep; });
      }
      return;
    }
    index.lists.decode(id - index.high, term);
    if (as_bits) {
      if (op == 'A') {
        ids.clear();
        for (u32 doc : term)
          if (has_bit(bits.data(), doc))
            ids.push_back(doc);
        as_bits = false;
      } else {
        for (u32 doc : term) {
          u64 bit = 1ull << (doc % 64);
          bits[doc / 64] = op == 'O' ? bits[doc / 64] | bit : bits[doc / 64] & ~bit;
        }
      }
    } else {
      if (op == 'A')
        intersect(ids, term, merged);
      else if (op == 'O')
        merge_union(ids, term, merged);
      else
        difference(ids, term, merged);
      ids.swap(merged);
    }
  }

  void write(std::string &out) const {
    if (as_bits)
      write_bits(out, bits);
    else
      write_ids(out, ids.size(), [&](size_t i) { return ids[i]; });
  }
};

int main() {
  mallopt(M_MMAP_THRESHOLD, 1 << 16);
  memset(low_bytes, 255, 32);
  cpu_set_t cpus;
  sched_getaffinity(0, sizeof cpus, &cpus);
  threads = CPU_COUNT(&cpus);
  size_t length;
  const u8 *text = map_input("/data/queries.txt", length);
  vector<size_t> bounds = query_chunks(text, length);
  size_t chunks = bounds.size() - 1;
  Table table(length);
  vector<ChunkPhrases> phrase_chunks(chunks);
  on_all_threads(chunks, [&](size_t c) {
    phrase_chunks[c] = read_chunk(text, bounds[c], bounds[c + 1], table);
  });

  input = map_input("/data/docs.txt", input_size);
  size_t docs = 0;
  if (input_size) {
    const u8 *last = input + input_size - 1;
    while (last > input && last[-1] != '\n')
      last--;
    docs = parse_id(last) + 1;
  }
  size_t words = docs / 64 + 1;
  Plan plan;
  {
    vector<u16> seen = sample_counts(input, table, docs);
    plan = plan_index(table, phrase_chunks, seen, docs / SAMPLE_SHARE);
  }

  auto bits = (u64 *)calloc(plan.high * words + 1, 8);
  vector<Worker> workers(threads);
  atomic<size_t> next_chunk{0};
  vector<std::thread> pool;
  for (auto &worker : workers)
    pool.emplace_back([&, w = &worker] {
      parse_docs(table, plan, bits, words, *w, next_chunk);
    });
  for (auto &thread : pool)
    thread.join();
  vector<Hit> hits;
  for (auto &worker : workers)
    hits.insert(hits.end(), worker.hits.begin(), worker.hits.end());
  Lists lists = build_lists(workers, plan.low, docs);
  Index index{plan.high, words, bits, lists};

  // A document can match a phrase more than once, but it counts once.
  std::sort(hits.begin(), hits.end());
  hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
  size_t phrase_count = plan.phrases.starts.size() - 1;
  vector<u32> matched(phrase_count + 1);
  for (const Hit &hit : hits)
    matched[hit.phrase + 1]++;
  for (size_t i = 1; i <= phrase_count; i++)
    matched[i] += matched[i - 1];

  vector<std::string> outputs(chunks);
  on_all_threads(chunks, [&](size_t c) {
    std::string &out = outputs[c];
    out.reserve(bounds[c + 1] - bounds[c]);
    Evaluator evaluator;
    size_t phrase = plan.phrase_bases[c];
    for (const u8 *p = text + bounds[c]; p < text + bounds[c + 1];) {
      if (*p == 'B') {
        size_t position = 0;
        u8 op = 0;
        p = scan_line(p, [&](const u8 *word, u64 hash) {
          if (position % 2 == 0)
            op = *word;
          else if (position == 1)
            evaluator.start(index, table.find(hash) & ID_MASK);
          else
            evaluator.apply(index, op, table.find(hash) & ID_MASK);
          position++;
        });
        evaluator.write(out);
      } else {
        const Hit *found = hits.data() + matched[phrase];
        write_ids(out, matched[phrase + 1] - matched[phrase],
                  [&](size_t i) { return found[i].doc; });
        phrase++;
        p = (const u8 *)memchr(p, '\n', text + length - p) + 1;
      }
    }
  });
  int fd = open("/out/results.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  for (auto &out : outputs)
    if (write(fd, out.data(), out.size()) != (ssize_t)out.size())
      abort();
  _exit(0);
}
