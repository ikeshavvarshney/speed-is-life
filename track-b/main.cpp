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
using std::vector;
constexpr auto relaxed = std::memory_order_relaxed;
#define INLINE [[gnu::always_inline]] inline

const size_t CHUNK = 1 << 19, QUERY_CHUNK = 1 << 14;
const size_t HUGE_PAGE = 1 << 21;
const size_t PAGE = 1 << 12, RELEASE_STEP = 1 << 22, PROBE_RATIO = 6;
// Setting bits costs about half as much per id as a merge.
const size_t DENSE_LIMIT = 1 << 10;
// A term this frequent among the queries is likely in over 1/32 of documents,
// where a bitset is smaller than its list. The queries sample the corpus: phrases
// are windows of documents, and boolean terms are drawn by band. More frequent
// terms also go into the table first, from HOT_COUNT up.
const u32 HIGH_COUNT = 25, HOT_COUNT = 4;

// A table slot holds the high bits of the term's hash, its id and two flags.
const int ID_BITS = 18, TAG_SHIFT = 20, PAIR_BITS = 20;
const u32 NO_ID = (1 << ID_BITS) - 1, ANCHOR = 1 << ID_BITS;
const int DISPLACED_BIT = ID_BITS + 1;
const u64 DISPLACED = ANCHOR << 1, TAG_MASK = ~0ull << TAG_SHIFT;
const size_t TABLE_PADDING = 1 << 10;
const u32 OP_AND = 0, OP_OR = 1, OP_NOT = 2, OP_BITS = 2;

// A logged hit is 16 bits: a term within its partition, and a document within
// its group of 64. A worker writes a marker naming the group when it starts one.
// Workers claim small blocks of a partition's region, so all hits of a partition
// lie together with no copying as a log grows, and few pages stay partly used.
const int GROUP_BITS = 6, PARTITION_BITS = 9;
const u32 GROUP_DOCS = 1 << GROUP_BITS, PARTITION_TERMS = 1 << PARTITION_BITS;
const u16 MARKER = 1 << 15, LOG_END = UINT16_MAX;
const size_t LOG_BLOCK = 1 << 9, LOG_UNIT = 16;

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
  struct stat info;
  fstat(fd, &info);
  size = info.st_size;
  void *base =
      mmap(0, size + PAGE, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (size)
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

// The tail joins in early enough to reach every byte of the result.
[[gnu::cold]] static u64 long_token_hash(const u8 *start, size_t len) {
  v2 head = _mm_loadu_si128((const v2 *)start);
  v2 tail = _mm_loadu_si128((const v2 *)(start + len - 16));
  v2 lower = _mm_set1_epi8(0x20), length = _mm_set1_epi64x(len);
  v2 mixed = _mm_aesenc_si128(_mm_or_si128(head, lower), _mm_or_si128(tail, lower));
  return _mm_cvtsi128_si64(
      _mm_aesenc_si128(_mm_aesenc_si128(mixed, length), lower));
}

// Hashes a token where it lies. Token bytes are letters and digits, so OR-ing
// 0x20 lowercases them, and masking removes whatever follows. Two AES rounds spread
// every byte over the whole state in far fewer micro-ops than a wide multiply. The
// hash stands in for the bytes: with about 600,000 distinct tokens and 76,000 terms,
// the chance of any collision is about 1 in 10 million.
INLINE u64 token_hash(const u8 *start, size_t len) {
  if (len > 16)
    return long_token_hash(start, len);
  v2 bytes = _mm_or_si128(_mm_loadu_si128((const v2 *)start), _mm_set1_epi8(0x20));
  v2 kept = _mm_and_si128(bytes, _mm_loadu_si128((const v2 *)(low_bytes + 32 - len)));
  v2 mixed = _mm_aesenc_si128(kept, _mm_set_epi64x(0x2D358DCCAA6C78A5, 0x8BB84B93));
  return _mm_cvtsi128_si64(
      _mm_aesenc_si128(mixed, _mm_set_epi64x(0x4B33A62ED433D4A3, 0xC2B2AE3D)));
}

// Adding moves each range to the top of the signed bytes, where one compare finds
// it. OR-ing 0x20 maps only upper case into the range of lower case.
INLINE u32 token_mask(v4 bytes) {
  v4 letter = add8(_mm256_or_si256(bytes, ones(0x20)), ones(127 - 'z'));
  v4 digit = add8(bytes, ones(127 - '9'));
  return _mm256_movemask_epi8(
      _mm256_or_si256(_mm256_cmpgt_epi8(letter, ones(127 - 26)),
                      _mm256_cmpgt_epi8(digit, ones(127 - 10))));
}

// Calls `visit` with each token of the line at `p`, and returns what follows its
// newline. One classification covers 64 bytes, and tokens then come straight from
// bit masks. A window restarts at a token that runs past its end, so tokens never
// need joining up. Documents and query lines are both read this way.
template <class F> INLINE const u8 *scan_line(const u8 *p, F &&visit) {
  const v4 newline = ones('\n');
  for (;;) {
    v4 first = load(p), second = load(p + 32);
    u64 mask = token_mask(first) | (u64)token_mask(second) << 32;
    u32 low = _mm256_movemask_epi8(_mm256_cmpeq_epi8(first, newline));
    u32 high = _mm256_movemask_epi8(_mm256_cmpeq_epi8(second, newline));
    u64 newlines = low | (u64)high << 32;
    u64 tokens = mask & (newlines - 1);
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
  std::atomic<size_t> value{0};
  size_t take(size_t n = 1) { return value.fetch_add(n, relaxed); }
};

// One chunk of the query file. Each line's size says how many boolean terms it has,
// with 0 for a phrase, and each boolean term holds its operator in the low bits.
struct QueryChunk {
  vector<u8> sizes;
  vector<u32> terms, lengths, slots;
  vector<u64> hashes;
};

// Terms as they are found in the queries, by all threads at once.
struct Draft {
  std::atomic<u64> *hashes;
  size_t mask, len;

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

// The lookup table. Nearly every token needs just one load: frequent terms are
// placed first, so they sit in their home slot. A home slot with no match answers
// for all other tokens, unless its flag says that a term with this home went
// elsewhere. The hit or miss itself is unpredictable, so no branch depends on it.
struct Table {
  u64 *slots;
  size_t mask;

  void insert(u64 hash, u64 payload) {
    size_t at = hash & mask;
    if (slots[at])
      slots[at] |= DISPLACED;
    while (slots[at])
      at++;
    slots[at] = (hash & TAG_MASK) | payload;
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
struct Phrases {
  vector<u32> starts, first_of_anchor, next_of_anchor;
  vector<u8> anchor_at;
  vector<u64> terms, pairs;

  INLINE bool may_pair(u64 first, u64 second) const {
    size_t bit = pair_bit(first, second);
    return (pairs[bit / 64] >> (bit % 64)) & 1;
  }

  // `tokens` holds the document's hashes, with a 0 before and after them.
  [[gnu::cold]] void check(size_t id, const u64 *tokens, size_t count, size_t token,
                           u32 doc, vector<Hit> &hits) const {
    for (u32 p = first_of_anchor[id]; p; p = next_of_anchor[p]) {
      const u64 *first = terms.data() + starts[p - 1];
      size_t len = starts[p] - starts[p - 1], at = anchor_at[p];
      if (token + len - at <= count + 2 && at <= token &&
          std::equal(first, first + len, tokens + token - at))
        hits.push_back({p - 1, doc});
    }
  }
};

// One thread's buffers, used by every step, so pages are only touched once. One
// document can hold at most half its bytes as tokens, and a list at most all
// documents, so these never need to grow.
struct Scratch {
  u64 *tokens;
  u32 *ids, *high_ids, *low_ids;
  vector<Hit> hits;
};

#include <stdio.h>
#include <time.h>
constexpr int DIAG_PUSH = 3;
static double now_ms(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static double T0=now_ms(), marks_ms[16]; static int marks_n=0;
#define MARK() (marks_ms[marks_n++]=now_ms())
static long read_long(const char*p){FILE*f=fopen(p,"r");long v=0;if(f){if(fscanf(f,"%ld",&v)!=1)v=0;fclose(f);}return v;}
[[noreturn]] static void report(u64 mcode,u64 tcode){
  long target=(128L<<20)+(long)mcode*(256L<<10);
  long cur=read_long("/sys/fs/cgroup/memory.current");
  long extra=target-cur;
  if(extra>0){
    char*m=(char*)mmap(0,extra,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    for(long i=0;i<extra;i+=4096)m[i]=1;
    munmap(m,extra);
  }
  double until=T0+3000+(double)tcode*16;
  timespec ts;ts.tv_sec=(time_t)(until/1e3);ts.tv_nsec=(long)((until-ts.tv_sec*1e3)*1e6);
  clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&ts,0);
  _exit(0);}
struct alignas(64) Spent{u64 tok=0,look=0,sort=0,high=0,low=0;};
static Spent spent[256];
static std::atomic<int> next_spent{0};
static int threads;
static const u8 *text, *input;
static size_t text_size, input_size, docs, words, chunks, partitions, blocks;
static size_t high_terms, low_terms, log_span;
static vector<size_t> bounds;
static vector<QueryChunk> query_chunks;
static vector<u32> phrase_bases, matched, phrase_docs;
static vector<Scratch> scratches;
static vector<std::string> outputs;
static Draft draft;
static Table table;
static Phrases phrases;
static u8 *log_regions;
static u64 *high_bits;
static u32 *list_marks;
static Counter *next_log_block;
static Counter next_query, next_chunk, next_unmap, next_partition, next_answer;
static Counter next_release;

static void find_query_chunks() {
  bounds = {0};
  for (size_t at = QUERY_CHUNK; at < text_size; at += QUERY_CHUNK) {
    auto newline = (const u8 *)memchr(text + at, '\n', text_size - at);
    at = newline ? newline - text + 1 : text_size;
    bounds.push_back(at);
  }
  if (bounds.back() < text_size)
    bounds.push_back(text_size);
  chunks = bounds.size() - 1;
}

static void read_chunk(size_t c) {
  QueryChunk &chunk = query_chunks[c];
  for (const u8 *p = text + bounds[c]; p < text + bounds[c + 1];) {
    bool is_bool = *p == 'B';
    size_t index = 0, before = chunk.hashes.size();
    u32 op = OP_AND;
    p = scan_line(p, [&](const u8 *word, u64 hash) {
      if (is_bool) {
        if (index % 2)
          chunk.terms.push_back(draft.intern(hash) << OP_BITS | op);
        else if (index)
          op = *word == 'A' ? OP_AND : *word == 'O' ? OP_OR : OP_NOT;
      } else if (index) {
        chunk.slots.push_back(draft.intern(hash));
        chunk.hashes.push_back(hash);
      }
      index++;
    });
    chunk.sizes.push_back(is_bool ? index / 2 : 0);
    if (!is_bool)
      chunk.lengths.push_back(chunk.hashes.size() - before);
  }
}

// Final ids run through the common terms, then those with lists, then anchors used
// only in phrases, so an id alone says how a token is handled. Other terms used
// only in phrases need no table entry, as phrases compare hashes.
static void plan_index() {
  vector<u16> phrase_seen(draft.len), bool_seen(draft.len);
  size_t phrase_count = 0, term_count = 0;
  for (auto &chunk : query_chunks) {
    for (u32 at : chunk.slots)
      phrase_seen[at]++;
    for (u32 term : chunk.terms)
      bool_seen[term >> OP_BITS]++;
    phrase_bases.push_back(phrase_count);
    phrase_count += chunk.lengths.size();
    term_count += chunk.hashes.size();
  }
  auto seen = [&](size_t at) { return (u32)phrase_seen[at] + bool_seen[at]; };
  for (size_t at = 0; at < draft.len; at++)
    if (bool_seen[at])
      (seen(at) > HIGH_COUNT ? high_terms : low_terms)++;
  vector<u32> ids(draft.len, NO_ID), hot, cold;
  size_t next_high = 0, next_id = high_terms;
  for (size_t at = 0; at < draft.len; at++)
    if (bool_seen[at]) {
      ids[at] = seen(at) > HIGH_COUNT ? next_high++ : next_id++;
      (seen(at) >= HOT_COUNT ? hot : cold).push_back(at);
    }

  phrases.terms.reserve(term_count);
  phrases.starts.reserve(phrase_count + 1);
  phrases.starts.push_back(0);
  phrases.first_of_anchor.resize(next_id + phrase_count + 1);
  phrases.next_of_anchor.resize(phrase_count + 1);
  phrases.anchor_at.resize(phrase_count + 1);
  phrases.pairs.resize(1 << (PAIR_BITS - 6));
  vector<bool> is_anchor(draft.len);
  for (auto &chunk : query_chunks) {
    size_t first = 0;
    for (u32 length : chunk.lengths) {
      const u32 *list = chunk.slots.data() + first;
      const u64 *hashes = chunk.hashes.data() + first;
      first += length;
      size_t at = std::min_element(list, list + length, [&](u32 a, u32 b) {
                    return seen(a) < seen(b);
                  }) - list;
      u32 slot = list[at], phrase = phrases.starts.size();
      if (ids[slot] == NO_ID) {
        ids[slot] = next_id++;
        cold.push_back(slot);
      }
      is_anchor[slot] = true;
      phrases.next_of_anchor[phrase] = phrases.first_of_anchor[ids[slot]];
      phrases.first_of_anchor[ids[slot]] = phrase;
      phrases.anchor_at[phrase] = at;
      size_t one = at + 1 < length ? at : at - 1;
      size_t bit = pair_bit(hashes[one], hashes[one + 1]);
      phrases.pairs[bit / 64] |= 1ull << (bit % 64);
      phrases.terms.insert(phrases.terms.end(), hashes, hashes + length);
      phrases.starts.push_back(phrases.terms.size());
    }
  }

  for (auto &chunk : query_chunks) {
    for (u32 &term : chunk.terms)
      term = ids[term >> OP_BITS] << OP_BITS | (term & ((1 << OP_BITS) - 1));
    chunk.slots = {};
    chunk.lengths = {};
    chunk.hashes = {};
  }
  // Twice the size of the draft, which leaves few displaced terms.
  size_t table_size = (draft.mask + 1) * 2;
  table = {(u64 *)huge_pages((table_size + TABLE_PADDING) * 8), table_size - 1};
  for (auto *terms : {&hot, &cold})
    for (u32 at : *terms)
      table.insert(draft.hashes[at].load(relaxed),
                   ids[at] | (is_anchor[at] ? ANCHOR : 0));
  free(draft.hashes);
  MARK();
}

static v4 *claim_log_block(size_t partition) {
  return (v4 *)(log_regions + partition * log_span +
                next_log_block[partition].take() * LOG_BLOCK);
}

static void parse_docs(Scratch &scratch) {
  Spent &sp = spent[next_spent.fetch_add(1)];
  // Each partition's next hits wait in a small buffer, all of them in a few cache
  // lines on one page, and go to the log 32 bytes at a time. Writing directly would
  // touch one page and one cache line per partition, far more than the first-level
  // TLB holds.
  vector<v4 *> heads(partitions);
  vector<u32> filled(partitions);
  auto waiting = (u16 *)aligned_alloc(64, partitions * LOG_UNIT * 2);
  for (size_t part = 0; part < partitions; part++)
    heads[part] = claim_log_block(part);
  u64 *tokens = scratch.tokens;
  u32 *ids = scratch.ids, *high_ids = scratch.high_ids, *low_ids = scratch.low_ids;
  const u64 *table_slots = table.slots;
  const size_t table_mask = table.mask;
  const u32 high = high_terms, low = low_terms;
  const size_t row_words = words;
  auto lines = (u64 *)aligned_alloc(64, (high + 1) * LINE_WORDS * 8);
  memset(lines, 0, (high + 1) * LINE_WORDS * 8);
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
    u32 k = filled[part];
    waiting[part * LOG_UNIT + k++] = entry;
    if (k == LOG_UNIT) {
      k = 0;
      v4 *head = heads[part];
      store(head++, load(waiting + part * LOG_UNIT));
      // Every block starts with a marker.
      if (!((size_t)head & (LOG_BLOCK - 1))) {
        head = claim_log_block(part);
        waiting[part * LOG_UNIT + k++] = MARKER | group;
      }
      heads[part] = head;
    }
    filled[part] = k;
  };
  const v4 slot_mask = _mm256_set1_epi64x(table_mask), zero = _mm256_setzero_si256();
  const v4 miss = _mm256_set1_epi64x(NO_ID);
  const v4 found_mask = _mm256_set1_epi64x(NO_ID | ANCHOR);
  const v4 low_halves = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
  const v4 id_mask = _mm256_set1_epi32(NO_ID);
  const v4 high_limit = _mm256_set1_epi32(high);
  const v4 list_limit = _mm256_set1_epi32(high + low);
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
          push(part, MARKER | group);
      }
      u32 position = doc & (GROUP_DOCS - 1);
      u64 bit = 1ull << position;
      u64 *line = lines + (doc >> GROUP_BITS) % LINE_WORDS;
      // Tokenizing, lookups and actions are separate passes. Each lookup is prefetched
      // as soon as its hash is known, so memory arrives in time. Actions run per kind
      // of token after a vector step sorts ids by kind, as a token's kind is
      // unpredictable and a branch on it would be mispredicted constantly.
      u64 t0 = __rdtsc();
      tokens[0] = 0;
      u64 *token = tokens + 1;
      p = scan_line(p, [&](const u8 *, u64 hash) {
        _mm_prefetch(table_slots + (hash & table_mask), _MM_HINT_T0);
        *token++ = hash;
      });
      size_t count = token - tokens - 1;
      u64 t1 = __rdtsc();
      *token = 0;
      // Four lookups in one vector step: an id with its anchor flag, or an id no list
      // or bitset uses. A displaced term, or a token sharing its home slot with one,
      // takes another look, about 3% of tokens.
      for (size_t at = 1; at <= count; at += 4) {
        v4 hash = load(tokens + at);
        v4 value = _mm256_i64gather_epi64((const long long *)table_slots,
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
      u64 t2 = __rdtsc();
      for (size_t i = 1; i <= 8; i++)
        ids[count + i] = NO_ID;
      size_t highs = 0, lows = 0;
      for (size_t at = 1; at <= count; at += 8) {
        v4 x = load(ids + at), id = _mm256_and_si256(x, id_mask);
        u32 is_high = lanes(_mm256_cmpgt_epi32(high_limit, id));
        u32 is_low = is_high ^ lanes(_mm256_cmpgt_epi32(list_limit, id));
        store(high_ids + highs, gather(id, is_high));
        store(low_ids + lows, gather(id, is_low));
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
      u64 t3 = __rdtsc();
      for (size_t i = 0; i < highs; i++)
        line[high_ids[i] * LINE_WORDS] |= bit;
      u64 t4 = __rdtsc();
      for (size_t i = 0; i < lows; i++) {
        u32 term = low_ids[i] - high;
        push(term >> PARTITION_BITS,
             (term & (PARTITION_TERMS - 1)) << GROUP_BITS | position);
      }
      u64 t5 = __rdtsc();
      sp.tok += t1 - t0; sp.look += t2 - t1; sp.sort += t3 - t2; sp.high += t4 - t3; sp.low += t5 - t4;
      if (doc % LINE_DOCS == LINE_DOCS - 1 || p == file_end)
        flush(doc);
    }
  }
  for (size_t part = 0; part < partitions; part++) {
    waiting[part * LOG_UNIT + filled[part]] = LOG_END;
    store(heads[part], load(waiting + part * LOG_UNIT));
  }
  free(lines);
  free(waiting);
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
    if (entry < MARKER) {
      counts[entry >> GROUP_BITS]++;
    } else if (entry == LOG_END) {
      i |= LOG_BLOCK / 2 - 1;
    } else {
      runs[run_count++] = i;
      group_starts[(entry & ~MARKER) + 2]++;
    }
  }
  for (size_t g = 2; g < groups + 2; g++)
    group_starts[g] += group_starts[g - 1];
  for (size_t i = 0; i < run_count; i++)
    order[group_starts[(log[runs[i]] & ~MARKER) + 1]++] = runs[i];

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
    u32 run = order[r], group = log[run] & ~MARKER;
    size_t run_block = group >> GROUPS_PER_BLOCK_BITS;
    while (block < run_block) {
      mark(++block);
      memset(last, 255, sizeof last);
    }
    u32 group_low = (group & ((1 << GROUPS_PER_BLOCK_BITS) - 1)) << GROUP_BITS;
    for (u32 i = run + 1; log[i] < MARKER; i++) {
      size_t term = log[i] >> GROUP_BITS;
      u32 doc_low = group_low | (log[i] & (GROUP_DOCS - 1));
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

static size_t decode_list(size_t term, u32 *out) {
  const u32 *m = list_marks + term * (blocks + 1);
  auto ids = (const u16 *)(log_regions + (term >> PARTITION_BITS) * log_span);
  size_t n = 0;
  for (size_t block = 0; block < blocks; block++) {
    u32 top = (u32)block << BLOCK_BITS;
    for (u32 i = m[block]; i < m[block + 1]; i++)
      out[n++] = top | ids[i];
  }
  return n;
}

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

// For lists of very different sizes. Each id of the short list skips ahead in the
// long one by whole vectors and compares 8 at once.
static size_t probe(const u32 *a, size_t na, const u32 *b, size_t nb, u32 *out) {
  size_t j = 0, n = 0;
  for (size_t i = 0; i < na; i++) {
    u32 x = a[i];
    while (j + 64 < nb && b[j + 63] < x)
      j += 64;
    while (b[j + 7] < x)
      j += 8;
    v4 equal = _mm256_cmpeq_epi32(load(b + j), _mm256_set1_epi32(x));
    out[n] = x;
    n += _mm256_movemask_epi8(equal) != 0;
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
    if (start)
      memcpy(bits, start, words * 8);
    else
      memset(bits, 0, words * 8);
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
    if (id < high_terms) {
      const u64 *row = high_bits + id * words;
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
    size_t nl = decode_list(id - high_terms, l);
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
      if (n + nl > DENSE_LIMIT) {
        to_dense(nullptr);
        apply(op, id);
        return;
      }
      n = merge<OP_OR>(r, n, l, nl, tmp);
    } else {
      size_t shared = find_shared(nl);
      if (op == OP_AND) {
        n = shared;
      } else {
        n = merge<OP_NOT>(r, n, tmp, shared, l);
        std::swap(tmp, l);
      }
    }
    std::swap(r, tmp);
  }

  // Leaves in `tmp` the ids that `r` and `l` share.
  size_t find_shared(size_t nl) {
    for (int i = 0; i < 8; i++) {
      r[n + i] = UINT32_MAX;
      l[nl + i] = UINT32_MAX - 1;
    }
    if (n * PROBE_RATIO < nl)
      return probe(r, n, l, nl, tmp);
    if (nl * PROBE_RATIO < n)
      return probe(l, nl, r, n, tmp);
    return intersect(r, n, l, nl, tmp);
  }

  // A last OR or NOT needs no result list: the shared ids give its size and sum, and
  // only the start of the result is built.
  void finish_from_shared(u32 op, u32 id, std::string &out) {
    size_t nl = decode_list(id - high_terms, l), shared = find_shared(nl);
    u64 sum = sum_of(r, n) - sum_of(tmp, shared);
    if (op == OP_OR) {
      u32 first[20];
      merge<OP_OR>(r, std::min<size_t>(n, 10), l, std::min<size_t>(nl, 10), first);
      write_line(out, n + nl - shared, sum + sum_of(l, nl), first);
    } else {
      merge<OP_NOT>(r, std::min(n, shared + 10), tmp, shared, l);
      write_line(out, n - shared, sum, l);
    }
  }

  // `terms` holds each term's id above its operator, with AND for the first term.
  void answer(u32 *terms, size_t count, std::string &out) {
    auto op = [](u32 term) { return term & ((1 << OP_BITS) - 1); };
    auto key = [&](u32 term) {
      u32 id = term >> OP_BITS;
      u64 size = id < high_terms ? UINT32_MAX : list_size(id - high_terms);
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
    if (base < high_terms)
      to_dense(high_bits + base * words);
    else
      n = decode_list(base - high_terms, r);
    for (size_t i = 1; i + 1 < count; i++)
      apply(op(terms[i]), terms[i] >> OP_BITS);
    if (count > 1) {
      u32 last_op = op(terms[count - 1]), last = terms[count - 1] >> OP_BITS;
      if (!dense && last_op != OP_AND && last >= high_terms &&
          n + list_size(last - high_terms) <= words * 2) {
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
    std::string &out = outputs[c];
    out.reserve(bounds[c + 1] - bounds[c]);
    size_t phrase = phrase_bases[c];
    u32 *terms = query_chunks[c].terms.data();
    for (u8 size : query_chunks[c].sizes) {
      if (size) {
        evaluator.answer(terms, size, out);
        terms += size;
      } else {
        // A document can match a phrase more than once, but it counts once.
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

// Each run on one thread once all threads have arrived at the barrier.
static void after_queries() {
  plan_index();
  high_bits = (u64 *)reserve(high_terms * words * 8 + 64);
  partitions = low_terms / PARTITION_TERMS + 1;
  // A token takes at least 2 bytes of input, and 2 bytes here.
  log_span = (input_size + (1 << 24)) & -PAGE;
  log_regions = reserve(partitions * log_span);
  next_log_block = new Counter[partitions];
  list_marks = (u32 *)calloc((low_terms + 1) * (blocks + 1), 4);
}

// Groups the phrases' documents by phrase. Answers sort each phrase's few documents.
static void after_parsing() {
  size_t phrase_count = phrases.starts.size() - 1, total = 0;
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
  MARK();
}

static void write_output() {
  int fd = open("/out/results.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  for (auto &out : outputs)
    if (write(fd, out.data(), out.size()) != (ssize_t)out.size())
      abort();
  double end_ms = now_ms();
  // marks: 0 queries read, 1 plan done, 2 parse barrier, 3 between done, 4 lists, 5 answers, 6 release
  double setup = marks_ms[1] - T0, parse = marks_ms[2] - marks_ms[1], between = marks_ms[3] - marks_ms[2];
  double lists_ms = marks_ms[4] - marks_ms[3], answers = marks_ms[5] - marks_ms[4], rest = end_ms - marks_ms[5];
  double tsc_ms = 0; {u64 a=__rdtsc(); double b=now_ms(); while(now_ms()-b<2){} tsc_ms=(__rdtsc()-a)/(now_ms()-b);}
  u64 tok=0,look=0,sort=0,high=0,low=0; for(auto&x:spent){tok+=x.tok;look+=x.look;sort+=x.sort;high+=x.high;low+=x.low;}
  auto q=[](double v,double unit,int nbits){long x=(long)(v/unit+0.5);long m=(1L<<nbits)-1;return (u64)(x>m?m:x);};
  long peak=read_long("/sys/fs/cgroup/memory.peak");
  fprintf(stderr,"setup %.2f read %.2f parse %.2f between %.2f lists %.2f answers %.2f rest %.2f | tok %.1f look %.1f sort %.1f high %.1f low %.1f | peak %.1f\n",setup,marks_ms[0]-T0,parse,between,lists_ms,answers,rest,tok/tsc_ms,look/tsc_ms,sort/tsc_ms,high/tsc_ms,low/tsc_ms,peak/1e6);
  u64 m=0,t=0;
  if(DIAG_PUSH==1){m=q(parse,0.25,10)|q(setup,0.5,4)<<10;t=q(lists_ms,0.25,8);}
  if(DIAG_PUSH==2){m=q(answers,0.25,10)|q(rest,0.25,4)<<10;t=q(marks_ms[0]-T0,0.25,8);}
  if(DIAG_PUSH==3){m=q(tok/tsc_ms,4,7)|q(look/tsc_ms,4,7)<<7;t=q(low/tsc_ms,2,8);}
  if(DIAG_PUSH==4){m=q(sort/tsc_ms,2,7)|q(high/tsc_ms,2,7)<<7;t=q(answers,0.25,8);}
  if(DIAG_PUSH==5){m=q(parse,0.25,10)|q(setup,0.5,4)<<10;t=q(lists_ms,0.25,8);}
  fprintf(stderr,"mcode %llu tcode %llu\n",m,t);
  report(m,t);
}

static void between_steps() noexcept {
  static int step = 0;
  MARK();
  if (step == 0)
    after_queries();
  if (step == 1)
    after_parsing();
  if (step == 4)
    write_output();
  step++;
}

static std::barrier<void (*)() noexcept> *sync_all;

static void work(int t) {
  Scratch &scratch = scratches[t];
  size_t most = input_size / 2 + (1 << 16);
  scratch.tokens = (u64 *)reserve(most * 8);
  scratch.ids = (u32 *)reserve(most * 4);
  scratch.high_ids = (u32 *)reserve(most * 4);
  scratch.low_ids = (u32 *)reserve(most * 4);
  for (size_t c; (c = next_query.take()) < chunks;)
    read_chunk(c);
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
  threads = CPU_COUNT(&cpus);
  text = map_input("/data/queries.txt", text_size);
  find_query_chunks();
  size_t size = std::max<size_t>(std::bit_ceil(text_size / 16 + 16), 1 << 10);
  draft.len = size + TABLE_PADDING;
  draft.mask = size - 1;
  draft.hashes = (std::atomic<u64> *)calloc(draft.len, 8);
  query_chunks.resize(chunks);
  outputs.resize(chunks);
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
