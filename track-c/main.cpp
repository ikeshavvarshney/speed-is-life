#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <fcntl.h>
#include <immintrin.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

using u8 = uint8_t;
using u32 = uint32_t;
// Not size_t's type, so writing one never forces sizes and masks to be reloaded.
using u64 = unsigned long long;
using v4 = __m256i;
using std::atomic;
constexpr auto relaxed = std::memory_order_relaxed;
#define INLINE [[gnu::always_inline]] inline

const size_t CHUNK = 1 << 18, REGION_CHUNK = 1 << 16, UNMAP_STEP = 1 << 23;
const size_t COUNTRIES = 256, HUGE_PAGE = 1 << 21;
// Region ids stop below this at every scale (spec section 2.3). Its own entry
// stays 0 and stands for every id above the table.
const u32 REGION_END = 1000000;
const size_t NAMES_AT = REGION_END + 64;
const size_t CODES = 26 * 26 * 26;
// The generator's pool of 100,000 names, plus the few that edge-case blocks
// add, fits at 77% load.
const int NAME_BITS = 17;
// Words of one country's bitset over name ids.
const size_t WORDS = (1 << NAME_BITS) / 64;
const size_t BLOCKS = 8, MAX_MARKS = BLOCKS * 64 + 64;
const size_t MAX_RECORDS = MAX_MARKS / 5 + 1;

// One worker's running values for one country, on one cache line. `sums` is
// added to as one vector: id sum, amount sum, count << 32 | name bytes, note
// bytes. Neither half of the third can carry into the other, as both stay below
// 2^31.
struct alignas(64) Group {
  u64 sums[4] = {};
  int min = INT32_MAX, max = INT32_MIN;
};

struct Rec {
  __m128i values; // id, amount, name bytes, note bytes
  u64 hash;
  u32 region;
};

// Input mapped over a larger anonymous mapping, so vector loads near the end of
// the file stay readable.
static const u8 *map_input(const char *path, size_t &size) {
  int fd = open(path, O_RDONLY);
  struct stat info;
  fstat(fd, &info);
  size = info.st_size;
  void *base =
      mmap(0, size + 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  mmap(base, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
  return (const u8 *)base;
}

// Zeroed memory on 2 MB boundaries. Lives until process exit.
static u8 *zeroed(size_t bytes) {
  auto raw = (size_t)mmap(0, bytes + HUGE_PAGE, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return (u8 *)((raw + HUGE_PAGE - 1) & -HUGE_PAGE);
}

static const u8 *input, *regions;
static size_t input_size, regions_size;
static u8 *by_id;
static atomic<u64> *slots;
static atomic<u64> *name_bits;
static Group *groups;
static int threads;
static atomic<u8> seen_codes[CODES];
static std::mutex codes_lock;
static int code_count;
static atomic<u32> distinct_names[COUNTRIES];
// Written by every thread, so kept away from what records read.
alignas(64) static atomic<size_t> next_region{1}; // The first piece holds the header.
alignas(64) static atomic<size_t> next_chunk;
alignas(64) static atomic<size_t> next_country;
alignas(64) static atomic<size_t> next_unmap;

// Claims pieces of regions.csv. by_id holds 1 + the country's index, and 0 for
// anything unknown. Ids are unique, so each byte has one writer.
static void load_regions() {
  for (;;) {
    size_t off = next_region.fetch_add(REGION_CHUNK, relaxed);
    if (off >= regions_size)
      return;
    size_t stop = std::min(off + REGION_CHUNK, regions_size), i = off;
    while (regions[i - 1] != '\n')
      i++;
    while (i < stop) {
      size_t id = 0;
      for (; regions[i] != ','; i++)
        id = id * 10 + regions[i] - '0';
      const u8 *c = regions + i + 1;
      int key = (c[0] - 'A') * 676 + (c[1] - 'A') * 26 + c[2] - 'A';
      if (!seen_codes[key].load(relaxed)) {
        std::lock_guard guard(codes_lock);
        if (!seen_codes[key].load(relaxed))
          seen_codes[key].store(++code_count, relaxed);
      }
      by_id[id] = seen_codes[key].load(relaxed);
      i += 5;
    }
  }
}

// The bytes of up to eight digits ending just before `end`, with leading
// positions zeroed. Field lengths are known from the structural masks, so
// nothing branches on the number of digits.
INLINE u64 digit_word(const u8 *end, size_t len) {
  auto word = _mm_loadl_epi64((const __m128i *)(end - 8));
  return _mm_cvtsi128_si64(word) & (~0ull << (-(len * 8) & 63));
}

// Converts three words of eight digits together. Each step merges pairs: digits
// to 2-digit values, then 4, then 8. Three scalar conversions would queue on the
// one multiply port.
INLINE v4 decode(u64 a, u64 b, u64 c) {
  // In the amount's lane, `ddddd.cc` loses its dot, which leaves plain cents.
  const v4 drop_dot = _mm256_setr_epi8(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, //
      -1, 0, 1, 2, 3, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
  v4 digits = _mm256_shuffle_epi8(
      _mm256_and_si256(_mm256_set_epi64x(0, c, b, a), _mm256_set1_epi8(15)),
      drop_dot);
  v4 x = _mm256_maddubs_epi16(digits, _mm256_set1_epi16(0x010A));
  x = _mm256_madd_epi16(x, _mm256_set1_epi32(0x00010064));
  x = _mm256_packus_epi32(x, x);
  return _mm256_madd_epi16(x, _mm256_set1_epi32(0x00012710));
}

// 32 set bytes, then 32 clear ones, to mask a vector to a length.
alignas(64) static u8 LOW_BYTES[64];

// Up to 32 bytes are masked and mixed at a fixed cost, as a loop per word would
// mispredict on names of varying length. Longer names, which never occur in
// practice, take the loop. Names hold no NUL, so the padding cannot make two
// names equal and the length need not be mixed in. Every byte passes through at
// least three AES rounds after it enters, as two left crafted names colliding,
// which would change a count.
INLINE u64 hash_name(const u8 *p, size_t len, __m128i key) {
  v4 bytes = _mm256_loadu_si256((const v4 *)p);
  auto mask = (const v4 *)(LOW_BYTES + 32 - len);
  if (len > 32) [[unlikely]]
    mask = (const v4 *)LOW_BYTES;
  bytes = _mm256_and_si256(bytes, _mm256_loadu_si256(mask));
  __m128i state = _mm_aesenc_si128(
      _mm_xor_si128(_mm256_castsi256_si128(bytes), key),
      _mm256_extracti128_si256(bytes, 1));
  for (size_t i = 32; i < len; i += 16) [[unlikely]] {
    auto tail = LOW_BYTES + 32 - std::min<size_t>(len - i, 16);
    state = _mm_aesenc_si128(
        state, _mm_and_si128(_mm_loadu_si128((const __m128i *)(p + i)),
                             _mm_loadu_si128((const __m128i *)tail)));
  }
  state = _mm_aesenc_si128(_mm_aesenc_si128(state, key), key);
  return _mm_cvtsi128_si64(_mm_aesenc_si128(state, key)) | 1;
}

// Lock-free table of 64-bit name hashes, whose slot number is the name's id.
// Comparing hashes instead of bytes never touches the input again. With 100,000
// names a collision has a chance of about 3e-10, and would only make one name
// count as another. A bucket is one cache line of eight hashes compared at once,
// so a hit does not branch on where the hash sits. A bucket fills in order, so a
// thread that sees a free way has not missed its hash further along, and losing
// the race for a way only means looking at the winner's hash.
INLINE size_t intern(atomic<u64> *table, u64 hash) {
  size_t bucket = hash >> (67 - NAME_BITS);
  v4 want = _mm256_set1_epi64x(hash);
  auto mask = [](v4 x, v4 y) -> u32 {
    return _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(x, y)));
  };
  for (;;) {
    auto at = table + bucket * 8;
    v4 low = _mm256_load_si256((const v4 *)at);
    v4 high = _mm256_load_si256((const v4 *)(at + 4));
    u32 hit = mask(low, want) | mask(high, want) << 4;
    if (hit)
      return bucket * 8 + __builtin_ctz(hit);
    v4 zero = _mm256_setzero_si256();
    u32 free = mask(low, zero) | mask(high, zero) << 4;
    if (!free) {
      bucket = (bucket + 1) & ((1 << (NAME_BITS - 3)) - 1);
      continue;
    }
    int way = __builtin_ctz(free);
    u64 actual = 0;
    if (at[way].compare_exchange_strong(actual, hash, relaxed) || actual == hash)
      return bucket * 8 + way;
  }
}

static void parse_chunk(size_t first, size_t stop, Group *mine) {
  const u8 *base = input;
  const u8 *ids = by_id;
  const __m128i top = _mm_set1_epi32(REGION_END);
  const __m128i three = _mm_setr_epi16(3, 0, 3, 0, 0, 0, 0, 0);
  const __m128i key = _mm_set_epi64x(0x8BB84B93962EACC9, 0x2D358DCCAA6C78A5);
  const auto table = slots;
  const auto bits = name_bits;
  u32 marks[MAX_MARKS];
  // Number of "" pairs in the field each mark ends. It stays mostly zero.
  u32 pairs[MAX_MARKS + 8] = {};
  alignas(32) Rec recs[MAX_RECORDS];
  const v4 finder = _mm256_setr_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, -1, 44, -1, -1, -1, //
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, -1, 44, -1, -1, -1);
  const v4 quote = _mm256_set1_epi8('"');
  const v4 one_count = _mm256_set_epi64x(0, 1ll << 32, 0, 0);
  auto mask = [](v4 x, v4 y) -> u64 {
    return (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, y));
  };
  size_t block = first, filled = 0, start = first;
  u64 inside = 0, closed = 0;
  for (;;) {
    for (size_t b = 0; b < BLOCKS; b++, block += 64) {
      // Bitmasks of one 64-byte block. A carry-less multiply by all ones is a
      // running XOR of the quote bits, set from an opening quote up to its
      // closing quote. A pair inside a value closes and at once reopens, while
      // fields always have a comma between them, so an opening quote right
      // after a closing quote is exactly a "" pair.
      v4 x = _mm256_loadu_si256((const v4 *)(base + block));
      v4 y = _mm256_loadu_si256((const v4 *)(base + block + 32));
      u64 quotes = mask(x, quote) | mask(y, quote) << 32;
      u64 seps = mask(_mm256_shuffle_epi8(finder, x), x) |
                 mask(_mm256_shuffle_epi8(finder, y), y) << 32;
      __m128i q = _mm_cvtsi64_si128(quotes);
      u64 running =
          _mm_cvtsi128_si64(_mm_clmulepi64_si128(q, _mm_set1_epi8(-1), 0)) ^
          inside;
      inside = (long long)running >> 63;
      u64 closers = quotes & ~running;
      u64 found_pairs = quotes & running & (closers << 1 | closed);
      closed = closers >> 63;
      seps &= ~running;
      // A quarter of blocks hold pairs, which a branch would keep mispredicting,
      // and under 2% hold more than two. So two are counted unconditionally: with
      // no pair left, the bit is 0 and the add does nothing.
      auto count_pair = [&] {
        u64 bit = found_pairs & -found_pairs;
        pairs[filled + __builtin_popcountll(seps & (bit - 1))] += bit != 0;
        found_pairs ^= bit;
      };
      count_pair();
      count_pair();
      while (found_pairs) [[unlikely]]
        count_pair();
      size_t k = filled;
      filled += __builtin_popcountll(seps);
      // Blocks hold about six marks. Writing eight every time beats a branch.
      do {
        for (int i = 0; i < 8; i++, k++, seps &= seps - 1)
          marks[k] = block + _tzcnt_u64(seps);
      } while (seps);
    }
    size_t whole = filled / 5, n = 0;
    bool done = false;
    for (size_t r = 0; r < whole; r++) {
      if (start >= stop) {
        done = true;
        break;
      }
      const u32 *m = marks + r * 5;
      size_t c1 = m[0], c4 = m[3], lf = m[4];
      // One subtraction gives every field's length plus one. Taking 3 saturating
      // removes a separator and two quotes, and leaves an empty field at 0.
      // Fields are far below 64 KB, so 16 bits are enough here.
      __m128i at = _mm_loadu_si128((const __m128i *)m);
      __m128i gaps =
          _mm_sub_epi32(_mm_loadu_si128((const __m128i *)(m + 1)), at);
      __m128i raw = _mm_subs_epu16(gaps, three);
      // Rare enough for a branch to cost less than a calculation every time.
      if (base[start] == 0xEF) [[unlikely]]
        start += 3;
      size_t end = lf - (base[lf - 1] == '\r'), amount_at = c4 + 1;
      int sign = 0;
      if (base[amount_at] == '-') [[unlikely]] {
        sign = -1;
        amount_at++;
      }
      v4 numbers = decode(digit_word(base + c1, c1 - start),
                          digit_word(base + c4, c4 - m[2] - 1),
                          digit_word(base + end, end - amount_at));
      __m128i low = _mm256_castsi256_si128(numbers);
      __m128i region = _mm_min_epu32(_mm_srli_si128(low, 4), top);
      int amount = (_mm256_extract_epi32(numbers, 4) ^ sign) - sign;
      u32 region_id = _mm_cvtsi128_si32(region);
      u64 hash = hash_name(base + c1 + 2, (u32)_mm_cvtsi128_si32(raw), key);
      _mm_prefetch((const char *)(ids + region_id), _MM_HINT_T0);
      _mm_prefetch((const char *)(table + (hash >> (67 - NAME_BITS)) * 8),
                   _MM_HINT_T0);
      // `raw` less the "" pairs is the two text lengths.
      __m128i texts = _mm_sub_epi32(
          raw, _mm_loadu_si128((const __m128i *)(pairs + r * 5 + 1)));
      recs[n].values =
          _mm_unpacklo_epi64(_mm_insert_epi32(low, amount, 1), texts);
      recs[n].hash = hash;
      recs[n++].region = region_id;
      start = lf + 1;
    }
    // Group prefetching: this pass touches only what the pass above prefetched,
    // so the cache misses of a batch overlap instead of stalling one at a time.
    for (size_t r = 0; r < n; r++) {
      const Rec &rec = recs[r];
      Group &group = mine[ids[rec.region]];
      auto sums = (v4 *)group.sums;
      // Every value but the amount is positive, so sign extending suits them all.
      _mm256_store_si256(
          sums, _mm256_add_epi64(*sums, _mm256_add_epi64(one_count,
                                        _mm256_cvtepi32_epi64(rec.values))));
      int amount = _mm_extract_epi32(rec.values, 1);
      // A new extreme is rare, so this avoids two writes per record.
      if (amount < group.min) [[unlikely]]
        group.min = amount;
      if (amount > group.max) [[unlikely]]
        group.max = amount;
      size_t name = intern(table, rec.hash);
      size_t word = ids[rec.region] * WORDS + name / 64;
      __builtin_prefetch(bits + word, 1);
      recs[r].hash = word;
      recs[r].region = name % 64;
    }
    // A set bit costs no write, and a write costs a locked instruction that
    // often has to fetch the line from another core, so it is the last step.
    for (size_t r = 0; r < n; r++) {
      auto &word = bits[recs[r].hash];
      u64 bit = 1ull << recs[r].region;
      if (!(word.load(relaxed) & bit))
        word.fetch_or(bit, relaxed);
    }
    if (done || start >= stop)
      return;
    // Under five marks remain, and the mark after them may already have pairs
    // waiting for it. Fixed moves cost less than calls to copy so little.
    u32 kept_pairs[5];
    __builtin_memcpy(kept_pairs, pairs + whole * 5, 20);
    __builtin_memcpy(marks, marks + whole * 5, 16);
    for (size_t i = 0; i <= filled; i += 8)
      _mm256_storeu_si256((v4 *)(pairs + i), _mm256_setzero_si256());
    __builtin_memcpy(pairs, kept_pairs, 20);
    filled -= whole * 5;
  }
}

// Past a chunk's start, the next record starts after an LF followed by a digit
// or BOM. Newlines inside notes are always followed by a lowercase word or a
// quote (spec section 2.2).
static size_t first_record(size_t off) {
  size_t i = off;
  while (i < input_size && !(input[i - 1] == '\n' && (u8(input[i] - '0') < 10 || input[i] == 0xEF)))
    i++;
  return i;
}

static void worker(int index, std::barrier<> *barrier) {
  load_regions();
  barrier->arrive_and_wait();
  Group *mine = groups + index * COUNTRIES;
  std::uninitialized_default_construct_n(mine, COUNTRIES);
  for (;;) {
    size_t off = next_chunk.fetch_add(CHUNK, relaxed);
    if (off >= input_size)
      break;
    size_t first = off ? first_record(off)
                       : (const u8 *)memchr(input, '\n', input_size) - input + 1;
    size_t stop = std::min(off + CHUNK, input_size);
    if (first < stop)
      parse_chunk(first, stop, mine);
  }
  barrier->arrive_and_wait();

  // Process exit would otherwise unmap every page of the input on one thread,
  // while the clock runs. Clearing the page tables now, on all threads, removes
  // most of that cost.
  for (;;) {
    size_t off = next_unmap.fetch_add(UNMAP_STEP, relaxed);
    if (off >= input_size)
      break;
    madvise((void *)(input + off), std::min(UNMAP_STEP, input_size - off),
            MADV_DONTNEED);
  }

  for (;;) {
    size_t country = next_country.fetch_add(1, relaxed);
    if (country >= COUNTRIES)
      break;
    u32 distinct = 0;
    for (size_t i = 0; i < WORDS; i++)
      distinct += __builtin_popcountll(name_bits[country * WORDS + i].load(relaxed));
    distinct_names[country].store(distinct, relaxed);
  }
  barrier->arrive_and_wait();
}

static char *money(char *out, long long cents) {
  u64 abs = cents < 0 ? -cents : cents;
  return out + sprintf(out, " %s%llu.%02llu", cents < 0 ? "-" : "", abs / 100, abs % 100);
}

int main() {
  memset(LOW_BYTES, 255, 32);
  cpu_set_t cpus;
  sched_getaffinity(0, sizeof cpus, &cpus);
  threads = CPU_COUNT(&cpus);
  regions = map_input("/data/regions.csv", regions_size);
  input = map_input("/data/records.csv", input_size);
  // The randomly accessed tables share a huge page, which avoids a TLB miss on
  // most records. They are touched before the workers start: threads racing to
  // fault one huge page are each charged 2 MB until the race resolves, which
  // shows up in peak memory.
  size_t tables_size = NAMES_AT + (8ull << NAME_BITS);
  by_id = zeroed(tables_size);
  by_id[0] = 0;
  slots = (atomic<u64> *)(by_id + NAMES_AT);
  // 241 groups fill two huge pages. Touched first, like the tables above.
  name_bits = (atomic<u64> *)zeroed(COUNTRIES * WORDS * 8);
  for (size_t page = 0; page < 2; page++)
    *(u8 *)(name_bits + page * HUGE_PAGE / 8) = 0;
  // The box has huge pages always on. From here, a first touch of a thread's stack
  // or the groups takes 4 KB, not 2 MB.
  prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
  groups = (Group *)zeroed(threads * COUNTRIES * sizeof(Group));
  std::barrier barrier(threads);
  for (int t = 1; t < threads; t++)
    std::thread(worker, t, &barrier).detach();
  worker(0, &barrier);

  // Countries in code order are already sorted, and _UNMATCHED sorts last.
  char text[32768], *out = text;
  for (size_t code = 0; code <= CODES; code++) {
    int country = code < CODES ? seen_codes[code].load(relaxed) : 0;
    if (!country && code < CODES)
      continue;
    char key[] = {char('A' + code / 676), char('A' + code / 26 % 26),
                  char('A' + code % 26), 0};
    Group total;
    for (int t = 0; t < threads; t++) {
      const Group &from = groups[t * COUNTRIES + country];
      for (int i = 0; i < 4; i++)
        total.sums[i] += from.sums[i];
      total.min = std::min(total.min, from.min);
      total.max = std::max(total.max, from.max);
    }
    u64 count = total.sums[2] >> 32;
    if (!count)
      continue;
    out += sprintf(out, "%s %llu %llu", code < CODES ? key : "_UNMATCHED", count, total.sums[0]);
    out = money(out, total.sums[1]);
    out = money(out, total.min);
    out = money(out, total.max);
    out += sprintf(out, " %u %u %llu\n", distinct_names[country].load(relaxed),
                   (u32)total.sums[2], total.sums[3]);
  }
  int fd = open("/out/aggregates.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (write(fd, text, out - text) != out - text)
    abort();
  _exit(0);
}
