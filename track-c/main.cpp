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
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using v4 = __m256i;
using std::atomic;
constexpr auto relaxed = std::memory_order_relaxed;
#define INLINE [[gnu::always_inline]] inline

const size_t CHUNK = 1 << 18, REGION_CHUNK = 1 << 16, UNMAP_STEP = 1 << 23;
const size_t COUNTRIES = 256, HUGE_PAGE = 1 << 21;
// The generator's pool of 100,000 names, plus the few that edge-case blocks
// add, fits at 77% load.
const int NAME_BITS = 17;
// Words of one country's bitset over name ids.
const size_t WORDS = (1 << NAME_BITS) / 64;
const size_t BLOCKS = 16, MAX_MARKS = BLOCKS * 64 + 64;
const size_t MAX_RECORDS = MAX_MARKS / 5 + 1;

// One worker's running values for one country, on one cache line. `sums` is
// added to as one vector: id sum, amount sum, count << 32 | name bytes, note
// bytes. Neither half of the third can carry into the other, as both stay below
// 2^31.
struct alignas(64) Group {
  u64 sums[4] = {};
  int min = INT32_MAX, max = INT32_MIN;
};

// Counters that every thread writes, away from what records read.
struct alignas(64) Counter {
  atomic<size_t> value{0};
  size_t take(size_t n) { return value.fetch_add(n, relaxed); }
};

struct Rec {
  v4 sums;
  u64 hash;
  u32 region;
  int amount;
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
static u8 *zeroed(size_t bytes, int advice) {
  auto raw = (size_t)mmap(0, bytes + HUGE_PAGE, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  auto aligned = (u8 *)((raw + HUGE_PAGE - 1) & -HUGE_PAGE);
  madvise(aligned, bytes, advice);
  return aligned;
}

static const u8 *input, *regions;
static size_t input_size, regions_size;
static u8 *by_id;
static u32 last_region;
static atomic<u64> *slots;
static atomic<u64> *name_bits;
static Group *groups;
static int threads;
static atomic<u8> seen_codes[26 * 26 * 26];
static std::mutex codes_lock;
static std::vector<std::string> codes;
static atomic<u32> distinct_names[COUNTRIES];
static Counter next_region, next_chunk, next_country, next_unmap;

// Claims pieces of regions.csv. by_id holds 1 + the country's index, and 0 for
// anything unknown. Its last entry stays 0 and stands for every id above the
// table. Ids are unique, so each byte has one writer.
static void load_regions() {
  for (;;) {
    size_t off = next_region.take(REGION_CHUNK);
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
      u8 country = seen_codes[key].load(relaxed);
      if (!country) {
        std::lock_guard guard(codes_lock);
        country = seen_codes[key].load(relaxed);
        if (!country) {
          codes.emplace_back((const char *)c, 3);
          country = codes.size();
          seen_codes[key].store(country, relaxed);
        }
      }
      by_id[id] = country;
      i += 5;
    }
  }
}

INLINE u64 load64(const u8 *p) {
  u64 word;
  __builtin_memcpy(&word, p, 8);
  return word;
}

// The bytes of up to eight digits ending just before `end`, with leading
// positions zeroed. Field lengths are known from the structural masks, so
// nothing branches on the number of digits.
INLINE u64 digit_word(const u8 *end, size_t len) {
  return load64(end - 8) & (~0ull << ((8 - len) * 8));
}

// Converts three words of eight digits together. Each step merges pairs: digits
// to 2-digit values, then 4, then 8. Three scalar conversions would queue on the
// one multiply port.
INLINE v4 decode(u64 a, u64 b, u64 c) {
  v4 digits = _mm256_and_si256(_mm256_set_epi64x(0, c, b, a),
                               _mm256_set1_epi8(15));
  v4 x = _mm256_maddubs_epi16(digits, _mm256_set1_epi16(0x010A));
  x = _mm256_madd_epi16(x, _mm256_set1_epi32(0x00010064));
  x = _mm256_packus_epi32(x, x);
  return _mm256_madd_epi16(x, _mm256_set1_epi32(0x00012710));
}

// 32 set bytes, then 32 clear ones, to mask a vector to a length.
static u8 LOW_BYTES[64];

// Up to 32 bytes are masked and mixed at a fixed cost, as a loop per word would
// mispredict on names of varying length. Longer names, which are rare, loop over
// the rest. Every byte passes through at least two AES rounds after it enters,
// which spreads it over the whole state.
INLINE u64 hash_name(const u8 *p, size_t len) {
  auto mask = (const v4 *)(LOW_BYTES + 32 - std::min<size_t>(len, 32));
  v4 bytes = _mm256_and_si256(_mm256_loadu_si256((const v4 *)p),
                              _mm256_loadu_si256(mask));
  __m128i key =
      _mm_set_epi64x(0x8BB84B93962EACC9, 0x2D358DCCAA6C78A5 ^ (long long)len);
  __m128i low = _mm_xor_si128(_mm256_castsi256_si128(bytes), key);
  __m128i state = _mm_aesenc_si128(low, _mm256_extracti128_si256(bytes, 1));
  for (size_t i = 32; i < len; i += 16) {
    auto tail = LOW_BYTES + 32 - std::min<size_t>(len - i, 16);
    state = _mm_aesenc_si128(
        state, _mm_and_si128(_mm_loadu_si128((const __m128i *)(p + i)),
                             _mm_loadu_si128((const __m128i *)tail)));
  }
  state = _mm_aesenc_si128(state, key);
  return _mm_cvtsi128_si64(_mm_aesenc_si128(state, key)) | 1;
}

// Lock-free table of 64-bit name hashes, whose slot number is the name's id.
// Comparing hashes instead of bytes never touches the input again. With 100,000
// names a collision has a chance of about 3e-10, and would only make one name
// count as another. A bucket is one cache line of eight hashes compared at once,
// so a hit does not branch on where the hash sits. A bucket fills in order, so a
// thread that sees a free way has not missed its hash further along, and losing
// the race for a way only means looking at the winner's hash.
INLINE size_t intern(u64 hash) {
  size_t bucket = hash >> (67 - NAME_BITS), probes = 0;
  v4 want = _mm256_set1_epi64x(hash);
  auto mask = [](v4 x, v4 y) -> u32 {
    return _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(x, y)));
  };
  for (;;) {
    auto at = slots + bucket * 8;
    v4 low = _mm256_load_si256((const v4 *)at);
    v4 high = _mm256_load_si256((const v4 *)(at + 4));
    u32 hit = mask(low, want) | mask(high, want) << 4;
    if (hit)
      return bucket * 8 + __builtin_ctz(hit);
    v4 zero = _mm256_setzero_si256();
    u32 free = mask(low, zero) | mask(high, zero) << 4;
    if (!free) {
      // The generator's pool is a constant: this only guards against hanging.
      if (++probes >> (NAME_BITS - 3))
        abort();
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
  const u32 top = last_region;
  u32 marks[MAX_MARKS];
  // Number of "" pairs in the field each mark ends. It stays mostly zero.
  u32 pairs[MAX_MARKS + 1] = {};
  alignas(32) Rec recs[MAX_RECORDS];
  const v4 finder = _mm256_setr_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, -1, 44, -1, -1, -1, //
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, -1, 44, -1, -1, -1);
  const v4 quote = _mm256_set1_epi8('"');
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
      for (int i = 0; i < 2 || found_pairs; i++) {
        u64 bit = found_pairs & -found_pairs;
        pairs[filled + __builtin_popcountll(seps & (bit - 1))] += bit != 0;
        found_pairs ^= bit;
      }
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
      size_t c1 = m[0], c2 = m[1], c3 = m[2], c4 = m[3], lf = m[4];
      size_t id_at = start + 3 * (base[start] == 0xEF);
      u32 name_raw = c2 - c1 - 1, note_raw = c3 - c2 - 1;
      name_raw -= std::min(name_raw, 2u);
      note_raw -= std::min(note_raw, 2u);
      size_t end = lf - (base[lf - 1] == '\r');
      int negative = base[c4 + 1] == '-';
      u64 text = digit_word(base + end, end - c4 - 1 - negative);
      v4 numbers = decode(digit_word(base + c1, c1 - id_at),
                          digit_word(base + c4, c4 - c3 - 1), text << 24);
      __m128i low = _mm256_castsi256_si128(numbers);
      u32 id = _mm_cvtsi128_si32(low);
      u32 region = std::min<u32>(_mm_extract_epi32(low, 1), top);
      u32 whole_part = _mm256_extract_epi32(numbers, 4);
      int cents = whole_part * 100 + ((text >> 48) & 15) * 10 + (text >> 56 & 15);
      int amount = (cents ^ -negative) + negative;
      u64 hash = hash_name(base + c1 + 2, name_raw);
      _mm_prefetch((const char *)(ids + region), _MM_HINT_T0);
      _mm_prefetch((const char *)(slots + (hash >> (67 - NAME_BITS)) * 8),
                   _MM_HINT_T0);
      recs[n].sums = _mm256_set_epi64x(
          note_raw - pairs[r * 5 + 2],
          (1ll << 32) | (name_raw - pairs[r * 5 + 1]), amount, id);
      recs[n].hash = hash;
      recs[n].region = region;
      recs[n++].amount = amount;
      start = lf + 1;
    }
    // Group prefetching: this pass touches only what the pass above prefetched,
    // so the cache misses of a batch overlap instead of stalling one at a time.
    for (size_t r = 0; r < n; r++) {
      const Rec &rec = recs[r];
      Group &group = mine[ids[rec.region]];
      auto sums = (v4 *)group.sums;
      _mm256_store_si256(sums, _mm256_add_epi64(*sums, rec.sums));
      group.min = std::min(group.min, rec.amount);
      group.max = std::max(group.max, rec.amount);
      size_t name = intern(rec.hash);
      size_t word = ids[rec.region] * WORDS + name / 64;
      __builtin_prefetch(name_bits + word, 1);
      recs[r].hash = word;
      recs[r].region = name % 64;
    }
    // A set bit costs no write, and a write costs a locked instruction that
    // often has to fetch the line from another core, so it is the last step.
    for (size_t r = 0; r < n; r++) {
      auto &word = name_bits[recs[r].hash];
      u64 bit = 1ull << recs[r].region;
      if (!(word.load(relaxed) & bit))
        word.fetch_or(bit, relaxed);
    }
    if (done || start >= stop)
      return;
    size_t left = filled - whole * 5;
    std::copy(marks + whole * 5, marks + filled, marks);
    // The mark after the last one found may already have pairs waiting for it.
    std::copy(pairs + whole * 5, pairs + filled + 1, pairs);
    std::fill(pairs + left + 1, pairs + filled + 1, 0);
    filled = left;
  }
}

// Past a chunk's start, the next record starts after an LF followed by a digit
// or BOM. Newlines inside notes are always followed by a lowercase word or a
// quote (spec section 2.2).
static size_t first_record(size_t off) {
  for (size_t i = off - 1; i + 1 < input_size; i++) {
    u8 next = input[i + 1];
    if (input[i] == '\n' && ((next >= '0' && next <= '9') || next == 0xEF))
      return i + 1;
  }
  return input_size;
}

static void worker(int index, std::barrier<> *barrier) {
  load_regions();
  barrier->arrive_and_wait();
  Group *mine = groups + index * COUNTRIES;
  std::uninitialized_default_construct_n(mine, COUNTRIES);
  for (;;) {
    size_t off = next_chunk.take(CHUNK);
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
    size_t off = next_unmap.take(UNMAP_STEP);
    if (off >= input_size)
      break;
    madvise((void *)(input + off), std::min(UNMAP_STEP, input_size - off),
            MADV_DONTNEED);
  }

  for (;;) {
    size_t country = next_country.take(1);
    if (country >= COUNTRIES)
      return;
    u32 distinct = 0;
    for (size_t i = 0; i < WORDS; i++)
      distinct += __builtin_popcountll(name_bits[country * WORDS + i].load(relaxed));
    distinct_names[country].store(distinct, relaxed);
  }
}

static char *money(char *out, long long cents) {
  auto abs = (unsigned long long)(cents < 0 ? -cents : cents);
  return out + sprintf(out, " %s%llu.%02llu", cents < 0 ? "-" : "", abs / 100,
                       abs % 100);
}

int main() {
  memset(LOW_BYTES, 255, 32);
  cpu_set_t cpus;
  sched_getaffinity(0, sizeof cpus, &cpus);
  threads = CPU_COUNT(&cpus);
  regions = map_input("/data/regions.csv", regions_size);
  input = map_input("/data/records.csv", input_size);
  size_t line = regions_size - 1;
  while (regions[line - 1] != '\n')
    line--;
  for (; regions[line] != ','; line++)
    last_region = last_region * 10 + regions[line] - '0';
  last_region++;
  // The randomly accessed tables share a huge page, which avoids a TLB miss on
  // most records. They are touched before the workers start: threads racing to
  // fault one huge page are each charged 2 MB until the race resolves, which
  // shows up in peak memory.
  size_t names_at = (last_region + 64) & -64ull;
  size_t tables_size = names_at + (8ull << NAME_BITS);
  by_id = zeroed(tables_size, MADV_HUGEPAGE);
  for (size_t page = 0; page < tables_size; page += HUGE_PAGE)
    by_id[page] = 0;
  slots = (atomic<u64> *)(by_id + names_at);
  // 241 groups fill two huge pages. Touched first, like the tables above.
  name_bits = (atomic<u64> *)zeroed(COUNTRIES * WORDS * 8, MADV_HUGEPAGE);
  for (size_t page = 0; page < 2; page++)
    *(u8 *)(name_bits + page * HUGE_PAGE / 8) = 0;
  groups = (Group *)zeroed(threads * COUNTRIES * sizeof(Group), MADV_NOHUGEPAGE);
  next_region.value = 1; // The first piece starts after the header.
  std::barrier barrier(threads);
  std::vector<std::thread> pool;
  for (int t = 1; t < threads; t++)
    pool.emplace_back(worker, t, &barrier);
  worker(0, &barrier);
  for (auto &thread : pool)
    thread.join();

  std::vector<std::pair<std::string, int>> keys{{"_UNMATCHED", 0}};
  for (size_t i = 0; i < codes.size(); i++)
    keys.push_back({codes[i], i + 1});
  std::sort(keys.begin(), keys.end());
  char text[32768], *out = text;
  for (auto &[key, country] : keys) {
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
    out += sprintf(out, "%s %llu %llu", key.c_str(), (unsigned long long)count,
                   (unsigned long long)total.sums[0]);
    out = money(out, (long long)total.sums[1]);
    out = money(out, total.min);
    out = money(out, total.max);
    out += sprintf(out, " %u %u %llu\n", distinct_names[country].load(relaxed),
                   (u32)total.sums[2], (unsigned long long)total.sums[3]);
  }
  int fd = open("/out/aggregates.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (write(fd, text, out - text) != out - text)
    abort();
  _exit(0);
}
