#include <algorithm>
#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

using u8 = uint8_t;
using u32 = uint32_t;
// Not size_t's type, so writing one never forces sizes and masks to be reloaded.
using u64 = unsigned long long;
using v4 = __m256i;
using std::atomic;
constexpr auto relaxed = std::memory_order_relaxed;
#define INLINE [[gnu::always_inline]] inline
#define LOAD(p) _mm_loadu_si128((const __m128i *)(p))

const size_t CHUNK = 1 << 18, REGION_CHUNK = 1 << 16, UNMAP = 1 << 23;
// 240 countries at every scale (spec section 2.3), plus _UNMATCHED. The box has 12
// threads, and more than that would only cost time.
const size_t COUNTRIES = 241, HUGE_PAGE = 1 << 21, MAX_THREADS = 12;
const size_t STACK = 1 << 13, LINE = 128;
// Region ids stop below this at every scale (spec section 2.3). Its own entry
// stays 0 and stands for every id above the table.
const u32 REGION_END = 1000000;
const size_t NAMES_AT = REGION_END + 64, CODES = 26 * 26 * 26;
// The generator's pool of 100,000 names, plus the few that edge-case blocks
// add, fits at 77% load.
const int NAME_BITS = 17;
// Words of one country's bitset over name ids.
const size_t WORDS = (1 << NAME_BITS) / 64;
// The shortest record is 11 bytes and has 5 marks, so a batch holds at most 466
// marks, plus 4 left over and 8 written past the last.
const size_t BLOCKS = 16, MAX_MARKS = 512, MAX_RECORDS = MAX_MARKS / 5 + 1;

// One worker's running values for one country, on one cache line. `sums` is
// added to as one vector: id sum, amount sum, count << 32 | name bytes, note
// bytes. Neither half of the third can carry into the other, as both stay below
// 2^31.
struct alignas(64) Group {
  u64 sums[4];
  int min, max;
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
  size = lseek(fd, 0, SEEK_END);
  // On a 2 MB boundary, so that each page table covers exactly 2 MB of the file.
  size_t raw = (size_t)mmap(0, size + 4096 + HUGE_PAGE, PROT_READ,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  auto base = (const u8 *)((raw + HUGE_PAGE - 1) & -HUGE_PAGE);
  mmap((void *)base, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
  // Page tables for the first of it. See the note in main.
  for (size_t off = 0; off < std::min(size, 8 * HUGE_PAGE); off += HUGE_PAGE)
    *(volatile const u8 *)(base + off);
  return base;
}

// Work split into pieces that threads claim as they become free. Threads wait by
// spinning, as they do so for microseconds, and waking from a futex takes longer.
struct Job {
  alignas(64) atomic<size_t> next, done;
};

// One mapping at a fixed address, so its addresses are constants and take no
// register. The address is above where the kernel may randomize the heap to (up
// to 1 GB), where libc's thread block lives, and below 2 GB, the limit for
// constants in instructions. Huge pages hold the first 6 MB, which avoids a TLB miss on most
// records. Groups and stacks fit in what the tables leave of those pages, so
// they cost no memory or page faults. A huge page needs 2 MB of mapping, so the
// rest takes 4 KB pages.
struct Memory {
  u8 by_id[NAMES_AT];
  atomic<u64> slots[1 << NAME_BITS], name_bits[COUNTRIES * WORDS];
  Group groups[MAX_THREADS * COUNTRIES];
  u8 stacks[MAX_THREADS][STACK];
  alignas(4096) u8 lengths[COUNTRIES];
  std::atomic_flag codes_lock;
  int code_count;
  atomic<bool> bits_ready;
  Job region_job, chunk_job, country_job;
  // One more than there are codes, so the last entry stands for _UNMATCHED.
  atomic<u8> seen_codes[CODES + 1];
  char code_of[COUNTRIES][3], lines[COUNTRIES][LINE], text[COUNTRIES * LINE];
};
#define mem (*(Memory *)0x60000000)

static const u8 *input, *regions;
static size_t input_size, regions_size, threads;

template <class F> INLINE void run(Job &job, size_t size, size_t step, F f) {
  for (size_t i; (i = job.next.fetch_add(step, relaxed)) < size; job.done += step)
    f(i);
  while (job.done < size)
    _mm_pause();
}

// One piece of regions.csv. by_id holds 1 + the country's index, and 0 for
// anything unknown. Ids are unique, so each byte has one writer.
static void load_regions(size_t off) {
  size_t i = off + 1, stop = std::min(i + REGION_CHUNK, regions_size);
  while (regions[i - 1] != '\n')
    i++;
  while (i < stop) {
    size_t id = 0;
    for (; regions[i] != ','; i++)
      id = id * 10 + regions[i] - '0';
    const u8 *c = regions + i + 1;
    int key = (c[0] - 'A') * 676 + (c[1] - 'A') * 26 + c[2] - 'A';
    if (!mem.seen_codes[key].load(relaxed)) {
      while (mem.codes_lock.test_and_set(std::memory_order_acquire))
        _mm_pause();
      if (!mem.seen_codes[key].load(relaxed)) {
        memcpy(mem.code_of[++mem.code_count], c, 3);
        mem.seen_codes[key].store(mem.code_count, relaxed);
      }
      mem.codes_lock.clear(std::memory_order_release);
    }
    mem.by_id[id] = mem.seen_codes[key].load(relaxed);
    i += 5;
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
  const v4 drop_dot = _mm256_set_epi64x(-1, 0x07060403020100FF, 0x0F0E0D0C0B0A0908,
                                        0x0706050403020100);
  v4 digits = _mm256_shuffle_epi8(
      _mm256_set_epi64x(0, c, b, a) & _mm256_set1_epi8(15), drop_dot);
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
  bytes &= _mm256_loadu_si256(mask);
  __m128i state = _mm_aesenc_si128(_mm256_castsi256_si128(bytes) ^ key,
                                   _mm256_extracti128_si256(bytes, 1));
  for (size_t i = 32; i < len; i += 16) [[unlikely]] {
    auto tail = LOW_BYTES + 32 - std::min<size_t>(len - i, 16);
    state = _mm_aesenc_si128(state, LOAD(p + i) & LOAD(tail));
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
  const u8 *base = input, *ids = mem.by_id;
  const __m128i top = _mm_set1_epi32(REGION_END);
  const __m128i three = _mm_setr_epi16(3, 0, 3, 0, 0, 0, 0, 0);
  const __m128i key = _mm_set_epi64x(0x8BB84B93962EACC9, 0x2D358DCCAA6C78A5);
  const auto table = mem.slots, bits = mem.name_bits;
  u32 marks[MAX_MARKS];
  // Number of "" pairs in the field each mark ends. It stays mostly zero.
  u32 pairs[MAX_MARKS + 8] = {};
  alignas(32) Rec recs[MAX_RECORDS];
  // Indexed by a byte's low bits, this returns the byte itself only for LF and comma.
  const long long lf_comma = 0xFFFFFF2CFF0AFFFF;
  const v4 finder = _mm256_set_epi64x(lf_comma, -1, lf_comma, -1);
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
      __m128i q = _mm_clmulepi64_si128(_mm_cvtsi64_si128(quotes),
                                       _mm_set1_epi8(-1), 0);
      u64 running = _mm_cvtsi128_si64(q) ^ inside;
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
    // The hardware prefetcher stops at each 4 KB page, so without this the first
    // lines of every page are demand misses to memory.
    if ((block & 4095) >= 4096 - BLOCKS * 64)
      for (int i = 1; i <= 6; i++)
        _mm_prefetch(base + (block | 4095) + i * 64, _MM_HINT_T0);
    size_t whole = filled / 5, n = 0;
    for (size_t r = 0; r < whole && start < stop; r++) {
      const u32 *m = marks + r * 5;
      size_t c1 = m[0], c4 = m[3], lf = m[4];
      // One subtraction gives every field's length plus one. Taking 3 saturating
      // removes a separator and two quotes, and leaves an empty field at 0.
      // Fields are far below 64 KB, so 16 bits are enough here.
      __m128i raw =
          _mm_subs_epu16(_mm_sub_epi32(LOAD(m + 1), LOAD(m)), three);
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
      _mm_prefetch(ids + region_id, _MM_HINT_T0);
      _mm_prefetch(table + (hash >> (67 - NAME_BITS)) * 8, _MM_HINT_T0);
      // `raw` less the "" pairs is the two text lengths.
      __m128i texts = _mm_sub_epi32(raw, LOAD(pairs + r * 5 + 1));
      recs[n++] = {_mm_unpacklo_epi64(_mm_insert_epi32(low, amount, 1), texts),
                   hash, region_id};
      start = lf + 1;
    }
    // Group prefetching: this pass touches only what the pass above prefetched,
    // so the cache misses of a batch overlap instead of stalling one at a time.
    for (size_t r = 0; r < n; r++) {
      const Rec &rec = recs[r];
      Group &group = mine[ids[rec.region]];
      auto sums = (v4 *)group.sums;
      // Every value but the amount is positive, so sign extending suits them all.
      *sums += one_count + _mm256_cvtepi32_epi64(rec.values);
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
    if (start >= stop)
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

// A space and a number written backward, ending just before `p`. With `dot` set
// to 2, the number is cents and prints as money, always with three digits.
static char *field(char *p, long long value, int dot = -1) {
  u64 left = value < 0 ? -value : value;
  for (int i = 0; left || i <= dot || !i; i++, left /= 10) {
    if (i == dot)
      *--p = '.';
    *--p = '0' + left % 10;
  }
  if (value < 0)
    *--p = '-';
  *--p = ' ';
  return p;
}

// Merges one country and formats its line. Each worker's group was last written
// by another core, so doing this on all threads overlaps those misses.
static void finish_country(size_t country) {
  // Process exit would otherwise unmap every page of the input on one thread,
  // while the clock runs. Clearing the page tables now, on all threads, removes
  // most of that cost. The first turns also take one piece of it each. Each call
  // interrupts every thread to flush its TLB, so calls must be few.
  // A call must stop at the end of the file, or it clears what is mapped above.
  size_t off = country * UNMAP;
  if (off < input_size)
    madvise((u8 *)input + off, std::min(UNMAP, input_size - off), MADV_DONTNEED);
  Group total{{}, INT32_MAX, INT32_MIN};
  for (size_t t = 0; t < threads; t++) {
    const Group &from = mem.groups[t * COUNTRIES + country];
    *(v4 *)total.sums += *(const v4 *)from.sums;
    total.min = std::min(total.min, from.min);
    total.max = std::max(total.max, from.max);
  }
  u64 count = total.sums[2] >> 32, distinct = 0;
  if (!count)
    return;
  for (size_t i = 0; i < WORDS; i++)
    distinct += __builtin_popcountll(mem.name_bits[country * WORDS + i]);
  char *end = mem.lines[country] + LINE, *p = end - 1;
  *p = '\n';
  p = field(p, total.sums[3]);
  p = field(p, (u32)total.sums[2]);
  p = field(p, distinct);
  p = field(p, total.max, 2);
  p = field(p, total.min, 2);
  p = field(p, total.sums[1], 2);
  p = field(p, total.sums[0]);
  p = field(p, count);
  size_t key = country ? 3 : 10;
  memcpy(p -= key, country ? mem.code_of[country] : "_UNMATCHED", key);
  mem.lengths[country] = end - p;
}

// Runs on a thread with no state of its own in libc, so it may call only what
// keeps none. Returning ends the thread with no stack to unmap or free.
static int worker(void *index) {
  run(mem.region_job, regions_size - 1, REGION_CHUNK, load_regions);
  while (!mem.bits_ready)
    _mm_pause();
  // See the note in main. The other threads start at the file's start, and this
  // moves far faster than they do, so it stays ahead of them.
  if (!index)
    for (size_t off = 0; off < input_size; off += HUGE_PAGE)
      *(volatile const u8 *)(input + off);
  Group *mine = mem.groups + (size_t)index * COUNTRIES;
  run(mem.chunk_job, input_size, CHUNK, [=](size_t off) {
    // The first search skips the header, as it starts with no LF before it.
    size_t first = first_record(off + !off);
    size_t stop = std::min(off + CHUNK, input_size);
    // Faults together here, not between the batches of a chunk, as each one costs
    // the parsing loop its cached state. The kernel maps 64 KB per fault.
    for (size_t at = off; at < stop; at += 1 << 16)
      *(volatile const u8 *)(input + at);
    if (first < stop)
      parse_chunk(first, stop, mine);
  });
  run(mem.country_job, COUNTRIES, 1, finish_country);
  return 0;
}

int main() {
  memset(LOW_BYTES, 255, 32);
  cpu_set_t cpus;
  sched_getaffinity(0, sizeof cpus, &cpus);
  threads = std::min<size_t>(CPU_COUNT(&cpus), MAX_THREADS);
  regions = map_input("/data/regions.csv", regions_size);
  input = map_input("/data/records.csv", input_size);
  if (mmap(&mem, sizeof(Memory), PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0) != &mem)
    abort();
  // The first memory charge on a CPU reserves 256 KB for that CPU, and peak memory
  // counts the reserve. So this thread faults what the workers would otherwise
  // be charged for: the huge pages, the small pages they write, and page tables
  // for the input, of which thread 0 makes the rest as they start on it.
  auto clear_page = [](int page) { ((volatile u8 *)&mem)[page * HUGE_PAGE] = 0; };
  clear_page(0);
  clear_page(2);
  madvise(mem.lengths, 1 << 16, MADV_POPULATE_WRITE);
  // Here and not in the workers, as a late one must not be merged half made.
  for (Group &group : mem.groups)
    group.min = INT32_MAX, group.max = INT32_MIN;
  for (size_t t = 1; t < threads; t++)
    clone(worker, mem.stacks[t] + STACK,
          CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD,
          (void *)t);
  // Only the bitsets are on this page, so the workers can load regions meanwhile.
  clear_page(1);
  mem.bits_ready = true;
  worker(0);

  // Countries in code order are already sorted, and _UNMATCHED sorts last.
  char *out = mem.text;
  for (size_t code = 0; code <= CODES; code++)
    if (int c = mem.seen_codes[code]; c || code == CODES) {
      memcpy(out, mem.lines[c] + LINE - mem.lengths[c], mem.lengths[c]);
      out += mem.lengths[c];
    }
  int fd = open("/out/aggregates.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (write(fd, mem.text, out - mem.text) != out - mem.text)
    abort();
  _exit(0);
}
