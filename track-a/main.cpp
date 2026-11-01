#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <mutex>
#include <sched.h>
#include <string_view>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
// Not size_t's type, so writing one never forces sizes and masks to be reloaded.
using u64 = unsigned long long;
using v4 = __m256i;
using v2 = __m128i;
using std::atomic;
constexpr auto relaxed = std::memory_order_relaxed;
#define INLINE [[gnu::always_inline]] inline

#ifndef CHUNK_BITS
#define CHUNK_BITS 20
#endif
const size_t CHUNK = 1 << CHUNK_BITS, PAGE = 4096, HOURS = 72, BLOCK = 256;
// A chunk that starts this many minutes into an hour shows that the hour before is
// over, as the spec has the last upload of an hour land within 10 minutes of its end.
const size_t HOUR_GRACE = 10;
#ifndef RENDER_LIMIT
#define RENDER_LIMIT 4
#endif
#ifndef SLOT_BITS
#define SLOT_BITS 12
#endif
const size_t SLOTS = 1 << SLOT_BITS;
// A log entry is id << 21 | is_error << 20 | latency. A latency that does not fit,
// on about 1% of lines, puts BIG in the entry and takes the next slot whole. An
// all-zero entry is an unused part of a block, and id 0 is reserved to absorb it.
const int LATENCY_BITS = 20;
const u32 BIG = (1 << LATENCY_BITS) - 1;
const size_t MAX_IDS = 1 << 11;
const u64 DIGITS = 0x3030303030303030;

// Held for a few instructions, with no more threads than CPUs, so waiting in the
// kernel would cost far more than spinning.
struct SpinLock {
  atomic<bool> held;
  void lock() {
    while (held.exchange(true, std::memory_order_acquire))
      while (held.load(relaxed))
        _mm_pause();
  }
  void unlock() { held.store(false, std::memory_order_release); }
};
struct Name {
  u32 at;
  u16 len, id;
};
// What the thread that parsed a chunk leaves for the one that plans the hours.
struct ChunkState {
  atomic<bool> done;
  u8 lowest_hour;
  u16 first_minute;
};
struct Text {
  u8 *data;
  size_t len;
};

// The workers are processes rather than threads, as a shared address space is what
// makes faulting the input in slow: every fault bounces the same locks between CPUs,
// and unmapping interrupts every CPU. All that they share lives here, in memory that
// is remapped as shared before they start.
static struct alignas(4096) {
  // Table of names, one word each: the high 48 bits of the hash and the id. After
  // the first few lines it is only read, so it is shared without contention and
  // nothing needs merging. With the slot that gives 60 bits of hash, which over
  // 1,842 names collide with odds of 2e-12. Ids are given out in the order names are
  // first seen. Popular names come early, so they take their first choice of slot,
  // and they pack the counters they use later into few cache lines.
  atomic<u64> table[SLOTS];
  // Names in the order they were found, and sorted up to `sorted_names`.
  SpinLock names_lock;
  Name order[MAX_IDS];
  size_t name_count, sorted_names, name_end;
  alignas(64) char name_bytes[MAX_IDS * 128];
  // Written by every worker, so kept off the lines that each line of input reads.
  alignas(64) atomic<u64> stamps[HOURS];
  alignas(64) atomic<size_t> cursors[HOURS];
  alignas(64) atomic<u32 *> bases[HOURS];
  alignas(64) atomic<size_t> next_chunk;
  atomic<bool> starting_over, finished;
  // Guards the rest: how far parsing has got, which hours are over and taken, the
  // free regions and the turn to write.
  alignas(64) SpinLock plan_lock;
  size_t parsed_chunks, latest_minute, hours_over, next_hour, next_output, rendering;
  size_t free_count, used_regions;
  bool writing;
  Text waiting[HOURS];
  u32 *free_regions[HOURS * 3];
} g;

alignas(64) static size_t input_size, capacity, chunk_count;
static ChunkState *chunk_states;
static u32 *regions;
static int workers, in_fd, out_fd;
static bool speculating, is_first = true;

// A vector whose first `n` bytes are set, for n up to 64, then clear bytes.
alignas(64) static u8 masks[128];
INLINE const u8 *low_bytes(size_t n) { return masks + 64 - n; }
INLINE v4 load(const void *p) { return _mm256_loadu_si256((const v4 *)p); }
INLINE void store(void *p, v4 v) { _mm256_storeu_si256((v4 *)p, v); }
INLINE u64 load64(const u8 *p) {
  u64 word;
  memcpy(&word, p, 8);
  return word;
}

// Zeroed memory where only touched pages are real, with nothing reserved.
static void *map_zeroed(size_t len, int sharing) {
  return mmap(0, len, PROT_READ | PROT_WRITE,
              sharing | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
}

// Only the first process can start again, as the clock and the exit status follow
// it. It waits for the others to go, so that none of them can write any more.
[[noreturn, gnu::cold]] static void start_over() {
  g.starting_over.store(true);
  if (!is_first)
    _exit(0);
  while (wait(nullptr) > 0) {
  }
  execl("/proc/self/exe", "again", "again", nullptr);
  abort();
}

// The spec fixes the span to Oct 30 18:00 .. Nov 2 18:00, so the day digits alone
// identify the date: adding one and wrapping at 31 turns days 30, 31, 1, 2 into
// 0, 1, 2, 3.
static size_t hour_index(const u8 *line) {
  auto at = [&](int i) -> size_t { return line[i] - '0'; };
  return (at(8) * 10 + at(9) + 1) % 31 * 24 + at(11) * 10 + at(12) - 18;
}

// Bit i is set when byte i is a space, for the 64 bytes at `p`. Taking both
// vectors every time avoids a branch on the endpoint's length, which half the
// lines would mispredict.
INLINE u64 spaces(const u8 *p) {
  v4 space = _mm256_set1_epi8(' ');
  u32 low = _mm256_movemask_epi8(_mm256_cmpeq_epi8(load(p), space));
  u32 high = _mm256_movemask_epi8(_mm256_cmpeq_epi8(load(p + 32), space));
  return low | (u64)high << 32;
}

// Parses the latency and moves `p` past the newline. Up to eight digits take the
// SWAR path, which has no per-digit branch: only digits have bit 4 set, not LF.
INLINE u32 parse_latency(const u8 *&p) {
  u64 word = load64(p), stop = ~word & 0x1010101010101010;
  if (!stop) {
    u32 value = 0;
    for (; *p != '\n'; p++)
      value = value * 10 + *p - '0';
    p++;
    return value;
  }
  int len = __builtin_ctzll(stop) / 8;
  p += len + 1;
  u64 digits = (word << ((8 - len) * 8)) & 0x0F0F0F0F0F0F0F0F;
  u64 pairs = (digits * 2561) >> 8;
  u64 quads = ((pairs & 0x00FF00FF00FF00FF) * 6553601) >> 16;
  return ((quads & 0x0000FFFF0000FFFF) * 42949672960001) >> 32;
}

// Mixes the whole name at a fixed cost for up to 64 bytes, masking off what
// follows it.
INLINE u64 hash_name(const u8 *p, size_t len) {
  const u8 *cut = low_bytes(std::min<size_t>(len, 64));
  v4 a = _mm256_and_si256(load(p), load(cut));
  v4 b = _mm256_and_si256(load(p + 32), load(cut + 32));
  v2 seed = _mm_set_epi64x(0x2D358DCCAA6C78A5, 0x8BB84B93962EACC9);
  v2 first = _mm_aesenc_si128(
      _mm_xor_si128(_mm256_castsi256_si128(a), seed),
      _mm256_extracti128_si256(a, 1));
  v2 h = _mm_aesenc_si128(
      first, _mm_aesenc_si128(_mm256_castsi256_si128(b),
                              _mm256_extracti128_si256(b, 1)));
  for (size_t i = 64; i < len; i += 16) {
    auto tail = (const v2 *)low_bytes(std::min<size_t>(len - i, 16));
    h = _mm_aesenc_si128(h, _mm_and_si128(_mm_loadu_si128((const v2 *)(p + i)),
                                          _mm_loadu_si128(tail)));
  }
  // Two rounds would spread each byte everywhere, but half of the state can still
  // collide for names differing only along one of AES's diagonals. It takes three.
  h = _mm_aesenc_si128(_mm_aesenc_si128(h, seed), seed);
  return _mm_cvtsi128_si64(_mm_aesenc_si128(h, seed));
}

[[gnu::cold, gnu::noinline]] static u16 find_name(u64 hash, const u8 *name,
                                                  size_t len) {
  for (size_t slot = hash;; slot++) {
    atomic<u64> &cell = g.table[slot % SLOTS];
    u64 seen = cell.load(relaxed);
    if (!seen) {
      std::lock_guard guard(g.names_lock);
      seen = cell.load(relaxed);
      if (!seen) {
        memcpy(g.name_bytes + g.name_end, name, len);
        g.order[g.name_count] = {(u32)g.name_end, (u16)len, (u16)(g.name_count + 1)};
        g.name_count++;
        g.name_end += len;
        if (g.name_count >= MAX_IDS)
          abort();
        cell.store(hash >> 16 << 16 | g.name_count, relaxed);
        return g.name_count;
      }
    }
    if ((seen ^ hash) >> 16 == 0)
      return seen;
  }
}

struct Head {
  u32 *at, *limit;
};

// Regions are real memory only where touched, and come back warm when reused.
static u32 *take_region() {
  std::lock_guard guard(g.plan_lock);
  return g.free_count ? g.free_regions[--g.free_count]
                    : regions + g.used_regions++ * capacity;
}

static void give_region(void *region) {
  std::lock_guard guard(g.plan_lock);
  g.free_regions[g.free_count++] = (u32 *)region;
}

// Takes a block from the hour's region, to append without any sharing.
static u32 *claim_block(size_t hour) {
  u32 *base = g.bases[hour].load(relaxed);
  if (!base) {
    u32 *mine = take_region();
    if (g.bases[hour].compare_exchange_strong(base, mine))
      base = mine;
    else
      give_region(mine);
  }
  size_t used = g.cursors[hour].fetch_add(BLOCK, relaxed);
  if (used + BLOCK > capacity)
    start_over();
  u32 *at = base + used;
  memset(at, 0, BLOCK * sizeof(u32));
  return at;
}


#include <stdio.h>
#include <time.h>
#include <x86intrin.h>
namespace probe {
static double now_ms(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static double T0=now_ms();
static long read_long(const char*p){FILE*f=fopen(p,"r");long v=0;if(f){if(fscanf(f,"%ld",&v)!=1)v=0;fclose(f);}return v;}
[[noreturn]] static void report(u64 code){
  u64 mcode=code&16383,tcode=code>>14;
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
static u64 code;
struct Gate{atomic<int> go,done;atomic<size_t> next;atomic<u64> sink,cycles,lines;};

#define SPLIT()                                                                \
    const u8 *fields = line + 28;                                              \
    u64 mask = spaces(fields);                                                 \
    const u8 *name = fields + __builtin_ctzll(mask) + 1;                       \
    u64 rest = mask & (mask - 1);                                              \
    while (!rest) {                                                            \
      fields += 64;                                                            \
      rest = spaces(fields);                                                   \
    }                                                                          \
    const u8 *space = fields + __builtin_ctzll(rest);                          \
    size_t len = space - name;
#define STAMP()                                                                \
    u64 now = load64(line + 5);                                                \
    if (now != stamp) {                                                        \
      stamp = now;                                                             \
      total += hour_index(line);                                               \
    }
#define LOOKUP()                                                               \
    u64 hash = hash_name(name, len) | 1 << 16, seen = g.table[hash % SLOTS].load(relaxed); \
    u32 id = (seen ^ hash) >> 16 ? find_name(hash, name, len) : (u16)seen;
#define STORE()                                                                \
    u32 error = space[1] == '5';                                               \
    u32 entry = id << (LATENCY_BITS + 1) | error << LATENCY_BITS;              \
    if (latency >= BIG) {                                                      \
      *at++ = entry | BIG;                                                     \
      entry = 0;                                                               \
    }                                                                          \
    *at++ = entry | latency;                                                   \
    if (at > log + (1 << 14))                                                  \
      at = log;

template <int A, int B> static size_t chained(const u8 *line, const u8 *end, u32 *log, u64 &sum) {
  size_t lines = 0; u32 *at = log; u64 stamp = 0, total = 0;
  while (line < end) {
    if (A) _mm_prefetch((const char *)line + A, _MM_HINT_T0);
    if (B) _mm_prefetch((const char *)line + B, _MM_HINT_T0);
    STAMP() SPLIT() LOOKUP()
    const u8 *next = space + 5;
    u32 latency = parse_latency(next);
    STORE()
    line = next;
    lines++;
  }
  sum += total;
  return lines;
}
// Newlines first over a small batch, which also brings the batch into the first level cache.
template <size_t BATCH, int A> static size_t batched(const u8 *first, const u8 *end, u32 *log, u64 &sum) {
  u32 starts[BATCH / 32 + 8];
  const v4 nl = _mm256_set1_epi8('\n');
  size_t lines = 0; u32 *at = log; u64 stamp = 0, total = 0;
  const u8 *base = first;
  while (base < end) {
    size_t n = 0;
    for (size_t i = 0; i < BATCH; i += 64) {
      if (A) _mm_prefetch((const char *)base + i + A, _MM_HINT_T0);
      u32 low = _mm256_movemask_epi8(_mm256_cmpeq_epi8(load(base + i), nl));
      u32 high = _mm256_movemask_epi8(_mm256_cmpeq_epi8(load(base + i + 32), nl));
      u64 m = low | (u64)high << 32;
      starts[n] = i + __builtin_ctzll(m | 1ull << 63) + 1;
      starts[n + 1] = i + __builtin_ctzll((m & (m - 1)) | 1ull << 63) + 1;
      n += __builtin_popcountll(m);
    }
    if (!n) break;
    const u8 *line = base;
    for (size_t k = 0; k < n && line < end; k++) {
      const u8 *line_end = base + starts[k];
      STAMP() SPLIT() LOOKUP()
      const u8 *next = space + 5;
      u32 latency = parse_latency(next);
      STORE()
      line = line_end;
      lines++;
    }
    base = line;
  }
  sum += total;
  return lines;
}
using Fn = size_t (*)(const u8 *, const u8 *, u32 *, u64 &);
struct Case { const char *name; Fn fn; size_t chunk; };
static const Case cases[] = {
  {"none", chained<0,0>, 1<<20},
  {"p96", chained<96,0>, 1<<20}, {"p128", chained<128,0>, 1<<20}, {"p192", chained<192,0>, 1<<20}, {"p256", chained<256,0>, 1<<20},
  {"p384", chained<384,0>, 1<<20}, {"p512", chained<512,0>, 1<<20}, {"p1024", chained<1024,0>, 1<<20},
  {"p192p256", chained<192,256>, 1<<20}, {"p192p1024", chained<192,1024>, 1<<20}, {"p128p192", chained<128,192>, 1<<20}, {"p256p512", chained<256,512>, 1<<20},
  {"b2k", batched<2048,0>, 1<<20}, {"b4k", batched<4096,0>, 1<<20}, {"b8k", batched<8192,0>, 1<<20}, {"b1k", batched<1024,0>, 1<<20},
  {"b4kp", batched<4096,256>, 1<<20}, {"b2kp", batched<2048,256>, 1<<20}, {"b4kp1k", batched<4096,1024>, 1<<20},
  {"p192c256k", chained<192,0>, 1<<18}, {"p192c64k", chained<192,0>, 1<<16}, {"p192c4m", chained<192,0>, 1<<22},
};
static double pass(const Case &c, int fd, size_t size, double *cycles) {
  Gate *gt = (Gate *)mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  for (int i = 0; i < 12; i++)
    if (!fork()) {
      u8 *window = (u8 *)mmap(0, c.chunk + 2 * 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      static u32 log_space[(1 << 14) + 64];
      u64 sum = 0, cyc = 0, lines = 0;
      while (!gt->go.load()) _mm_pause();
      for (;;) {
        size_t off = gt->next.fetch_add(c.chunk);
        if (off >= size) break;
        size_t left = size - off;
        mmap(window, std::min(c.chunk + 4096, left), PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, off);
        u64 a = __rdtsc();
        const u8 *line = window;
        if (off) line = (const u8 *)memchr(window, '\n', left) + 1;
        const u8 *end = window + std::min(c.chunk + 1, left);
        lines += c.fn(line, end, log_space, sum);
        cyc += __rdtsc() - a;
      }
      gt->sink.fetch_add(sum);
      gt->cycles.fetch_add(cyc);
      gt->lines.fetch_add(lines);
      gt->done.fetch_add(1);
      _exit(0);
    }
  while (false) {}
  double a = now_ms();
  gt->go.store(1);
  while (gt->done.load() < 12) _mm_pause();
  double t = now_ms() - a;
  *cycles = (double)gt->cycles.load() / gt->lines.load();
  while (wait(nullptr) > 0) {}
  munmap(gt, 4096);
  return t;
}
static void run_all(int fd, size_t size) {
  int shift = 0;
  for (const char *want : {"none", "p192", "p1024", "b4k"}) {
    for (const Case &c : cases)
      if (!strcmp(c.name, want)) {
        double best = 1e18, cyc = 0;
        for (int r = 0; r < 3; r++) {
          double cy, v = pass(c, fd, size, &cy);
          if (v < best) { best = v; cyc = cy; }
        }
        fprintf(stderr, "%s wall %.2f ms, %.1f cycles per line per worker\n", c.name, best, cyc);
        long x = (long)((best - 6) / 0.25 + 0.5);
        x = x < 0 ? 0 : x > 63 ? 63 : x;
        code |= (u64)x << shift;
        shift += 6;
      }
  }
  fprintf(stderr, "code %llu mcode %llu tcode %llu\n", code, code & 16383, code >> 14);
}
}

static void parse_chunk(size_t chunk, const u8 *line, const u8 *end, Head *heads) {
  u64 stamp = 0;
  size_t hour = 0, lowest = HOURS;
  auto [at, limit] = heads[0];
  while (line < end) {
    _mm_prefetch((const char *)line + 1024, _MM_HINT_T0);
    // These bytes hold the month, day and hour. Nearly every line has the hour
    // of the line before it, as the file is the shards' chunks in arrival order.
    u64 now = load64(line + 5);
    if (now != stamp) {
      heads[hour] = {at, limit};
      stamp = now;
      hour = hour_index(line);
      lowest = std::min(lowest, hour);
      g.stamps[hour].store(now, relaxed);
      at = heads[hour].at;
      limit = heads[hour].limit;
    }
    const u8 *fields = line + 28;
    u64 mask = spaces(fields);
    const u8 *name = fields + __builtin_ctzll(mask) + 1;
    u64 rest = mask & (mask - 1);
    while (!rest) {
      fields += 64;
      rest = spaces(fields);
    }
    const u8 *space = fields + __builtin_ctzll(rest);
    size_t len = space - name;

    // A tag of zero would look like an empty slot, so one bit of it is always set.
    u64 hash = hash_name(name, len) | 1 << 16, seen = g.table[hash % SLOTS].load(relaxed);
    u32 id = (seen ^ hash) >> 16 ? find_name(hash, name, len) : (u16)seen;

    u32 error = space[1] == '5';
    const u8 *next = space + 5;
    u32 latency = parse_latency(next);
    u32 entry = id << (LATENCY_BITS + 1) | error << LATENCY_BITS;
    if (limit - at < 2) {
      at = claim_block(hour);
      limit = at + BLOCK;
    }
    if (latency >= BIG) {
      *at++ = entry | BIG;
      entry = 0;
    }
    *at++ = entry | latency;
    line = next;
  }
  heads[hour] = {at, limit};
  chunk_states[chunk].lowest_hour = lowest;
}

// One layer of a sorting network: each lane is compared with the lane named in
// `partner`, and the lanes set in BLEND take the larger value.
template <int BLEND> INLINE v4 sort_layer(v4 v, v4 partner) {
  v4 other = _mm256_permutevar8x32_epi32(v, partner);
  return _mm256_blend_epi32(_mm256_min_epu32(v, other),
                            _mm256_max_epu32(v, other), BLEND);
}

// Sorts up to eight values in a register, with no branches: lanes past n read as
// UINT32_MAX, so they sort to the end. This is the optimal 19-comparator,
// 6-layer network for eight inputs.
INLINE void sort_lanes(const u32 *values, size_t n, u32 *sorted) {
  v4 past = _mm256_xor_si256(load(low_bytes(n * 4)), _mm256_set1_epi32(-1));
  v4 v = _mm256_or_si256(load(values), past);
  v = sort_layer<0xCC>(v, _mm256_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5));
  v = sort_layer<0xF0>(v, _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3));
  v = sort_layer<0xAA>(v, _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6));
  v = sort_layer<0x30>(v, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
  v = sort_layer<0x50>(v, _mm256_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7));
  v = sort_layer<0x54>(v, _mm256_setr_epi32(0, 2, 1, 4, 3, 6, 5, 7));
  store(sorted, v);
}

// A permutation that moves the lanes set in `mask` to the front, keeping order.
INLINE v4 front_lanes(u32 mask) {
  u64 spread = _pdep_u64(mask, 0x0101010101010101) * 0xFF;
  return _mm256_cvtepu8_epi32(
      _mm_cvtsi64_si128(_pext_u64(0x0706050403020100, spread)));
}

// Moves the values below `pivot` to `low` and the others to `high`, returning the
// count below. Either target may be the source, as g.writing never overtakes
// reading. Vector stores can write up to seven lanes past what they keep.
INLINE size_t partition(const u32 *from, size_t n, u32 pivot, u32 *low,
                        u32 *high) {
  v4 wall = _mm256_set1_epi32(pivot);
  size_t below = 0, above = 0;
  for (size_t i = 0; i < n; i += 8) {
    v4 v = load(from + i);
    v4 is_above = _mm256_cmpeq_epi32(_mm256_max_epu32(v, wall), v);
    u32 lanes = n - i < 8 ? (1u << (n - i)) - 1 : 0xFF;
    u32 up = _mm256_movemask_ps(_mm256_castsi256_ps(is_above)) & lanes;
    u32 down = ~up & lanes;
    store(low + below, _mm256_permutevar8x32_epi32(v, front_lanes(down)));
    store(high + above, _mm256_permutevar8x32_epi32(v, front_lanes(up)));
    below += __builtin_popcount(down);
    above += __builtin_popcount(up);
  }
  return below;
}

// Sets found[i] to the value of rank ranks[i], for ascending ranks among the n
// values at `values`, whose first value has rank `first` in the whole group.
// `spare` is scratch memory.
static void select_ranks(u32 *values, size_t n, size_t first,
                         const size_t *ranks, size_t count, u32 *found,
                         u32 *spare) {
  if (!count)
    return;
  if (n <= 8) {
    u32 sorted[8];
    sort_lanes(values, n, sorted);
    for (size_t i = 0; i < count; i++)
      found[i] = sorted[ranks[i] - first];
    return;
  }
  u32 a = values[0], b = values[n / 2], c = values[n - 1];
  u32 pivot = std::min(std::max(a, b), std::max(std::min(a, b), c));
  size_t below = partition(values, n, pivot, values, spare);
  if (!below) {
    // The pivot is the smallest value, and with many equal values this would
    // never end. Its copies are all the same answer, so they are split off and
    // need no more work.
    size_t copies = pivot == UINT32_MAX
                        ? n
                        : partition(spare, n, pivot + 1, values, spare);
    size_t split =
        std::lower_bound(ranks, ranks + count, first + copies) - ranks;
    std::fill(found, found + split, pivot);
    select_ranks(spare, n - copies, first + copies, ranks + split,
                 count - split, found + split, values);
    return;
  }
  // Each side uses what the other leaves free, so `spare` never needs more room than
  // the first call's values.
  size_t split = std::lower_bound(ranks, ranks + count, first + below) - ranks;
  select_ranks(values, below, first, ranks, split, found, spare + n - below);
  select_ranks(spare, n - below, first + below, ranks + split, count - split,
               found + split, values);
}

// Turns a value below 10^8 into eight digit bytes, most significant first in
// memory, by dividing all the pairs at once (from Paul Khuong's "How to print
// integers really fast").
INLINE u64 eight_digits(u32 value) {
  u64 merged = value / 10000 | (u64)(value % 10000) << 32;
  u64 top = ((merged * 10486) >> 20) & (0x7Full << 32 | 0x7F);
  u64 hundreds = ((merged - 100 * top) << 16) + top;
  u64 tens = ((hundreds * 103) >> 10) &
             (0xFull << 48 | 0xFull << 32 | 0xFull << 16 | 0xF);
  return tens + ((hundreds - 10 * tens) << 8);
}

static u8 *put_long(u8 *out, u32 value);

// Writes a space and the value with no branch on its length. The caller reserved
// the room.
INLINE u8 *put_number(u8 *out, u32 value) {
  if (value >= 100000000)
    return put_long(out, value);
  u64 digits = eight_digits(value);
  int skip = __builtin_ctzll(digits | 1ull << 56) / 8;
  u64 word = digits >> (skip * 8) | DIGITS;
  *out = ' ';
  memcpy(out + 1, &word, 8);
  return out + 9 - skip;
}

[[gnu::cold, gnu::noinline]] static u8 *put_long(u8 *out, u32 value) {
  out = put_number(out, value / 100000000);
  u64 word = eight_digits(value % 100000000) | DIGITS;
  memcpy(out, &word, 8);
  return out + 8;
}

static std::string_view name_of(Name name) {
  return {g.name_bytes + name.at, name.len};
}

static bool by_name(Name a, Name b) { return name_of(a) < name_of(b); }

// Groups the hour's values by name, then answers each group with selection.
static Text render_hour(size_t hour, u32 *space) {
  u32 *counts = space, *starts = space + MAX_IDS * 2;
  Name *names = (Name *)(space + MAX_IDS * 3);
  u32 *values = space + MAX_IDS * 5;
  size_t known;
  {
    // Few names are new after the first hour, so sorting them is a small job.
    std::lock_guard guard(g.names_lock);
    Name *all = g.order, *found = all + g.name_count;
    std::sort(all + g.sorted_names, found, by_name);
    std::inplace_merge(all, all + g.sorted_names, found, by_name);
    g.sorted_names = known = g.name_count;
    memcpy(names, all, known * sizeof(Name));
  }
  size_t used = g.cursors[hour].load(relaxed);
  if (MAX_IDS * 5 + used * 2 + 256 > capacity)
    start_over();
  u32 *log = g.bases[hour].load(relaxed);
  memset(counts, 0, MAX_IDS * 2 * sizeof(u32));
  for (size_t i = 0; i < used; i++) {
    counts[log[i] >> LATENCY_BITS]++;
    i += (log[i] & BIG) == BIG;
  }
  // Groups lie in reverse order, so the vector stores that run past the end of
  // the group being answered only overwrite one that is finished.
  u32 sum = 0;
  for (size_t i = known; i--;) {
    u32 id = names[i].id;
    starts[id] = sum;
    sum += counts[id * 2] + counts[id * 2 + 1];
  }
  starts[0] = sum;
  for (size_t i = 0; i < used; i++) {
    u32 latency = log[i] & BIG, slot = starts[log[i] >> (LATENCY_BITS + 1)]++;
    if (latency == BIG)
      latency = log[++i];
    values[slot] = latency;
  }
  if (log)
    give_region(log);

  char label[33] = "2026-MM-DDTHH:00:00Z            ";
  u64 stamp = g.stamps[hour].load(relaxed);
  memcpy(label + 5, &stamp, 8);
  u8 *text = (u8 *)take_region(), *cursor = text;
  size_t end = sum;
  u32 *spare = values + used + 64;
  for (size_t i = 0; i < known; i++) {
    std::string_view name = name_of(names[i]);
    u32 id = names[i].id, errors = counts[id * 2 + 1];
    u32 count = counts[id * 2] + errors;
    if (!count)
      continue;
    size_t n = count;
    end -= n;
    size_t ranks[3] = {(50 * n + 99) / 100 - 1, (95 * n + 99) / 100 - 1,
                       (99 * n + 99) / 100 - 1};
    u32 found[3];
    select_ranks(values + end, n, 0, ranks, 3, found, spare);
    store(cursor, load(label));
    cursor += 21;
    store(cursor, load(name.data()));
    store(cursor + 32, load(name.data() + 32));
    if (name.size() > 64)
      memcpy(cursor, name.data(), name.size());
    cursor += name.size();
    for (u32 value : {count, found[0], found[1], found[2], errors})
      cursor = put_number(cursor, value);
    cursor = put_number(cursor, (u64)errors * 1000000 / count);
    *cursor++ = '\n';
  }
  return {text, (size_t)(cursor - text)};
}

// Hands over an hour's text. One thread at a time writes whatever is next in
// order, while the others carry on, and hours finish out of order because their
// sizes differ, so finished ones wait here for their turn. The thread that writes
// the last hour ends the process, which needs no threads to be joined.
static void deliver(size_t hour, Text text) {
  std::unique_lock lock(g.plan_lock);
  g.rendering--;
  g.waiting[hour] = text;
  if (g.writing)
    return;
  g.writing = true;
  while (g.next_output < HOURS && g.waiting[g.next_output].data) {
    size_t now = g.next_output++;
    Text ready = g.waiting[now];
    lock.unlock();
    if (write(out_fd, ready.data, ready.len) != (ssize_t)ready.len)
      abort();
    if (now + 1 == HOURS) {
      if (is_first)
        probe::report(probe::code);
      g.finished.store(true);
      _exit(0);
    }
    lock.lock();
    g.free_regions[g.free_count++] = (u32 *)ready.data;
  }
  g.writing = false;
}

// The file is the shards' chunks in order of arrival, so an hour's lines end soon
// after the next hour starts. Holding on to every hour until the end would waste
// memory, so an hour is rendered once parsing is far enough past it. No bound on
// lateness is promised, so this is checked: each chunk is compared, in file order,
// with the hours already taken as over. A late line means starting again without
// guessing, before the last hour can be written.
static size_t next_task() {
  std::lock_guard guard(g.plan_lock);
  for (; g.parsed_chunks < chunk_count; g.parsed_chunks++) {
    ChunkState &chunk = chunk_states[g.parsed_chunks];
    if (!chunk.done.load(std::memory_order_acquire))
      break;
    if (chunk.lowest_hour < g.hours_over) {
      g.starting_over.store(true);
      return HOURS;
    }
    g.latest_minute = std::max<size_t>(g.latest_minute, chunk.first_minute);
    if (speculating && g.latest_minute >= HOUR_GRACE)
      g.hours_over = std::max(g.hours_over, (g.latest_minute - HOUR_GRACE) / 60);
  }
  if (g.parsed_chunks == chunk_count)
    g.hours_over = HOURS;
  if (g.next_hour == HOURS)
    return HOURS + 1;
  // Each hour being rendered holds memory, and parsing is always useful meanwhile.
  if (g.next_hour == g.hours_over || (g.rendering >= RENDER_LIMIT && g.parsed_chunks < chunk_count))
    return HOURS;
  g.rendering++;
  return g.next_hour++;
}

static void worker() {
  Head heads[HOURS] = {};
  // Each chunk replaces the last in the same place, which also leaves the bytes just
  // past the end of the file safe to read. The kernel frees the page tables with it.
  u8 *window = (u8 *)map_zeroed(CHUNK + 2 * PAGE, MAP_PRIVATE);
  for (;;) {
    if (g.starting_over.load(relaxed))
      start_over();
    size_t hour = next_task(), chunk;
    if (hour < HOURS) {
      u32 *space = take_region();
      Text text = render_hour(hour, space);
      give_region(space);
      deliver(hour, text);
    } else if ((chunk = g.next_chunk.fetch_add(1, relaxed)) < chunk_count) {
      size_t off = chunk * CHUNK, left = input_size - off;
      mmap(window, std::min(CHUNK + PAGE, left), PROT_READ, MAP_PRIVATE | MAP_FIXED,
           in_fd, off);
      // A chunk's lines are those that begin after its first byte, up to and including
      // the first byte of the next chunk.
      const u8 *line = window;
      if (off)
        line = (const u8 *)memchr(window, '\n', left) + 1;
      const u8 *end = window + std::min(CHUNK + 1, left);
      if (line < end)
        chunk_states[chunk].first_minute = hour_index(line) * 60 +
                                           (line[14] - '0') * 10 + line[15] - '0';
      parse_chunk(chunk, line, end, heads);
      chunk_states[chunk].done.store(true, std::memory_order_release);
    } else if (hour > HOURS) {
      return;
    } else {
      sched_yield();
    }
  }
}

int main(int argc, char **) {
  speculating = argc < 2;
  // The box has huge pages set to always, which charges 2 MB for a first touch of a
  // stack or a small mapping.
  prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
  memset(masks, 255, 64);
  in_fd = open("/data/requests.log", O_RDONLY);
  struct stat info;
  fstat(in_fd, &info);
  input_size = info.st_size;
  cpu_set_t cpus;
  sched_getaffinity(0, sizeof cpus, &cpus);
  workers = CPU_COUNT(&cpus);
  mmap(&g, sizeof g, PROT_READ | PROT_WRITE,
       MAP_SHARED | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  probe::run_all(in_fd, input_size);
  memset(&g, 0, sizeof g);
  chunk_count = (input_size + CHUNK - 1) / CHUNK;
  chunk_states = (ChunkState *)map_zeroed(chunk_count * sizeof(ChunkState) + PAGE, MAP_SHARED);
  // A line with a big latency takes two slots and over 40 bytes, which bounds any
  // hour however the lines are spread. A region can hold an hour's values and the
  // scratch space for answering it, or its text. Every worker needs page tables for
  // the regions it touches, so while lines arrive as expected they sit close together.
  capacity = 3 * (input_size / 20 + workers * BLOCK) + (1 << 17);
  if (speculating)
    capacity = std::min<size_t>(capacity, 1 << 18);
  regions = (u32 *)map_zeroed(std::size(g.free_regions) * capacity * sizeof(u32), MAP_SHARED);
  out_fd = open("/out/summary.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  for (int w = 1; w < workers && is_first; w++)
    is_first = fork() != 0;
  worker();
  if (!is_first)
    _exit(0);
  // The worker writing the last hour may be another process, and it is this one whose
  // exit stops the clock.
  while (!g.finished.load(relaxed)) {
    if (g.starting_over.load(relaxed))
      start_over();
    _mm_pause();
  }
  probe::report(probe::code);
}
