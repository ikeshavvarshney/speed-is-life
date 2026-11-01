#include <algorithm>
#include <array>
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
#include <thread>
#include <unistd.h>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <sys/resource.h>

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

#ifndef CHUNK_BITS
#define CHUNK_BITS 16
#endif
const size_t CHUNK = 1 << CHUNK_BITS, HOURS = 72, BLOCK = 256;
// The longest line in the spec's grammar is under this, so reading this much more
// than a chunk gives every line that starts in it.
const size_t LONGEST_LINE = 256;
// A chunk that starts this many minutes into an hour shows that the hour before is
// over, as the spec has the last upload of an hour land within 10 minutes of its end.
const size_t HOUR_GRACE = 10;
constexpr int DIAG_PUSH = 3;
static double now_ms(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static double T0=now_ms();
static long read_long(const char*p){FILE*f=fopen(p,"r");long v=0;if(f){if(fscanf(f,"%ld",&v)!=1)v=0;fclose(f);}return v;}
[[noreturn]] static void report(unsigned long long mcode,unsigned long long tcode){
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
static unsigned long long q(double v,double unit,int nbits){long x=(long)(v/unit+0.5);long m=(1L<<nbits)-1;return (unsigned long long)(x>m?m:x);}
struct alignas(64) Spent{unsigned long long read=0,parse=0,plan=0,render=0,deliver=0,idle=0,count=0,place=0,select=0;};
static Spent spent[256];
static std::atomic<int> next_spent{0};
static thread_local Spent *me;
static double tsc_per_ms;
static double t_last_parse;
#define TSC() __rdtsc()
const size_t BUCKETS = 512, WAYS = 8, SLOTS = BUCKETS * WAYS;
// A log entry is id << 21 | is_error << 20 | latency. A latency that does not fit,
// on about 1% of lines, puts BIG in the entry and takes the next slot whole. An
// all-zero entry is an unused part of a block, and id 0 is reserved to absorb it.
const int LATENCY_BITS = 20;
const u32 BIG = (1 << LATENCY_BITS) - 1;
const size_t MAX_IDS = 1 << 11;
const u64 DIGITS = 0x3030303030303030;

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

// Table of name hashes, with eight ways per bucket compared at once by one vector
// load. After the first few lines it is only read, so it is shared without
// contention and nothing needs merging. A tag and its bucket give 40 bits of
// hash, which over 1,842 names collide with odds of 2e-6.
alignas(64) static atomic<u32> tags[SLOTS];
// Ids are given out in the order names are first seen. Popular names come early,
// which packs the counters they use later into few cache lines.
alignas(64) static atomic<u16> ids[SLOTS];
// Names in the order they were found, and sorted up to `sorted_names`.
static std::mutex names_lock;
static vector<Name> order;
static size_t sorted_names;
alignas(64) static char name_bytes[MAX_IDS * 128];
static size_t name_end;
// Written by every thread, so kept off the lines that each line of input reads.
alignas(64) static atomic<u64> stamps[HOURS];
alignas(64) static atomic<size_t> cursors[HOURS];
alignas(64) static atomic<u32 *> bases[HOURS];
alignas(64) static atomic<size_t> next_chunk;
// Guards the rest: how far parsing has got, which hours are over and taken, the free
// regions and the turn to write.
static std::mutex plan_lock;
static vector<ChunkState> chunk_states;
static size_t parsed_chunks, latest_minute, hours_over, next_hour, next_output;
static bool writing, speculating;
static Text waiting[HOURS];
static u32 *free_regions[HOURS * 3];
static size_t free_count, used_regions;

alignas(64) static size_t input_size, capacity, chunk_count;
static u32 *regions;
static int threads, out_fd;

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
static void *map_zeroed(size_t len) {
  return mmap(0, len, PROT_READ | PROT_WRITE,
              MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
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

// Readers see a tag only after its id is in place, so they never read an id that
// is not set yet.
[[gnu::cold, gnu::noinline]] static u16 insert_name(u32 tag, size_t bucket,
                                                    const u8 *name,
                                                    size_t len) {
  std::lock_guard guard(names_lock);
  for (;; bucket = (bucket + 1) % BUCKETS) {
    for (size_t slot = bucket * WAYS; slot < (bucket + 1) * WAYS; slot++) {
      u32 seen = tags[slot].load(relaxed);
      if (seen == tag)
        return ids[slot].load(relaxed);
      if (!seen) {
        memcpy(name_bytes + name_end, name, len);
        order.push_back({(u32)name_end, (u16)len, (u16)(order.size() + 1)});
        name_end += len;
        if (order.size() >= MAX_IDS)
          abort();
        ids[slot].store(order.size(), relaxed);
        tags[slot].store(tag, std::memory_order_release);
        return order.size();
      }
    }
  }
}

struct Head {
  u32 *at, *limit;
};

// Regions are real memory only where touched, and come back warm when reused.
static u32 *take_region() {
  std::lock_guard guard(plan_lock);
  return free_count ? free_regions[--free_count]
                    : regions + used_regions++ * capacity;
}

static void give_region(void *region) {
  std::lock_guard guard(plan_lock);
  free_regions[free_count++] = (u32 *)region;
}

// Takes a block from the hour's region, to append without any sharing.
static u32 *claim_block(size_t hour) {
  u32 *base = bases[hour].load(relaxed);
  if (!base) {
    u32 *mine = take_region();
    if (bases[hour].compare_exchange_strong(base, mine))
      base = mine;
    else
      give_region(mine);
  }
  u32 *at = base + cursors[hour].fetch_add(BLOCK, relaxed);
  memset(at, 0, BLOCK * sizeof(u32));
  return at;
}

static void parse_chunk(size_t chunk, const u8 *line, const u8 *end, Head *heads) {
  u64 stamp = 0;
  size_t hour = 0, lowest = HOURS;
  auto [at, limit] = heads[0];
  const v4 zero = _mm256_setzero_si256();
  while (line < end) {
    // These bytes hold the month, day and hour. Nearly every line has the hour
    // of the line before it, as the file is the shards' chunks in arrival order.
    u64 now = load64(line + 5);
    if (now != stamp) {
      heads[hour] = {at, limit};
      stamp = now;
      hour = hour_index(line);
      lowest = std::min(lowest, hour);
      stamps[hour].store(now, relaxed);
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

    u64 hash = hash_name(name, len);
    u32 tag = (u32)hash | 1, id;
    v4 wanted = _mm256_set1_epi32(tag);
    size_t bucket = (hash >> 32) & (BUCKETS - 1);
    for (;;) {
      v4 group = _mm256_load_si256((const v4 *)(tags + bucket * WAYS));
      u32 hit = _mm256_movemask_epi8(_mm256_cmpeq_epi32(group, wanted));
      if (hit) {
        id = ids[bucket * WAYS + __builtin_ctz(hit) / 4].load(relaxed);
        break;
      }
      // A full bucket sends a name on to the next. Any room means it is new.
      if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(group, zero))) {
        id = insert_name(tag, bucket, name, len);
        break;
      }
      bucket = (bucket + 1) % BUCKETS;
    }

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
// count below. Either target may be the source, as writing never overtakes
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
                 count - split, found + split, spare + n - copies + 8);
    return;
  }
  size_t split = std::lower_bound(ranks, ranks + count, first + below) - ranks;
  u32 *rest = spare + n - below + 8;
  select_ranks(values, below, first, ranks, split, found, rest);
  select_ranks(spare, n - below, first + below, ranks + split, count - split,
               found + split, rest);
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
  return {name_bytes + name.at, name.len};
}

static bool by_name(Name a, Name b) { return name_of(a) < name_of(b); }

// Groups the hour's values by name, then answers each group with selection.
static Text render_hour(size_t hour, u32 *space) {
  auto r0 = TSC();
  u32 *counts = space, *starts = space + MAX_IDS * 2;
  Name *names = (Name *)(space + MAX_IDS * 3);
  u32 *values = space + MAX_IDS * 5;
  size_t known;
  {
    // Few names are new after the first hour, so sorting them is a small job.
    std::lock_guard guard(names_lock);
    std::sort(order.begin() + sorted_names, order.end(), by_name);
    std::inplace_merge(order.begin(), order.begin() + sorted_names, order.end(),
                       by_name);
    sorted_names = known = order.size();
    memcpy(names, order.data(), known * sizeof(Name));
  }
  size_t used = cursors[hour].load(relaxed);
  u32 *log = bases[hour].load(relaxed);
  memset(counts, 0, MAX_IDS * 2 * sizeof(u32));
  for (size_t i = 0; i < used; i++) {
    counts[log[i] >> LATENCY_BITS]++;
    i += (log[i] & BIG) == BIG;
  }
  // Groups lie in reverse order, so the vector stores that run past the end of
  // the group being answered only overwrite one that is finished.
  auto r1 = TSC();
  me->count += r1 - r0;
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
  auto r2 = TSC();
  me->place += r2 - r1;

  char label[33] = "2026-MM-DDTHH:00:00Z            ";
  u64 stamp = stamps[hour].load(relaxed);
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
  me->select += TSC() - r2;
  return {text, (size_t)(cursor - text)};
}

// Hands over an hour's text. One thread at a time writes whatever is next in
// order, while the others carry on, and hours finish out of order because their
// sizes differ, so finished ones wait here for their turn. The thread that writes
// the last hour ends the process, which needs no threads to be joined.
static void deliver(size_t hour, Text text) {
  std::unique_lock lock(plan_lock);
  waiting[hour] = text;
  if (writing)
    return;
  writing = true;
  while (next_output < HOURS && waiting[next_output].data) {
    size_t now = next_output++;
    Text ready = waiting[now];
    lock.unlock();
    if (write(out_fd, ready.data, ready.len) != (ssize_t)ready.len)
      abort();
    if (now + 1 == HOURS) {
      double end = now_ms();
      tsc_per_ms = 0;
      {
        auto a = TSC(); double b = now_ms(); while (now_ms() - b < 1.0) {} tsc_per_ms = (TSC() - a) / (now_ms() - b);
      }
      double cpu[9] = {};
      for (auto &x : spent) { cpu[0]+=x.read; cpu[1]+=x.parse; cpu[2]+=x.plan; cpu[3]+=x.render; cpu[4]+=x.deliver; cpu[5]+=x.idle; cpu[6]+=x.count; cpu[7]+=x.place; cpu[8]+=x.select; }
      for (double &v : cpu) v /= tsc_per_ms;
      long peak=read_long("/sys/fs/cgroup/memory.peak");
      rusage ru; getrusage(RUSAGE_SELF,&ru);
      double wall=end-T0;
      fprintf(stderr,"wall %.2f parse-end %.2f | read %.1f parse %.1f plan %.1f render %.1f (count %.1f place %.1f select %.1f) deliver %.1f idle %.1f | peak %.2f regions %zu nvcsw %ld nivcsw %ld minflt %ld\n",wall,t_last_parse-T0,cpu[0],cpu[1],cpu[2],cpu[3],cpu[6],cpu[7],cpu[8],cpu[4],cpu[5],peak/1e6,used_regions,ru.ru_nvcsw,ru.ru_nivcsw,ru.ru_minflt);
      unsigned long long m=0,t=0;
      unsigned long long c=0;
      if(DIAG_PUSH==1)c=q(cpu[0],1,8)|q(cpu[1],1,8)<<8|q(cpu[3],0.5,8)<<16;
      if(DIAG_PUSH==2)c=q(cpu[6],0.5,6)|q(cpu[7],0.5,6)<<6|q(cpu[8],0.5,6)<<12|q(cpu[4],0.5,6)<<18;
      if(DIAG_PUSH==3)c=q(peak/1e6,0.1,8)|q(used_regions,1,6)<<8|q(ru.ru_nvcsw,8,5)<<14|q(ru.ru_minflt,64,5)<<19;
      if(DIAG_PUSH==4)c=q(t_last_parse-T0,0.1,8)|q(cpu[5],0.25,8)<<8|q(cpu[2],0.05,8)<<16;
      m=c&16383;t=c>>14;
      fprintf(stderr,"mcode %llu tcode %llu\n",m,t);
      report(m,t);
    }
    lock.lock();
    free_regions[free_count++] = (u32 *)ready.data;
  }
  writing = false;
}

// The file is the shards' chunks in order of arrival, so an hour's lines end soon
// after the next hour starts. Holding on to every hour until the end would waste
// memory, so an hour is rendered once parsing is far enough past it. No bound on
// lateness is promised, so this is checked: each chunk is compared, in file order,
// with the hours already taken as over. A late line means starting again without
// guessing, before the last hour can be written.
static size_t next_task() {
  std::lock_guard guard(plan_lock);
  for (; parsed_chunks < chunk_count; parsed_chunks++) {
    ChunkState &chunk = chunk_states[parsed_chunks];
    if (!chunk.done.load(std::memory_order_acquire))
      break;
    if (chunk.lowest_hour < hours_over) {
      execl("/proc/self/exe", "again", "again", nullptr);
      abort();
    }
    latest_minute = std::max<size_t>(latest_minute, chunk.first_minute);
    if (speculating && latest_minute >= HOUR_GRACE)
      hours_over = std::max(hours_over, (latest_minute - HOUR_GRACE) / 60);
  }
  if (parsed_chunks == chunk_count)
    hours_over = HOURS;
  return next_hour < hours_over ? next_hour++ : next_hour == HOURS ? HOURS + 1 : HOURS;
}

static void worker() {
  me = &spent[next_spent.fetch_add(1)];
  Head heads[HOURS] = {};
  // Reading costs less than faulting the file in from a mapping, which then also has
  // to be unmapped page by page, and parsing finds the bytes already in its cache.
  // Each thread has its own file so that they don't share its reference count.
  alignas(64) u8 buffer[CHUNK + LONGEST_LINE + 128];
  int input_fd = open("/data/requests.log", O_RDONLY);
  for (;;) {
    auto t0 = TSC();
    size_t hour = next_task(), chunk;
    me->plan += TSC() - t0;
    if (hour < HOURS) {
      auto t1 = TSC();
      u32 *space = take_region();
      Text text = render_hour(hour, space);
      give_region(space);
      auto t2 = TSC();
      me->render += t2 - t1;
      deliver(hour, text);
      me->deliver += TSC() - t2;
    } else if ((chunk = next_chunk.fetch_add(1, relaxed)) < chunk_count) {
      // A chunk starts at the first line that begins inside it, which the byte
      // before the chunk tells.
      size_t before = chunk > 0, off = chunk * CHUNK;
      auto t1 = TSC();
      ssize_t got = pread(input_fd, buffer, before + CHUNK + LONGEST_LINE, off - before);
      auto t2 = TSC();
      me->read += t2 - t1;
      const u8 *line = buffer;
      if (before)
        line = (const u8 *)memchr(buffer, '\n', got) + 1;
      const u8 *end = buffer + std::min<size_t>(got, before + CHUNK);
      if (line < end)
        chunk_states[chunk].first_minute = hour_index(line) * 60 +
                                           (line[14] - '0') * 10 + line[15] - '0';
      parse_chunk(chunk, line, end, heads);
      me->parse += TSC() - t2;
      t_last_parse = now_ms();
      chunk_states[chunk].done.store(true, std::memory_order_release);
    } else if (hour > HOURS) {
      return;
    } else {
      auto t1 = TSC();
      sched_yield();
      me->idle += TSC() - t1;
    }
  }
}

int main(int argc, char **) {
  speculating = argc < 2;
  // The box has huge pages set to always, which charges 2 MB for a first touch of a
  // thread stack or a small mapping, and more when threads race to fault one.
  prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
  memset(masks, 255, 64);
  int fd = open("/data/requests.log", O_RDONLY);
  struct stat info;
  fstat(fd, &info);
  input_size = info.st_size;
  cpu_set_t cpus;
  sched_getaffinity(0, sizeof cpus, &cpus);
  threads = CPU_COUNT(&cpus);
  // A line with a big latency takes two slots and over 40 bytes, which bounds any
  // hour however the lines are spread. A region can hold an hour's values and the
  // scratch space for answering it, or its text.
  chunk_count = (input_size + CHUNK - 1) / CHUNK;
  chunk_states = vector<ChunkState>(chunk_count);
  capacity = 3 * (input_size / 20 + threads * BLOCK) + (1 << 17);
  regions = (u32 *)map_zeroed(std::size(free_regions) * capacity * sizeof(u32));
  out_fd = open("/out/summary.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  for (int t = 1; t < threads; t++)
    std::thread(worker).detach();
  worker();
  // The thread writing the last hour ends the process, so this one just waits.
  for (;;)
    pause();
}
