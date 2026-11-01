#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <mutex>
#include <string_view>
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

const size_t CHUNK = 1 << 18, UNMAP_STEP = 1 << 23, HOURS = 72, BLOCK = 256;
const size_t BUCKETS = 512, WAYS = 8, SLOTS = BUCKETS * WAYS;
// A log entry is id << 21 | is_error << 20 | latency. Bigger latencies, about 1%
// of lines, go to a side list. An all-zero entry is an unused part of a block,
// and id 0 is reserved to absorb it.
const int LATENCY_BITS = 20;
const size_t MAX_IDS = 1 << 11;
const u64 DIGITS = 0x3030303030303030;

struct Big {
  u32 entry, latency;
};
using BigLists = std::array<vector<Big>, HOURS>;
struct Span {
  u32 at, len;
};
struct Named {
  std::string_view name;
  u32 id;
};
// An hour's text. It is allocated by hand so that no time goes to zeroing it.
struct Text {
  u8 *data = nullptr;
  size_t len = 0, capacity = 0;
};

// Table of name hashes, with eight ways per bucket compared at once by one vector
// load. After the first few lines it is only read, so it is shared without
// contention and nothing needs merging. A tag and its bucket give 40 bits of
// hash, which over 1,842 names collide with odds of 2e-6.
#include <stdio.h>
#include <time.h>
constexpr int DIAG_PUSH = 2;
static double now_ms(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static double T0=now_ms(), t_spawned, t_parsed, t_rendered_last;
static long read_long(const char*p){FILE*f=fopen(p,"r");long v=0;if(f){if(fscanf(f,"%ld",&v)!=1)v=0;fclose(f);}return v;}
static long peak_parsed=0;
struct alignas(64) Spent{double parse=0,unmap=0,render=0,deliver=0,wait=0; u64 count=0,place=0,select=0,print=0;};
static Spent spent[256];
static atomic<int> next_spent{0};
static atomic<int> waiting_texts{0}, most_waiting{0};
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
alignas(64) static atomic<u32> tags[SLOTS];
// Ids are given out in the order names are first seen. Popular names come early,
// which packs the counters they use later into few cache lines.
alignas(64) static atomic<u16> ids[SLOTS];
// Each name's offset and length in the block of name bytes. Id n is at n - 1.
static std::mutex names_lock;
static vector<Span> spans;
static vector<char> name_bytes;
// Written by every thread, so kept off the lines that each line of input reads.
alignas(64) static atomic<u64> stamps[HOURS];
alignas(64) static atomic<size_t> cursors[HOURS];
alignas(64) static atomic<size_t> next_chunk, next_unmap, next_hour, next_worker, parsed;
static std::mutex big_lock;
static vector<BigLists> all_big;
static vector<Named> order; // Each name and its id, in output order.
static atomic<bool> ready;
static std::mutex output_lock;
static size_t next_output;
static bool writing;
static Text waiting[HOURS];
static vector<Text> spare_texts;

alignas(64) static const u8 *input;
static size_t input_size, capacity;
static u32 *logs;
static int threads, out_fd;

// A vector whose first `n` bytes are set, for n up to 64, then clear bytes.
alignas(64) static u8 masks[128];
INLINE const u8 *low_bytes(size_t n) { return masks + 64 - n; }
INLINE u32 *region(size_t index) { return logs + index * capacity; }
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
        spans.push_back({(u32)name_bytes.size(), (u32)len});
        name_bytes.insert(name_bytes.end(), name, name + len);
        if (spans.size() >= MAX_IDS)
          abort();
        ids[slot].store(spans.size(), relaxed);
        tags[slot].store(tag, std::memory_order_release);
        return spans.size();
      }
    }
  }
}

struct Head {
  u32 *at, *limit;
};

static void parse_chunk(size_t first, size_t stop, Head *heads, BigLists &big) {
  const u8 *line = input + first, *end = input + stop;
  u64 stamp = 0;
  size_t hour = 0;
  auto [at, limit] = heads[0];
  const v4 zero = _mm256_setzero_si256();
  while (line < end) {
    _mm_prefetch((const char *)line + 1024, _MM_HINT_T0);
    // These bytes hold the month, day and hour. Nearly every line has the hour
    // of the line before it, as the file is the shards' chunks in arrival order.
    u64 now = load64(line + 5);
    if (now != stamp) {
      heads[hour] = {at, limit};
      stamp = now;
      hour = hour_index(line);
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
    if (latency >> LATENCY_BITS) {
      big[hour].push_back({entry, latency});
    } else {
      if (at == limit) {
        // Takes a block from the hour's region, to append without any sharing.
        at = region(hour) + cursors[hour].fetch_add(BLOCK, relaxed);
        limit = at + BLOCK;
      }
      *at++ = entry | latency;
    }
    line = next;
  }
  heads[hour] = {at, limit};
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

// Groups the hour's values by name in `values`, then answers each group with
// selection.
static void render_hour(size_t hour, u32 *values, u32 *tables, Text &out) {
  u32 *counts = tables, *starts = tables + MAX_IDS * 2;
  size_t used = cursors[hour].load(relaxed), extra = 0;
  const u32 *log = region(hour);
  memset(counts, 0, MAX_IDS * 2 * sizeof(u32));
  for (size_t i = 0; i < used; i++)
    counts[log[i] >> LATENCY_BITS]++;
  for (const BigLists &lists : all_big)
    for (const Big &big : lists[hour]) {
      counts[big.entry >> LATENCY_BITS]++;
      extra++;
    }
  // Groups lie in reverse order, so the vector stores that run past the end of
  // the group being answered only overwrite one that is finished.
  u32 sum = 0;
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    starts[it->id] = sum;
    sum += counts[it->id * 2] + counts[it->id * 2 + 1];
  }
  starts[0] = sum;
  for (size_t i = 0; i < used; i++)
    values[starts[log[i] >> (LATENCY_BITS + 1)]++] =
        log[i] & ((1 << LATENCY_BITS) - 1);
  for (const BigLists &lists : all_big)
    for (const Big &big : lists[hour])
      values[starts[big.entry >> (LATENCY_BITS + 1)]++] = big.latency;

  char label[33] = "2026-MM-DDTHH:00:00Z            ";
  u64 stamp = stamps[hour].load(relaxed);
  memcpy(label + 5, &stamp, 8);
  // Each line is the label, the name, six numbers with separators and a newline,
  // and vector writes can pass the end of the last one.
  size_t need = order.size() * 88 + name_bytes.size() + 64;
  if (out.capacity < need) {
    free(out.data);
    out.data = (u8 *)malloc(need);
    out.capacity = need;
  }
  u8 *cursor = out.data;
  size_t end = sum;
  u32 *spare = values + used + extra + 64;
  for (const auto &[name, id] : order) {
    u32 errors = counts[id * 2 + 1], count = counts[id * 2] + errors;
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
  out.len = cursor - out.data;
}

// Hands over an hour's text. One thread at a time writes whatever is next in
// order, while the others carry on rendering, so writing the file does not become
// a serial step at the end. Hours finish out of order because their sizes differ,
// so finished ones wait here for their turn. The thread that writes the last hour
// ends the process, which needs no threads to be joined.
static void deliver(size_t hour, Text text) {
  std::unique_lock lock(output_lock);
  waiting[hour] = text;
  if (writing)
    return;
  writing = true;
  while (next_output < HOURS && waiting[next_output].data) {
    size_t now = next_output++;
    Text ready_text = waiting[now];
    waiting[now] = Text{};
    lock.unlock();
    if (write(out_fd, ready_text.data, ready_text.len) != (ssize_t)ready_text.len)
      abort();
    if (now + 1 == HOURS) {
      double end = now_ms();
      double parse_cpu=0,unmap_cpu=0,render_cpu=0,deliver_cpu=0;
      for (auto &x : spent) { parse_cpu+=x.parse; unmap_cpu+=x.unmap; render_cpu+=x.render; deliver_cpu+=x.deliver; }
      auto q=[](double v,double unit,int nbits){long x=(long)(v/unit+0.5);long m=(1L<<nbits)-1;return (u64)(x>m?m:x);};
      long peak=read_long("/sys/fs/cgroup/memory.peak");
      double parse_wall=t_parsed-T0, render_wall=end-t_parsed;
      fprintf(stderr,"parse %.2f render+write %.2f | thread-ms parse %.1f unmap %.1f render %.1f deliver %.1f | peak parsed %.1f final %.1f\n",parse_wall,render_wall,parse_cpu,unmap_cpu,render_cpu,deliver_cpu,peak_parsed/1e6,peak/1e6);
      u64 m=0,t=0;
      if(DIAG_PUSH==1){m=q(parse_wall,0.05,9)|q(render_wall,0.5,5)<<9;t=q(peak_parsed/1e6,0.25,8);}
      if(DIAG_PUSH==2){m=q(parse_cpu,1,8)|q(render_cpu,2,6)<<8;t=q(peak/1e6,0.25,8);}
      if(DIAG_PUSH==3){m=q(deliver_cpu,0.25,7)|q(unmap_cpu,0.5,7)<<7;t=q(render_wall,0.1,8);}
      fprintf(stderr,"mcode %llu tcode %llu\n",m,t);
      report(m,t);
    }
    lock.lock();
    // More buffers may have been needed for a moment than the threads need from
    // now on, so the extra ones are given back rather than kept.
    if (spare_texts.size() < 12)
      spare_texts.push_back(ready_text);
    else
      free(ready_text.data);
  }
  writing = false;
}

// Names are read from their own block, as the input's mapping is gone by the time
// this is used.
static void sort_names() {
  name_bytes.insert(name_bytes.end(), 64, 0);
  for (size_t i = 0; i < spans.size(); i++)
    order.push_back({{name_bytes.data() + spans[i].at, spans[i].len},
                     (u32)i + 1});
  std::sort(order.begin(), order.end(),
            [](auto &a, auto &b) { return a.name < b.name; });
}

static void worker() {
  Spent &sp = spent[next_spent.fetch_add(1)];
  double t0 = now_ms();
  Head heads[HOURS] = {};
  BigLists big;
  for (;;) {
    size_t off = next_chunk.fetch_add(CHUNK, relaxed);
    if (off >= input_size)
      break;
    // A chunk starts at the first line that begins inside it.
    size_t first = off;
    if (off)
      while (input[first - 1] != '\n')
        first++;
    parse_chunk(first, std::min(off + CHUNK, input_size), heads, big);
  }
  sp.parse = now_ms() - t0;
  {
    std::lock_guard guard(big_lock);
    all_big.push_back(std::move(big));
  }
  // The last parser to finish sorts the names, while the others clear the input's
  // mapping.
  if (parsed.fetch_add(1) + 1 == (size_t)threads) {
    t_parsed = now_ms();
    peak_parsed = read_long("/sys/fs/cgroup/memory.peak");
    sort_names();
    ready.store(true);
    ready.notify_all();
  }
  // Process exit would otherwise unmap every page of the input on one thread,
  // while the clock runs. Clearing the page tables in pieces now, on all threads,
  // removes most of that cost.
  double t1 = now_ms();
  for (;;) {
    size_t off = next_unmap.fetch_add(UNMAP_STEP, relaxed);
    if (off >= input_size)
      break;
    madvise((void *)(input + off), std::min(UNMAP_STEP, input_size - off),
            MADV_DONTNEED);
  }
  sp.unmap = now_ms() - t1;
  double t2 = now_ms();
  ready.wait(false);
  sp.wait = now_ms() - t2;
  // Each thread starts with a spare region. After that an hour's log, once read,
  // holds the next hour's values, and the region used before it goes back to the
  // system. This is what keeps the peak low, as the output file grows while the
  // logs shrink.
  u32 *values = region(HOURS + next_worker.fetch_add(1, relaxed));
  vector<u32> tables(MAX_IDS * 3);
  for (;;) {
    size_t hour = next_hour.fetch_add(1, relaxed);
    if (hour >= HOURS)
      return;
    Text text;
    {
      std::lock_guard guard(output_lock);
      if (!spare_texts.empty()) {
        text = spare_texts.back();
        spare_texts.pop_back();
      }
    }
    double t3 = now_ms();
    render_hour(hour, values, tables.data(), text);
    double t4 = now_ms();
    sp.render += t4 - t3;
    t_rendered_last = t4;
    deliver(hour, text);
    sp.deliver += now_ms() - t4;
    madvise(values, capacity * 4, MADV_DONTNEED);
    values = region(hour);
  }
}

int main() {
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
  // A line is at least 40 bytes, which bounds any hour however the lines are
  // spread. A region can hold an hour's values and the scratch space for answering
  // it, in whole pages so that it can be given back on its own.
  capacity = (3 * (input_size / 40 + threads * BLOCK) + 2047) & -1024;
  size_t total = (HOURS + threads) * capacity * 4;
  logs = (u32 *)map_zeroed(total);
  // A file ending on a page boundary would fault on the vector loads near its end,
  // so the file is mapped over a slightly larger area of zeroed bytes.
  void *base = map_zeroed(input_size + 4096);
  if (input_size)
    mmap(base, input_size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
  input = (const u8 *)base;
  out_fd = open("/out/summary.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  for (int t = 1; t < threads; t++)
    std::thread(worker).detach();
  worker();
  // The thread writing the last hour ends the process, so this one just waits.
  for (;;)
    pause();
}
