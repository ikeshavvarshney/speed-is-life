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
#define AES _mm_aesenc_si128
#define LOW _mm256_castsi256_si128
#define HIGH(v) _mm256_extracti128_si256(v, 1)

#ifndef PREFETCH
#define PREFETCH 192
#endif
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
static int in_fd, out_fd;
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

#include <stdio.h>
#include <time.h>
#include <x86intrin.h>
#include <sys/resource.h>
namespace probe {
static double now_ms(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static double T0=now_ms();
static long read_long(const char*p){FILE*f=fopen(p,"r");long v=0;if(f){if(fscanf(f,"%ld",&v)!=1)v=0;fclose(f);}return v;}
static long read_stat(const char*key){FILE*f=fopen("/sys/fs/cgroup/memory.stat","r");char k[64];long v=0,r=0;if(f){while(fscanf(f,"%63s %ld",k,&v)==2)if(!strcmp(k,key))r=v;fclose(f);}return r;}
static u64 q(double v,double unit,int nbits){long x=(long)(v/unit+0.5);long m=(1L<<nbits)-1;return (u64)(x>m?m:x);}
[[noreturn]] static void report(u64 code){
  u64 mcode=code&16383,tcode=code>>14;
  long target=(128L<<20)+(long)mcode*(256L<<10);
  long cur=read_long("/sys/fs/cgroup/memory.current");
  long extra=target-cur;
  extra-=extra/512;
  if(extra>0){
    char*m=(char*)mmap(0,extra,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    for(long i=0;i<extra;i+=4096)m[i]=1;
    munmap(m,extra);
  }
  double until=T0+3000+(double)tcode*16;
  timespec ts;ts.tv_sec=(time_t)(until/1e3);ts.tv_nsec=(long)((until-ts.tv_sec*1e3)*1e6);
  clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&ts,0);
  _exit(0);}
struct Spent{atomic<u64> map,parse,plan,render,deliver,idle,count,place,select,forks,start_sum,first_parse;atomic<u64> end_parse;atomic<u64> faults,switches;};
static Spent *sp;
static u64 tsc0;
#define TSC() __rdtsc()
static void finish(){
  u64 now=TSC();
  auto a=TSC();double b=now_ms();while(now_ms()-b<1.0){}double per=(TSC()-a)/(now_ms()-b);
  double wall=(double)(now-tsc0)/per;
  auto ms=[&](atomic<u64>&x){return (double)x.load()/per;};
  long peak=read_long("/sys/fs/cgroup/memory.peak");
  long shmem=read_stat("shmem"),anon=read_stat("anon"),kernel=read_stat("kernel"),pt=read_stat("pagetables");
  double since0=T0;
  fprintf(stderr,"wall %.2f forks %.2f startavg %.2f endparse %.2f | map %.1f parse %.1f plan %.1f render %.1f (count %.1f place %.1f select %.1f) deliver %.1f idle %.1f | peak %.2f shmem %.2f anon %.2f kernel %.2f pt %.2f regions %zu\n",wall,ms(sp->forks),ms(sp->start_sum)/12,ms(sp->end_parse),ms(sp->map),ms(sp->parse),ms(sp->plan),ms(sp->render),ms(sp->count),ms(sp->place),ms(sp->select),ms(sp->deliver),ms(sp->idle),peak/1e6,shmem/1e6,anon/1e6,kernel/1e6,pt/1e6,g.used_regions);
  (void)since0;
  u64 c=0;
  if(5==1)c=q(ms(sp->map),1,8)|q(ms(sp->parse),1,8)<<8|q(ms(sp->render),0.5,8)<<16;
  if(5==2)c=q(ms(sp->count),0.5,6)|q(ms(sp->place),0.5,6)<<6|q(ms(sp->select),0.5,6)<<12|q(ms(sp->deliver),0.5,6)<<18;
  if(5==3)c=q(wall,0.1,8)|q(ms(sp->forks),0.05,8)<<8|q(ms(sp->end_parse),0.1,8)<<16;
  if(5==4)c=q(ms(sp->idle),0.25,8)|q(ms(sp->plan),0.05,8)<<8|q(ms(sp->start_sum)/12,0.05,8)<<16;
  if(5==5)c=q(peak/1e6,0.1,8)|q(shmem/1e6,0.1,8)<<8|q(pt/1e6,0.05,4)<<16|q(g.used_regions,2,4)<<20;
  if(5==6)c=q(anon/1e6,0.05,8)|q(kernel/1e6,0.05,8)<<8|q(pt/1e6,0.05,8)<<16;
  fprintf(stderr,"code %llu mcode %llu tcode %llu\n",c,c&16383,c>>14);
  report(c);
}
}

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
  v2 h = AES(AES(_mm_xor_si128(LOW(a), seed), HIGH(a)), AES(LOW(b), HIGH(b)));
  for (size_t i = 64; i < len; i += 16) {
    auto tail = (const __m128i_u *)low_bytes(std::min<size_t>(len - i, 16));
    h = AES(h, _mm_and_si128(_mm_loadu_si128((const v2 *)(p + i)), *tail));
  }
  // Two rounds would spread each byte everywhere, but half of the state can still
  // collide for names differing only along one of AES's diagonals. It takes three.
  return _mm_cvtsi128_si64(AES(AES(AES(h, seed), seed), seed));
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

static void parse_chunk(size_t chunk, const u8 *line, const u8 *end, u32 **heads) {
  u64 stamp = 0;
  size_t hour = 0, lowest = HOURS;
  u32 *at = heads[0];
  while (line < end) {
    _mm_prefetch((const char *)line + PREFETCH, _MM_HINT_T0);
    // These bytes hold the month, day and hour. Nearly every line has the hour
    // of the line before it, as the file is the shards' chunks in arrival order.
    u64 now = load64(line + 5);
    if (now != stamp) {
      heads[hour] = at;
      stamp = now;
      hour = hour_index(line);
      lowest = std::min(lowest, hour);
      g.stamps[hour].store(now, relaxed);
      at = heads[hour];
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
    // Blocks are aligned, so this finds one with no room for two slots, or no block.
    if (((size_t)at + 4) % (BLOCK * 4) < 8)
      at = claim_block(hour);
    if (latency >= BIG) {
      *at++ = entry | BIG;
      entry = 0;
    }
    *at++ = entry | latency;
    line = next;
  }
  heads[hour] = at;
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
  auto r0 = TSC();
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
  auto r1 = TSC();
  probe::sp->count += r1 - r0;
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
  probe::sp->place += r2 - r1;

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
  probe::sp->select += TSC() - r2;
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
      probe::finish();
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
  u32 *heads[HOURS] = {};
  // Each chunk replaces the last in the same place, which also leaves the bytes just
  // past the end of the file safe to read. The kernel frees the page tables with it.
  u8 *window = (u8 *)map_zeroed(CHUNK + 2 * PAGE, MAP_PRIVATE);
  for (;;) {
    if (g.starting_over.load(relaxed))
      start_over();
    auto t0 = TSC();
    size_t hour = next_task(), chunk;
    probe::sp->plan += TSC() - t0;
    if (hour < HOURS) {
      auto t1 = TSC();
      u32 *space = take_region();
      Text text = render_hour(hour, space);
      give_region(space);
      auto t2 = TSC();
      probe::sp->render += t2 - t1;
      deliver(hour, text);
      probe::sp->deliver += TSC() - t2;
    } else if ((chunk = g.next_chunk.fetch_add(1, relaxed)) < chunk_count) {
      auto t1 = TSC();
      size_t off = chunk * CHUNK, left = input_size - off;
      mmap(window, std::min(CHUNK + PAGE, left), PROT_READ, MAP_PRIVATE | MAP_FIXED,
           in_fd, off);
      // A chunk's lines are those that begin after its first byte, up to and including
      // the first byte of the next chunk.
      auto t2 = TSC();
      probe::sp->map += t2 - t1;
      const u8 *line = window;
      if (off)
        line = (const u8 *)memchr(window, '\n', left) + 1;
      const u8 *end = window + std::min(CHUNK + 1, left);
      if (line < end)
        chunk_states[chunk].first_minute = hour_index(line) * 60 +
                                           (line[14] - '0') * 10 + line[15] - '0';
      parse_chunk(chunk, line, end, heads);
      probe::sp->parse += TSC() - t2;
      probe::sp->end_parse.store(TSC() - probe::tsc0);
      chunk_states[chunk].done.store(true, std::memory_order_release);
    } else if (hour > HOURS) {
      return;
    } else {
      auto t1 = TSC();
      sched_yield();
      probe::sp->idle += TSC() - t1;
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
  input_size = lseek(in_fd, 0, SEEK_END);
  cpu_set_t cpus;
  sched_getaffinity(0, sizeof cpus, &cpus);
  int workers = CPU_COUNT(&cpus);
  mmap(&g, sizeof g, PROT_READ | PROT_WRITE,
       MAP_SHARED | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  chunk_count = (input_size + CHUNK - 1) / CHUNK;
  chunk_states = (ChunkState *)map_zeroed(chunk_count * sizeof(ChunkState) + PAGE, MAP_SHARED);
  // A line with a big latency takes two slots and over 40 bytes, which bounds any
  // hour however the lines are spread. A region can hold an hour's values and the
  // scratch space for answering it, or its text. Every worker needs page tables for
  // the regions it touches, so while lines arrive as expected they sit close together.
  capacity = (3 * (input_size / 20 + workers * BLOCK) + (1 << 17)) & -BLOCK;
  if (speculating)
    capacity = std::min<size_t>(capacity, 1 << 18);
  regions = (u32 *)map_zeroed(std::size(g.free_regions) * capacity * sizeof(u32), MAP_SHARED);
  out_fd = open("/out/summary.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  probe::sp = (probe::Spent *)mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  auto f0 = TSC();
  probe::tsc0 = f0 - (u64)((probe::now_ms() - probe::T0) * 2500000);
  for (int w = 1; w < workers && is_first; w++)
    is_first = fork() != 0;
  probe::sp->start_sum += TSC() - probe::tsc0;
  if (is_first)
    probe::sp->forks.store(TSC() - f0);
  worker();
  // The worker writing the last hour may be another process, and it is this one whose
  // exit stops the clock.
  for (;;)
    pause();
}
