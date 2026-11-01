#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <mutex>
#include <sched.h>
#include <string_view>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <thread>
#include <unistd.h>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
// Not size_t's type, so writing one never forces sizes and masks to be reloaded.
using u64 = unsigned long long;
using v4 = __m256i;
using v2 = __m128i;
using unaligned_u64 [[gnu::aligned(1), gnu::may_alias]] = u64;
using std::atomic;
constexpr auto relaxed = std::memory_order_relaxed;
#define INLINE [[gnu::always_inline]] inline
#define AES _mm_aesenc_si128
#define LOW _mm256_castsi256_si128
#define HIGH(v) _mm256_extracti128_si256(v, 1)

const size_t CHUNK = 1 << 16, HOURS = 72, BLOCK = 256;
// The longest line in the spec's grammar is under this, so reading this much more
// than a chunk gives every line that starts in it.
const size_t LONGEST_LINE = 256;
// A chunk that starts this many minutes into an hour shows that the hour before is
// over, as the spec has the last upload of an hour land within 10 minutes of its end.
const size_t HOUR_GRACE = 10;
const size_t SLOTS = 1 << 13;
// A log entry is id << 21 | is_error << 20 | latency. A latency that does not fit,
// on about 1% of lines, puts BIG in the entry and takes the next slot whole. An
// all-zero entry is an unused part of a block, and id 0 is reserved to absorb it.
const int LATENCY_BITS = 20;
const u32 BIG = (1 << LATENCY_BITS) - 1;
const size_t MAX_IDS = 1 << 11;
const u64 DIGITS = 0x3030303030303030;
const size_t NO_RANK = -1;

struct Name {
  u32 at;
  u16 len, id;
};
struct Text {
  u8 *data;
  size_t len;
};

// Table of names, one word each: the high 48 bits of the hash and the id. After the
// first few lines it is only read, so it is shared without contention and nothing
// needs merging. With the slot that gives 60 bits of hash, which over 1,842 names
// collide with odds of 2e-12. Ids are given out in the order names are first seen.
// Popular names come early, so they take their first choice of slot, and they pack
// the counters they use later into few cache lines.
alignas(64) static atomic<u64> table[SLOTS];
// Names in the order they were found, and sorted up to `sorted_names`.
static std::mutex names_lock;
static Name order[MAX_IDS];
static size_t name_count, sorted_names, name_end;
alignas(64) static char name_bytes[MAX_IDS * 128];
// Written by every thread, so kept off the lines that each line of input reads.
alignas(64) static atomic<u64> stamps[HOURS];
alignas(64) static atomic<size_t> cursors[HOURS];
alignas(64) static atomic<u32 *> bases[HOURS];
alignas(64) static atomic<size_t> next_chunk;
// Guards the rest: how far parsing has got, which hours are over and taken, the free
// regions and the turn to write.
alignas(64) static std::mutex plan_lock;
static size_t parsed_chunks, latest_minute, hours_over, next_hour, next_output;
static size_t free_count, used_regions;
static bool writing, speculating;
static Text waiting[HOURS];
static u32 *free_regions[HOURS * 3];

alignas(64) static size_t input_size, capacity, chunk_count;
// What the thread that parsed a chunk leaves for planning the hours: its first line's
// minute, its lowest hour and a bit to say that it is done.
static atomic<u32> *chunk_states;
const u32 DONE = 1 << 31;
static u32 *regions;
static int out_fd;

// A vector whose first `n` bytes are set, for n up to 128, then clear bytes.
alignas(64) static u8 masks[192];
INLINE const u8 *low_bytes(size_t n) { return masks + 128 - n; }
INLINE v4 load(const void *p) { return _mm256_loadu_si256((const v4 *)p); }
INLINE void store(void *p, v4 v) { _mm256_storeu_si256((v4 *)p, v); }
INLINE u64 load64(const u8 *p) { return *(const unaligned_u64 *)p; }

// Zeroed memory where only touched pages are real, with nothing reserved.
static void *map_zeroed(size_t len) {
  return mmap(0, len, PROT_READ | PROT_WRITE,
              MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
}

// The spec fixes the span to Oct 30 18:00 .. Nov 2 18:00, so the day digits alone
// identify the date: adding one and wrapping at 31 turns days 30, 31, 1, 2 into
// 0, 1, 2, 3.
static size_t hour_index(const u8 *line) {
  return (line[8] * 10 + line[9] - 527) % 31 * 24 + line[11] * 10 + line[12] - 546;
}

// Bit i is set when byte i is a space, for the 64 bytes at `p`. Taking both
// vectors every time avoids a branch on the endpoint's length, which half the
// lines would mispredict.
INLINE u64 spaces(const u8 *p, v4 space) {
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

// Mixes up to 64 bytes of a name at a fixed cost, masking off what follows it.
INLINE v2 mix_name(const u8 *p, size_t len, v2 h) {
  const u8 *cut = low_bytes(len);
  v4 a = _mm256_and_si256(load(p), load(cut));
  v4 b = _mm256_and_si256(load(p + 32), load(cut + 32));
  return AES(AES(_mm_xor_si128(LOW(a), h), HIGH(a)), AES(LOW(b), HIGH(b)));
}

// Two rounds after the last input would spread each byte everywhere, but half of the
// state can still collide for names differing only along one of AES's diagonals. It
// takes three.
INLINE u64 hash_name(const u8 *p, size_t len) {
  v2 seed = _mm_set_epi64x(0x2D358DCCAA6C78A5, 0x8BB84B93962EACC9);
  v2 h = mix_name(p, len, seed);
  if (len > 64)
    h = mix_name(p + 64, len - 64, h);
  return _mm_cvtsi128_si64(AES(AES(AES(h, seed), seed), seed));
}

[[gnu::cold, gnu::noinline]] static u16 find_name(u64 hash, const u8 *name,
                                                  size_t len) {
  for (size_t slot = hash;; slot++) {
    atomic<u64> &cell = table[slot % SLOTS];
    u64 seen = cell.load(relaxed);
    if (!seen) {
      std::lock_guard guard(names_lock);
      seen = cell.load(relaxed);
      if (!seen) {
        memcpy(name_bytes + name_end, name, len);
        order[name_count] = {(u32)name_end, (u16)len, (u16)(name_count + 1)};
        name_count++;
        name_end += len;
        if (name_count >= MAX_IDS)
          abort();
        cell.store(hash >> 16 << 16 | name_count, relaxed);
        return name_count;
      }
    }
    if ((seen ^ hash) >> 16 == 0)
      return seen;
  }
}

// Regions are real memory only where touched, and come back warm when reused.
static u32 *take_region() {
  std::lock_guard guard(plan_lock);
  return free_count ? free_regions[--free_count]
                    : regions + used_regions++ * capacity;
}

// The output file grows until the end, so that is when the peak is set, and from
// mid-run memory left in a free region is worth more to the score than saved faults.
static void give_region(void *region) {
  if (next_hour >= HOURS / 2)
    madvise(region, capacity * sizeof(u32), MADV_DONTNEED);
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

// One walk along the lines of a chunk.
struct Walk {
  const u8 *line;
  u32 **heads;
  u32 *at;
  u64 stamp;
  size_t hour;
};

INLINE void parse_line(Walk &walk, size_t &lowest, v4 space_vector) {
  const u8 *line = walk.line;
  // The buffer was just copied into the second cache level, and misses in the first
  // stall every load on this line, so the input ahead is brought in early.
  _mm_prefetch((const char *)line + 256, _MM_HINT_T0);
  // These bytes hold the month, day and hour. Nearly every line has the hour
  // of the line before it, as the file is the shards' chunks in arrival order.
  u64 now = load64(line + 5);
  if (now != walk.stamp) {
    walk.heads[walk.hour] = walk.at;
    walk.stamp = now;
    walk.hour = hour_index(line);
    lowest = std::min(lowest, walk.hour);
    stamps[walk.hour].store(now, relaxed);
    walk.at = walk.heads[walk.hour];
  }
  const u8 *fields = line + 28;
  u64 mask = spaces(fields, space_vector);
  const u8 *name = fields + __builtin_ctzll(mask) + 1;
  u64 rest = mask & (mask - 1);
  while (!rest) {
    fields += 64;
    rest = spaces(fields, space_vector);
  }
  const u8 *space = fields + __builtin_ctzll(rest);
  size_t len = space - name;

  u64 hash = hash_name(name, len), seen = table[hash % SLOTS].load(relaxed);
  u32 id = (seen ^ hash) >> 16 ? find_name(hash, name, len) : (u16)seen;

  u32 error = space[1] == '5';
  const u8 *next = space + 5;
  u32 latency = parse_latency(next);
  u32 entry = id << (LATENCY_BITS + 1) | error << LATENCY_BITS;
  u32 *at = walk.at;
  // Blocks are aligned, so this finds one with no room for two slots, or no block.
  if (((size_t)at + 4) % (BLOCK * 4) < 8)
    at = claim_block(walk.hour);
  if (latency >= BIG) {
    *at++ = entry | BIG;
    entry = 0;
  }
  *at++ = entry | latency;
  walk.at = at;
  walk.line = next;
}

// Where a line ends depends on the line before it, so one walk has to finish each
// line before the core can start the next. Two walks take turns, which lets the core
// work on a line from each at once.
static size_t parse_chunk(const u8 *line, const u8 *end, u32 **heads) {
  size_t lowest = HOURS;
  const v4 space_vector = _mm256_set1_epi8(' ');
  const u8 *mid = line + (end - line) / 2;
  mid = (const u8 *)memchr(mid, '\n', end - mid);
  mid = mid ? mid + 1 : end;
  Walk a{line, heads, heads[0], 0, 0}, b{mid, heads + HOURS, heads[HOURS], 0, 0};
  while (a.line < mid && b.line < end) {
    parse_line(a, lowest, space_vector);
    parse_line(b, lowest, space_vector);
  }
  while (a.line < mid)
    parse_line(a, lowest, space_vector);
  while (b.line < end)
    parse_line(b, lowest, space_vector);
  a.heads[a.hour] = a.at;
  b.heads[b.hour] = b.at;
  return lowest;
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
// values at `values`, whose first value has rank `first` in the whole group. The
// ranks are followed by three of NO_RANK, so finding how many fall on each side takes
// no branch, and it is the same for what is written: a call writes three answers, and
// any that are not its own are written again by the calls for higher ranks. `spare`
// is scratch memory.
static void select_ranks(u32 *values, size_t n, size_t first,
                         const size_t *ranks, u32 *found, u32 *spare) {
  if (ranks[0] >= first + n)
    return;
  if (n <= 8) {
    u32 sorted[8];
    sort_lanes(values, n, sorted);
    for (int i = 0; i < 3; i++)
      found[i] = sorted[(ranks[i] - first) & 7];
    return;
  }
  u32 a = values[0], b = values[n / 2], c = values[n - 1];
  u32 pivot = std::min(std::max(a, b), std::max(std::min(a, b), c));
  size_t below = partition(values, n, pivot, values, spare);
  auto below_rank = [&](size_t rank) {
    return (ranks[0] < rank) + (ranks[1] < rank) + (ranks[2] < rank);
  };
  if (!below) {
    // The pivot is the smallest value, and with many equal values this would
    // never end. Its copies are all the same answer, so they are split off and
    // need no more work.
    size_t copies = pivot == UINT32_MAX
                        ? n
                        : partition(spare, n, pivot + 1, values, spare);
    found[0] = found[1] = found[2] = pivot;
    size_t split = below_rank(first + copies);
    select_ranks(spare, n - copies, first + copies, ranks + split, found + split,
                 values);
    return;
  }
  // Each side uses what the other leaves free, so `spare` never needs more room than
  // the first call's values.
  size_t split = below_rank(first + below);
  select_ranks(values, below, first, ranks, found, spare + n - below);
  select_ranks(spare, n - below, first + below, ranks + split, found + split, values);
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

// Writes a space and the value with no branch on its length. The caller reserved
// the room.
INLINE u8 *put_number(u8 *out, u32 value) {
  if (value >= 100000000) [[unlikely]]
    return out + sprintf((char *)out, " %u", value);
  u64 digits = eight_digits(value);
  int skip = __builtin_ctzll(digits | 1ull << 56) / 8;
  u64 word = digits >> (skip * 8) | DIGITS;
  *out = ' ';
  memcpy(out + 1, &word, 8);
  return out + 9 - skip;
}

static std::string_view name_of(Name name) {
  return {name_bytes + name.at, name.len};
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
    std::lock_guard guard(names_lock);
    Name *all = order, *found = all + name_count;
    std::sort(all + sorted_names, found, by_name);
    std::inplace_merge(all, all + sorted_names, found, by_name);
    sorted_names = known = name_count;
    memcpy(names, all, known * sizeof(Name));
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
  u64 stamp = stamps[hour].load(relaxed);
  memcpy(label + 5, &stamp, 8);
  u8 *text = (u8 *)take_region(), *cursor = text;
  size_t end = sum;
  u32 *spare = values + used + 64;
  for (size_t i = 0; i < known; i++) {
    std::string_view name = name_of(names[i]);
    u32 id = names[i].id, errors = counts[id * 2 + 1];
    size_t n = counts[id * 2] + errors;
    if (!n)
      continue;
    end -= n;
    size_t ranks[6] = {(50 * n + 99) / 100 - 1, (95 * n + 99) / 100 - 1,
                       (99 * n + 99) / 100 - 1, NO_RANK, NO_RANK, NO_RANK};
    u32 found[6];
    select_ranks(values + end, n, 0, ranks, found, spare);
    store(cursor, load(label));
    cursor += 21;
    store(cursor, load(name.data()));
    store(cursor + 32, load(name.data() + 32));
    if (name.size() > 64)
      memcpy(cursor, name.data(), name.size());
    cursor += name.size();
    for (u32 value : {(u32)n, found[0], found[1], found[2], errors})
      cursor = put_number(cursor, value);
    cursor = put_number(cursor, errors * 1000000ull / n);
    *cursor++ = '\n';
  }
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
    if (now + 1 == HOURS)
      _exit(0);
    give_region(ready.data);
    lock.lock();
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
    u32 state = chunk_states[parsed_chunks].load(std::memory_order_acquire);
    if (!(state & DONE))
      break;
    if ((state >> 16 & 255) < hours_over) {
      execl("/proc/self/exe", "again", "again", nullptr);
      abort();
    }
    latest_minute = std::max<size_t>(latest_minute, state & 65535);
    if (speculating && latest_minute >= HOUR_GRACE)
      hours_over = std::max(hours_over, (latest_minute - HOUR_GRACE) / 60);
  }
  if (parsed_chunks == chunk_count)
    hours_over = HOURS;
  return next_hour < hours_over ? next_hour++ : next_hour == HOURS ? HOURS + 1 : HOURS;
}

static void worker() {
  u32 *heads[HOURS * 2] = {};
  // Reading costs less than faulting the file in from one mapping that all threads
  // share, which then also has to be unmapped page by page. Each thread has its own
  // open file so that they don't share its reference count.
  alignas(64) u8 buffer[CHUNK + LONGEST_LINE + 128];
  int fd = open("/data/requests.log", O_RDONLY);
  for (;;) {
    size_t hour = next_task(), chunk;
    if (hour < HOURS) {
      u32 *space = take_region();
      Text text = render_hour(hour, space);
      give_region(space);
      deliver(hour, text);
    } else if ((chunk = next_chunk.fetch_add(1, relaxed)) < chunk_count) {
      size_t off = chunk * CHUNK;
      ssize_t got = pread(fd, buffer, CHUNK + LONGEST_LINE, off);
      // A chunk's lines are those that begin after its first byte, up to and including
      // the first byte of the next chunk.
      const u8 *line = buffer;
      if (off)
        line = (const u8 *)memchr(buffer, '\n', got) + 1;
      const u8 *end = buffer + std::min<size_t>(CHUNK + 1, got);
      u32 minute = 0;
      if (line < end)
        minute = hour_index(line) * 60 + line[14] * 10 + line[15] - 528;
      size_t lowest = parse_chunk(line, end, heads);
      chunk_states[chunk].store(DONE | lowest << 16 | minute, std::memory_order_release);
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
  // thread stack or a small mapping.
  prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
  memset(masks, 255, 128);
  input_size = lseek(open("/data/requests.log", O_RDONLY), 0, SEEK_END);
  cpu_set_t cpus;
  sched_getaffinity(0, sizeof cpus, &cpus);
  int workers = CPU_COUNT(&cpus);
  chunk_count = (input_size + CHUNK - 1) / CHUNK;
  chunk_states = (atomic<u32> *)map_zeroed(chunk_count * 4 + 1);
  // A line with a big latency takes two slots and over 40 bytes, which bounds any
  // hour however the lines are spread. A region can hold an hour's values and the
  // scratch space for answering it, or its text, in whole pages.
  capacity = (3 * (input_size / 20 + workers * BLOCK) + (1 << 17)) & -1024;
  regions = (u32 *)map_zeroed(std::size(free_regions) * capacity * sizeof(u32));
  out_fd = open("/out/summary.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  for (int w = 1; w < workers; w++)
    std::thread(worker).detach();
  worker();
  // The thread writing the last hour ends the process, so this one just waits.
  for (;;)
    pause();
}
