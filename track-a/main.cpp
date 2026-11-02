#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <mutex>
#include <pthread.h>
#include <string_view>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned;
// Not size_t's type, so writing one never forces sizes and masks to be reloaded.
using u64 = unsigned long long;
using v4 = __m256i;
using v2 = __m128i;
using std::atomic;
constexpr auto relaxed = std::memory_order_relaxed;
#define INLINE [[gnu::always_inline]] inline
#define AES _mm_aesenc_si128

// The box's hardware threads.
const size_t WORKERS = 12, STACK = 1 << 18;
const size_t CHUNK = 1 << 18, CLEAR_STEP = 1 << 25, HOURS = 72, BLOCK = 256;
const size_t SLOTS = 1 << 13, MAX_IDS = 1 << 11, NONE = -1;
// A log entry is id << 21 | is_error << 20 | latency. A latency that does not fit,
// on about 1% of lines, puts BIG in the entry and takes the next slot whole. An
// all-zero entry is an unused part of a block, and id 0 is reserved to absorb it.
const u32 BIG = (1 << 20) - 1, DONE = 1u << 31;

using Text = std::string_view;

// Table of names, one word each: the high 48 bits of the hash and the id. After the
// first few lines it is only read, so it is shared without contention and nothing
// needs merging. With the slot that gives 60 bits of hash, which over 1,842 names
// collide with odds of 2e-12. Ids are given out in the order names are first seen.
// `stamps`, `cursors` and `next_chunk` are written by every thread, so each is kept
// off the lines that each line of input reads.
alignas(64) static atomic<u64> table[SLOTS], stamps[HOURS];
alignas(64) static atomic<size_t> cursors[HOURS], next_chunk, next_clear;
alignas(64) static atomic<u32 *> bases[HOURS];
// Names in the order they were found, and sorted up to `sorted`. Each is a word:
// its offset in name_bytes << 32 | its length << 16 | its id. `plan_lock` guards the
// rest: how far parsing has got, which hours are over and taken, and the free regions.
alignas(64) static std::mutex names_lock, plan_lock;
static u64 order[MAX_IDS];
static size_t name_count, sorted, name_end;
alignas(64) static char name_bytes[MAX_IDS * 128];
// `grace`: a chunk that starts this many minutes into an hour shows that the hour
// before is over, as the spec has the last upload of an hour land within 10 minutes
// of its end. After a restart it is too long for any hour to end before all parsing has.
static size_t parsed, hours_over, next_hour, pooled, carved, grace;
// What the thread that parsed a chunk leaves for planning the hours: its first line's
// minute, its lowest hour and a bit to say that it is done. Then each hour's size plus
// one, which also says that its text is ready.
alignas(64) static atomic<u32> chunk_states[1 << 14], text_sizes[HOURS];
static u8 *texts[HOURS];

alignas(64) static size_t input_size, capacity;
static const u8 *input;
static u32 *regions, *pool[HOURS * 3];
// A vector whose first `n` bytes are set, for n up to 128, then clear bytes. After
// them, 16 zeros and 16 of 15, which keep the low half of the last bytes of a vector.
alignas(64) static u8 masks[224];
// For each set of lanes, a permutation that moves them to the front, keeping order,
// and the other lanes after them.
static u64 moves[256];

INLINE const u8 *low_bytes(size_t n) { return masks + 128 - n; }
INLINE v4 load(const void *p) { return _mm256_loadu_si256((const v4 *)p); }
INLINE v2 load2(const void *p) { return _mm_loadu_si128((const v2 *)p); }
INLINE void store(void *p, v4 v) { _mm256_storeu_si256((v4 *)p, v); }

// Zeroed readable and writable memory where only touched pages are real, with nothing
// reserved.
static void *map_zeroed(size_t len) {
  return mmap(0, len, 3, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
}

// The spec fixes the span to Oct 30 18:00 .. Nov 2 18:00, so the day digits alone
// identify the date: adding one and wrapping at 31 turns days 30, 31, 1, 2 into
// 0, 1, 2, 3.
static size_t hour_index(const u8 *l) {
  return (l[8] * 10 + l[9] - 527) % 31 * 24 + l[11] * 10 + l[12] - 546;
}

// Bit i is set when byte i is a space, for the 64 bytes at `p`. Taking both
// vectors every time avoids a branch on the endpoint's length, which half the
// lines would mispredict.
INLINE u32 spaces32(v4 v) {
  return _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(' ')));
}
INLINE u64 spaces(const u8 *p) {
  return spaces32(load(p)) | (u64)spaces32(load(p + 32)) << 32;
}

// Parses the latency and moves `p` past the newline, with no branch on its length:
// the 16 bytes before the newline end in the digits, so what comes before them is
// masked off and the digits are merged in pairs, fours and eights, each step a single
// instruction.
INLINE u32 parse_latency(const u8 *&p) {
  v2 newline = _mm_cmpeq_epi8(load2(p), _mm_set1_epi8('\n'));
  int len = __builtin_ctz(_mm_movemask_epi8(newline));
  p += len + 1;
  v2 digits = _mm_and_si128(load2(masks + 192 + len), load2(p - 17));
  v2 pairs = _mm_maddubs_epi16(digits, _mm_set1_epi16(10 | 1 << 8));
  v2 fours = _mm_madd_epi16(pairs, _mm_set1_epi32(100 | 1 << 16));
  fours = _mm_packus_epi32(fours, fours);
  // 75536 is 10000 | 1 << 16, the weights of the fours.
  u64 parts = _mm_cvtsi128_si64(_mm_madd_epi16(fours, _mm_set1_epi32(75536)));
  return (u32)parts * 100000000 + (parts >> 32);
}

// Mixes up to 64 bytes of a name at a fixed cost, masking off what follows it.
INLINE v2 mix_name(const u8 *p, size_t len, v2 h) {
  const u8 *cut = low_bytes(len);
  v4 a = _mm256_and_si256(load(p), load(cut));
  v4 b = _mm256_and_si256(load(p + 32), load(cut + 32));
  v2 a0 = _mm256_castsi256_si128(a), b0 = _mm256_castsi256_si128(b);
  return AES(AES(_mm_xor_si128(a0, h), _mm256_extracti128_si256(a, 1)),
             AES(b0, _mm256_extracti128_si256(b, 1)));
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

[[gnu::cold]] static u32 find_name(u64 hash, const u8 *name, size_t len) {
  for (size_t slot = hash;; slot++) {
    atomic<u64> &cell = table[slot % SLOTS];
    u64 seen = cell.load(relaxed);
    if (!seen) {
      std::lock_guard guard(names_lock);
      if (!(seen = cell.load(relaxed))) {
        memcpy(name_bytes + name_end, name, len);
        order[name_count++] = name_end << 32 | len << 16 | (name_count + 1);
        name_end += len;
        cell.store(seen = hash >> 16 << 16 | name_count, relaxed);
      }
    }
    if ((seen ^ hash) >> 16 == 0)
      return (u16)seen;
  }
}

// Regions are real memory only where touched, and come back warm when reused.
static u32 *take_region() {
  std::lock_guard guard(plan_lock);
  return pooled ? pool[--pooled] : regions + carved++ * capacity;
}

// The output file grows until the end, so that is when the peak is set. Only near the
// end is memory left in a free region worth more to the score than the saved faults.
static void give_region(const void *region) {
  if (next_hour >= HOURS * 5 / 6)
    madvise((void *)region, capacity * 4, MADV_DONTNEED);
  std::lock_guard guard(plan_lock);
  pool[pooled++] = (u32 *)region;
}

// Takes a block from the hour's region, to append without any sharing.
static u32 *claim_block(size_t hour) {
  u32 *base = bases[hour].load(relaxed);
  if (!base) {
    u32 *mine = take_region();
    bases[hour].compare_exchange_strong(base, mine) ? void(base = mine)
                                                    : give_region(mine);
  }
  u32 *at = base + cursors[hour].fetch_add(BLOCK, relaxed);
  memset(at, 0, BLOCK * 4);
  return at;
}

// One walk along the lines of a chunk.
struct Walk {
  const u8 *line;
  u32 **heads, *at;
  u64 stamp;
  size_t hour;
};

INLINE void parse_line(Walk &walk, size_t &lowest) {
  const u8 *line = walk.line;
  // The hardware prefetcher stops at page boundaries.
  _mm_prefetch((const char *)line + 1024, _MM_HINT_T0);
  // These bytes hold the month, day and hour. Nearly every line has the hour
  // of the line before it, as the file is the shards' chunks in arrival order.
  u64 now = _mm_cvtsi128_si64(load2(line + 5));
  if (now != walk.stamp) {
    walk.heads[walk.hour] = walk.at;
    walk.stamp = now;
    walk.hour = hour_index(line);
    lowest = std::min(lowest, walk.hour);
    stamps[walk.hour].store(now, relaxed);
    walk.at = walk.heads[walk.hour];
  }
  const u8 *fields = line + 28;
  u64 mask = spaces(fields);
  const u8 *name = fields + __builtin_ctzll(mask) + 1;
  u64 rest = mask & (mask - 1);
  while (!rest)
    rest = spaces(fields += 64);
  const u8 *space = fields + __builtin_ctzll(rest);
  size_t len = space - name;

  u64 hash = hash_name(name, len), seen = table[hash % SLOTS].load(relaxed);
  u32 id = (seen ^ hash) >> 16 ? find_name(hash, name, len) : (u16)seen;

  u32 entry = id << 21 | (space[1] == '5') << 20;
  walk.line = space + 5;
  u32 latency = parse_latency(walk.line);
  // Blocks are aligned, so this finds one with no room for two slots, or no block.
  if (((size_t)walk.at + 4) % (BLOCK * 4) < 8)
    walk.at = claim_block(walk.hour);
  if (latency >= BIG)
    *walk.at++ = entry | BIG, entry = 0;
  *walk.at++ = entry | latency;
}

// Where a line ends depends on the line before it, so one walk has to finish each
// line before the core can start the next. Two walks take turns, which lets the core
// work on a line from each at once.
static size_t parse_chunk(const u8 *line, const u8 *end, u32 **heads) {
  size_t lowest = HOURS;
  const u8 *mid = std::find(line + (end - line) / 2, end, '\n');
  mid += mid < end;
  Walk a{line, heads, heads[0]}, b{mid, heads + HOURS, heads[HOURS]};
  while (a.line < mid && b.line < end) {
    parse_line(a, lowest);
    parse_line(b, lowest);
  }
  while (a.line < mid)
    parse_line(a, lowest);
  while (b.line < end)
    parse_line(b, lowest);
  a.heads[a.hour] = a.at;
  b.heads[b.hour] = b.at;
  return lowest;
}

// One layer of a sorting network: each lane is compared with the lane named in
// `partner`, and the lanes set in B take the larger value.
template <int B> INLINE v4 sort_layer(v4 v, v4 partner) {
  v4 w = _mm256_permutevar8x32_epi32(v, partner);
  return _mm256_blend_epi32(_mm256_min_epu32(v, w), _mm256_max_epu32(v, w), B);
}

// Sorts up to eight values in a register, with no branches: lanes past n read as
// UINT32_MAX, so they sort to the end. This is the optimal 19-comparator,
// 6-layer network for eight inputs.
INLINE v4 sort_lanes(const u32 *values, size_t n) {
  v4 past = _mm256_xor_si256(load(low_bytes(n * 4)), _mm256_set1_epi32(-1));
  v4 v = _mm256_or_si256(load(values), past);
  v = sort_layer<0xCC>(v, _mm256_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5));
  v = sort_layer<0xF0>(v, _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3));
  v = sort_layer<0xAA>(v, _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6));
  v = sort_layer<0x30>(v, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
  v = sort_layer<0x50>(v, _mm256_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7));
  return sort_layer<0x54>(v, _mm256_setr_epi32(0, 2, 1, 4, 3, 6, 5, 7));
}

// Moves the values below `pivot` to the front of `values` and the others downwards
// from `top`, returning the count below. One permutation serves both: it puts the low
// lanes first and the high ones last, and each store keeps its own end. Writing low
// values never overtakes reading, and anything else a store touches is free.
INLINE size_t partition(u32 *values, size_t n, u32 pivot, u32 *top) {
  v4 wall = _mm256_set1_epi32(pivot);
  size_t above = 0;
  for (size_t i = 0; i < n; i += 8) {
    v4 v = load(values + i);
    u32 lanes = n - i < 8 ? (1u << (n - i)) - 1 : 255;
    u32 up = lanes & _mm256_movemask_ps(_mm256_castsi256_ps(
                         _mm256_cmpeq_epi32(_mm256_max_epu32(v, wall), v)));
    v2 move = _mm_cvtsi64_si128(moves[255 ^ up]);
    v = _mm256_permutevar8x32_epi32(v, _mm256_cvtepu8_epi32(move));
    store(values + i - above, v);
    store(top - above - 8, v);
    above += __builtin_popcount(up);
  }
  return n - above;
}

// Sets found[i] to the value of rank ranks[i], for ascending ranks among the n values
// at `values`, whose first value has rank `first` in the whole group. The ranks are
// followed by three of -1, so finding how many fall on each side takes no branch, and
// it is the same for what is written: a pass writes three answers, and any that are not
// its own are written again later for higher ranks. Memory below `top` is free, and
// holds at most one copy of each value at a time, so it never needs more than n + 8.
static void select_ranks(u32 *values, size_t n, size_t first, const size_t *ranks,
                         u32 *found, u32 *top) {
  while (ranks[0] < first + n) {
    if (n <= 8) {
      u32 sorted[8];
      store(sorted, sort_lanes(values, n));
      for (int i = 0; i < 3; i++)
        found[i] = sorted[(ranks[i] - first) & 7];
      return;
    }
    u32 a = values[0], b = values[n / 2], c = values[n - 1];
    u32 pivot = std::min(std::max(a, b), std::max(std::min(a, b), c));
    size_t below = partition(values, n, pivot, top), above = n - below;
    if (!below) {
      // The pivot is the smallest value, and with many equal values this would never
      // end. Such values are few and rare enough for a plain sort.
      std::sort(top - n, top);
      for (int i = 0; i < 3; i++)
        found[i] = top[std::min(ranks[i] - first, n - 1) - n];
      return;
    }
    size_t last = first + below;
    size_t split = (ranks[0] < last) + (ranks[1] < last) + (ranks[2] < last);
    select_ranks(values, below, first, ranks, found, top - above);
    memcpy(values, top - above, above * 4);
    n = above, first = last, ranks += split, found += split;
  }
}

// Turns a value below 10^8 into eight digit bytes, most significant first in
// memory, by dividing all the pairs at once (from Paul Khuong's "How to print
// integers really fast").
INLINE u64 eight_digits(u32 value) {
  u64 merged = value / 10000 | (u64)(value % 10000) << 32;
  u64 top = ((merged * 10486) >> 20) & (0x7Full << 32 | 0x7F);
  u64 hundreds = ((merged - 100 * top) << 16) + top;
  u64 tens = ((hundreds * 103) >> 10) & 0x000F000F000F000F;
  return tens + ((hundreds - 10 * tens) << 8);
}

// Writes a space and the value with no branch on its length. The caller reserved
// the room.
INLINE u8 *put_number(u8 *out, u32 value) {
  if (value >= 100000000) [[unlikely]]
    return out + sprintf((char *)out, " %u", value);
  u64 digits = eight_digits(value);
  int skip = __builtin_ctzll(digits | 1ull << 56) / 8;
  u64 word = digits >> (skip * 8) | 0x3030303030303030;
  *out = ' ';
  memcpy(out + 1, &word, 8);
  return out + 9 - skip;
}

static Text name_of(u64 n) { return {name_bytes + (n >> 32), (u16)(n >> 16)}; }
static bool by_name(u64 a, u64 b) { return name_of(a) < name_of(b); }

// Groups the hour's values by name, then answers each group with selection.
static void render_hour(size_t hour, u32 *space) {
  u32 *counts = space, *starts = space + MAX_IDS * 2;
  u64 *names = (u64 *)(space + MAX_IDS * 3);
  u32 *values = space + MAX_IDS * 5;
  // Few names are new after the first hour, so sorting them is a small job.
  names_lock.lock();
  std::sort(order + sorted, order + name_count, by_name);
  std::inplace_merge(order, order + sorted, order + name_count, by_name);
  size_t known = sorted = name_count;
  memcpy(names, order, known * 8);
  names_lock.unlock();
  size_t used = cursors[hour].load(relaxed);
  u32 *log = bases[hour].load(relaxed);
  memset(counts, 0, MAX_IDS * 8);
  for (size_t i = 0; i < used; i++) {
    // Other cores wrote these entries, so each new line takes long to arrive.
    _mm_prefetch((const char *)(log + i + 64), _MM_HINT_T0);
    counts[log[i] >> 20]++;
    i += (log[i] & BIG) == BIG;
  }
  // Groups lie in reverse order, so the vector stores that run past the end of
  // the group being answered only overwrite one that is finished.
  u32 sum = 0, biggest = 0;
  for (size_t i = known; i--;) {
    u32 id = (u16)names[i], n = counts[id * 2] + counts[id * 2 + 1];
    starts[id] = sum;
    sum += n;
    biggest = std::max(biggest, n);
  }
  starts[0] = sum;
  // Scratch for selection goes right after the values, so it touches few new pages.
  u32 *top = values + sum + biggest + 16;
  for (size_t i = 0; i < used; i++) {
    u32 latency = log[i] & BIG, slot = starts[log[i] >> 21]++;
    if (latency == BIG)
      latency = log[++i];
    values[slot] = latency;
  }
  if (log)
    give_region(log);
  char label[33] = "2026-MM-DDTHH:00:00Z            ";
  memcpy(label + 5, &stamps[hour], 8);
  u8 *text = (u8 *)take_region(), *cursor = text;
  auto rank = [](size_t p, size_t n) { return (p * n + 99) / 100 - 1; };
  for (size_t i = 0; i < known; i++) {
    u32 id = (u16)names[i], errors = counts[id * 2 + 1];
    size_t n = counts[id * 2] + errors;
    if (!n)
      continue;
    sum -= n;
    size_t ranks[6] = {rank(50, n), rank(95, n), rank(99, n), NONE, NONE, NONE};
    u32 found[6];
    select_ranks(values + sum, n, 0, ranks, found, top);
    store(cursor, load(label));
    cursor += 21;
    Text name = name_of(names[i]);
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
  texts[hour] = text;
  text_sizes[hour] = cursor - text + 1;
}

// The file is the shards' chunks in order of arrival, so an hour's lines end soon
// after the next hour starts. Holding on to every hour until the end would waste
// memory, so an hour is rendered once parsing is far enough past it. No bound on
// lateness is promised, so this is checked: each chunk is compared, in file order,
// with the hours already taken as over. A late line means starting again without
// guessing, before the last hour can be written.
static size_t next_task() {
  std::lock_guard guard(plan_lock);
  for (u32 state; (state = chunk_states[parsed]) & DONE; parsed++) {
    if ((state >> 16 & 255) < hours_over)
      execl("/proc/self/exe", "again", "again", nullptr), _exit(1);
    size_t over = (std::max<size_t>(state & 65535, grace) - grace) / 60;
    hours_over = std::max(hours_over, over);
  }
  if (parsed * CHUNK >= input_size)
    hours_over = HOURS;
  return next_hour < hours_over ? next_hour++ : HOURS + (next_hour == HOURS);
}

// Parses a chunk or clears a piece of the input, and says if there was any left.
static bool parse_or_clear(u32 **heads) {
  // No hour can be answered until every chunk before it is done, so if a thread stalls
  // the others would only pile up logs, and peak memory is the worst of five runs.
  if (next_chunk.load(relaxed) > parsed + 48 && hours_over < HOURS)
    return false;
  size_t chunk = next_chunk.fetch_add(1, relaxed), off = chunk * CHUNK;
  if (off < input_size) {
    // A chunk's lines are those that begin after its first byte, up to and including
    // the first byte of the next chunk.
    const u8 *line = off ? (u8 *)memchr(input + off, '\n', 256) + 1 : input;
    const u8 *end = input + std::min(off + CHUNK + 1, input_size);
    u32 minute = line < end ? hour_index(line) * 60 + line[14] * 10 + line[15] - 528 : 0;
    u32 lowest = parse_chunk(line, end, heads);
    chunk_states[chunk] = DONE | lowest << 16 | minute;
    return true;
  }
  // Exit would clear the input's page tables on one thread, with the clock running.
  // Each call interrupts every thread to flush its TLB, so calls are few and large,
  // and the last stops at the end of the file rather than clear what lies above it.
  off = next_clear.fetch_add(CLEAR_STEP, relaxed);
  size_t len = std::min(CLEAR_STEP, input_size - std::min(off, input_size));
  return len && (madvise((void *)(input + off), len, MADV_DONTNEED), true);
}

static void *worker(void *) {
  u32 *heads[HOURS * 2] = {};
  for (;;) {
    size_t hour = next_task();
    if (hour < HOURS) {
      u32 *space = take_region();
      render_hour(hour, space);
      give_region(space);
    } else if (!parse_or_clear(heads)) {
      if (hour > HOURS)
        return nullptr;
      sched_yield();
    }
  }
}

int main(int argc, char **) {
  grace = argc < 2 ? 10 : 1 << 20;
  // The box has huge pages set to always, which charges 2 MB for a first touch of a
  // thread stack or a small mapping.
  prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
  memset(masks, 255, 128);
  memset(masks + 208, 15, 16);
  for (u64 m = 0, all = 0x0706050403020100; m < 256; m++) {
    u64 ones = _pdep_u64(m, 0x0101010101010101) * 255;
    u64 half = _mm_popcnt_u64(m) * 4;
    moves[m] = _pext_u64(all, ones) | _pext_u64(all, ~ones) << half << half;
  }
  int fd = open("/data/requests.log", O_RDONLY);
  input_size = lseek(fd, 0, SEEK_END);
  // Mapped over a larger anonymous mapping, so vector loads past the end stay readable.
  input = (const u8 *)map_zeroed(input_size + (1 << 13));
  mmap((void *)input, input_size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
  // A line with a big latency takes two slots and over 40 bytes, which bounds any
  // hour however the lines are spread. A region can hold an hour's values and the
  // scratch space for answering it, or its text, in whole pages.
  capacity = (3 * (input_size / 20 + WORKERS * 2 * BLOCK) + (1 << 17)) & -1024;
  regions = (u32 *)map_zeroed(std::size(pool) * capacity * 4);
  // This thread makes the input's page tables, far faster than the workers can use
  // them, so that no worker waits on one or is charged for it. Each thread's stack is
  // faulted here too, and needs no unmapping when the thread ends.
  size_t table = 0;
  for (; table < 1 << 24; table += 1 << 21)
    *(volatile const u8 *)(input + std::min(table, input_size));
  auto stacks = (char *)map_zeroed(WORKERS * STACK);
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  for (size_t w = 1; w < WORKERS; w++) {
    memset(stacks + w * STACK - 16384, 0, 16384);
    pthread_attr_setstack(&attr, stacks + (w - 1) * STACK, STACK);
    pthread_t thread;
    pthread_create(&thread, &attr, worker, nullptr);
  }
  for (; table < input_size + (1 << 21); table += 1 << 21)
    *(volatile const u8 *)(input + std::min(table, input_size));
  int out = open("/out/summary.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  // This thread alone writes, in order, and parses while the next hour is not ready.
  // Writing the last hour ends the process, which needs no threads to be joined.
  u32 *heads[HOURS * 2] = {};
  for (size_t hour = 0; hour < HOURS;) {
    u32 size = text_sizes[hour];
    if (size) {
      if (write(out, texts[hour], size - 1) != size - 1)
        _exit(1);
      give_region(texts[hour++]);
    } else if (!parse_or_clear(heads))
      sched_yield();
  }
  _exit(0);
}
