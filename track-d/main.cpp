#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
// Not size_t's type, so writing one never forces sizes and masks to be reloaded.
using u64 = unsigned long long;
using std::atomic;
using std::vector;
constexpr auto relaxed = std::memory_order_relaxed;
#define INLINE [[gnu::always_inline]] inline

const size_t CHUNK = 1 << 18, PLACE_AHEAD = 16, HUGE_PAGE = 1 << 21;

// Each node has four 4-byte slots at a computed address, so finding its edges
// takes no lookup and can be prefetched from the node id alone. Over 98% of
// nodes have at most four edges. A slot holds head + 1 above 11 bits of weight,
// and 0 when empty. Edges that do not fit, or that weigh too much, go to a side
// list that the node's last used slot points to with an ESCAPE weight.
const size_t SLOTS = 4;
const int WEIGHT_BITS = 11;
const u32 ESCAPE = (1 << WEIGHT_BITS) - 1, LIST_END = UINT32_MAX;

// Labels pack a distance below a 21-bit node. The generated graphs stay far
// below the 2^43 this allows for distances.
const int DIST_BITS = 64 - 21;
const u64 NO_DIST = (1ull << DIST_BITS) - 1, KEY_MASK = ~NO_DIST;
const size_t LABELS_START = 1 << 12, LABELS_LOCAL = 1 << 14;
const size_t LABELS_SHARED = 1 << 19;

// Dial's bucket queue window, which a single light edge cannot jump past.
const size_t WINDOW = 1 << 12, WORDS = WINDOW / 64;

struct Edge {
  u32 first, second;
};
struct Loose {
  u32 node, head, weight;
};
struct Slot {
  u64 side[2];
};
using Slots = atomic<u32> *;

struct Graph {
  Slots slots;
  vector<Edge> side_list;
};

// Mapped over a larger anonymous mapping so word loads near the end of the file
// stay readable.
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

// Zeroed memory on 2 MB boundaries with huge pages requested, as graph accesses
// are random.
static Slots zeroed_slots(size_t len) {
  size_t bytes = (len * 4 + HUGE_PAGE - 1) & -HUGE_PAGE;
  auto raw = (size_t)mmap(0, bytes + HUGE_PAGE, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  auto aligned = (void *)((raw + HUGE_PAGE - 1) & -HUGE_PAGE);
  madvise(aligned, bytes, MADV_HUGEPAGE);
  return (Slots)aligned;
}

// Only digits have bit 4 set among the bytes that can follow a number, so up to
// seven digits decode without a branch per digit.
INLINE u64 parse_digits(const u8 *&p) {
  u64 word;
  memcpy(&word, p, 8);
  u64 stop = ~word & 0x1010101010101010;
  if (!stop) {
    u64 value = 0;
    for (; *p >= '0' && *p <= '9'; p++)
      value = value * 10 + *p - '0';
    return value;
  }
  int len = __builtin_ctzll(stop) / 8;
  p += len;
  u64 digits = (word << ((8 - len) * 8)) & 0x0F0F0F0F0F0F0F0F;
  u64 pairs = (digits * 2561) >> 8;
  u64 quads = ((pairs & 0x00FF00FF00FF00FF) * 6553601) >> 16;
  return ((quads & 0x0000FFFF0000FFFF) * 42949672960001) >> 32;
}

INLINE void place(Slots slots, u32 node, u32 head, u32 weight,
                  vector<Loose> &loose) {
  if (weight < ESCAPE) {
    u32 entry = (head + 1) << WEIGHT_BITS | weight;
    Slots own = slots + node * SLOTS;
    for (size_t i = 0; i < SLOTS; i++) {
      u32 empty = 0;
      if (own[i].load(relaxed) == 0 &&
          own[i].compare_exchange_strong(empty, entry, relaxed))
        return;
    }
  }
  loose.push_back({node, head, weight});
}

struct Reader {
  vector<Loose> loose[2];
};
static const u8 *input;
static size_t input_size, first_edge;
static atomic<size_t> next_chunk;
static Slots graph_slots[2];

// Places edges directly as they are parsed, with no log. Each placement is
// prefetched and made some edges later, so that the cache misses of these
// random writes overlap.
static void read_edges(Reader *reader) {
  Loose ring[PLACE_AHEAD];
  for (auto &edge : ring)
    edge.node = UINT32_MAX;
  size_t at = 0;
  auto place_edge = [&](const Loose &e) {
    place(graph_slots[0], e.node, e.head, e.weight, reader->loose[0]);
    place(graph_slots[1], e.head, e.node, e.weight, reader->loose[1]);
  };
  for (;;) {
    size_t off = next_chunk.fetch_add(CHUNK, relaxed);
    if (off >= input_size)
      break;
    size_t i = off;
    if (off != first_edge)
      while (input[i - 1] != '\n')
        i++;
    size_t stop = std::min(off + CHUNK, input_size);
    while (i < stop) {
      const u8 *p = input + i;
      u32 tail = parse_digits(p);
      p++;
      u32 head = parse_digits(p);
      p++;
      u32 weight = parse_digits(p);
      i = p - input + 1;
      // A self-loop never shortens a path.
      if (tail == head)
        continue;
      __builtin_prefetch(graph_slots[0] + tail * SLOTS, 1);
      __builtin_prefetch(graph_slots[1] + head * SLOTS, 1);
      Loose old = ring[at];
      ring[at] = {tail, head, weight};
      at = (at + 1) % PLACE_AHEAD;
      if (old.node != UINT32_MAX)
        place_edge(old);
    }
  }
  for (auto &edge : ring)
    if (edge.node != UINT32_MAX)
      place_edge(edge);
}

// Moves each node's loose edges to the side list, with its last slot pointing
// there.
static Graph finish_graph(Slots slots, vector<Loose> loose) {
  std::sort(loose.begin(), loose.end(),
            [](auto &a, auto &b) { return a.node < b.node; });
  vector<Edge> list;
  list.reserve(loose.size() * 2);
  for (size_t i = 0; i < loose.size();) {
    u32 node = loose[i].node;
    Slots own = slots + node * SLOTS;
    u32 index = list.size() + 1;
    size_t used = 0;
    while (used < SLOTS - 1 && own[used].load(relaxed))
      used++;
    u32 displaced = own[used].load(relaxed);
    if (displaced)
      list.push_back({(displaced >> WEIGHT_BITS) - 1, displaced & ESCAPE});
    for (; i < loose.size() && loose[i].node == node; i++)
      list.push_back({loose[i].head, loose[i].weight});
    list.push_back({LIST_END, 0});
    own[used].store(index << WEIGHT_BITS | ESCAPE, relaxed);
  }
  return {slots, std::move(list)};
}

static std::mutex pool_lock;
static vector<std::pair<Slot *, size_t>> label_pool;

static Slot *zeroed_labels(size_t len) {
  return (Slot *)calloc(len, sizeof(Slot));
}

// Maps nodes to their forward and backward distances in one table, so a
// relaxation that looks at both directions touches one slot. Node ids are
// random, so their low bits index it directly. Each thread owns a small table.
// The few searches that outgrow it borrow a large one from a shared pool, so
// memory follows how many large searches overlap rather than the thread count,
// and no search pays to map or unmap memory. All past `mask` is always zero.
struct Labels {
  size_t capacity = LABELS_LOCAL, mask = LABELS_START - 1, used = 0;
  Slot *slots = zeroed_labels(LABELS_LOCAL), *local = nullptr;

  void reset() {
    memset(slots, 0, (mask + 1) * sizeof(Slot));
    mask = LABELS_START - 1;
    used = 0;
    if (capacity > LABELS_LOCAL) {
      std::lock_guard guard(pool_lock);
      label_pool.push_back({slots, capacity});
      slots = local;
      capacity = LABELS_LOCAL;
    }
  }

  INLINE Slot &entry(u32 node) {
    u64 key = (u64)(node + 1) << DIST_BITS;
    for (size_t at = node & mask;; at = (at + 1) & mask) {
      Slot &slot = slots[at];
      u64 found = slot.side[0] & KEY_MASK;
      if (found == key)
        return slot;
      if (!found) {
        slot = {{key | NO_DIST, NO_DIST}};
        used++;
        return slot;
      }
    }
  }

  [[gnu::cold]] void grow() {
    size_t old_size = mask + 1, size = old_size * 2;
    // Sized exactly: growing it by doubling would hold two copies at the peak.
    vector<Slot> live;
    live.reserve(used);
    for (size_t i = 0; i < old_size; i++)
      if (slots[i].side[0])
        live.push_back(slots[i]);
    memset(slots, 0, old_size * sizeof(Slot));
    if (size > capacity) {
      if (capacity == LABELS_LOCAL) {
        local = slots;
        std::lock_guard guard(pool_lock);
        if (label_pool.empty()) {
          slots = zeroed_labels(LABELS_SHARED);
          capacity = LABELS_SHARED;
        } else {
          std::tie(slots, capacity) = label_pool.back();
          label_pool.pop_back();
        }
      } else {
        free(slots);
        slots = zeroed_labels(size);
        capacity = size;
      }
    }
    mask = size - 1;
    for (const Slot &slot : live) {
      size_t at = ((slot.side[0] >> DIST_BITS) - 1) & mask;
      while (slots[at].side[0])
        at = (at + 1) & mask;
      slots[at] = slot;
    }
  }
};

INLINE void set_dist(Slot &slot, int side, u64 dist) {
  slot.side[side] = (slot.side[side] & KEY_MASK) | dist;
}

// Dial's bucket queue: one bucket per distance, with a bitmap to find the next
// used bucket. Pushing and popping take constant time and no comparisons, as
// Dijkstra only ever pushes distances at or above the last one popped. The few
// heavier edges wait in a side list until the window reaches them.
struct Queue {
  vector<u32> heads = vector<u32>(WINDOW);
  u64 used[WORDS] = {};
  vector<Edge> pool = vector<Edge>(1); // {next, node}
  u32 free_item = 0;
  u64 base = 0;
  size_t len = 0;
  vector<std::pair<u64, u32>> far;

  void reset() {
    memset(used, 0, sizeof used);
    pool.resize(1);
    far.clear();
    free_item = 0;
    base = 0;
    len = 0;
  }

  INLINE void insert(u64 dist, u32 node) {
    size_t bucket = dist & (WINDOW - 1), word = bucket / 64;
    u64 bit = 1ull << (bucket % 64);
    u32 head = used[word] & bit ? heads[bucket] : 0;
    used[word] |= bit;
    u32 index = free_item;
    if (index) {
      free_item = pool[index].first;
      pool[index] = {head, node};
    } else {
      pool.push_back({head, node});
      index = pool.size() - 1;
    }
    heads[bucket] = index;
  }

  INLINE void push(u64 dist, u32 node) {
    len++;
    if (dist - base < WINDOW)
      insert(dist, node);
    else
      far.push_back({dist, node});
  }

  // The smallest distance held. Call this only when the queue is not empty, and
  // not between a pop and the pushes that follow it, as pushes must not fall
  // below `base`.
  INLINE u64 top() {
    size_t bucket = base & (WINDOW - 1);
    if (!((used[bucket / 64] >> (bucket % 64)) & 1) || !far.empty())
      advance();
    return base;
  }

  // The distance of the next used bucket, or UINT64_MAX if there is none.
  INLINE u64 next_used() const {
    size_t bucket = base & (WINDOW - 1), word = bucket / 64, shift = bucket % 64;
    u64 rest = used[word] >> shift;
    if (rest)
      return base + __builtin_ctzll(rest);
    u64 offset = 64 - shift;
    for (size_t i = 1; i <= WORDS; i++, offset += 64) {
      u64 bits = used[(word + i) % WORDS];
      if (bits)
        return base + offset + __builtin_ctzll(bits);
    }
    return UINT64_MAX;
  }

  [[gnu::cold]] void advance() {
    u64 lowest = next_used();
    for (auto &item : far)
      lowest = std::min(lowest, item.first);
    if (lowest == UINT64_MAX)
      return;
    base = lowest;
    for (size_t i = 0; i < far.size();) {
      if (far[i].first - base < WINDOW) {
        auto [dist, node] = far[i];
        far[i] = far.back();
        far.pop_back();
        insert(dist, node);
      } else {
        i++;
      }
    }
  }

  // Removes a node at distance `base`, which `top` has just found.
  INLINE u32 pop() {
    size_t bucket = base & (WINDOW - 1);
    u32 index = heads[bucket];
    auto [next, node] = pool[index];
    heads[bucket] = next;
    if (!next)
      used[bucket / 64] &= ~(1ull << (bucket % 64));
    pool[index].first = free_item;
    free_item = index;
    len--;
    return node;
  }
};

struct Search {
  Labels labels;
  Queue queues[2];
};

INLINE void relax(Search &search, int side, const Graph &graph, u32 head,
                  u64 next, u64 far_top, u64 &best) {
  if (next + far_top >= best)
    return;
  Slot &slot = search.labels.entry(head);
  if (next < (slot.side[side] & NO_DIST)) {
    set_dist(slot, side, next);
    u64 across = slot.side[1 - side] & NO_DIST;
    if (across != NO_DIST)
      best = std::min(best, next + across);
    search.queues[side].push(next, head);
    __builtin_prefetch(graph.slots + head * SLOTS);
    if (search.labels.used * 8 > search.labels.mask * 7)
      search.labels.grow();
  }
}

// Bidirectional Dijkstra: two balls of half the radius settle far fewer nodes
// than one. An empty queue means that side reached everything it can, so
// unreachable targets end immediately.
static long long shortest_path(const Graph *graphs, u32 s, u32 t,
                               Search &search) {
  if (s == t)
    return 0;
  Labels &labels = search.labels;
  labels.reset();
  u32 starts[2] = {s, t};
  for (int side = 0; side < 2; side++) {
    set_dist(labels.entry(starts[side]), side, 0);
    search.queues[side].reset();
    search.queues[side].push(0, starts[side]);
  }
  u64 best = UINT64_MAX;
  while (search.queues[0].len && search.queues[1].len) {
    u64 dists[2] = {search.queues[0].top(), search.queues[1].top()};
    if (dists[0] + dists[1] >= best)
      break;
    int side = search.queues[0].len > search.queues[1].len;
    u64 dist = dists[side], far_top = dists[1 - side];
    Queue &queue = search.queues[side];
    // A stale entry needs no check: all its relaxations simply fail.
    u32 node = queue.pop();
    const Graph &graph = graphs[side];
    // Labels of unseen neighbours are the main cost, as node ids are random.
    // Each node's edges were prefetched when it was pushed, so the labels of the
    // neighbours of the node most likely to come next can be prefetched now. A
    // later push may overtake it, which costs nothing.
    u64 coming = queue.next_used();
    if (coming != UINT64_MAX) {
      u32 next_node = queue.pool[queue.heads[coming & (WINDOW - 1)]].second;
      for (size_t i = 0; i < SLOTS; i++) {
        u32 slot = graph.slots[next_node * SLOTS + i].load(relaxed);
        __builtin_prefetch(labels.slots +
                           (((slot >> WEIGHT_BITS) - 1) & labels.mask));
      }
    }
    for (size_t i = 0; i < SLOTS; i++) {
      u32 entry = graph.slots[node * SLOTS + i].load(relaxed);
      if (!entry)
        break;
      u32 head = (entry >> WEIGHT_BITS) - 1, weight = entry & ESCAPE;
      if (weight == ESCAPE) {
        for (const Edge *e = &graph.side_list[head]; e->first != LIST_END; e++)
          relax(search, side, graph, e->first, dist + e->second, far_top, best);
        break;
      }
      relax(search, side, graph, head, dist + weight, far_top, best);
    }
  }
  return best == UINT64_MAX ? -1 : (long long)best;
}

int main() {
  input = map_input("/data/graph.txt", input_size);
  const u8 *p = input;
  size_t n = parse_digits(p);
  p++;
  parse_digits(p);
  first_edge = p - input + 1;
  next_chunk = first_edge;
  cpu_set_t cpus;
  sched_getaffinity(0, sizeof cpus, &cpus);
  int threads = CPU_COUNT(&cpus);

  size_t text_size;
  const u8 *text = map_input("/data/queries.txt", text_size);
  vector<Edge> queries;
  for (const u8 *q = text; q < text + text_size; q++) {
    u32 s = parse_digits(q);
    q++;
    queries.push_back({s, (u32)parse_digits(q)});
  }

  graph_slots[0] = zeroed_slots(n * SLOTS);
  graph_slots[1] = zeroed_slots(n * SLOTS);
  vector<Reader> readers(threads);
  vector<std::thread> pool;
  for (auto &reader : readers)
    pool.emplace_back(read_edges, &reader);
  vector<Loose> loose[2];
  for (int t = 0; t < threads; t++) {
    pool[t].join();
    for (int side = 0; side < 2; side++)
      loose[side].insert(loose[side].end(), readers[t].loose[side].begin(),
                         readers[t].loose[side].end());
  }
  Graph graphs[2];
  std::thread finisher([&] {
    graphs[1] = finish_graph(graph_slots[1], std::move(loose[1]));
  });
  graphs[0] = finish_graph(graph_slots[0], std::move(loose[0]));
  finisher.join();

  vector<long long> answers(queries.size());
  atomic<size_t> next_query{0};
  pool.clear();
  for (int t = 0; t < threads; t++)
    pool.emplace_back([&] {
      Search search;
      for (;;) {
        size_t q = next_query.fetch_add(1, relaxed);
        if (q >= queries.size())
          return;
        answers[q] =
            shortest_path(graphs, queries[q].first, queries[q].second, search);
      }
    });
  for (auto &thread : pool)
    thread.join();

  std::string out;
  for (long long answer : answers)
    out += std::to_string(answer) + '\n';
  int fd = open("/out/distances.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (write(fd, out.data(), out.size()) != (ssize_t)out.size())
    abort();
  _exit(0);
}
