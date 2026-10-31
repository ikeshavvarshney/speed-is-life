use std::arch::x86_64::*;
use std::ffi::c_void;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicUsize, Ordering::Relaxed};

const CHUNK: usize = 1 << 18;
const PLACE_AHEAD: usize = 16;

// Each node has four 4-byte slots at a computed address, so finding its edges takes no lookup and
// can be prefetched from the node id alone. Over 98% of nodes have at most four edges. A slot
// holds head + 1 above 11 bits of weight, and 0 when empty. Edges that do not fit, or that weigh
// too much, go to a side list that the node's last used slot points to with an ESCAPE weight.
const SLOTS: usize = 4;
const WEIGHT_BITS: u32 = 11;
const ESCAPE: u32 = (1 << WEIGHT_BITS) - 1;
const LIST_END: u32 = u32::MAX;

// Labels pack a distance below a 21-bit node. The generated graphs stay far below the 2^43 this
// allows for distances.
const NODE_BITS: u32 = 21;
const DIST_BITS: u32 = 64 - NODE_BITS;
const NO_DIST: u64 = (1 << DIST_BITS) - 1;
const KEY_MASK: u64 = !NO_DIST;
const LABELS_START: usize = 1 << 12;
const LABELS_LOCAL: usize = 1 << 14;
const LABELS_SHARED: usize = 1 << 19;

unsafe extern "C" {
    #[link_name = "_exit"]
    fn exit(code: i32) -> !;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
}

const PROT_READ_WRITE: i32 = 3;
const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;
const MAP_FIXED: i32 = 0x10;
const MAP_ANONYMOUS: i32 = 0x20;

// Mapped over a larger anonymous mapping so word loads near the end of the file stay readable.
fn map_input(file: &File, size: usize) -> &'static [u8] {
    unsafe {
        let base = mmap(std::ptr::null_mut(), size + 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        mmap(base, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, file.as_raw_fd(), 0);
        std::slice::from_raw_parts(base as *const u8, size)
    }
}

// Zeroed memory on 2 MB boundaries with huge pages requested, as graph accesses are random.
fn zeroed_slots(len: usize) -> &'static [AtomicU32] {
    const HUGE: usize = 1 << 21;
    const MADV_HUGEPAGE: i32 = 14;
    let bytes = (len * 4).next_multiple_of(HUGE);
    unsafe {
        let raw = mmap(std::ptr::null_mut(), bytes + HUGE, PROT_READ_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        let aligned = (raw as usize).next_multiple_of(HUGE) as *mut c_void;
        madvise(aligned, bytes, MADV_HUGEPAGE);
        std::slice::from_raw_parts(aligned as *const AtomicU32, len)
    }
}

// Only digits have bit 4 set among the bytes that can follow a number, so up to seven digits
// decode without a branch per digit.
#[inline(always)]
fn parse_digits(p: *const u8) -> (u64, *const u8) {
    let word = unsafe { p.cast::<u64>().read_unaligned() };
    let stop = !word & 0x1010_1010_1010_1010;
    if stop == 0 {
        let (mut value, mut p) = (0u64, p);
        unsafe {
            while (*p).is_ascii_digit() {
                value = value * 10 + (*p - b'0') as u64;
                p = p.add(1);
            }
        }
        return (value, p);
    }
    let len = stop.trailing_zeros() as usize / 8;
    let digits = (word << ((8 - len) * 8)) & 0x0F0F_0F0F_0F0F_0F0F;
    let pairs = digits.wrapping_mul(2561) >> 8;
    let quads = (pairs & 0x00FF_00FF_00FF_00FF).wrapping_mul(6553601) >> 16;
    let value = (quads & 0x0000_FFFF_0000_FFFF).wrapping_mul(42949672960001) >> 32;
    (value, unsafe { p.add(len) })
}

struct Loose {
    node: u32,
    head: u32,
    weight: u32,
}

#[inline(always)]
fn place(slots: &[AtomicU32], node: u32, head: u32, weight: u32, loose: &mut Vec<Loose>) {
    if weight < ESCAPE {
        let entry = (head + 1) << WEIGHT_BITS | weight;
        let at = node as usize * SLOTS;
        for slot in unsafe { slots.get_unchecked(at..at + SLOTS) } {
            if slot.load(Relaxed) == 0 && slot.compare_exchange(0, entry, Relaxed, Relaxed).is_ok() {
                return;
            }
        }
    }
    loose.push(Loose { node, head, weight });
}

// Places edges directly as they are parsed, with no log. Each placement is prefetched and made
// some edges later, so that the cache misses of these random writes overlap.
fn read_edges(input: &[u8], start: usize, graphs: [&[AtomicU32]; 2], threads: usize) -> [Vec<Loose>; 2] {
    let next = AtomicUsize::new(start);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let next = &next;
                scope.spawn(move || {
                    let mut loose = [Vec::new(), Vec::new()];
                    let mut ring = [(u32::MAX, 0, 0); PLACE_AHEAD];
                    let mut at = 0;
                    let base = input.as_ptr();
                    loop {
                        let off = next.fetch_add(CHUNK, Relaxed);
                        if off >= input.len() {
                            for &(tail, head, weight) in ring.iter().filter(|edge| edge.0 != u32::MAX) {
                                place(graphs[0], tail, head, weight, &mut loose[0]);
                                place(graphs[1], head, tail, weight, &mut loose[1]);
                            }
                            return loose;
                        }
                        let mut i = off;
                        if off != start {
                            i += input[off - 1..].iter().position(|&b| b == b'\n').unwrap();
                        }
                        let stop = (off + CHUNK).min(input.len());
                        while i < stop {
                            unsafe {
                                let (tail, p) = parse_digits(base.add(i));
                                let (head, p) = parse_digits(p.add(1));
                                let (weight, p) = parse_digits(p.add(1));
                                i = p.offset_from(base) as usize + 1;
                                let (tail, head, weight) = (tail as u32, head as u32, weight as u32);
                                // A self-loop never shortens a path.
                                if tail == head {
                                    continue;
                                }
                                _mm_prefetch::<_MM_HINT_ET0>(graphs[0].as_ptr().add(tail as usize * SLOTS).cast());
                                _mm_prefetch::<_MM_HINT_ET0>(graphs[1].as_ptr().add(head as usize * SLOTS).cast());
                                let old = std::mem::replace(ring.get_unchecked_mut(at), (tail, head, weight));
                                at = (at + 1) % PLACE_AHEAD;
                                if old.0 != u32::MAX {
                                    place(graphs[0], old.0, old.1, old.2, &mut loose[0]);
                                    place(graphs[1], old.1, old.0, old.2, &mut loose[1]);
                                }
                            }
                        }
                    }
                })
            })
            .collect();
        handles.into_iter().map(|handle| handle.join().unwrap()).collect::<Vec<_>>()
    });
    let mut merged = [Vec::new(), Vec::new()];
    for [forward, backward] in results {
        merged[0].extend(forward);
        merged[1].extend(backward);
    }
    merged
}

struct Graph {
    slots: &'static [AtomicU32],
    side_list: Vec<(u32, u32)>,
}

// Moves each node's loose edges to the side list, with its last slot pointing there.
fn finish_graph(slots: &'static [AtomicU32], mut loose: Vec<Loose>) -> Graph {
    loose.sort_unstable_by_key(|edge| edge.node);
    let mut side_list = Vec::with_capacity(loose.len() * 2);
    let mut i = 0;
    while i < loose.len() {
        let node = loose[i].node as usize;
        let own = &slots[node * SLOTS..(node + 1) * SLOTS];
        let index = side_list.len() as u32 + 1;
        let used = own.iter().position(|slot| slot.load(Relaxed) == 0).unwrap_or(SLOTS - 1);
        let displaced = own[used].load(Relaxed);
        if displaced != 0 {
            side_list.push(((displaced >> WEIGHT_BITS) - 1, displaced & ESCAPE));
        }
        while i < loose.len() && loose[i].node as usize == node {
            side_list.push((loose[i].head, loose[i].weight));
            i += 1;
        }
        side_list.push((LIST_END, 0));
        own[used].store(index << WEIGHT_BITS | ESCAPE, Relaxed);
    }
    Graph { slots, side_list }
}

// Maps nodes to their forward and backward distances in one table, so a relaxation that looks at
// both directions touches one slot. Node ids are random, so their low bits index it directly.
// Each thread owns a small table. The few searches that outgrow it borrow a large one from a
// shared pool, so memory follows how many large searches overlap rather than the thread count,
// and no search pays to map or unmap memory. Everything past `mask` is always zero.
struct Labels<'a> {
    slots: Vec<[u64; 2]>,
    mask: usize,
    used: usize,
    local: Vec<[u64; 2]>,
    pool: &'a Mutex<Vec<Vec<[u64; 2]>>>,
}

impl<'a> Labels<'a> {
    fn new(pool: &'a Mutex<Vec<Vec<[u64; 2]>>>) -> Self {
        Self { slots: vec![[0; 2]; LABELS_LOCAL], mask: LABELS_START - 1, used: 0, local: Vec::new(), pool }
    }

    fn reset(&mut self) {
        self.slots[..=self.mask].fill([0; 2]);
        self.mask = LABELS_START - 1;
        self.used = 0;
        if self.slots.len() > LABELS_LOCAL {
            let local = std::mem::take(&mut self.local);
            let shared = std::mem::replace(&mut self.slots, local);
            self.pool.lock().unwrap().push(shared);
        }
    }

    #[inline(always)]
    fn entry(&mut self, node: u32) -> *mut [u64; 2] {
        let key = (node as u64 + 1) << DIST_BITS;
        let mut at = node as usize & self.mask;
        loop {
            let slot = unsafe { self.slots.get_unchecked_mut(at) };
            let found = slot[0] & KEY_MASK;
            if found == key {
                return slot;
            }
            if found == 0 {
                *slot = [key | NO_DIST, NO_DIST];
                self.used += 1;
                return slot;
            }
            at = (at + 1) & self.mask;
        }
    }

    #[cold]
    fn grow(&mut self) {
        let old_size = self.mask + 1;
        let live: Vec<[u64; 2]> = self.slots[..old_size].iter().copied().filter(|slot| slot[0] != 0).collect();
        self.slots[..old_size].fill([0; 2]);
        let size = old_size * 2;
        if size > self.slots.len() {
            if self.slots.len() == LABELS_LOCAL {
                let shared = self.pool.lock().unwrap().pop().unwrap_or_else(|| vec![[0; 2]; LABELS_SHARED]);
                self.local = std::mem::replace(&mut self.slots, shared);
            } else {
                self.slots = vec![[0; 2]; size];
            }
        }
        self.mask = size - 1;
        for slot in live {
            let mut at = (slot[0] >> DIST_BITS) as usize - 1 & self.mask;
            while self.slots[at][0] != 0 {
                at = (at + 1) & self.mask;
            }
            self.slots[at] = slot;
        }
    }
}

#[inline(always)]
fn set_dist(slot: &mut [u64; 2], side: usize, dist: u64) {
    slot[side] = slot[side] & KEY_MASK | dist;
}

// Dial's bucket queue: one bucket per distance, in a window that a single edge cannot jump past,
// with a bitmap to find the next used bucket. Pushing and popping take constant time and no
// comparisons, as Dijkstra only ever pushes distances at or above the last one popped. The few
// heavier edges wait in a side list until the window reaches them.
const WINDOW: usize = 1 << 12;
const WINDOW_MASK: u64 = WINDOW as u64 - 1;

struct Queue {
    heads: Vec<u32>,
    used: [u64; WINDOW / 64],
    pool: Vec<(u32, u32)>,
    free: u32,
    base: u64,
    len: usize,
    far: Vec<(u64, u32)>,
}

impl Queue {
    fn new() -> Self {
        Self { heads: vec![0; WINDOW], used: [0; WINDOW / 64], pool: vec![(0, 0)], free: 0, base: 0, len: 0, far: Vec::new() }
    }

    fn reset(&mut self) {
        self.used = [0; WINDOW / 64];
        self.pool.truncate(1);
        self.far.clear();
        self.free = 0;
        self.base = 0;
        self.len = 0;
    }

    #[inline(always)]
    fn insert(&mut self, dist: u64, node: u32) {
        let bucket = (dist & WINDOW_MASK) as usize;
        let (word, bit) = (bucket / 64, 1u64 << (bucket % 64));
        unsafe {
            let head = if *self.used.get_unchecked(word) & bit != 0 { *self.heads.get_unchecked(bucket) } else { 0 };
            *self.used.get_unchecked_mut(word) |= bit;
            let index = if self.free != 0 {
                let index = self.free;
                self.free = self.pool.get_unchecked(index as usize).0;
                *self.pool.get_unchecked_mut(index as usize) = (head, node);
                index
            } else {
                self.pool.push((head, node));
                self.pool.len() as u32 - 1
            };
            *self.heads.get_unchecked_mut(bucket) = index;
        }
    }

    #[inline(always)]
    fn push(&mut self, dist: u64, node: u32) {
        self.len += 1;
        if dist - self.base < WINDOW as u64 {
            self.insert(dist, node);
        } else {
            self.far.push((dist, node));
        }
    }

    // The smallest distance held. Call this only when the queue is not empty, and not between a
    // pop and the pushes that follow it, as pushes must not fall below `base`.
    #[inline(always)]
    fn top(&mut self) -> u64 {
        let bucket = (self.base & WINDOW_MASK) as usize;
        if self.used[bucket / 64] >> (bucket % 64) & 1 == 0 || !self.far.is_empty() {
            self.advance();
        }
        self.base
    }

    #[inline(always)]
    fn next_used(&self) -> Option<u64> {
        let bucket = (self.base & WINDOW_MASK) as usize;
        let (word, shift) = (bucket / 64, bucket % 64);
        let rest = self.used[word] >> shift;
        if rest != 0 {
            return Some(self.base + rest.trailing_zeros() as u64);
        }
        let mut offset = 64 - shift as u64;
        for i in 1..=WINDOW / 64 {
            let bits = self.used[(word + i) % (WINDOW / 64)];
            if bits != 0 {
                return Some(self.base + offset + bits.trailing_zeros() as u64);
            }
            offset += 64;
        }
        None
    }

    // The node most likely to come next. A later push may overtake it, which costs nothing.
    #[inline(always)]
    fn peek(&self) -> Option<u32> {
        let dist = self.next_used()?;
        let index = unsafe { *self.heads.get_unchecked((dist & WINDOW_MASK) as usize) };
        Some(unsafe { self.pool.get_unchecked(index as usize).1 })
    }

    #[cold]
    fn advance(&mut self) {
        let near = self.next_used();
        let far = self.far.iter().map(|item| item.0).min();
        self.base = match (near, far) {
            (Some(near), Some(far)) => near.min(far),
            (Some(near), None) => near,
            (None, Some(far)) => far,
            (None, None) => return,
        };
        let mut i = 0;
        while i < self.far.len() {
            if self.far[i].0 - self.base < WINDOW as u64 {
                let (dist, node) = self.far.swap_remove(i);
                self.insert(dist, node);
            } else {
                i += 1;
            }
        }
    }

    // Removes a node at distance `base`, which `top` has just found.
    #[inline(always)]
    fn pop(&mut self) -> u32 {
        let bucket = (self.base & WINDOW_MASK) as usize;
        unsafe {
            let index = *self.heads.get_unchecked(bucket);
            let (next, node) = *self.pool.get_unchecked(index as usize);
            *self.heads.get_unchecked_mut(bucket) = next;
            if next == 0 {
                *self.used.get_unchecked_mut(bucket / 64) &= !(1u64 << (bucket % 64));
            }
            self.pool.get_unchecked_mut(index as usize).0 = self.free;
            self.free = index;
            self.len -= 1;
            node
        }
    }
}

struct Search<'a> {
    labels: Labels<'a>,
    queues: [Queue; 2],
}

#[inline(always)]
fn relax(search: &mut Search, side: usize, graph: &Graph, head: u32, next: u64, far_top: u64, best: &mut u64) {
    if next + far_top < *best {
        let slot = unsafe { &mut *search.labels.entry(head) };
        if next < slot[side] & NO_DIST {
            set_dist(slot, side, next);
            let across = slot[1 - side] & NO_DIST;
            if across != NO_DIST {
                *best = (*best).min(next + across);
            }
            search.queues[side].push(next, head);
            unsafe { _mm_prefetch::<_MM_HINT_T0>(graph.slots.as_ptr().add(head as usize * SLOTS).cast()) };
            if search.labels.used * 8 > search.labels.mask * 7 {
                search.labels.grow();
            }
        }
    }
}

// Bidirectional Dijkstra: two balls of half the radius settle far fewer nodes than one. An empty
// heap means that side reached everything it can, so unreachable targets end immediately.
fn shortest_path(graphs: &[Graph; 2], s: u32, t: u32, search: &mut Search) -> i64 {
    if s == t {
        return 0;
    }
    search.labels.reset();
    for (side, start) in [s, t].into_iter().enumerate() {
        set_dist(unsafe { &mut *search.labels.entry(start) }, side, 0);
        search.queues[side].reset();
        search.queues[side].push(0, start);
    }
    let mut best = u64::MAX;
    loop {
        if search.queues[0].len == 0 || search.queues[1].len == 0 {
            break;
        }
        let (dist0, dist1) = (search.queues[0].top(), search.queues[1].top());
        if dist0 + dist1 >= best {
            break;
        }
        let side = (search.queues[0].len > search.queues[1].len) as usize;
        let (dist, far_top) = if side == 0 { (dist0, dist1) } else { (dist1, dist0) };
        // A stale entry needs no check: all its relaxations simply fail.
        let node = search.queues[side].pop();
        let graph = &graphs[side];
        // Labels of unseen neighbours are the main cost, as node ids are random. Each node's edges
        // were prefetched when it was pushed, so the labels of the next node's neighbours can be
        // prefetched now, in time for when it is processed.
        if let Some(coming) = search.queues[side].peek() {
            let at = coming as usize * SLOTS;
            for slot in unsafe { graph.slots.get_unchecked(at..at + SLOTS) } {
                let label = (slot.load(Relaxed) >> WEIGHT_BITS).wrapping_sub(1) as usize & search.labels.mask;
                unsafe { _mm_prefetch::<_MM_HINT_T0>(search.labels.slots.as_ptr().add(label).cast()) };
            }
        }
        for slot in unsafe { graph.slots.get_unchecked(node as usize * SLOTS..(node as usize + 1) * SLOTS) } {
            let entry = slot.load(Relaxed);
            if entry == 0 {
                break;
            }
            let (head, weight) = ((entry >> WEIGHT_BITS) - 1, entry & ESCAPE);
            if weight == ESCAPE {
                let mut at = head as usize;
                while graph.side_list[at].0 != LIST_END {
                    let (head, weight) = graph.side_list[at];
                    relax(search, side, graph, head, dist + weight as u64, far_top, &mut best);
                    at += 1;
                }
                break;
            }
            relax(search, side, graph, head, dist + weight as u64, far_top, &mut best);
        }
    }
    if best == u64::MAX { -1 } else { best as i64 }
}

fn main() {
    let file = File::open("/data/graph.txt").unwrap();
    let size = file.metadata().unwrap().len() as usize;
    let input = map_input(&file, size);
    let (n, p) = parse_digits(input.as_ptr());
    let (_, p) = parse_digits(unsafe { p.add(1) });
    let start = unsafe { p.offset_from(input.as_ptr()) } as usize + 1;
    let n = n as usize;
    let threads = std::thread::available_parallelism().map_or(1, |c| c.get());

    let text = std::fs::read("/data/queries.txt").unwrap();
    let mut queries = Vec::with_capacity(text.len() / 12);
    let mut i = 0;
    while i < text.len() {
        let (s, p) = parse_digits(unsafe { text.as_ptr().add(i) });
        let (t, p) = parse_digits(unsafe { p.add(1) });
        i = unsafe { p.offset_from(text.as_ptr()) } as usize + 1;
        queries.push((s as u32, t as u32));
    }

    let (forward, backward) = (zeroed_slots(n * SLOTS), zeroed_slots(n * SLOTS));
    let [loose_forward, loose_backward] = read_edges(input, start, [forward, backward], threads);
    let graphs = std::thread::scope(|scope| {
        let handle = scope.spawn(|| finish_graph(backward, loose_backward));
        [finish_graph(forward, loose_forward), handle.join().unwrap()]
    });

    let answers: Vec<AtomicI64> = queries.iter().map(|_| AtomicI64::new(0)).collect();
    let pool = Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut search = Search { labels: Labels::new(&pool), queues: [Queue::new(), Queue::new()] };
                loop {
                    let q = next.fetch_add(1, Relaxed);
                    let Some(&(s, t)) = queries.get(q) else { return };
                    answers[q].store(shortest_path(&graphs, s, t, &mut search), Relaxed);
                }
            });
        }
    });

    let mut out = String::with_capacity(answers.len() * 8);
    for answer in &answers {
        out.push_str(&answer.load(Relaxed).to_string());
        out.push('\n');
    }
    std::fs::write("/out/distances.txt", out).unwrap();
    unsafe { exit(0) };
}
