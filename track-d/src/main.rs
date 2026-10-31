use std::ffi::c_void;
use std::fs::File;
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering::Relaxed};

const CHUNK: usize = 1 << 18;
const PARTITIONS: usize = 64;
const BLOCK: usize = 13107;
const TABLE_START: usize = 1 << 11;
const TABLE_KEEP: usize = 1 << 17;

unsafe extern "C" {
    #[link_name = "_exit"]
    fn exit(code: i32) -> !;
    fn mallopt(param: i32, value: i32) -> i32;
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
}

const M_MMAP_THRESHOLD: i32 = -3;

// Mapped over a larger anonymous mapping so word loads near the end of the file stay readable.
fn map_input(file: &File, size: usize) -> &'static [u8] {
    const PROT_READ: i32 = 1;
    const MAP_PRIVATE: i32 = 2;
    const MAP_FIXED: i32 = 0x10;
    const MAP_ANONYMOUS: i32 = 0x20;
    unsafe {
        let base = mmap(std::ptr::null_mut(), size + 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        mmap(base, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, file.as_raw_fd(), 0);
        std::slice::from_raw_parts(base as *const u8, size)
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

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct Edge {
    tail: u16,
    head: u32,
    weight: u32,
}

#[derive(Default)]
struct Bucket {
    blocks: Vec<Box<MaybeUninit<[Edge; BLOCK]>>>,
    used: usize,
    total: usize,
}

impl Bucket {
    #[inline(always)]
    fn push(&mut self, edge: Edge) {
        if self.used == 0 {
            self.blocks.push(Box::new_uninit());
        }
        unsafe {
            let block = self.blocks.last_mut().unwrap_unchecked().as_mut_ptr();
            block.cast::<Edge>().add(self.used).write(edge);
        }
        self.used = (self.used + 1) % BLOCK;
        self.total += 1;
    }

    fn edges(&self) -> impl Iterator<Item = &[Edge]> {
        self.blocks.iter().enumerate().map(|(i, block)| {
            let n = if i + 1 == self.blocks.len() && self.used != 0 { self.used } else { BLOCK };
            unsafe { std::slice::from_raw_parts(block.as_ptr().cast::<Edge>(), n) }
        })
    }
}

struct Graph {
    offsets: Vec<u32>,
    edges: Vec<(u32, u32)>,
    has_in: Vec<AtomicU64>,
}

struct Shared<T>(*mut T);
unsafe impl<T> Sync for Shared<T> {}

fn zeroed_atomics(len: usize) -> Vec<AtomicU64> {
    let mut zeros = std::mem::ManuallyDrop::new(vec![0u64; len]);
    unsafe { Vec::from_raw_parts(zeros.as_mut_ptr().cast(), len, len) }
}

// Sorts edges into node ranges as they are parsed, so that the graph can then be built one range
// at a time by a single thread, inside its cache, with no shared counters.
fn parse_edges(input: &'static [u8], start: usize, shift: u32, threads: usize) -> Vec<Vec<Bucket>> {
    let next = AtomicUsize::new(start);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let next = &next;
                scope.spawn(move || {
                    let mut buckets: Vec<Bucket> = (0..PARTITIONS).map(|_| Bucket::default()).collect();
                    let base = input.as_ptr();
                    loop {
                        let off = next.fetch_add(CHUNK, Relaxed);
                        if off >= input.len() {
                            return buckets;
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
                                let bucket = buckets.get_unchecked_mut((tail >> shift) as usize);
                                bucket.push(Edge {
                                    tail: (tail & ((1 << shift) - 1)) as u16,
                                    head: head as u32,
                                    weight: weight as u32,
                                });
                                i = p.offset_from(base) as usize + 1;
                            }
                        }
                    }
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    })
}

fn build_graph(parsed: Vec<Vec<Bucket>>, n: usize, shift: u32, wanted: &[u64], threads: usize) -> Graph {
    let mut partitions: Vec<Vec<Bucket>> = (0..PARTITIONS).map(|_| Vec::with_capacity(threads)).collect();
    for buckets in parsed {
        for (p, bucket) in buckets.into_iter().enumerate() {
            partitions[p].push(bucket);
        }
    }
    let mut bases = vec![0usize; PARTITIONS + 1];
    for p in 0..PARTITIONS {
        bases[p + 1] = bases[p] + partitions[p].iter().map(|b| b.total).sum::<usize>();
    }
    let total = bases[PARTITIONS];
    let tasks: Vec<Mutex<Vec<Bucket>>> = partitions.into_iter().map(Mutex::new).collect();
    let mut offsets = vec![0u32; n + 1];
    let mut edges = vec![(0u32, 0u32); total];
    let has_in = zeroed_atomics(n / 64 + 1);
    let offsets_ptr = Shared(offsets.as_mut_ptr());
    let edges_ptr = Shared(edges.as_mut_ptr());
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (next, tasks, bases, has_in, offsets_ptr, edges_ptr) =
                (&next, &tasks, &bases, &has_in, &offsets_ptr, &edges_ptr);
            scope.spawn(move || {
                let mut cursors = vec![0u32; 1 << shift];
                loop {
                    let p = next.fetch_add(1, Relaxed);
                    if p >= PARTITIONS {
                        return;
                    }
                    let buckets = std::mem::take(&mut *tasks[p].lock().unwrap());
                    cursors.fill(0);
                    for bucket in &buckets {
                        for block in bucket.edges() {
                            for edge in block {
                                cursors[edge.tail as usize] += 1;
                            }
                        }
                    }
                    let first = p << shift;
                    let last = ((p + 1) << shift).min(n);
                    let mut sum = bases[p] as u32;
                    // Each node range and edge range has one owner: the thread that claimed p.
                    for node in first..last {
                        let degree = cursors[node - first];
                        cursors[node - first] = sum;
                        unsafe { *offsets_ptr.0.add(node) = sum };
                        sum += degree;
                    }
                    for bucket in &buckets {
                        for block in bucket.edges() {
                            for edge in block {
                                let (head, weight) = (edge.head, edge.weight);
                                let cursor = &mut cursors[edge.tail as usize];
                                unsafe { *edges_ptr.0.add(*cursor as usize) = (head, weight) };
                                *cursor += 1;
                                let node = head as usize;
                                if wanted[node / 64] >> (node % 64) & 1 != 0 {
                                    has_in[node / 64].fetch_or(1 << (node % 64), Relaxed);
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    offsets[n] = total as u32;
    Graph { offsets, edges, has_in }
}

// Open-addressing node-to-distance map. Most searches settle few nodes, so a small table stays in
// cache where a distance array per thread would be megabytes of misses and memory.
struct Distances {
    keys: Vec<u32>,
    values: Vec<u64>,
    mask: usize,
    used: usize,
}

impl Distances {
    fn new() -> Self {
        Self { keys: vec![0; TABLE_START], values: vec![0; TABLE_START], mask: TABLE_START - 1, used: 0 }
    }

    fn reset(&mut self) {
        if self.keys.len() > TABLE_KEEP {
            *self = Self::new();
        } else if self.used > 0 {
            self.keys.fill(0);
            self.used = 0;
        }
    }

    #[inline(always)]
    fn slot(&self, node: u32) -> usize {
        let key = node + 1;
        let mut slot = (key.wrapping_mul(0x9E37_79B1) >> 7) as usize & self.mask;
        loop {
            let seen = unsafe { *self.keys.get_unchecked(slot) };
            if seen == key || seen == 0 {
                return slot;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn get(&self, node: u32) -> u64 {
        let slot = self.slot(node);
        if self.keys[slot] == 0 { u64::MAX } else { self.values[slot] }
    }

    // Stores the distance if it is an improvement, and says whether it was.
    #[inline(always)]
    fn improve(&mut self, node: u32, distance: u64) -> bool {
        let slot = self.slot(node);
        let key = unsafe { *self.keys.get_unchecked(slot) };
        if key != 0 {
            let value = unsafe { self.values.get_unchecked_mut(slot) };
            if distance >= *value {
                return false;
            }
            *value = distance;
            return true;
        }
        self.keys[slot] = node + 1;
        self.values[slot] = distance;
        self.used += 1;
        if self.used * 2 > self.keys.len() {
            self.grow();
        }
        true
    }

    #[cold]
    fn grow(&mut self) {
        let size = self.keys.len() * 4;
        let (old_keys, old_values) = (
            std::mem::replace(&mut self.keys, vec![0; size]),
            std::mem::replace(&mut self.values, vec![0; size]),
        );
        self.mask = size - 1;
        for (&key, &value) in old_keys.iter().zip(&old_values) {
            if key != 0 {
                let slot = self.slot(key - 1);
                self.keys[slot] = key;
                self.values[slot] = value;
            }
        }
    }
}

// Four children per node: shallower than a binary heap, and siblings share a cache line.
#[derive(Default)]
struct Heap {
    items: Vec<(u64, u32)>,
}

impl Heap {
    #[inline(always)]
    fn push(&mut self, item: (u64, u32)) {
        let mut i = self.items.len();
        self.items.push(item);
        while i > 0 {
            let parent = (i - 1) / 4;
            if self.items[parent].0 <= item.0 {
                break;
            }
            self.items[i] = self.items[parent];
            i = parent;
        }
        self.items[i] = item;
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<(u64, u32)> {
        let top = *self.items.first()?;
        let last = self.items.pop().unwrap();
        let n = self.items.len();
        if n > 0 {
            let mut i = 0;
            loop {
                let first = 4 * i + 1;
                if first >= n {
                    break;
                }
                let mut best = first;
                for child in first + 1..(first + 4).min(n) {
                    if self.items[child].0 < self.items[best].0 {
                        best = child;
                    }
                }
                if self.items[best].0 >= last.0 {
                    break;
                }
                self.items[i] = self.items[best];
                i = best;
            }
            self.items[i] = last;
        }
        Some(top)
    }
}

fn shortest_path(graph: &Graph, s: u32, t: u32, dist: &mut Distances, heap: &mut Heap) -> i64 {
    if s == t {
        return 0;
    }
    // Without this, a target nothing points to would make the search settle the whole graph.
    if graph.has_in[t as usize / 64].load(Relaxed) >> (t % 64) & 1 == 0 {
        return -1;
    }
    dist.reset();
    heap.items.clear();
    dist.improve(s, 0);
    heap.push((0, s));
    let mut bound = u64::MAX;
    while let Some((d, u)) = heap.pop() {
        if u == t {
            return d as i64;
        }
        if d > dist.get(u) {
            continue;
        }
        let begin = graph.offsets[u as usize] as usize;
        let end = graph.offsets[u as usize + 1] as usize;
        for &(v, w) in &graph.edges[begin..end] {
            let next = d + w as u64;
            if next < bound && dist.improve(v, next) {
                if v == t {
                    bound = next;
                }
                heap.push((next, v));
            }
        }
    }
    -1
}

fn main() {
    unsafe { mallopt(M_MMAP_THRESHOLD, 1 << 16) };
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
    let mut wanted = vec![0u64; n / 64 + 1];
    let mut i = 0;
    while i < text.len() {
        let (s, p) = parse_digits(unsafe { text.as_ptr().add(i) });
        let (t, p) = parse_digits(unsafe { p.add(1) });
        i = unsafe { p.offset_from(text.as_ptr()) } as usize + 1;
        wanted[t as usize / 64] |= 1 << (t % 64);
        queries.push((s as u32, t as u32));
    }

    // Partitions need at most PARTITIONS ranges of at most 2^16 nodes.
    let bits = usize::BITS - n.max(1).leading_zeros();
    let shift = bits.saturating_sub(PARTITIONS.trailing_zeros()).max(1);
    assert!(shift <= 16);
    let parsed = parse_edges(input, start, shift, threads);
    let graph = build_graph(parsed, n, shift, &wanted, threads);

    let answers: Vec<AtomicI64> = queries.iter().map(|_| AtomicI64::new(0)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut dist = Distances::new();
                let mut heap = Heap::default();
                loop {
                    let q = next.fetch_add(1, Relaxed);
                    let Some(&(s, t)) = queries.get(q) else { return };
                    answers[q].store(shortest_path(&graph, s, t, &mut dist, &mut heap), Relaxed);
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
