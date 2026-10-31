use std::arch::x86_64::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::mem::MaybeUninit;
use std::ffi::c_void;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

const CHUNK: usize = 1 << 18;
const HOURS: usize = 72;
const TS_LEN: usize = 27;
const ERR_FLAG: u16 = 1 << 15;
// Large enough that malloc serves each block with mmap, so freeing returns it to the kernel.
const BLOCK: usize = 1 << 15;

struct Block {
    latency: [MaybeUninit<u32>; BLOCK],
    id: [MaybeUninit<u16>; BLOCK],
}

#[derive(Default)]
struct Bucket {
    full: Vec<Box<MaybeUninit<Block>>>,
    last: usize,
}

impl Bucket {
    #[inline]
    fn push(&mut self, latency: u32, id: u16) {
        if self.last == BLOCK || self.full.is_empty() {
            self.full.push(Box::new_uninit());
            self.last = 0;
        }
        unsafe {
            let block = self.full.last_mut().unwrap_unchecked().as_mut_ptr();
            (*block).latency[self.last].write(latency);
            (*block).id[self.last].write(id);
        }
        self.last += 1;
    }

    fn drain(&self, mut visit: impl FnMut(&[u32], &[u16])) {
        for (i, block) in self.full.iter().enumerate() {
            let n = if i + 1 == self.full.len() { self.last } else { BLOCK };
            unsafe {
                let block = block.as_ptr();
                let latency = std::slice::from_raw_parts((*block).latency.as_ptr().cast(), n);
                let id = std::slice::from_raw_parts((*block).id.as_ptr().cast(), n);
                visit(latency, id);
            }
        }
    }
}

struct Interner {
    slots: Vec<u16>,
    entries: Vec<(u64, u32, u32)>,
    bytes: Vec<u8>,
}

impl Interner {
    fn new() -> Self {
        Self { slots: vec![0; 1 << 13], entries: Vec::new(), bytes: Vec::new() }
    }

    #[inline]
    fn intern(&mut self, key: &[u8]) -> u16 {
        let hash = hash(key);
        let mask = self.slots.len() - 1;
        let mut slot = hash as usize & mask;
        loop {
            let id = unsafe { *self.slots.get_unchecked(slot) } as usize;
            if id == 0 {
                return self.insert(slot, hash, key);
            }
            let (h, off, len) = unsafe { *self.entries.get_unchecked(id - 1) };
            if h == hash
                && unsafe { self.bytes.get_unchecked(off as usize..(off + len) as usize) } == key
            {
                return id as u16 - 1;
            }
            slot = (slot + 1) & mask;
        }
    }

    #[cold]
    fn insert(&mut self, slot: usize, hash: u64, key: &[u8]) -> u16 {
        self.entries.push((hash, self.bytes.len() as u32, key.len() as u32));
        self.bytes.extend_from_slice(key);
        self.slots[slot] = self.entries.len() as u16;
        assert!(self.entries.len() * 2 <= self.slots.len());
        self.entries.len() as u16 - 1
    }

    fn get(&self, id: usize) -> &[u8] {
        let (_, off, len) = self.entries[id];
        &self.bytes[off as usize..(off + len) as usize]
    }
}

#[inline]
fn load64(p: *const u8) -> u64 {
    unsafe { p.cast::<u64>().read_unaligned() }
}

// Mixes length with the first and last eight bytes; templates mostly differ at one of the ends.
#[inline]
fn hash(key: &[u8]) -> u64 {
    let len = key.len();
    let (head, tail) = if len >= 8 {
        (load64(key.as_ptr()), load64(unsafe { key.as_ptr().add(len - 8) }))
    } else {
        (load64(key.as_ptr()) & ((1u64 << (len * 8)) - 1), 0)
    };
    let mixed = (head ^ (len as u64) << 56).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ tail.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    mixed ^ mixed >> 29
}

// The spec fixes the span to Oct 30 18:00 .. Nov 2 18:00, so the day digits alone identify the date.
#[inline]
fn hour_index(line: *const u8) -> usize {
    let at = |i| unsafe { (*line.add(i) - b'0') as usize };
    let day = at(8) * 10 + at(9);
    let day_index = if day >= 30 { day - 30 } else { day + 1 };
    day_index * 24 + at(11) * 10 + at(12) - 18
}

#[inline]
fn find_space(mut p: *const u8) -> *const u8 {
    unsafe {
        let spaces = _mm256_set1_epi8(b' ' as i8);
        loop {
            let chunk = _mm256_loadu_si256(p.cast());
            let mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, spaces)) as u32;
            if mask != 0 {
                return p.add(mask.trailing_zeros() as usize);
            }
            p = p.add(32);
        }
    }
}

// Parses the latency and returns it with a pointer past the newline. Up to eight digits take
// the SWAR path, which has no per-digit branch: only '0'..'9' have bit 4 set, not '\n'.
#[inline]
fn parse_latency(p: *const u8) -> (u32, *const u8) {
    let word = load64(p);
    let stop = !word & 0x1010_1010_1010_1010;
    if stop == 0 {
        let (mut value, mut p) = (0u32, p);
        unsafe {
            while *p != b'\n' {
                value = value * 10 + (*p - b'0') as u32;
                p = p.add(1);
            }
            return (value, p.add(1));
        }
    }
    let len = stop.trailing_zeros() as usize / 8;
    let digits = if len == 0 { 0 } else { (word << ((8 - len) * 8)) & 0x0F0F_0F0F_0F0F_0F0F };
    let pairs = digits.wrapping_mul(2561) >> 8;
    let quads = (pairs & 0x00FF_00FF_00FF_00FF).wrapping_mul(6553601) >> 16;
    let value = (quads & 0x0000_FFFF_0000_FFFF).wrapping_mul(42949672960001) >> 32;
    (value as u32, unsafe { p.add(len + 1) })
}

fn parse_chunk(base: *const u8, first: usize, stop: usize, sink: &mut Worker) {
    let mut i = first;
    while i < stop {
        unsafe {
            let line = base.add(i);
            let bucket = sink.buckets.get_unchecked_mut(hour_index(line));
            // Methods are at most seven bytes, so the separating space is inside this word.
            let word = load64(line.add(TS_LEN + 1)) ^ 0x2020_2020_2020_2020;
            let zero = word.wrapping_sub(0x0101_0101_0101_0101) & !word & 0x8080_8080_8080_8080;
            let start = line.add(TS_LEN + 2 + zero.trailing_zeros() as usize / 8);
            let space = find_space(start);
            let len = space.offset_from(start) as usize;
            let id = sink.names.intern(std::slice::from_raw_parts(start, len));
            let flag = if *space.add(1) == b'5' { ERR_FLAG } else { 0 };
            let (latency, next) = parse_latency(space.add(5));
            bucket.push(latency, id | flag);
            i = next.offset_from(base) as usize;
        }
    }
}

struct Worker {
    buckets: Vec<Bucket>,
    names: Interner,
}

// A file ending on a page boundary would fault on the vector loads near its end, so the file is
// mapped over a slightly larger anonymous mapping that provides zeroed bytes after it.
fn map_input(file: &File, size: usize) -> *const u8 {
    const PROT_READ: i32 = 1;
    const MAP_PRIVATE: i32 = 2;
    const MAP_FIXED: i32 = 0x10;
    const MAP_ANONYMOUS: i32 = 0x20;
    unsafe {
        let base = mmap(std::ptr::null_mut(), size + 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if size > 0 {
            mmap(base, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, file.as_raw_fd(), 0);
        }
        base as *const u8
    }
}

fn parse_file(base: *const u8, size: usize, threads: usize) -> Vec<Worker> {
    let next = AtomicUsize::new(0);
    let input = Shared(base as *mut u8);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let (next, input) = (&next, &input);
                scope.spawn(move || {
                    let mut sink = Worker {
                        buckets: (0..HOURS).map(|_| Bucket::default()).collect(),
                        names: Interner::new(),
                    };
                    let bytes = unsafe { std::slice::from_raw_parts(input.0 as *const u8, size) };
                    loop {
                        let off = next.fetch_add(CHUNK, Relaxed);
                        if off >= size {
                            return sink;
                        }
                        let first = if off == 0 {
                            0
                        } else {
                            off + bytes[off - 1..].iter().position(|&b| b == b'\n').unwrap()
                        };
                        parse_chunk(input.0, first, (off + CHUNK).min(size), &mut sink);
                    }
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    })
}

#[inline]
fn push_int(out: &mut Vec<u8>, mut value: u64) {
    let mut digits = [0u8; 20];
    let mut i = digits.len();
    loop {
        i -= 1;
        digits[i] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    out.extend_from_slice(&digits[i..]);
}

fn hour_label(hour: usize) -> [u8; 21] {
    let total = hour + 18;
    let (month, day) = [(b'0', 30), (b'0', 31), (b'1', 1), (b'1', 2)][total / 24];
    let hh = total % 24;
    let mut label = *b"2026-1X-DDTHH:00:00Z ";
    label[6] = if month == b'0' { b'0' } else { b'1' };
    label[8] = b'0' + day / 10;
    label[9] = b'0' + day % 10;
    label[11] = b'0' + (hh / 10) as u8;
    label[12] = b'0' + (hh % 10) as u8;
    label
}

// Fills `flat` so each endpoint's latencies are contiguous, then answers it with selection.
fn render_hour(
    hour: usize,
    workers: &mut [&mut Bucket],
    maps: &[Vec<u32>],
    endpoints: &[&[u8]],
    state: &mut HourState,
) -> Vec<u8> {
    let HourState { counts, errors, flat } = state;
    counts.fill(0);
    errors.fill(0);
    let mut total = 0;
    for (bucket, map) in workers.iter().zip(maps) {
        bucket.drain(|_, ids| {
            total += ids.len();
            for &id in ids {
                let rank = map[(id & !ERR_FLAG) as usize] as usize;
                counts[rank] += 1;
                errors[rank] += (id >> 15) as u32;
            }
        });
    }
    let mut sum = 0;
    for count in counts.iter_mut() {
        let n = *count;
        *count = sum;
        sum += n;
    }
    flat.resize(total, 0);
    for (bucket, map) in workers.iter_mut().zip(maps) {
        bucket.drain(|latency, ids| {
            for (&value, &id) in latency.iter().zip(ids) {
                let rank = map[(id & !ERR_FLAG) as usize] as usize;
                flat[counts[rank] as usize] = value;
                counts[rank] += 1;
            }
        });
        **bucket = Bucket::default();
    }

    let label = hour_label(hour);
    let mut out = Vec::new();
    let mut begin = 0;
    for rank in 0..endpoints.len() {
        let end = counts[rank] as usize;
        if begin == end {
            continue;
        }
        let values = &mut flat[begin..end];
        let n = values.len();
        begin = end;
        let rank_of = |p: usize| (p * n + 99) / 100 - 1;
        let (r50, r95, r99) = (rank_of(50), rank_of(95), rank_of(99));
        let (left, &mut p99, _) = values.select_nth_unstable(r99);
        let (left, p95) = if r95 == r99 {
            (left, p99)
        } else {
            let (l, &mut v, _) = left.select_nth_unstable(r95);
            (l, v)
        };
        let p50 = if r50 == r95 { p95 } else { *left.select_nth_unstable(r50).1 };
        let errs = errors[rank] as u64;
        out.extend_from_slice(&label);
        out.extend_from_slice(endpoints[rank]);
        for value in [n as u64, p50 as u64, p95 as u64, p99 as u64, errs] {
            out.push(b' ');
            push_int(&mut out, value);
        }
        out.push(b' ');
        push_int(&mut out, errs * 1_000_000 / n as u64);
        out.push(b'\n');
    }
    out
}

struct HourState {
    counts: Vec<u32>,
    errors: Vec<u32>,
    flat: Vec<u32>,
}

struct Shared<T>(*mut T);
unsafe impl<T> Sync for Shared<T> {}

unsafe extern "C" {
    #[link_name = "_exit"]
    fn exit(code: i32) -> !;
    fn mallopt(param: i32, value: i32) -> i32;
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
}

const M_MMAP_THRESHOLD: i32 = -3;

fn main() {
    unsafe { mallopt(M_MMAP_THRESHOLD, 1 << 16) };
    let file = File::open("/data/requests.log").unwrap();
    let size = file.metadata().unwrap().len() as usize;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut parsed = parse_file(map_input(&file, size), size, threads);

    let name_tables: Vec<Interner> = parsed
        .iter_mut()
        .map(|w| std::mem::replace(&mut w.names, Interner { slots: vec![], entries: vec![], bytes: vec![] }))
        .collect();
    let mut ranks: HashMap<&[u8], u32> = HashMap::with_capacity(4096);
    for names in &name_tables {
        for id in 0..names.entries.len() {
            ranks.entry(names.get(id)).or_insert(0);
        }
    }
    let mut endpoints: Vec<&[u8]> = ranks.keys().copied().collect();
    endpoints.sort_unstable();
    for (rank, name) in endpoints.iter().enumerate() {
        ranks.insert(name, rank as u32);
    }
    let maps: Vec<Vec<u32>> = name_tables
        .iter()
        .map(|names| (0..names.entries.len()).map(|id| ranks[names.get(id)]).collect())
        .collect();

    let width = endpoints.len();
    let next_hour = AtomicUsize::new(0);
    let parsed_ptr = Shared(parsed.as_mut_ptr());
    let count = parsed.len();
    let mut outputs: Vec<(usize, Vec<u8>)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let (next_hour, parsed_ptr, maps, endpoints) =
                    (&next_hour, &parsed_ptr, &maps, &endpoints);
                scope.spawn(move || {
                    let mut state = HourState {
                        counts: vec![0; width],
                        errors: vec![0; width],
                        flat: Vec::new(),
                    };
                    let mut done = Vec::new();
                    loop {
                        let hour = next_hour.fetch_add(1, Relaxed);
                        if hour >= HOURS {
                            return done;
                        }
                        // Each hour index is claimed once, so this thread alone owns these buckets.
                        let mut buckets: Vec<&mut Bucket> = (0..count)
                            .map(|w| unsafe {
                                &mut *(*parsed_ptr.0.add(w)).buckets.as_mut_ptr().add(hour)
                            })
                            .collect();
                        done.push((hour, render_hour(hour, &mut buckets, maps, endpoints, &mut state)));
                    }
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().unwrap()).collect()
    });
    outputs.sort_unstable_by_key(|(hour, _)| *hour);

    let mut output = File::create("/out/summary.txt").unwrap();
    for (_, out) in &outputs {
        output.write_all(out).unwrap();
    }
    unsafe { exit(0) };
}
