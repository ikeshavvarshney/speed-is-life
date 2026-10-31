use std::arch::x86_64::*;
use std::ffi::c_void;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

const CHUNK: usize = 1 << 18;
const BOM_FIRST: u8 = 0xEF;
// Enough for the generator's pool of 100,000 names; a bigger pool restarts with a bigger table.
const START_BITS: u32 = 17;

unsafe extern "C" {
    #[link_name = "_exit"]
    fn exit(code: i32) -> !;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
}

// Mapped over a larger anonymous mapping so vector loads near the end of the file stay readable.
fn map_input(file: &File, size: usize) -> *const u8 {
    const PROT_READ: i32 = 1;
    const MAP_PRIVATE: i32 = 2;
    const MAP_FIXED: i32 = 0x10;
    const MAP_ANONYMOUS: i32 = 0x20;
    unsafe {
        let base = mmap(std::ptr::null_mut(), size + 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        mmap(base, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, file.as_raw_fd(), 0);
        base as *const u8
    }
}

#[inline(always)]
fn load64(p: *const u8) -> u64 {
    unsafe { p.cast::<u64>().read_unaligned() }
}

// Decodes digits up to the next delimiter. Only digits have bit 4 set among the bytes that can
// follow a number, so up to seven digits take a branch-free SWAR path.
#[inline(always)]
fn parse_digits(p: *const u8) -> (u64, *const u8) {
    let word = load64(p);
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

// Returns the closing quote of a quoted field and how many "" pairs it held. Each pair is two
// raw bytes but one value byte, so lengths need no unescaping.
#[inline(always)]
fn scan_quoted(mut p: *const u8) -> (*const u8, usize) {
    let mut escapes = 0;
    unsafe {
        let quotes = _mm256_set1_epi8(b'"' as i8);
        loop {
            let chunk = _mm256_loadu_si256(p.cast());
            let mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, quotes)) as u32;
            if mask == 0 {
                p = p.add(32);
                continue;
            }
            let quote = p.add(mask.trailing_zeros() as usize);
            if *quote.add(1) != b'"' {
                return (quote, escapes);
            }
            escapes += 1;
            p = quote.add(2);
        }
    }
}

// Reads a text field at `p`. Returns its raw start, raw length, value length and the delimiter.
#[inline(always)]
fn read_text(p: *const u8) -> (*const u8, usize, usize, *const u8) {
    unsafe {
        if *p == b'"' {
            let start = p.add(1);
            let (end, escapes) = scan_quoted(start);
            let raw = end.offset_from(start) as usize;
            (start, raw, raw - escapes, end.add(1))
        } else {
            let mut end = p;
            while *end != b',' {
                end = end.add(1);
            }
            let raw = end.offset_from(p) as usize;
            (p, raw, raw, end)
        }
    }
}

#[inline(always)]
fn mum(a: u64, b: u64) -> u64 {
    let product = a as u128 * b as u128;
    product as u64 ^ (product >> 64) as u64
}

const LOW_BYTES: [u8; 64] = {
    let mut table = [0u8; 64];
    let mut i = 0;
    while i < 32 {
        table[i] = 0xFF;
        i += 1;
    }
    table
};

// Names vary in length, so a loop per word mispredicts constantly. Up to 32 bytes are masked and
// mixed at a fixed cost. Longer names, which are rare, loop over the rest.
#[inline(always)]
fn hash_name(p: *const u8, len: usize) -> u64 {
    unsafe {
        let mask = _mm256_loadu_si256(LOW_BYTES.as_ptr().add(32 - len.min(32)).cast());
        let bytes = _mm256_and_si256(_mm256_loadu_si256(p.cast()), mask);
        let word0 = _mm256_extract_epi64::<0>(bytes) as u64;
        let word1 = _mm256_extract_epi64::<1>(bytes) as u64;
        let word2 = _mm256_extract_epi64::<2>(bytes) as u64;
        let word3 = _mm256_extract_epi64::<3>(bytes) as u64;
        let mut h = mum(word0 ^ 0x2D35_8DCC_AA6C_78A5, word1 ^ 0x8BB8_4B93_962E_ACC9)
            ^ mum(word2 ^ 0x4B33_A62E_D433_D4A3, word3 ^ len as u64 ^ 0xC2B2_AE3D_27D4_EB4F);
        let mut i = 32;
        while i < len {
            let left = len - i;
            let word = load64(p.add(i)) & if left >= 8 { u64::MAX } else { (1u64 << (left * 8)) - 1 };
            h = mum(h ^ word, 0x8BB8_4B93_962E_ACC9);
            i += 8;
        }
        mum(h, 0x9E37_79B9_7F4A_7C15).max(1)
    }
}

// Lock-free table of 64-bit name hashes, whose slot number is the name's id. Comparing hashes
// instead of bytes never touches the input again, which is what keeps this fast. With 100,000
// names a collision has a chance of about 3e-10, and would only make one name count as another.
struct Names {
    slots: &'static [AtomicU64],
    shift: u32,
    used: AtomicUsize,
}

impl Names {
    fn new(bits: u32) -> Self {
        let slots = zeroed_atomics(1 << bits);
        Self { slots, shift: 64 - bits, used: AtomicUsize::new(0) }
    }

    fn overfull(&self) -> bool {
        self.used.load(Relaxed) > self.slots.len() / 8 * 7
    }

    #[inline(always)]
    fn prefetch(&self, hash: u64) {
        let slot = self.slots[(hash >> self.shift) as usize].as_ptr();
        unsafe { _mm_prefetch::<_MM_HINT_T0>(slot as *const i8) };
    }

    #[inline(always)]
    fn intern(&self, hash: u64) -> usize {
        let mask = self.slots.len() - 1;
        let mut slot = (hash >> self.shift) as usize;
        loop {
            let entry = unsafe { self.slots.get_unchecked(slot) };
            let mut seen = entry.load(Relaxed);
            if seen == 0 {
                match entry.compare_exchange(0, hash, Relaxed, Relaxed) {
                    Ok(_) => {
                        self.used.fetch_add(1, Relaxed);
                        return slot;
                    }
                    Err(actual) => seen = actual,
                }
            }
            if seen == hash {
                return slot;
            }
            slot = (slot + 1) & mask;
        }
    }
}

// Zeroed anonymous memory on 2 MB boundaries, with huge pages requested: random access over
// several megabytes would otherwise pay a TLB miss on most records. Lives until process exit.
fn zeroed_atomics(len: usize) -> &'static [AtomicU64] {
    const HUGE: usize = 1 << 21;
    const MADV_HUGEPAGE: i32 = 14;
    let bytes = (len * 8).next_multiple_of(HUGE);
    unsafe {
        let raw = mmap(std::ptr::null_mut(), bytes + HUGE, 3, 0x22, -1, 0) as usize;
        let aligned = raw.next_multiple_of(HUGE) as *mut c_void;
        madvise(aligned, bytes, MADV_HUGEPAGE);
        std::slice::from_raw_parts(aligned as *const AtomicU64, len)
    }
}

#[derive(Clone, Copy)]
struct Totals {
    count: u64,
    id_sum: u64,
    amount_sum: i64,
    amount_min: i64,
    amount_max: i64,
    name_bytes: u64,
    note_bytes: u64,
}

const EMPTY: Totals = Totals {
    count: 0,
    id_sum: 0,
    amount_sum: 0,
    amount_min: i64::MAX,
    amount_max: i64::MIN,
    name_bytes: 0,
    note_bytes: 0,
};

struct Regions {
    by_id: Vec<u8>,
    codes: Vec<[u8; 3]>,
}

// by_id holds 1 + the country's index, and 0 for anything unknown. Ids are listed ascending, so
// the last line sizes the table. Countries fit u8: the spec has 240 of them.
fn load_regions() -> Regions {
    let data = std::fs::read("/data/regions.csv").unwrap();
    let last = data[..data.len() - 1].iter().rposition(|&b| b == b'\n').unwrap() + 1;
    let (max_id, _) = parse_digits(data[last..].as_ptr());
    let mut by_id = vec![0u8; max_id as usize + 1];
    let mut seen = vec![0u8; 26 * 26 * 26];
    let mut codes = Vec::with_capacity(255);
    let mut i = data.iter().position(|&b| b == b'\n').unwrap() + 1;
    while i < data.len() {
        let (id, comma) = parse_digits(unsafe { data.as_ptr().add(i) });
        let at = unsafe { comma.offset_from(data.as_ptr()) } as usize + 1;
        let code = [data[at], data[at + 1], data[at + 2]];
        let key = code.iter().fold(0, |k, &c| k * 26 + (c - b'A') as usize);
        if seen[key] == 0 {
            codes.push(code);
            seen[key] = codes.len() as u8;
        }
        by_id[id as usize] = seen[key];
        i = at + 4;
    }
    Regions { by_id, codes }
}

struct Shared {
    input: *const u8,
    names: Names,
    bits: &'static [AtomicU64],
    words: usize,
}
unsafe impl Sync for Shared {}

const PIPE: usize = 16;
const NO_COUNTRY: u32 = u32::MAX;

// Name lookups and bitset writes are random accesses that miss cache on nearly every record. Each
// is prefetched here and performed PIPE records later, so the misses overlap instead of stalling.
struct Pipe {
    names: [(u64, u32); PIPE],
    bits: [(u32, u64); PIPE],
    head: usize,
}

impl Pipe {
    fn new() -> Self {
        Self { names: [(0, NO_COUNTRY); PIPE], bits: [(0, 0); PIPE], head: 0 }
    }

    #[inline(always)]
    fn step(&mut self, shared: &Shared, hash: u64, country: u32) {
        let head = self.head;
        self.head = (head + 1) % PIPE;
        let (old_hash, old_country) = self.names[head];
        self.names[head] = (hash, country);
        if country != NO_COUNTRY && hash != 0 {
            shared.names.prefetch(hash);
        }
        if old_country == NO_COUNTRY {
            return;
        }
        // One bitset per country over name ids. Bit 0 is the empty name, however it was quoted.
        let bit = if old_hash == 0 { 0 } else { shared.names.intern(old_hash) + 1 };
        let word = old_country as usize * shared.words + bit / 64;
        unsafe {
            _mm_prefetch::<_MM_HINT_T0>(shared.bits.as_ptr().add(word).cast());
        }
        let (old_word, old_mask) = self.bits[head];
        self.bits[head] = (word as u32, 1u64 << (bit % 64));
        Self::set(shared, old_word, old_mask);
    }

    #[inline(always)]
    fn set(shared: &Shared, word: u32, mask: u64) {
        if mask != 0 {
            let word = unsafe { shared.bits.get_unchecked(word as usize) };
            if word.load(Relaxed) & mask == 0 {
                word.fetch_or(mask, Relaxed);
            }
        }
    }

    fn drain(&mut self, shared: &Shared) {
        for _ in 0..PIPE {
            self.step(shared, 0, NO_COUNTRY);
        }
        for i in 0..PIPE {
            let (word, mask) = self.bits[i];
            Self::set(shared, word, mask);
            self.bits[i] = (0, 0);
        }
    }
}

fn parse_chunk(shared: &Shared, by_id: &[u8], first: usize, stop: usize, totals: &mut [Totals]) {
    let base = shared.input;
    let mut pipe = Pipe::new();
    let mut i = first;
    while i < stop {
        unsafe {
            let mut p = base.add(i);
            if *p == BOM_FIRST {
                p = p.add(3);
            }
            let (id, comma) = parse_digits(p);
            let (name, name_raw, name_len, comma) = read_text(comma.add(1));
            let (_, _, note_len, comma) = read_text(comma.add(1));
            let (region, comma) = parse_digits(comma.add(1));
            let negative = *comma.add(1) == b'-';
            let (whole, dot) = parse_digits(comma.add(1 + negative as usize));
            let cents = (whole * 100
                + (*dot.add(1) - b'0') as u64 * 10
                + (*dot.add(2) - b'0') as u64) as i64;
            let amount = if negative { -cents } else { cents };
            let end = dot.add(3);
            i = end.offset_from(base) as usize + 1 + (*end == b'\r') as usize;

            let country = by_id.get(region as usize).copied().unwrap_or(0) as usize;
            let total = totals.get_unchecked_mut(country);
            total.count += 1;
            total.id_sum += id;
            total.amount_sum += amount;
            total.amount_min = total.amount_min.min(amount);
            total.amount_max = total.amount_max.max(amount);
            total.name_bytes += name_len as u64;
            total.note_bytes += note_len as u64;

            let hash = if name_raw == 0 { 0 } else { hash_name(name, name_raw) };
            pipe.step(shared, hash, country as u32);
        }
    }
    pipe.drain(shared);
}

// Past a chunk's start, the next record starts after an LF followed by a digit or BOM. Newlines
// inside notes are always followed by a lowercase word or a quote (spec section 2.2).
fn first_record(input: &[u8], off: usize) -> usize {
    let mut i = off - 1;
    while i + 1 < input.len() {
        if input[i] == b'\n' && (input[i + 1].is_ascii_digit() || input[i + 1] == BOM_FIRST) {
            return i + 1;
        }
        i += 1;
    }
    input.len()
}

fn run(input: &[u8], regions: &Regions, bits: u32, threads: usize) -> Option<(Vec<Totals>, Shared)> {
    let countries = regions.codes.len() + 1;
    let words = ((1usize << bits) + 64) / 64;
    let shared = Shared {
        input: input.as_ptr(),
        names: Names::new(bits),
        bits: zeroed_atomics(words * countries),
        words,
    };
    let next = AtomicUsize::new(0);
    let header_end = input.iter().position(|&b| b == b'\n').unwrap() + 1;
    let merged = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut totals = vec![EMPTY; countries];
                    loop {
                        let off = next.fetch_add(CHUNK, Relaxed);
                        if off >= input.len() || shared.names.overfull() {
                            return totals;
                        }
                        let first = if off == 0 { header_end } else { first_record(input, off) };
                        let stop = (off + CHUNK).min(input.len());
                        parse_chunk(&shared, &regions.by_id, first, stop, &mut totals);
                    }
                })
            })
            .collect();
        let mut merged = vec![EMPTY; countries];
        for worker in workers {
            for (into, from) in merged.iter_mut().zip(worker.join().unwrap()) {
                into.count += from.count;
                into.id_sum += from.id_sum;
                into.amount_sum += from.amount_sum;
                into.amount_min = into.amount_min.min(from.amount_min);
                into.amount_max = into.amount_max.max(from.amount_max);
                into.name_bytes += from.name_bytes;
                into.note_bytes += from.note_bytes;
            }
        }
        merged
    });
    (!shared.names.overfull()).then_some((merged, shared))
}

fn money(cents: i64) -> String {
    let sign = if cents < 0 { "-" } else { "" };
    let abs = cents.unsigned_abs();
    format!("{sign}{}.{:02}", abs / 100, abs % 100)
}

fn main() {
    let regions = load_regions();
    let file = File::open("/data/records.csv").unwrap();
    let size = file.metadata().unwrap().len() as usize;
    let input = unsafe { std::slice::from_raw_parts(map_input(&file, size), size) };
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut bits = START_BITS;
    let (totals, shared) = loop {
        match run(input, &regions, bits, threads) {
            Some(done) => break done,
            None => bits += 2,
        }
    };

    let mut keys: Vec<(&[u8], usize)> = vec![(b"_UNMATCHED".as_slice(), 0)];
    keys.extend(regions.codes.iter().enumerate().map(|(i, code)| (code.as_slice(), i + 1)));
    keys.sort_unstable();
    let mut out = Vec::with_capacity(keys.len() * 96);
    for (key, country) in keys {
        let total = &totals[country];
        if total.count == 0 {
            continue;
        }
        let bitset = &shared.bits[country * shared.words..(country + 1) * shared.words];
        let distinct: u32 = bitset.iter().map(|word| word.load(Relaxed).count_ones()).sum();
        out.extend_from_slice(key);
        let line = format!(
            " {} {} {} {} {} {} {} {}\n",
            total.count,
            total.id_sum,
            money(total.amount_sum),
            money(total.amount_min),
            money(total.amount_max),
            distinct,
            total.name_bytes,
            total.note_bytes
        );
        out.extend_from_slice(line.as_bytes());
    }
    std::fs::write("/out/aggregates.txt", &out).unwrap();
    unsafe { exit(0) };
}
