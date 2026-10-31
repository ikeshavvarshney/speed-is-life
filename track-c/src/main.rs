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

#[inline(always)]
fn swar(word: u64) -> u64 {
    let digits = word & 0x0F0F_0F0F_0F0F_0F0F;
    let pairs = digits.wrapping_mul(2561) >> 8;
    let quads = (pairs & 0x00FF_00FF_00FF_00FF).wrapping_mul(6553601) >> 16;
    (quads & 0x0000_FFFF_0000_FFFF).wrapping_mul(42949672960001) >> 32
}

// Decodes up to eight digits that end just before `end`. Field lengths are known from the
// structural masks, so there is no scan and no branch on the number of digits.
#[inline(always)]
fn digits_before(end: *const u8, len: usize) -> u64 {
    swar(load64(unsafe { end.sub(8) }) & (u64::MAX << ((8 - len) * 8)))
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
        mum(h, 0x9E37_79B9_7F4A_7C15) | 1
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
        let slot = unsafe { self.slots.get_unchecked((hash >> self.shift) as usize) }.as_ptr();
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

// by_id holds 1 + the country's index, and 0 for anything unknown. Its last entry stays 0 and
// stands for every id above the table.
fn load_regions() -> Regions {
    let data = std::fs::read("/data/regions.csv").unwrap();
    let last = data[..data.len() - 1].iter().rposition(|&b| b == b'\n').unwrap() + 1;
    let max_id = data[last..].iter().take_while(|b| b.is_ascii_digit()).fold(0usize, |v, b| v * 10 + (b - b'0') as usize);
    let mut by_id = vec![0u8; max_id + 2];
    let mut seen = vec![0u8; 26 * 26 * 26];
    let mut codes = Vec::with_capacity(255);
    let mut i = data.iter().position(|&b| b == b'\n').unwrap() + 1;
    while i < data.len() {
        let mut id = 0usize;
        while data[i] != b',' {
            id = id * 10 + (data[i] - b'0') as usize;
            i += 1;
        }
        let at = i + 1;
        let code = [data[at], data[at + 1], data[at + 2]];
        let key = code.iter().fold(0, |k, &c| k * 26 + (c - b'A') as usize);
        if seen[key] == 0 {
            codes.push(code);
            seen[key] = codes.len() as u8;
        }
        by_id[id] = seen[key];
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

const BLOCKS: usize = 16;
// A record is at least 11 bytes, of which 5 are structural.
const MAX_MARKS: usize = BLOCKS * 64 + 64;
const MAX_RECORDS: usize = MAX_MARKS / 5 + 1;

// One 64-byte block as bitmasks: quotes, and commas or LFs outside quotes. A carry-less multiply
// by all ones is a running XOR of the quote bits, so it marks exactly the bytes inside quotes;
// a "" pair toggles twice and leaves the state alone.
#[inline(always)]
unsafe fn block_masks(p: *const u8, inside: &mut u64) -> (u64, u64) {
    unsafe {
        let finder = _mm256_setr_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0A, -1, 0x2C, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0A, -1, 0x2C, -1, -1, -1,
        );
        let quote = _mm256_set1_epi8(b'"' as i8);
        let (a, b) = (_mm256_loadu_si256(p.cast()), _mm256_loadu_si256(p.add(32).cast()));
        let mask = |x, y| _mm256_movemask_epi8(_mm256_cmpeq_epi8(x, y)) as u32 as u64;
        let quotes = mask(a, quote) | mask(b, quote) << 32;
        let seps = mask(_mm256_shuffle_epi8(finder, a), a) | mask(_mm256_shuffle_epi8(finder, b), b) << 32;
        let q = _mm_set_epi64x(0, quotes as i64);
        let running = _mm_cvtsi128_si64(_mm_clmulepi64_si128::<0>(q, _mm_set1_epi8(-1))) as u64 ^ *inside;
        *inside = ((running as i64) >> 63) as u64;
        (quotes, seps & !running)
    }
}

#[derive(Clone, Copy, Default)]
struct Rec {
    id: u32,
    region: u32,
    amount: i32,
    name_len: u32,
    note_len: u32,
    bit: u32,
    hash: u64,
}

// A quoted field holds q quotes: two delimiters and q/2 - 1 pairs, so its value is raw - q/2 - 1
// bytes. An unquoted field here is always empty, with q = 0 and raw = 0.
#[inline(always)]
fn value_len(raw: u32, quotes: u32) -> u32 {
    (raw - quotes / 2).saturating_sub(1)
}

fn parse_chunk(shared: &Shared, by_id: &[u8], first: usize, stop: usize, totals: &mut [Totals]) {
    let base = shared.input;
    let last_region = by_id.len() as u64 - 1;
    let mut marks = [0u32; MAX_MARKS];
    let mut counts = [0u32; MAX_MARKS];
    let mut recs = [Rec::default(); MAX_RECORDS];
    let (mut block, mut filled, mut inside, mut seen) = (first, 0usize, 0u64, 0u32);
    let mut start = first;
    loop {
        unsafe {
            for _ in 0..BLOCKS {
                let (quotes, mut seps) = block_masks(base.add(block), &mut inside);
                let found = seps.count_ones() as usize;
                let mut k = filled;
                // Blocks hold about six marks. Writing eight every time beats a branch on the count.
                loop {
                    for _ in 0..8 {
                        *marks.get_unchecked_mut(k) = block as u32 + seps.trailing_zeros();
                        *counts.get_unchecked_mut(k) = seen + (quotes & (seps ^ seps.wrapping_sub(1))).count_ones();
                        seps &= seps.wrapping_sub(1);
                        k += 1;
                    }
                    if seps == 0 {
                        break;
                    }
                }
                filled += found;
                seen += quotes.count_ones();
                block += 64;
            }
            let whole = filled / 5;
            let mut n = 0;
            let mut done = false;
            for r in 0..whole {
                if start >= stop {
                    done = true;
                    break;
                }
                let m = marks.as_ptr().add(r * 5);
                let c = counts.as_ptr().add(r * 5);
                let (c1, c2, c3, c4, lf) =
                    (*m as usize, *m.add(1) as usize, *m.add(2) as usize, *m.add(3) as usize, *m.add(4) as usize);
                let id_at = start + 3 * (*base.add(start) == BOM_FIRST) as usize;
                let id = digits_before(base.add(c1), c1 - id_at);
                let name_raw = (c2 - c1 - 1) as u32;
                let note_raw = (c3 - c2 - 1) as u32;
                let name_len = value_len(name_raw, *c.add(1) - *c);
                let note_len = value_len(note_raw, *c.add(2) - *c.add(1));
                let region = digits_before(base.add(c4), c4 - c3 - 1).min(last_region);
                let end = lf - (*base.add(lf - 1) == b'\r') as usize;
                let negative = *base.add(c4 + 1) == b'-';
                let text = load64(base.add(end - 8)) & (u64::MAX << ((8 - (end - c4 - 1 - negative as usize)) * 8));
                let cents = swar(text << 24) * 100 + ((text >> 48) & 15) * 10 + ((text >> 56) & 15);
                let amount = if negative { -(cents as i32) } else { cents as i32 };
                let hash = hash_name(base.add(c1 + 2), name_raw.saturating_sub(2) as usize);
                _mm_prefetch::<_MM_HINT_T0>(by_id.as_ptr().add(region as usize).cast());
                shared.names.prefetch(hash);
                *recs.get_unchecked_mut(n) =
                    Rec { id: id as u32, region: region as u32, amount, name_len, note_len, bit: 0, hash };
                n += 1;
                start = lf + 1;
            }
            // Group prefetching: each pass touches only what the pass before it prefetched, so the
            // cache misses of a whole batch overlap instead of stalling one at a time.
            for rec in recs[..n].iter_mut() {
                let country = *by_id.get_unchecked(rec.region as usize) as usize;
                let total = totals.get_unchecked_mut(country);
                let amount = rec.amount as i64;
                total.count += 1;
                total.id_sum += rec.id as u64;
                total.amount_sum += amount;
                total.amount_min = total.amount_min.min(amount);
                total.amount_max = total.amount_max.max(amount);
                total.name_bytes += rec.name_len as u64;
                total.note_bytes += rec.note_len as u64;
                let bit = shared.names.intern(rec.hash);
                let word = country * shared.words + bit / 64;
                _mm_prefetch::<7>(shared.bits.as_ptr().add(word).cast());
                rec.region = word as u32;
                rec.bit = bit as u32 % 64;
            }
            for rec in recs[..n].iter() {
                let word = shared.bits.get_unchecked(rec.region as usize);
                let mask = 1u64 << rec.bit;
                if word.load(Relaxed) & mask == 0 {
                    word.fetch_or(mask, Relaxed);
                }
            }
            if done || start >= stop {
                return;
            }
            let left = filled - whole * 5;
            for i in 0..left {
                marks[i] = marks[whole * 5 + i];
                counts[i] = counts[whole * 5 + i];
            }
            filled = left;
        }
    }
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
    let words = (1usize << bits) / 64;
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
                        if first < stop {
                            parse_chunk(&shared, &regions.by_id, first, stop, &mut totals);
                        }
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
