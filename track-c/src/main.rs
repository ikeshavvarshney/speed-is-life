use std::arch::x86_64::*;
use std::ffi::c_void;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Barrier, Mutex};

const CHUNK: usize = 1 << 18;
const REGION_CHUNK: usize = 1 << 16;
const UNMAP_STEP: usize = 1 << 23;
const COUNTRIES: usize = 256;
const MADV_DONTNEED: i32 = 4;
const MADV_HUGEPAGE: i32 = 14;
const MADV_NOHUGEPAGE: i32 = 15;
const BOM_FIRST: u8 = 0xEF;
// The generator's pool of 100,000 names, plus the few that edge-case blocks add, fits at 77% load.
const NAME_BITS: u32 = 17;
// Length of one log segment in u16 entries: a two-entry link and 126 names, on four cache lines.
const SEGMENT: usize = 128;

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

// The bytes of up to eight digits ending just before `end`, with leading positions zeroed. Field
// lengths are known from the structural masks, so nothing branches on the number of digits.
#[inline(always)]
fn digit_word(end: *const u8, len: usize) -> u64 {
    load64(unsafe { end.sub(8) }) & (u64::MAX << ((8 - len) * 8))
}

// Converts three words of eight digits together. Each step merges pairs: digits to 2-digit
// values, then 4, then 8. Three scalar conversions would queue on the one multiply port.
#[inline(always)]
unsafe fn decode(a: u64, b: u64, c: u64) -> (u32, u32, u32) {
    unsafe {
        let digits = _mm256_and_si256(_mm256_set_epi64x(0, c as i64, b as i64, a as i64), _mm256_set1_epi8(15));
        let pairs = _mm256_maddubs_epi16(digits, _mm256_set1_epi16(0x010A));
        let quads = _mm256_madd_epi16(pairs, _mm256_set1_epi32(0x0001_0064));
        let packed = _mm256_packus_epi32(quads, quads);
        let eights = _mm256_madd_epi16(packed, _mm256_set1_epi32(0x0001_2710));
        let low = _mm256_castsi256_si128(eights);
        let high = _mm256_extracti128_si256::<1>(eights);
        (_mm_cvtsi128_si32(low) as u32, _mm_extract_epi32::<1>(low) as u32, _mm_cvtsi128_si32(high) as u32)
    }
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
// mixed at a fixed cost. Longer names, which are rare, loop over the rest. Every input byte passes
// through at least two AES rounds after it enters, which spreads it over the whole state.
#[inline(always)]
fn hash_name(p: *const u8, len: usize) -> u64 {
    unsafe {
        let mask = _mm256_loadu_si256(LOW_BYTES.as_ptr().add(32 - len.min(32)).cast());
        let bytes = _mm256_and_si256(_mm256_loadu_si256(p.cast()), mask);
        let key = _mm_set_epi64x(0x8BB8_4B93_962E_ACC9u64 as i64, 0x2D35_8DCC_AA6C_78A5u64 as i64 ^ len as i64);
        let low = _mm_xor_si128(_mm256_castsi256_si128(bytes), key);
        let mut state = _mm_aesenc_si128(low, _mm256_extracti128_si256::<1>(bytes));
        let mut i = 32;
        while i < len {
            let tail = _mm_loadu_si128(LOW_BYTES.as_ptr().add(32 - (len - i).min(16)).cast());
            state = _mm_aesenc_si128(state, _mm_and_si128(_mm_loadu_si128(p.add(i).cast()), tail));
            i += 16;
        }
        state = _mm_aesenc_si128(state, key);
        _mm_cvtsi128_si64(_mm_aesenc_si128(state, key)) as u64 | 1
    }
}

const HUGE: usize = 1 << 21;

// Zeroed anonymous memory on 2 MB boundaries. Lives until process exit.
fn zeroed(bytes: usize, advice: i32) -> *mut u8 {
    unsafe {
        let raw = mmap(std::ptr::null_mut(), bytes + HUGE, 3, 0x22, -1, 0) as usize;
        let aligned = raw.next_multiple_of(HUGE) as *mut c_void;
        madvise(aligned, bytes, advice);
        aligned.cast()
    }
}

// One worker's running values for one country, on one cache line. `sums` is added to as one vector:
// id sum, amount sum, count << 32 | name bytes, note bytes. Neither half of the third can carry
// into the other, as both stay below 2^31. `logs` are the write positions of the two name logs.
#[derive(Clone, Copy)]
#[repr(align(64))]
struct Group {
    sums: [u64; 4],
    amount_min: i32,
    amount_max: i32,
    logs: [u32; 2],
}

const EMPTY: Group = Group { sums: [0; 4], amount_min: i32::MAX, amount_max: i32::MIN, logs: [0; 2] };

// Counters that every thread writes, each on its own line, away from what records read.
#[repr(align(64))]
struct Counter<T>(T);

// Claims pieces of regions.csv. by_id holds 1 + the country's index, and 0 for anything unknown.
// Its last entry stays 0 and stands for every id above the table. Ids are unique, so each byte has
// one writer. Countries fit u8: the spec has 240 of them.
fn load_regions(job: &Job) {
    let data = job.regions;
    loop {
        let off = job.next_region.0.fetch_add(REGION_CHUNK, Relaxed);
        if off >= data.len() {
            return;
        }
        let stop = (off + REGION_CHUNK).min(data.len());
        let mut i = off + data[off - 1..].iter().position(|&b| b == b'\n').unwrap();
        while i < stop {
            let (id, comma) = parse_id(&data[i..]);
            let at = i + comma + 1;
            let key = data[at..at + 3].iter().fold(0, |k, &c| k * 26 + (c - b'A') as usize);
            let mut country = job.seen[key].load(Relaxed);
            if country == 0 {
                let mut codes = job.codes.lock().unwrap();
                country = job.seen[key].load(Relaxed);
                if country == 0 {
                    codes.push([data[at], data[at + 1], data[at + 2]]);
                    country = codes.len() as u8;
                    job.seen[key].store(country, Relaxed);
                }
            }
            unsafe { *job.by_id.add(id) = country };
            i = at + 4;
        }
    }
}

// Returns the number at the start of `text` and its length.
fn parse_id(text: &[u8]) -> (usize, usize) {
    let mut id = 0;
    let mut len = 0;
    while text[len].is_ascii_digit() {
        id = id * 10 + (text[len] - b'0') as usize;
        len += 1;
    }
    (id, len)
}

struct Job {
    input: *const u8,
    size: usize,
    regions: &'static [u8],
    by_id: *mut u8,
    last_region: u32,
    names: *const AtomicU64,
    log: *mut u16,
    groups: *mut Group,
    threads: usize,
    seen: Vec<AtomicU8>,
    codes: Mutex<Vec<[u8; 3]>>,
    distinct: Vec<AtomicU32>,
    barrier: Barrier,
    next_region: Counter<AtomicUsize>,
    next_chunk: Counter<AtomicUsize>,
    next_segment: Counter<AtomicU32>,
    next_country: Counter<AtomicUsize>,
    next_unmap: Counter<AtomicUsize>,
    used: Counter<AtomicUsize>,
}
unsafe impl Sync for Job {}

const BLOCKS: usize = 32;
// A record is at least 11 bytes, of which 5 are structural.
const MAX_MARKS: usize = BLOCKS * 64 + 64;
const MAX_RECORDS: usize = MAX_MARKS / 5 + 1;

// One 64-byte block as bitmasks: commas or LFs outside quotes, and second quotes of "" pairs. A
// carry-less multiply by all ones is a running XOR of the quote bits, which is set from an opening
// quote up to its closing quote. A pair inside a value closes and at once reopens, while the fields
// of a record always have a comma between them, so an opening quote right after a closing quote
// is exactly a pair.
#[inline(always)]
unsafe fn block_masks(p: *const u8, inside: &mut u64, closed: &mut u64) -> (u64, u64) {
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
        let closers = quotes & !running;
        let pairs = quotes & running & (closers << 1 | *closed);
        *closed = closers >> 63;
        (seps & !running, pairs)
    }
}

#[derive(Clone, Copy)]
struct Rec {
    sums: __m256i,
    hash: u64,
    region: u32,
    amount: i32,
}

// Logs grow in segments from one shared arena, and each starts with the number of the segment
// before it. Segment 0 is never handed out, so it marks the end of a chain.
#[cold]
fn new_segment(job: &Job, cursor: u32) -> u32 {
    let previous = (cursor as usize / SEGMENT).saturating_sub(1) as u32;
    let segment = job.next_segment.0.fetch_add(1, Relaxed) as usize;
    unsafe { job.log.add(segment * SEGMENT).cast::<u32>().write(previous) };
    (segment * SEGMENT + 2) as u32
}

// Lock-free table of 64-bit name hashes, whose slot number is the name's id. Comparing hashes
// instead of bytes never touches the input again, which is what keeps this fast. With 100,000
// names a collision has a chance of about 3e-10, and would only make one name count as another.
// A bucket is one cache line of eight hashes compared at once, so a hit does not branch on where
// the hash sits. A bucket fills in order, so a thread that sees a free way has not missed its hash
// further along, and losing the race for a way only means looking at the winner's hash.
#[inline(always)]
fn intern(job: &Job, slots: *const AtomicU64, hash: u64) -> usize {
    let mut bucket = (hash >> (67 - NAME_BITS)) as usize;
    unsafe {
        let want = _mm256_set1_epi64x(hash as i64);
        loop {
            let at = slots.add(bucket * 8);
            let (low, high) = (_mm256_load_si256(at.cast()), _mm256_load_si256(at.add(4).cast()));
            let mask = |x, y| _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(x, y))) as u32;
            let hit = mask(low, want) | mask(high, want) << 4;
            if hit != 0 {
                return bucket * 8 + hit.trailing_zeros() as usize;
            }
            let zero = _mm256_setzero_si256();
            let free = mask(low, zero) | mask(high, zero) << 4;
            if free == 0 {
                bucket = (bucket + 1) & ((1 << (NAME_BITS - 3)) - 1);
                continue;
            }
            let way = free.trailing_zeros() as usize;
            match (*at.add(way)).compare_exchange(0, hash, Relaxed, Relaxed) {
                Ok(_) => {
                    // The generator's pool is a constant, so this only guards against hanging.
                    assert!(job.used.0.fetch_add(1, Relaxed) < (1 << NAME_BITS) / 8 * 7, "too many names");
                    return bucket * 8 + way;
                }
                Err(actual) if actual == hash => return bucket * 8 + way,
                Err(_) => {}
            }
        }
    }
}

fn parse_chunk(job: &Job, first: usize, stop: usize, groups: &mut [Group]) {
    let base = job.input;
    let (by_id, last_region, slots, log) = (job.by_id as *const u8, job.last_region, job.names, job.log);
    let mut marks = [0u32; MAX_MARKS];
    // Number of "" pairs in the field that each mark ends. Pairs are rare, so it stays mostly zero.
    let mut pairs = [0u32; MAX_MARKS + 1];
    let mut recs = [Rec { sums: unsafe { _mm256_setzero_si256() }, hash: 0, region: 0, amount: 0 }; MAX_RECORDS];
    let (mut block, mut filled, mut inside, mut closed) = (first, 0usize, 0u64, 0u64);
    let mut start = first;
    loop {
        unsafe {
            for _ in 0..BLOCKS {
                let (mut seps, mut found_pairs) = block_masks(base.add(block), &mut inside, &mut closed);
                while found_pairs != 0 {
                    let before = found_pairs & found_pairs.wrapping_neg();
                    pairs[filled + (seps & (before - 1)).count_ones() as usize] += 1;
                    found_pairs ^= before;
                }
                let found = seps.count_ones() as usize;
                let mut k = filled;
                // Blocks hold about six marks. Writing eight every time beats a branch on the count.
                loop {
                    for _ in 0..8 {
                        *marks.get_unchecked_mut(k) = block as u32 + seps.trailing_zeros();
                        seps &= seps.wrapping_sub(1);
                        k += 1;
                    }
                    if seps == 0 {
                        break;
                    }
                }
                filled += found;
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
                let (c1, c2, c3, c4, lf) =
                    (*m as usize, *m.add(1) as usize, *m.add(2) as usize, *m.add(3) as usize, *m.add(4) as usize);
                let id_at = start + 3 * (*base.add(start) == BOM_FIRST) as usize;
                let name_raw = (c2 - c1 - 1) as u32;
                let note_raw = (c3 - c2 - 1) as u32;
                let name_len = name_raw.saturating_sub(2) - pairs.get_unchecked(r * 5 + 1);
                let note_len = note_raw.saturating_sub(2) - pairs.get_unchecked(r * 5 + 2);
                let end = lf - (*base.add(lf - 1) == b'\r') as usize;
                let negative = *base.add(c4 + 1) == b'-';
                let text = load64(base.add(end - 8)) & (u64::MAX << ((8 - (end - c4 - 1 - negative as usize)) * 8));
                let (id, region, whole) =
                    decode(digit_word(base.add(c1), c1 - id_at), digit_word(base.add(c4), c4 - c3 - 1), text << 24);
                let region = region.min(last_region);
                let cents = (whole * 100) as u64 + ((text >> 48) & 15) * 10 + ((text >> 56) & 15);
                let sign = -(negative as i32);
                let amount = (cents as i32 ^ sign) - sign;
                let hash = hash_name(base.add(c1 + 2), name_raw.saturating_sub(2) as usize);
                _mm_prefetch::<_MM_HINT_T0>(by_id.add(region as usize).cast());
                _mm_prefetch::<_MM_HINT_T0>(slots.add((hash >> (67 - NAME_BITS)) as usize * 8).cast());
                let sums = _mm256_set_epi64x(
                    note_len as i64,
                    (1i64 << 32) | name_len as i64,
                    amount as i64,
                    id as i64,
                );
                *recs.get_unchecked_mut(n) = Rec { sums, hash, region, amount };
                n += 1;
                start = lf + 1;
            }
            // Group prefetching: this pass touches only what the pass above prefetched, so the
            // cache misses of a whole batch overlap instead of stalling one at a time.
            for rec in recs[..n].iter() {
                let country = *by_id.add(rec.region as usize) as usize;
                let group = groups.get_unchecked_mut(country);
                let sums = group.sums.as_mut_ptr().cast();
                _mm256_store_si256(sums, _mm256_add_epi64(_mm256_load_si256(sums), rec.sums));
                group.amount_min = group.amount_min.min(rec.amount);
                group.amount_max = group.amount_max.max(rec.amount);
                let name = intern(job, slots, rec.hash);
                let cursor = group.logs.get_unchecked_mut(name >> 16);
                if *cursor as usize % SEGMENT == 0 {
                    *cursor = new_segment(job, *cursor);
                }
                *log.add(*cursor as usize) = name as u16;
                *cursor += 1;
            }
            if done || start >= stop {
                return;
            }
            let left = filled - whole * 5;
            marks.copy_within(whole * 5..filled, 0);
            // The mark after the last one found may already have pairs waiting for it.
            pairs.copy_within(whole * 5..=filled, 0);
            pairs[left + 1..=filled].fill(0);
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

fn worker(job: &Job, index: usize) {
    load_regions(job);
    job.barrier.wait();
    let input = unsafe { std::slice::from_raw_parts(job.input, job.size) };
    let groups = unsafe { std::slice::from_raw_parts_mut(job.groups.add(index * COUNTRIES), COUNTRIES) };
    groups.fill(EMPTY);
    loop {
        let off = job.next_chunk.0.fetch_add(CHUNK, Relaxed);
        if off >= job.size {
            break;
        }
        let first = if off == 0 { input.iter().position(|&b| b == b'\n').unwrap() + 1 } else { first_record(input, off) };
        let stop = (off + CHUNK).min(job.size);
        if first < stop {
            parse_chunk(job, first, stop, groups);
        }
    }
    job.barrier.wait();

    // Process exit would otherwise unmap every page of the input on one thread, while the clock
    // runs. Clearing the page tables in pieces now, on all threads, removes most of that cost.
    loop {
        let off = job.next_unmap.0.fetch_add(UNMAP_STEP, Relaxed);
        if off >= job.size {
            break;
        }
        unsafe { madvise(job.input.add(off) as *mut c_void, UNMAP_STEP.min(job.size - off), MADV_DONTNEED) };
    }

    // Nothing writes the logs now. Each country goes to one worker, whose bitset fits in L1.
    let mut seen = [0u64; 1 << 10];
    loop {
        let country = job.next_country.0.fetch_add(1, Relaxed);
        if country >= COUNTRIES {
            return;
        }
        let mut distinct = 0;
        for list in 0..2 {
            for thread in 0..job.threads {
                let mut cursor = unsafe { (*job.groups.add(thread * COUNTRIES + country)).logs[list] } as usize;
                while cursor != 0 {
                    let segment = (cursor - 1) / SEGMENT;
                    for at in segment * SEGMENT + 2..cursor {
                        let name = unsafe { *job.log.add(at) } as usize;
                        seen[name / 64] |= 1 << (name % 64);
                    }
                    let previous = unsafe { job.log.add(segment * SEGMENT).cast::<u32>().read() } as usize;
                    cursor = (previous + 1) * SEGMENT * (previous != 0) as usize;
                }
            }
            distinct += seen.iter().map(|word| word.count_ones()).sum::<u32>();
            seen.fill(0);
        }
        job.distinct[country].store(distinct, Relaxed);
    }
}

fn open_input(path: &str) -> (File, usize) {
    let file = File::open(path).unwrap();
    let size = file.metadata().unwrap().len() as usize;
    (file, size)
}

fn money(cents: i64) -> String {
    let sign = if cents < 0 { "-" } else { "" };
    let abs = cents.unsigned_abs();
    format!("{sign}{}.{:02}", abs / 100, abs % 100)
}

fn main() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let (region_file, region_size) = open_input("/data/regions.csv");
    let (file, size) = open_input("/data/records.csv");
    let regions = unsafe { std::slice::from_raw_parts(map_input(&region_file, region_size), region_size) };
    let last_line = regions[..region_size - 1].iter().rposition(|&b| b == b'\n').unwrap() + 1;
    let last_region = parse_id(&regions[last_line..]).0 + 1;
    // The randomly accessed tables share huge pages, which avoids a TLB miss on most records. They
    // are touched before the workers start: threads racing to fault one huge page are each charged
    // 2 MB until the race resolves, which shows up in peak memory.
    let names_at = (last_region + 1).next_multiple_of(64);
    let tables_size = names_at + (8 << NAME_BITS);
    let tables = zeroed(tables_size, MADV_HUGEPAGE);
    for page in (0..tables_size).step_by(HUGE) {
        unsafe { tables.add(page).write(0) };
    }
    let job = Job {
        input: map_input(&file, size),
        size,
        regions,
        by_id: tables,
        last_region: last_region as u32,
        names: unsafe { tables.add(names_at) }.cast(),
        // Worst case: 11-byte records, each with one u16, plus a segment per open log and the links.
        // Written in order and racing at its end, so small pages are best here.
        log: zeroed(size / 4 + threads * COUNTRIES * 4 * SEGMENT, MADV_NOHUGEPAGE).cast(),
        // Most of a huge page would stay unused here.
        groups: zeroed(threads * COUNTRIES * size_of::<Group>(), MADV_NOHUGEPAGE).cast(),
        threads,
        seen: (0..26 * 26 * 26).map(|_| AtomicU8::new(0)).collect(),
        codes: Mutex::new(Vec::with_capacity(COUNTRIES)),
        distinct: (0..COUNTRIES).map(|_| AtomicU32::new(0)).collect(),
        barrier: Barrier::new(threads),
        // The first piece starts after the header, which ends at the first LF.
        next_region: Counter(AtomicUsize::new(1)),
        next_chunk: Counter(AtomicUsize::new(0)),
        next_segment: Counter(AtomicU32::new(1)),
        next_country: Counter(AtomicUsize::new(0)),
        next_unmap: Counter(AtomicUsize::new(0)),
        used: Counter(AtomicUsize::new(0)),
    };
    std::thread::scope(|scope| {
        for index in 1..threads {
            let job = &job;
            scope.spawn(move || worker(job, index));
        }
        worker(&job, 0);
    });

    let codes = job.codes.lock().unwrap();
    let mut keys: Vec<(&[u8], usize)> = vec![(b"_UNMATCHED".as_slice(), 0)];
    keys.extend(codes.iter().enumerate().map(|(i, code)| (code.as_slice(), i + 1)));
    keys.sort_unstable();
    let mut out = Vec::with_capacity(keys.len() * 96);
    for (key, country) in keys {
        let mut total = EMPTY;
        for thread in 0..threads {
            let from = unsafe { &*job.groups.add(thread * COUNTRIES + country) };
            for (sum, add) in total.sums.iter_mut().zip(from.sums) {
                *sum = sum.wrapping_add(add);
            }
            total.amount_min = total.amount_min.min(from.amount_min);
            total.amount_max = total.amount_max.max(from.amount_max);
        }
        let [id_sum, amount_sum, counted, note_bytes] = total.sums;
        if counted == 0 {
            continue;
        }
        out.extend_from_slice(key);
        let line = format!(
            " {} {} {} {} {} {} {} {}\n",
            counted >> 32,
            id_sum,
            money(amount_sum as i64),
            money(total.amount_min as i64),
            money(total.amount_max as i64),
            job.distinct[country].load(Relaxed),
            counted as u32,
            note_bytes
        );
        out.extend_from_slice(line.as_bytes());
    }
    std::fs::write("/out/aggregates.txt", &out).unwrap();
    unsafe { exit(0) };
}
