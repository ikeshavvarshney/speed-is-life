use std::arch::x86_64::*;
use std::ffi::c_void;
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::ptr::null_mut;
use std::slice::from_raw_parts;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Mutex, OnceLock};

const CHUNK: usize = 1 << 18;
const UNMAP_STEP: usize = 1 << 23;
const HOURS: usize = 72;
const BLOCK: usize = 256;
const BUCKETS: usize = 512;
const WAYS: usize = 8;
const SLOTS: usize = BUCKETS * WAYS;
// A log entry is id << 21 | is_error << 20 | latency. Bigger latencies, about 1% of lines, go to a
// side list. An all-zero entry is an unused part of a block, and id 0 is reserved to absorb it.
const LATENCY_BITS: u32 = 20;
const MAX_IDS: usize = 1 << 11;
const DIGITS: u64 = 0x3030_3030_3030_3030;

type BigLists = [Vec<(u32, u32)>; HOURS];
type Ptr = *mut u32;

#[repr(align(32))]
struct Tags([AtomicU32; SLOTS]);

// Table of name hashes, with eight ways per bucket compared at once by one vector load. After the
// first few lines it is only read, so it is shared without contention and nothing needs merging.
// A tag and its bucket give 40 bits of hash, which over 1,842 names collide with odds of 2e-6.
static TAGS: Tags = Tags([const { AtomicU32::new(0) }; SLOTS]);
// Ids are given out in the order names are first seen. Popular names come early, which packs the
// counters they use later into few cache lines.
static IDS: [AtomicU16; SLOTS] = [const { AtomicU16::new(0) }; SLOTS];
// Each name's offset and length in the block of name bytes. Id n is at index n - 1.
static NAMES: Mutex<(Vec<(u32, u32)>, Vec<u8>)> = Mutex::new((Vec::new(), Vec::new()));
static STAMPS: [AtomicU64; HOURS] = [const { AtomicU64::new(0) }; HOURS];
static CURSORS: [AtomicUsize; HOURS] = [const { AtomicUsize::new(0) }; HOURS];
static NEXT_CHUNK: AtomicUsize = AtomicUsize::new(0);
static NEXT_UNMAP: AtomicUsize = AtomicUsize::new(0);
static NEXT_HOUR: AtomicUsize = AtomicUsize::new(0);
static NEXT_WORKER: AtomicUsize = AtomicUsize::new(0);
static PARSED: AtomicUsize = AtomicUsize::new(0);
static BIG: Mutex<Vec<BigLists>> = Mutex::new(Vec::new());
static READY: OnceLock<Ready> = OnceLock::new();
static OUTPUT: Mutex<Output> =
    Mutex::new(Output { next: 0, writing: false, waiting: [const { None }; HOURS], spare: Vec::new() });

struct Ready {
    // Each name and its id, in output order.
    order: Vec<(&'static [u8], u32)>,
    big: Vec<BigLists>,
}

struct Output {
    next: usize,
    writing: bool,
    waiting: [Option<Vec<u8>>; HOURS],
    spare: Vec<Vec<u8>>,
}

struct Job {
    input: *const u8,
    size: usize,
    logs: Ptr,
    capacity: usize,
    threads: usize,
    file: File,
}
unsafe impl Sync for Job {}

impl Job {
    fn region(&self, index: usize) -> Ptr {
        unsafe { self.logs.add(index * self.capacity) }
    }
}

#[repr(C)]
struct Masks([u8; 64], [u8; 64]);
static MASKS: Masks = Masks([0xFF; 64], [0; 64]);

// A vector whose first `n` bytes are set, for n up to 64, followed by clear bytes.
#[inline(always)]
unsafe fn low_bytes(n: usize) -> *const u8 {
    unsafe { (&raw const MASKS).cast::<u8>().add(64 - n) }
}

unsafe extern "C" {
    #[link_name = "_exit"]
    fn exit(code: i32) -> !;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
}

const MADV_DONTNEED: i32 = 4;
const MADV_NOHUGEPAGE: i32 = 15;

// Zeroed memory where only touched pages are real: readable, writable, private, anonymous, with
// nothing reserved up front.
unsafe fn map_zeroed(len: usize) -> *mut c_void {
    unsafe { mmap(null_mut(), len, 3, 0x4022, -1, 0) }
}

#[inline(always)]
fn load64(p: *const u8) -> u64 {
    unsafe { p.cast::<u64>().read_unaligned() }
}

#[inline(always)]
unsafe fn load(p: *const u8) -> __m256i {
    unsafe { _mm256_loadu_si256(p.cast()) }
}

#[inline(always)]
unsafe fn store(p: *mut u8, v: __m256i) {
    unsafe { _mm256_storeu_si256(p.cast(), v) }
}

// The spec fixes the span to Oct 30 18:00 .. Nov 2 18:00, so the day digits alone identify the
// date: adding one and wrapping at 31 turns days 30, 31, 1, 2 into 0, 1, 2, 3.
fn hour_index(line: *const u8) -> usize {
    let at = |i| unsafe { (*line.add(i) - b'0') as usize };
    (at(8) * 10 + at(9) + 1) % 31 * 24 + at(11) * 10 + at(12) - 18
}

// Bit i is set when byte i is a space, for the 64 bytes at `p`. Taking both vectors every time
// avoids a branch on the endpoint's length, which half the lines would mispredict.
#[inline(always)]
unsafe fn spaces(p: *const u8) -> u64 {
    unsafe {
        let space = _mm256_set1_epi8(b' ' as i8);
        let low = _mm256_movemask_epi8(_mm256_cmpeq_epi8(load(p), space)) as u32 as u64;
        low | (_mm256_movemask_epi8(_mm256_cmpeq_epi8(load(p.add(32)), space)) as u64) << 32
    }
}

// Parses the latency and returns it with a pointer past the newline. Up to eight digits take the
// SWAR path, which has no per-digit branch: only '0'..'9' have bit 4 set, not '\n'.
#[inline(always)]
unsafe fn parse_latency(p: *const u8) -> (u32, *const u8) {
    let word = load64(p);
    let stop = !word & 0x1010_1010_1010_1010;
    unsafe {
        if stop == 0 {
            let (mut value, mut p) = (0u32, p);
            while *p != b'\n' {
                value = value * 10 + (*p - b'0') as u32;
                p = p.add(1);
            }
            return (value, p.add(1));
        }
        let len = stop.trailing_zeros() as usize / 8;
        let digits = (word << ((8 - len) * 8)) & 0x0F0F_0F0F_0F0F_0F0F;
        let pairs = digits.wrapping_mul(2561) >> 8;
        let quads = (pairs & 0x00FF_00FF_00FF_00FF).wrapping_mul(6553601) >> 16;
        let value = (quads & 0x0000_FFFF_0000_FFFF).wrapping_mul(42949672960001) >> 32;
        (value as u32, p.add(len + 1))
    }
}

// Mixes the whole name at a fixed cost for up to 64 bytes, masking off what follows it.
#[inline(always)]
unsafe fn hash_name(p: *const u8, len: usize) -> u64 {
    unsafe {
        let cut = low_bytes(len.min(64));
        let a = _mm256_and_si256(load(p), load(cut));
        let b = _mm256_and_si256(load(p.add(32)), load(cut.add(32)));
        let seed = _mm_set_epi64x(0x2D35_8DCC_AA6C_78A5, 0x8BB8_4B93_962E_ACC9u64 as i64);
        let (a0, a1) = (_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a));
        let (b0, b1) = (_mm256_castsi256_si128(b), _mm256_extracti128_si256::<1>(b));
        let mut h = _mm_aesenc_si128(_mm_aesenc_si128(_mm_xor_si128(a0, seed), a1), _mm_aesenc_si128(b0, b1));
        let mut i = 64;
        while i < len {
            let cut = low_bytes((len - i).min(16));
            h = _mm_aesenc_si128(h, _mm_and_si128(_mm_loadu_si128(p.add(i).cast()), _mm_loadu_si128(cut.cast())));
            i += 16;
        }
        // One round only spreads a byte over four, so two are needed before keeping 64 bits.
        _mm_cvtsi128_si64(_mm_aesenc_si128(_mm_aesenc_si128(h, seed), seed)) as u64
    }
}

// Readers see a tag only after its id is in place, so they never read an id that is not set yet.
#[cold]
#[inline(never)]
fn insert_name(tag: u32, mut bucket: usize, name: &[u8]) -> u16 {
    let (spans, bytes) = &mut *NAMES.lock().unwrap();
    loop {
        for slot in bucket * WAYS..(bucket + 1) * WAYS {
            let seen = TAGS.0[slot].load(Relaxed);
            if seen == tag {
                return IDS[slot].load(Relaxed);
            }
            if seen == 0 {
                spans.push((bytes.len() as u32, name.len() as u32));
                bytes.extend_from_slice(name);
                assert!(spans.len() < MAX_IDS);
                IDS[slot].store(spans.len() as u16, Relaxed);
                TAGS.0[slot].store(tag, Release);
                return spans.len() as u16;
            }
        }
        bucket = (bucket + 1) % BUCKETS;
    }
}

// Takes a block from an hour's region, to append to it without any sharing.
#[cold]
fn claim(job: &Job, hour: usize) -> (Ptr, Ptr) {
    let at = unsafe { job.region(hour).add(CURSORS[hour].fetch_add(BLOCK, Relaxed)) };
    (at, unsafe { at.add(BLOCK) })
}

fn parse_chunk(job: &Job, first: usize, stop: usize, heads: &mut [(Ptr, Ptr); HOURS], big: &mut BigLists) {
    let mut line = unsafe { job.input.add(first) };
    let end = unsafe { job.input.add(stop) };
    let (mut stamp, mut hour) = (0, 0);
    let (mut at, mut limit) = heads[0];
    let tags = TAGS.0.as_ptr();
    while line < end {
        unsafe {
            _mm_prefetch::<_MM_HINT_T0>(line.add(1024).cast());
            // These bytes hold the month, day and hour. Nearly every line has the hour of the
            // line before it, as the file is the shards' chunks in order of arrival.
            let now = load64(line.add(5));
            if now != stamp {
                heads[hour] = (at, limit);
                stamp = now;
                hour = hour_index(line);
                STAMPS[hour].store(now, Relaxed);
                (at, limit) = heads[hour];
            }
            let mut fields = line.add(28);
            let mask = spaces(fields);
            let name = fields.add(mask.trailing_zeros() as usize + 1);
            let mut rest = mask & (mask - 1);
            while rest == 0 {
                fields = fields.add(64);
                rest = spaces(fields);
            }
            let space = fields.add(rest.trailing_zeros() as usize);
            let len = space.offset_from(name) as usize;

            let hash = hash_name(name, len);
            let tag = hash as u32 | 1;
            let wanted = _mm256_set1_epi32(tag as i32);
            let mut bucket = (hash >> 32) as usize & (BUCKETS - 1);
            let id = loop {
                let group = _mm256_load_si256(tags.add(bucket * WAYS).cast());
                let hit = _mm256_movemask_epi8(_mm256_cmpeq_epi32(group, wanted)) as u32;
                if hit != 0 {
                    break IDS[bucket * WAYS + hit.trailing_zeros() as usize / 4].load(Relaxed);
                }
                // A full bucket sends a name on to the next one. Any room here means it is new.
                if _mm256_movemask_epi8(_mm256_cmpeq_epi32(group, _mm256_setzero_si256())) != 0 {
                    break insert_name(tag, bucket, from_raw_parts(name, len));
                }
                bucket = (bucket + 1) % BUCKETS;
            } as u32;

            let error = (*space.add(1) == b'5') as u32;
            let (latency, next) = parse_latency(space.add(5));
            let entry = id << (LATENCY_BITS + 1) | error << LATENCY_BITS;
            if latency >> LATENCY_BITS != 0 {
                big[hour].push((entry, latency));
            } else {
                if at == limit {
                    (at, limit) = claim(job, hour);
                }
                *at = entry | latency;
                at = at.add(1);
            }
            line = next;
        }
    }
    heads[hour] = (at, limit);
}

// One layer of a sorting network: each lane is compared with the lane named in `partner`, and the
// lanes set in BLEND take the larger value.
#[inline(always)]
unsafe fn sort_layer<const BLEND: i32>(v: __m256i, partner: [i32; 8]) -> __m256i {
    unsafe {
        let other = _mm256_permutevar8x32_epi32(v, _mm256_loadu_si256(partner.as_ptr().cast()));
        _mm256_blend_epi32::<BLEND>(_mm256_min_epu32(v, other), _mm256_max_epu32(v, other))
    }
}

// Sorts up to eight values in a register, with no branches: lanes past n read as u32::MAX, so
// they sort to the end. This is the optimal 19-comparator, 6-layer network for eight inputs.
#[inline(always)]
unsafe fn sort_lanes(values: Ptr, n: usize) -> [u32; 8] {
    unsafe {
        let past = _mm256_xor_si256(load(low_bytes(n * 4)), _mm256_set1_epi32(-1));
        let v = _mm256_or_si256(load(values.cast()), past);
        let v = sort_layer::<0xCC>(v, [2, 3, 0, 1, 6, 7, 4, 5]);
        let v = sort_layer::<0xF0>(v, [4, 5, 6, 7, 0, 1, 2, 3]);
        let v = sort_layer::<0xAA>(v, [1, 0, 3, 2, 5, 4, 7, 6]);
        let v = sort_layer::<0x30>(v, [0, 1, 4, 5, 2, 3, 6, 7]);
        let v = sort_layer::<0x50>(v, [0, 4, 2, 6, 1, 5, 3, 7]);
        let v = sort_layer::<0x54>(v, [0, 2, 1, 4, 3, 6, 5, 7]);
        std::mem::transmute(v)
    }
}

// A permutation that moves the lanes set in `mask` to the front, keeping their order.
#[inline(always)]
unsafe fn front_lanes(mask: u32) -> __m256i {
    unsafe {
        let spread = _pdep_u64(mask as u64, 0x0101_0101_0101_0101).wrapping_mul(0xFF);
        _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(_pext_u64(0x0706_0504_0302_0100, spread) as i64))
    }
}

// Moves the values below `pivot` to `low` and the others to `high`, returning the count below.
// Either target may be the source, as writing never overtakes reading. Vector stores can write
// up to seven lanes past the end of what they keep.
#[inline(always)]
unsafe fn partition(from: Ptr, n: usize, pivot: u32, low: Ptr, high: Ptr) -> usize {
    unsafe {
        let wall = _mm256_set1_epi32(pivot as i32);
        let (mut below, mut above, mut i) = (0, 0, 0);
        while i < n {
            let v = load(from.add(i).cast());
            let is_above = _mm256_cmpeq_epi32(_mm256_max_epu32(v, wall), v);
            let lanes = if n - i < 8 { (1u32 << (n - i)) - 1 } else { 0xFF };
            let up = _mm256_movemask_ps(_mm256_castsi256_ps(is_above)) as u32 & lanes;
            let down = !up & lanes;
            store(low.add(below).cast(), _mm256_permutevar8x32_epi32(v, front_lanes(down)));
            store(high.add(above).cast(), _mm256_permutevar8x32_epi32(v, front_lanes(up)));
            below += down.count_ones() as usize;
            above += up.count_ones() as usize;
            i += 8;
        }
        below
    }
}

// Sets found[i] to the value of rank ranks[i], for ascending ranks among the n values at `values`,
// whose first value has rank `first` in the whole group. `spare` is scratch memory.
unsafe fn select(values: Ptr, n: usize, first: usize, ranks: &[usize], found: &mut [u32], spare: Ptr) {
    if ranks.is_empty() {
        return;
    }
    unsafe {
        if n <= 8 {
            let sorted = sort_lanes(values, n);
            for (rank, value) in ranks.iter().zip(found) {
                *value = sorted[rank - first];
            }
            return;
        }
        let (a, b, c) = (*values, *values.add(n / 2), *values.add(n - 1));
        let pivot = a.max(b).min(a.min(b).max(c));
        let below = partition(values, n, pivot, values, spare);
        if below == 0 {
            // The pivot is the smallest value, and with many equal values this would never end.
            // Its copies are all the same answer, so they are split off and need no more work.
            let copies = if pivot == u32::MAX { n } else { partition(spare, n, pivot + 1, values, spare) };
            let split = ranks.partition_point(|&rank| rank < first + copies);
            found[..split].fill(pivot);
            let rest = spare.add(n - copies + 8);
            return select(spare, n - copies, first + copies, &ranks[split..], &mut found[split..], rest);
        }
        let split = ranks.partition_point(|&rank| rank < first + below);
        let (lower, upper) = found.split_at_mut(split);
        let rest = spare.add(n - below + 8);
        select(values, below, first, &ranks[..split], lower, rest);
        select(spare, n - below, first + below, &ranks[split..], upper, rest);
    }
}

// Turns a value below 10^8 into eight digit bytes, most significant first in memory, by dividing
// all the pairs at once (from Paul Khuong's "How to print integers really fast").
#[inline(always)]
fn eight_digits(value: u32) -> u64 {
    let merged = (value / 10000) as u64 | ((value % 10000) as u64) << 32;
    let top = ((merged * 10486) >> 20) & (0x7F << 32 | 0x7F);
    let hundreds = ((merged - 100 * top) << 16) + top;
    let tens = ((hundreds * 103) >> 10) & (0xF << 48 | 0xF << 32 | 0xF << 16 | 0xF);
    tens + ((hundreds - 10 * tens) << 8)
}

// Writes a space and the value with no branch on its length. The caller reserved the room.
#[inline(always)]
unsafe fn put_number(out: *mut u8, value: u32) -> *mut u8 {
    unsafe {
        if value >= 100_000_000 {
            return put_long(out, value);
        }
        let digits = eight_digits(value);
        let skip = (digits | 1 << 56).trailing_zeros() as usize / 8;
        *out = b' ';
        out.add(1).cast::<u64>().write_unaligned(digits >> (skip * 8) | DIGITS);
        out.add(9 - skip)
    }
}

#[cold]
#[inline(never)]
unsafe fn put_long(out: *mut u8, value: u32) -> *mut u8 {
    unsafe {
        let out = put_number(out, value / 100_000_000);
        out.cast::<u64>().write_unaligned(eight_digits(value % 100_000_000) | DIGITS);
        out.add(8)
    }
}

// Groups the hour's values by name in `values`, then answers each group with selection.
fn render_hour(job: &Job, hour: usize, values: Ptr, tables: &mut [u32], out: &mut Vec<u8>) {
    let ready = READY.wait();
    let (counts, starts) = tables.split_at_mut(MAX_IDS * 2);
    let used = CURSORS[hour].load(Relaxed);
    let log = unsafe { from_raw_parts(job.region(hour), used) };
    let big = ready.big.iter().flat_map(|lists| lists[hour].iter());
    counts.fill(0);
    for &entry in log {
        unsafe { *counts.get_unchecked_mut((entry >> LATENCY_BITS) as usize) += 1 };
    }
    let mut extra = 0;
    for &(entry, _) in big.clone() {
        counts[(entry >> LATENCY_BITS) as usize] += 1;
        extra += 1;
    }
    // Groups lie in reverse order, so the vector stores that run past the end of the group being
    // answered only overwrite one that is finished.
    let mut sum = 0;
    for &(_, id) in ready.order.iter().rev() {
        starts[id as usize] = sum;
        sum += counts[id as usize * 2] + counts[id as usize * 2 + 1];
    }
    starts[0] = sum;
    for &entry in log {
        unsafe {
            let at = starts.get_unchecked_mut((entry >> (LATENCY_BITS + 1)) as usize);
            *values.add(*at as usize) = entry & ((1 << LATENCY_BITS) - 1);
            *at += 1;
        }
    }
    for &(entry, latency) in big {
        let at = &mut starts[(entry >> (LATENCY_BITS + 1)) as usize];
        unsafe { *values.add(*at as usize) = latency };
        *at += 1;
    }

    let mut label = *b"2026-MM-DDTHH:00:00Z            ";
    label[5..13].copy_from_slice(&STAMPS[hour].load(Relaxed).to_le_bytes());
    out.clear();
    // Each line is the label, the name, six numbers with separators and a newline, and vector
    // writes can pass the end of the last one.
    out.reserve(ready.order.len() * 88 + ready.order.iter().map(|item| item.0.len()).sum::<usize>() + 64);
    let mut cursor = out.as_mut_ptr();
    let mut end = sum as usize;
    let spare = unsafe { values.add(used + extra + 64) };
    for &(name, id) in ready.order.iter() {
        let errors = counts[id as usize * 2 + 1];
        let count = counts[id as usize * 2] + errors;
        if count == 0 {
            continue;
        }
        let n = count as usize;
        end -= n;
        let rank_of = |p: usize| (p * n + 99) / 100 - 1;
        let mut found = [0u32; 3];
        unsafe {
            select(values.add(end), n, 0, &[rank_of(50), rank_of(95), rank_of(99)], &mut found, spare);
            store(cursor, load(label.as_ptr()));
            cursor = cursor.add(21);
            store(cursor, load(name.as_ptr()));
            store(cursor.add(32), load(name.as_ptr().add(32)));
            if name.len() > 64 {
                std::ptr::copy_nonoverlapping(name.as_ptr(), cursor, name.len());
            }
            cursor = cursor.add(name.len());
            for value in [count, found[0], found[1], found[2], errors] {
                cursor = put_number(cursor, value);
            }
            cursor = put_number(cursor, (errors as u64 * 1_000_000 / count as u64) as u32);
            *cursor = b'\n';
            cursor = cursor.add(1);
        }
    }
    unsafe { out.set_len(cursor.offset_from(out.as_ptr()) as usize) };
}

// Hands over an hour's text. One thread at a time writes whatever is next in order, while the
// others carry on rendering, so writing the file does not become a serial step at the end. Hours
// finish out of order because their sizes differ, so finished ones wait here for their turn. The
// thread that writes the last hour ends the process, which needs no threads to be joined.
fn deliver(job: &Job, hour: usize, text: Vec<u8>) {
    let mut output = OUTPUT.lock().unwrap();
    output.waiting[hour] = Some(text);
    if output.writing {
        return;
    }
    output.writing = true;
    while output.next < HOURS {
        let next = output.next;
        let Some(text) = output.waiting[next].take() else { break };
        output.next += 1;
        drop(output);
        (&job.file).write_all(&text).unwrap();
        if next + 1 == HOURS {
            unsafe { exit(0) };
        }
        output = OUTPUT.lock().unwrap();
        // More buffers may have been needed for a moment than the threads need from now on, so
        // the extra ones are given back rather than kept.
        if output.spare.len() < 12 {
            output.spare.push(text);
        }
    }
    output.writing = false;
}

// Names are read from their own block, as the input's mapping is gone by the time this is used.
fn sorted_names(big: Vec<BigLists>) -> Ready {
    let (spans, bytes) = &mut *NAMES.lock().unwrap();
    bytes.extend_from_slice(&[0; 64]);
    let bytes: &'static [u8] = std::mem::take(bytes).leak();
    let name = |&(at, len): &(u32, u32)| &bytes[at as usize..(at + len) as usize];
    let mut order: Vec<_> = spans.iter().map(name).zip(1u32..).collect();
    order.sort_unstable();
    Ready { order, big }
}

fn worker(job: &Job) {
    let mut heads = [(null_mut(), null_mut()); HOURS];
    let mut big: BigLists = [const { Vec::new() }; HOURS];
    let bytes = unsafe { from_raw_parts(job.input, job.size) };
    loop {
        let off = NEXT_CHUNK.fetch_add(CHUNK, Relaxed);
        if off >= job.size {
            break;
        }
        // A chunk starts at the first line that begins inside it.
        let before = &bytes[off.saturating_sub(1)..];
        let first = off + before.iter().position(|&b| off == 0 || b == b'\n').unwrap();
        parse_chunk(job, first, (off + CHUNK).min(job.size), &mut heads, &mut big);
    }
    BIG.lock().unwrap().push(big);
    // The last parser to finish sorts the names, while the others clear the input's mapping.
    if PARSED.fetch_add(1, AcqRel) + 1 == job.threads {
        READY.get_or_init(|| sorted_names(std::mem::take(&mut *BIG.lock().unwrap())));
    }
    // Process exit would otherwise unmap every page of the input on one thread, while the clock
    // runs. Clearing the page tables in pieces now, on all threads, removes most of that cost.
    loop {
        let off = NEXT_UNMAP.fetch_add(UNMAP_STEP, Relaxed);
        if off >= job.size {
            break;
        }
        unsafe { madvise(job.input.add(off) as *mut c_void, UNMAP_STEP.min(job.size - off), MADV_DONTNEED) };
    }
    READY.wait();
    // Each thread starts with a spare region. After that an hour's log, once read, holds the next
    // hour's values, and the region used before it goes back to the system. This is what keeps
    // the peak low, as the output file grows while the logs shrink.
    let mut region = job.region(HOURS + NEXT_WORKER.fetch_add(1, Relaxed));
    let mut tables = vec![0; MAX_IDS * 3];
    loop {
        let hour = NEXT_HOUR.fetch_add(1, Relaxed);
        if hour >= HOURS {
            return;
        }
        let mut text = OUTPUT.lock().unwrap().spare.pop().unwrap_or_default();
        render_hour(job, hour, region, &mut tables, &mut text);
        deliver(job, hour, text);
        unsafe { madvise(region.cast(), job.capacity * 4, MADV_DONTNEED) };
        region = job.region(hour);
    }
}

fn main() {
    let input = File::open("/data/requests.log").unwrap();
    let size = input.metadata().unwrap().len() as usize;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    // A line is at least 40 bytes, which bounds any hour however the lines are spread. A region
    // can hold an hour's values and the scratch space for answering it, in whole pages so that it
    // can be given back on its own.
    let capacity = (3 * (size / 40 + threads * BLOCK) + 1024).next_multiple_of(1024);
    let total = (HOURS + threads) * capacity * 4;
    let job = unsafe {
        let logs = map_zeroed(total);
        // These regions are sparse, and with huge pages each would take at least 2 MB.
        madvise(logs, total, MADV_NOHUGEPAGE);
        // A file ending on a page boundary would fault on the vector loads near its end, so the
        // file is mapped over a slightly larger area that provides zeroed bytes after it.
        let base = map_zeroed(size + 4096);
        if size > 0 {
            mmap(base, size, 1, 0x12, input.as_raw_fd(), 0);
        }
        let file = File::create("/out/summary.txt").unwrap();
        Job { input: base as *const u8, size, logs: logs.cast(), capacity, threads, file }
    };
    std::thread::scope(|scope| {
        for _ in 1..threads {
            scope.spawn(|| worker(&job));
        }
        worker(&job);
    });
}
