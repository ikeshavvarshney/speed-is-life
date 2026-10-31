use std::arch::x86_64::*;
use std::ffi::c_void;
use std::fs::File;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

const CHUNK: usize = 1 << 18;
const QUERY_CHUNK: usize = 1 << 14;
const PREFETCH_AHEAD: usize = 12;
const SAMPLE_SHARE: usize = 512;
// A term in more than 1/32 of documents is smaller as a bitset than in the temporary hit log.
const HIGH_SHARE: usize = 32;

// A table slot holds the high bits of the term's hash, an anchor flag and the term's id. While
// queries are read, the lowest bit instead records that a boolean query uses the term.
const ID_BITS: u32 = 18;
const ID_MASK: u64 = (1 << ID_BITS) - 1;
const IN_BOOL: u64 = 1;
const ANCHOR_FLAG: u64 = 1 << ID_BITS;
const TAG_ONE: u64 = ANCHOR_FLAG << 1;
const TAG_MASK: u64 = !(TAG_ONE - 1);
const TABLE_PADDING: usize = 1 << 10;

// A logged hit is 16 bits: a term within its partition, and a document within its group of 64.
// A worker writes a marker naming the group each time it starts one.
const GROUP_BITS: u32 = 6;
const GROUP_DOCS: u32 = 1 << GROUP_BITS;
const PARTITION_BITS: u32 = 9;
const PARTITION_TERMS: usize = 1 << PARTITION_BITS;
const MARKER: u16 = 1 << 15;
const BUCKET_START: usize = 1 << 15;
const NO_GROUP: u32 = u32::MAX;

// Document ids in lists are stored as 16 bits within a block of 65,536 documents.
const BLOCK_BITS: u32 = 16;
const GROUPS_PER_BLOCK_BITS: u32 = BLOCK_BITS - GROUP_BITS;

const HASH_0: u64 = 0x2D35_8DCC_AA6C_78A5;
const HASH_1: u64 = 0x8BB8_4B93_962E_ACC9;
const HASH_2: u64 = 0x4B33_A62E_D433_D4A3;
const HASH_3: u64 = 0xC2B2_AE3D_27D4_EB4F;

unsafe extern "C" {
    #[link_name = "_exit"]
    fn exit(code: i32) -> !;
    fn mallopt(param: i32, value: i32) -> i32;
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
}

const M_MMAP_THRESHOLD: i32 = -3;

// Mapped over a larger anonymous mapping so vector loads near the end of the file stay readable.
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

const LOW_BYTES: [u8; 64] = {
    let mut table = [0u8; 64];
    let mut i = 0;
    while i < 32 {
        table[i] = 0xFF;
        i += 1;
    }
    table
};

#[inline(always)]
fn mum(a: u64, b: u64) -> u64 {
    let product = a as u128 * b as u128;
    product as u64 ^ (product >> 64) as u64
}

// Hashes a token where it lies. Token bytes are letters and digits, so OR-ing 0x20 lowercases
// them, and masking removes whatever follows. Terms are at most 32 bytes, so a longer token
// hashes to a value no term has. The hash stands in for the bytes: with about 600,000 distinct
// tokens and 76,000 terms, the chance of any collision is about 1 in 10 million.
#[inline(always)]
unsafe fn token_hash(start: *const u8, len: usize) -> u64 {
    unsafe {
        if len <= 16 {
            let bytes = _mm_or_si128(_mm_loadu_si128(start.cast()), _mm_set1_epi8(0x20));
            let kept = _mm_and_si128(bytes, _mm_loadu_si128(LOW_BYTES.as_ptr().add(32 - len).cast()));
            let (low, high) = (_mm_cvtsi128_si64(kept) as u64, _mm_extract_epi64::<1>(kept) as u64);
            return mum(low ^ HASH_0, high ^ HASH_1);
        }
        long_token_hash(start, len)
    }
}

#[cold]
unsafe fn long_token_hash(start: *const u8, len: usize) -> u64 {
    if len > 32 {
        return 0;
    }
    unsafe {
        let bytes = _mm256_or_si256(_mm256_loadu_si256(start.cast()), _mm256_set1_epi8(0x20));
        let kept = _mm256_and_si256(bytes, _mm256_loadu_si256(LOW_BYTES.as_ptr().add(32 - len).cast()));
        let [a, b, c, d]: [u64; 4] = std::mem::transmute(kept);
        mum(a ^ HASH_0, b ^ HASH_1) ^ mum(c ^ HASH_2, d ^ HASH_3)
    }
}

// Classifies by nibble lookups: the two tables only share a set bit for letters and digits.
#[inline(always)]
unsafe fn token_mask(bytes: __m256i) -> u32 {
    unsafe {
        let low = _mm256_setr_epi8(
            5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 2, 2, 2, 2, 2, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 2, 2, 2, 2, 2,
        );
        let high = _mm256_setr_epi8(
            0, 0, 0, 1, 2, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        let nibbles = _mm256_and_si256(_mm256_srli_epi16::<4>(bytes), _mm256_set1_epi8(0x0F));
        let both = _mm256_and_si256(_mm256_shuffle_epi8(low, bytes), _mm256_shuffle_epi8(high, nibbles));
        !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(both, _mm256_setzero_si256())) as u32)
    }
}

// Calls `visit` with each token of the line at `p`, and returns what follows its newline. One
// classification covers 64 bytes, and tokens then come straight from bit masks. A window restarts
// at a token that runs past its end, so tokens never need joining up. Documents and query lines
// are both read this way.
#[inline(always)]
fn scan_line(mut p: *const u8, mut visit: impl FnMut(*const u8, u64)) -> *const u8 {
    unsafe {
        let newline = _mm256_set1_epi8(b'\n' as i8);
        loop {
            let (first, second) = (_mm256_loadu_si256(p.cast()), _mm256_loadu_si256(p.add(32).cast()));
            let mask = token_mask(first) as u64 | (token_mask(second) as u64) << 32;
            let newlines = _mm256_movemask_epi8(_mm256_cmpeq_epi8(first, newline)) as u32 as u64
                | (_mm256_movemask_epi8(_mm256_cmpeq_epi8(second, newline)) as u32 as u64) << 32;
            let tokens = mask & newlines.wrapping_sub(1) & !newlines;
            let mut starts = tokens & !(tokens << 1);
            let mut ends = !tokens & (tokens << 1);
            while ends != 0 {
                let start = starts.trailing_zeros() as usize;
                let len = ends.trailing_zeros() as usize - start;
                visit(p.add(start), token_hash(p.add(start), len));
                starts &= starts - 1;
                ends &= ends - 1;
            }
            if newlines != 0 {
                return p.add(newlines.trailing_zeros() as usize + 1);
            }
            if starts == 1 {
                p = p.add(64);
                while (*p).is_ascii_alphanumeric() {
                    p = p.add(1);
                }
            } else {
                p = p.add(if starts == 0 { 64 } else { starts.trailing_zeros() as usize });
            }
        }
    }
}

#[inline(always)]
fn parse_id(mut p: *const u8) -> (u32, *const u8) {
    let mut id = 0u32;
    unsafe {
        while *p != b'\t' {
            id = id * 10 + (*p - b'0') as u32;
            p = p.add(1);
        }
        (id, p.add(1))
    }
}

// Runs `work` for each chunk on all threads, and returns the results in chunk order.
fn for_each_chunk<T: Send>(chunks: usize, threads: usize, work: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let results: Vec<Mutex<Option<T>>> = (0..chunks).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let chunk = next.fetch_add(1, Relaxed);
                    if chunk >= chunks {
                        return;
                    }
                    *results[chunk].lock().unwrap() = Some(work(chunk));
                }
            });
        }
    });
    results.into_iter().map(|result| result.into_inner().unwrap().unwrap()).collect()
}

// The one lookup table, about 1 MB, with everything in an 8-byte slot. A slot is found by the
// hash's low bits and confirmed by its high bits. Query reading fills it from all threads.
struct Table {
    slots: Vec<AtomicU64>,
    mask: usize,
}

impl Table {
    fn new(query_bytes: usize) -> Self {
        let size = (query_bytes / 16 + 16).next_power_of_two().max(1 << 10);
        let mut zeros = std::mem::ManuallyDrop::new(vec![0u64; size + TABLE_PADDING]);
        let slots = unsafe { Vec::from_raw_parts(zeros.as_mut_ptr().cast(), size + TABLE_PADDING, size + TABLE_PADDING) };
        Self { slots, mask: size - 1 }
    }

    fn intern(&self, hash: u64, flag: u64) -> u32 {
        let tag = (hash | TAG_ONE) & TAG_MASK;
        let mut at = hash as usize & self.mask;
        loop {
            let slot = &self.slots[at];
            let mut value = slot.load(Relaxed);
            if value == 0 {
                match slot.compare_exchange(0, tag | flag, Relaxed, Relaxed) {
                    Ok(_) => return at as u32,
                    Err(actual) => value = actual,
                }
            }
            if value & TAG_MASK == tag {
                if value & flag != flag {
                    slot.fetch_or(flag, Relaxed);
                }
                return at as u32;
            }
            at += 1;
        }
    }

    #[inline(always)]
    fn prefetch(&self, hash: u64) {
        unsafe { _mm_prefetch::<_MM_HINT_T0>(self.slots.as_ptr().add(hash as usize & self.mask).cast()) };
    }

    #[inline(always)]
    fn position(&self, hash: u64) -> Option<usize> {
        let tag = (hash | TAG_ONE) & TAG_MASK;
        let mut at = hash as usize & self.mask;
        loop {
            let value = self.slots[at].load(Relaxed);
            if value & TAG_MASK == tag {
                return Some(at);
            }
            if value == 0 {
                return None;
            }
            at += 1;
        }
    }

    // Compares four slots at once. Tokens missing from the table are unpredictable, so no branch
    // decides the result, which is then an id that no list or bitset uses.
    #[inline(always)]
    fn find(&self, hash: u64) -> u64 {
        let tag = (hash | TAG_ONE) & TAG_MASK;
        let mut at = hash as usize & self.mask;
        unsafe {
            let tags = _mm256_set1_epi64x(tag as i64);
            let tag_mask = _mm256_set1_epi64x(TAG_MASK as i64);
            loop {
                let slots = self.slots.as_ptr().add(at).cast::<u64>();
                let group = _mm256_loadu_si256(slots.cast());
                let equal = _mm256_cmpeq_epi64(_mm256_and_si256(group, tag_mask), tags);
                let hits = _mm256_movemask_pd(_mm256_castsi256_pd(equal)) as u32;
                let empty = _mm256_cmpeq_epi64(group, _mm256_setzero_si256());
                if hits | _mm256_movemask_pd(_mm256_castsi256_pd(empty)) as u32 != 0 {
                    let found = ((hits != 0) as u64).wrapping_neg();
                    return *slots.add((hits | 16).trailing_zeros() as usize) & found | ID_MASK & !found;
                }
                at += 4;
            }
        }
    }
}

// All phrases are known before any document is read, so they are matched live during tokenizing.
// Any document matching a phrase must contain its rarest term, its anchor, so nothing happens
// until an anchor token appears. No positions are stored, no documents are re-read, and
// phrase-only terms need no lists at all. Nearly all anchor tokens match nothing, so a small
// bitset of (term, neighbour) pairs rejects them before the phrase tables are touched.
#[derive(Default)]
struct Phrases {
    starts: Vec<u32>,
    terms: Vec<u64>,
    anchor_starts: Vec<u32>,
    anchored: Vec<(u32, u32)>,
    pairs: Vec<u64>,
}

const PAIR_BITS: u32 = 20;

#[inline(always)]
fn pair_bit(first: u64, second: u64) -> usize {
    (mum(first ^ HASH_2, second ^ HASH_3) >> (64 - PAIR_BITS)) as usize
}

impl Phrases {
    #[inline(always)]
    fn may_pair(&self, first: u64, second: u64) -> bool {
        let bit = pair_bit(first, second);
        unsafe { *self.pairs.get_unchecked(bit / 64) >> (bit % 64) & 1 != 0 }
    }

    // `tokens` holds the document's hashes, with a 0 before and after them.
    #[cold]
    fn check(&self, id: usize, tokens: &[u64], at_token: usize, doc: u32, hits: &mut Vec<(u32, u32)>) {
        let (a, b) = (self.anchor_starts[id] as usize, self.anchor_starts[id + 1] as usize);
        for &(phrase, at) in &self.anchored[a..b] {
            let (first, last) = (self.starts[phrase as usize] as usize, self.starts[phrase as usize + 1] as usize);
            let (earlier, later) = self.terms[first..last].split_at(at as usize);
            if tokens[at_token..].starts_with(later) && tokens[..at_token].ends_with(earlier) {
                hits.push((phrase, doc));
            }
        }
    }
}

struct Shared<T>(*mut T);
unsafe impl<T> Sync for Shared<T> {}

#[derive(Default)]
struct ChunkPhrases {
    lengths: Vec<u32>,
    hashes: Vec<u64>,
    slots: Vec<u32>,
}

fn query_chunks(text: &[u8]) -> Vec<usize> {
    let mut bounds = vec![0];
    let mut at = QUERY_CHUNK;
    while at < text.len() {
        at += text[at..].iter().position(|&b| b == b'\n').map_or(text.len() - at, |i| i + 1);
        bounds.push(at);
        at += QUERY_CHUNK;
    }
    if *bounds.last().unwrap() < text.len() {
        bounds.push(text.len());
    }
    bounds
}

// Records every term in the table. Boolean queries need nothing else now, as they are read again
// from the text when they are answered.
fn read_chunk(text: &[u8], range: Range<usize>, table: &Table) -> ChunkPhrases {
    let mut chunk = ChunkPhrases::default();
    let mut pos = range.start;
    while pos < range.end {
        let is_bool = text[pos] == b'B';
        let mut index = 0;
        let before = chunk.hashes.len();
        let end = scan_line(unsafe { text.as_ptr().add(pos) }, |_, hash| {
            if is_bool {
                if index % 2 == 1 {
                    table.intern(hash, IN_BOOL);
                }
            } else if index > 0 {
                chunk.slots.push(table.intern(hash, 0));
                chunk.hashes.push(hash);
            }
            index += 1;
        });
        if !is_bool {
            chunk.lengths.push((chunk.hashes.len() - before) as u32);
        }
        pos = unsafe { end.offset_from(text.as_ptr()) } as usize;
    }
    chunk
}

fn doc_count(input: &[u8]) -> usize {
    if input.is_empty() {
        return 0;
    }
    let start = input[..input.len() - 1].iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    parse_id(input[start..].as_ptr()).0 as usize + 1
}

// Counts tokens rather than documents, which is close enough to decide how to store a term.
fn sample_counts(input: &[u8], table: &Table, docs: usize) -> Vec<u16> {
    let mut seen = vec![0u16; table.slots.len()];
    let mut pos = input.as_ptr();
    for _ in 0..docs / SAMPLE_SHARE {
        let (_, text) = parse_id(pos);
        pos = scan_line(text, |_, hash| {
            if let Some(at) = table.position(hash) {
                seen[at] = seen[at].saturating_add(1);
            }
        });
    }
    seen
}

struct Plan {
    table: Table,
    phrases: Phrases,
    phrase_bases: Vec<u32>,
    high: usize,
    low: usize,
}

// Final ids run through the common terms, then those with lists, then phrase-only terms, so an
// id alone says how a token is handled.
fn plan_index(mut table: Table, chunks: Vec<ChunkPhrases>, seen: &[u16], sample: usize) -> Plan {
    let threshold = (sample / HIGH_SHARE).max(1) as u16;
    let slots = unsafe { &mut *(table.slots.as_mut_slice() as *mut [AtomicU64] as *mut [u64]) };
    let (mut high, mut low, mut terms) = (0, 0, 0);
    for (at, &value) in slots.iter().enumerate() {
        if value != 0 {
            terms += 1;
            if value & IN_BOOL != 0 {
                if seen[at] > threshold { high += 1 } else { low += 1 }
            }
        }
    }
    let (mut next_high, mut next_low, mut next_other) = (0, high, high + low);
    for (at, value) in slots.iter_mut().enumerate() {
        if *value != 0 {
            let next = if *value & IN_BOOL == 0 {
                &mut next_other
            } else if seen[at] > threshold {
                &mut next_high
            } else {
                &mut next_low
            };
            *value = (*value & TAG_MASK) | *next as u64;
            *next += 1;
        }
    }

    let phrase_bases: Vec<u32> = chunks
        .iter()
        .scan(0, |base, chunk| {
            let this = *base;
            *base += chunk.lengths.len() as u32;
            Some(this)
        })
        .collect();
    let phrase_count: usize = chunks.iter().map(|chunk| chunk.lengths.len()).sum();
    let mut phrases = Phrases {
        starts: Vec::with_capacity(phrase_count + 1),
        terms: Vec::with_capacity(chunks.iter().map(|chunk| chunk.hashes.len()).sum()),
        anchor_starts: vec![0; terms + 2],
        anchored: vec![(0, 0); phrase_count],
        pairs: vec![0; 1 << (PAIR_BITS - 6)],
    };
    phrases.starts.push(0);
    let mut anchors: Vec<(u32, u32)> = Vec::with_capacity(phrase_count);
    for chunk in &chunks {
        let mut first = 0;
        for &length in &chunk.lengths {
            let (list, hashes) = (&chunk.slots[first..first + length as usize], &chunk.hashes[first..first + length as usize]);
            first += length as usize;
            let at = (0..list.len()).min_by_key(|&i| seen[list[i] as usize]).unwrap();
            let slot = &mut slots[list[at] as usize];
            *slot |= ANCHOR_FLAG;
            let id = *slot & ID_MASK;
            phrases.anchor_starts[id as usize + 2] += 1;
            anchors.push((id as u32, at as u32));
            let (one, two) = if at + 1 < list.len() { (at, at + 1) } else { (at - 1, at) };
            let bit = pair_bit(hashes[one], hashes[two]);
            phrases.pairs[bit / 64] |= 1 << (bit % 64);
            phrases.terms.extend_from_slice(hashes);
            phrases.starts.push(phrases.terms.len() as u32);
        }
    }
    for i in 2..terms + 2 {
        phrases.anchor_starts[i] += phrases.anchor_starts[i - 1];
    }
    for (phrase, &(id, at)) in anchors.iter().enumerate() {
        let place = &mut phrases.anchor_starts[id as usize + 1];
        phrases.anchored[*place as usize] = (phrase as u32, at);
        *place += 1;
    }
    Plan { table, phrases, phrase_bases, high, low }
}

// A growable log written through raw pointers, so a write costs one store and one compare.
struct Bucket {
    log: Vec<u16>,
    head: *mut u16,
    end: *mut u16,
}

impl Bucket {
    fn new(capacity: usize) -> Self {
        let mut log = Vec::with_capacity(capacity);
        let head = log.as_mut_ptr();
        Self { log, head, end: head.wrapping_add(capacity) }
    }

    #[inline(always)]
    fn push(&mut self, entry: u16) {
        unsafe {
            *self.head = entry;
            self.head = self.head.add(1);
        }
        if self.head == self.end {
            self.grow();
        }
    }

    #[cold]
    fn grow(&mut self) {
        let len = self.finish();
        self.log.reserve(len);
        self.head = unsafe { self.log.as_mut_ptr().add(len) };
        self.end = unsafe { self.log.as_mut_ptr().add(self.log.capacity()) };
    }

    fn finish(&mut self) -> usize {
        let len = unsafe { self.head.offset_from(self.log.as_ptr()) } as usize;
        unsafe { self.log.set_len(len) };
        len
    }
}

struct Worker {
    buckets: Vec<Vec<u16>>,
    hits: Vec<(u32, u32)>,
}

fn parse_docs(input: &'static [u8], plan: &Plan, bits: &mut [u64], words: usize, threads: usize) -> Vec<Worker> {
    let next = AtomicUsize::new(0);
    let bits_ptr = Shared(bits.as_mut_ptr());
    let (high, low) = (plan.high as u64, plan.low as u64);
    let partitions = plan.low / PARTITION_TERMS + 1;
    let (high_index, sink) = (plan.high, partitions);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let (next, bits_ptr) = (&next, &bits_ptr);
                scope.spawn(move || {
                    // One extra bucket takes the writes of tokens that need no list. It is never
                    // read and costs only one page, as its head is reset often.
                    let mut buckets: Vec<Bucket> = (0..=partitions).map(|_| Bucket::new(BUCKET_START)).collect();
                    let mut hits = Vec::new();
                    let mut tokens: Vec<u64> = Vec::with_capacity(1 << 12);
                    // Common terms set one bit per token, so 64 documents are gathered locally.
                    let mut columns = vec![0u64; plan.high + 1];
                    let mut group = NO_GROUP;
                    let flush = |columns: &mut [u64], group: u32| {
                        for (row, column) in columns[..high_index].iter_mut().enumerate() {
                            if *column != 0 {
                                unsafe {
                                    let word = bits_ptr.0.add(row * words + group as usize);
                                    AtomicU64::from_ptr(word).fetch_or(*column, Relaxed);
                                }
                                *column = 0;
                            }
                        }
                    };
                    let base = input.as_ptr();
                    loop {
                        let off = next.fetch_add(CHUNK, Relaxed);
                        if off >= input.len() {
                            if group != NO_GROUP {
                                flush(&mut columns, group);
                            }
                            buckets.iter_mut().for_each(|bucket| {
                                bucket.finish();
                            });
                            buckets.pop();
                            let buckets = buckets.into_iter().map(|bucket| bucket.log).collect();
                            return Worker { buckets, hits };
                        }
                        let mut i = if off == 0 {
                            0
                        } else {
                            off + input[off - 1..].iter().position(|&b| b == b'\n').unwrap()
                        };
                        let stop = (off + CHUNK).min(input.len());
                        while i < stop {
                            let (doc, text) = parse_id(unsafe { base.add(i) });
                            if doc >> GROUP_BITS != group {
                                if group != NO_GROUP {
                                    flush(&mut columns, group);
                                }
                                group = doc >> GROUP_BITS;
                                buckets.iter_mut().for_each(|bucket| bucket.push(MARKER | group as u16));
                                buckets[sink].head = buckets[sink].log.as_mut_ptr();
                            }
                            let position = doc & (GROUP_DOCS - 1);
                            let bit = 1u64 << position;
                            // Tokenizing and lookups are separate passes, so each lookup can be
                            // prefetched far enough ahead for memory to arrive in time.
                            tokens.clear();
                            tokens.push(0);
                            let end = scan_line(text, |_, hash| tokens.push(hash));
                            let count = tokens.len() - 1;
                            tokens.resize(count + 2 + PREFETCH_AHEAD, 0);
                            for at_token in 1..=count {
                                unsafe {
                                    plan.table.prefetch(*tokens.get_unchecked(at_token + PREFETCH_AHEAD));
                                    let hash = *tokens.get_unchecked(at_token);
                                    let slot = plan.table.find(hash);
                                    let id = slot & ID_MASK;
                                    if slot & ANCHOR_FLAG != 0
                                        && (plan.phrases.may_pair(hash, *tokens.get_unchecked(at_token + 1))
                                            || plan.phrases.may_pair(*tokens.get_unchecked(at_token - 1), hash))
                                    {
                                        plan.phrases.check(id as usize, &tokens, at_token, doc, &mut hits);
                                    }
                                    // A token's kind is unpredictable, so every token does both writes,
                                    // and tokens of the other kinds land in a slot nothing reads.
                                    *columns.get_unchecked_mut(id.min(high) as usize) |= bit;
                                    let term = id.wrapping_sub(high);
                                    let target = if term < low { (term >> PARTITION_BITS) as usize } else { sink };
                                    let entry = ((term as usize & (PARTITION_TERMS - 1)) as u16) << GROUP_BITS | position as u16;
                                    buckets.get_unchecked_mut(target).push(entry);
                                }
                            }
                            i = unsafe { end.offset_from(base) } as usize;
                        }
                    }
                })
            })
            .collect();
        handles.into_iter().map(|handle| handle.join().unwrap()).collect()
    })
}

struct Lists {
    ids: Vec<u16>,
    marks: Vec<u32>,
    blocks: usize,
}

impl Lists {
    #[inline]
    fn decode(&self, term: usize, out: &mut Vec<u32>) {
        out.clear();
        for block in 0..self.blocks {
            let at = term * (self.blocks + 1) + block;
            let (a, b) = (self.marks[at] as usize, self.marks[at + 1] as usize);
            let top = (block as u32) << BLOCK_BITS;
            out.extend(self.ids[a..b].iter().map(|&low| top | low as u32));
        }
    }
}

struct Segment {
    group: u32,
    first: u16,
    bucket: u32,
    start: u32,
    end: u32,
}

// One thread builds each partition of terms inside its cache, and frees its logs as it goes.
// Runs of hits are visited by group, so every list comes out sorted without sorting.
fn build_lists(workers: Vec<Vec<Vec<u16>>>, terms: usize, docs: usize, threads: usize) -> Lists {
    let partitions = terms / PARTITION_TERMS + 1;
    let mut logs: Vec<Vec<Vec<u16>>> = (0..partitions).map(|_| Vec::with_capacity(threads)).collect();
    for buckets in workers {
        for (p, bucket) in buckets.into_iter().enumerate() {
            logs[p].push(bucket);
        }
    }
    let mut bases = vec![0usize; partitions + 1];
    for p in 0..partitions {
        bases[p + 1] = bases[p] + logs[p].iter().map(Vec::len).sum::<usize>();
    }
    let blocks = (docs >> BLOCK_BITS) + 1;
    let tasks: Vec<Mutex<Vec<Vec<u16>>>> = logs.into_iter().map(Mutex::new).collect();
    let mut ids = vec![0u16; bases[partitions]];
    let mut marks = vec![0u32; (terms + 1) * (blocks + 1)];
    let ids_ptr = Shared(ids.as_mut_ptr());
    let marks_ptr = Shared(marks.as_mut_ptr());
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (next, tasks, bases, ids_ptr, marks_ptr) = (&next, &tasks, &bases, &ids_ptr, &marks_ptr);
            scope.spawn(move || {
                let mut segments: Vec<Segment> = Vec::new();
                loop {
                    let p = next.fetch_add(1, Relaxed);
                    if p >= partitions {
                        return;
                    }
                    let buckets = std::mem::take(&mut *tasks[p].lock().unwrap());
                    let mut counts = [0u32; PARTITION_TERMS];
                    segments.clear();
                    for (b, bucket) in buckets.iter().enumerate() {
                        let mut open = None;
                        for (i, &entry) in bucket.iter().enumerate() {
                            if entry & MARKER != 0 {
                                if let Some(segment) = open.take() {
                                    close_segment(&mut segments, segment, i, bucket);
                                }
                                open = Some(((entry & !MARKER) as u32, b as u32, i as u32 + 1));
                            } else {
                                counts[(entry >> GROUP_BITS) as usize] += 1;
                            }
                        }
                        if let Some(segment) = open {
                            close_segment(&mut segments, segment, bucket.len(), bucket);
                        }
                    }
                    segments.sort_unstable_by_key(|s| (s.group, s.first));

                    let mut cursors = [0u32; PARTITION_TERMS];
                    let mut sum = bases[p] as u32;
                    for (cursor, &count) in cursors.iter_mut().zip(&counts) {
                        *cursor = sum;
                        sum += count;
                    }
                    let first_term = p << PARTITION_BITS;
                    let last_term = (first_term + PARTITION_TERMS).min(terms);
                    let mark = |block: usize, cursors: &[u32; PARTITION_TERMS]| {
                        for term in first_term..last_term {
                            let at = term * (blocks + 1) + block;
                            unsafe { *marks_ptr.0.add(at) = cursors[term - first_term] };
                        }
                    };
                    mark(0, &cursors);
                    let mut block = 0;
                    let mut last = [u32::MAX; PARTITION_TERMS];
                    for segment in &segments {
                        let segment_block = (segment.group >> GROUPS_PER_BLOCK_BITS) as usize;
                        while block < segment_block {
                            block += 1;
                            mark(block, &cursors);
                            last = [u32::MAX; PARTITION_TERMS];
                        }
                        let group_low = (segment.group & ((1 << GROUPS_PER_BLOCK_BITS) - 1)) << GROUP_BITS;
                        let bucket = &buckets[segment.bucket as usize];
                        for &entry in &bucket[segment.start as usize..segment.end as usize] {
                            let term = (entry >> GROUP_BITS) as usize;
                            let low = group_low | (entry as u32 & (GROUP_DOCS - 1));
                            // A term repeated within a document appears once per token.
                            if last[term] != low {
                                last[term] = low;
                                unsafe { *ids_ptr.0.add(cursors[term] as usize) = low as u16 };
                                cursors[term] += 1;
                            }
                        }
                    }
                    while block < blocks {
                        block += 1;
                        mark(block, &cursors);
                    }
                }
            });
        }
    });
    Lists { ids, marks, blocks }
}

fn close_segment(segments: &mut Vec<Segment>, open: (u32, u32, u32), end: usize, bucket: &[u16]) {
    let (group, b, start) = open;
    if start as usize != end {
        let first = bucket[start as usize] & (GROUP_DOCS as u16 - 1);
        segments.push(Segment { group, first, bucket: b, start, end: end as u32 });
    }
}

struct Index {
    high: usize,
    words: usize,
    bits: Vec<u64>,
    lists: Lists,
}

#[derive(Default)]
struct Evaluator {
    as_bits: bool,
    ids: Vec<u32>,
    term: Vec<u32>,
    merged: Vec<u32>,
    bits: Vec<u64>,
}

#[inline(always)]
fn has_bit(bits: &[u64], id: u32) -> bool {
    bits[id as usize / 64] >> (id % 64) & 1 != 0
}

#[inline(always)]
fn set_bit(bits: &mut [u64], id: u32) {
    bits[id as usize / 64] |= 1 << (id % 64);
}

#[inline(always)]
fn clear_bit(bits: &mut [u64], id: u32) {
    bits[id as usize / 64] &= !(1 << (id % 64));
}

fn intersect(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if large.len() > small.len() * 16 {
        let mut rest = large;
        for &id in small {
            let at = rest.partition_point(|&x| x < id);
            rest = &rest[at..];
            if rest.first() == Some(&id) {
                out.push(id);
            }
        }
    } else {
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            let (x, y) = (a[i], b[j]);
            if x == y {
                out.push(x);
            }
            i += (x <= y) as usize;
            j += (y <= x) as usize;
        }
    }
}

fn union(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let (x, y) = (a[i], b[j]);
        out.push(x.min(y));
        i += (x <= y) as usize;
        j += (y <= x) as usize;
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

fn difference(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let mut rest = b;
    for &id in a {
        let at = rest.partition_point(|&x| x < id);
        rest = &rest[at..];
        if rest.first() != Some(&id) {
            out.push(id);
        }
    }
}

impl Evaluator {
    fn start(&mut self, index: &Index, id: usize) {
        self.as_bits = id < index.high;
        if self.as_bits {
            self.bits.clear();
            self.bits.extend_from_slice(&index.bits[id * index.words..(id + 1) * index.words]);
        } else {
            index.lists.decode(id - index.high, &mut self.ids);
        }
    }

    // Applies one operator, strictly left to right. `op` is the first byte of its word.
    fn apply(&mut self, index: &Index, op: u8, id: usize) {
        if !self.as_bits && self.ids.is_empty() && op != b'O' {
            return;
        }
        if id < index.high {
            let term = &index.bits[id * index.words..(id + 1) * index.words];
            if self.as_bits {
                match op {
                    b'A' => self.bits.iter_mut().zip(term).for_each(|(x, y)| *x &= y),
                    b'O' => self.bits.iter_mut().zip(term).for_each(|(x, y)| *x |= y),
                    _ => self.bits.iter_mut().zip(term).for_each(|(x, y)| *x &= !y),
                }
            } else {
                match op {
                    b'A' => self.ids.retain(|&doc| has_bit(term, doc)),
                    b'O' => {
                        self.bits.clear();
                        self.bits.extend_from_slice(term);
                        self.ids.iter().for_each(|&doc| set_bit(&mut self.bits, doc));
                        self.as_bits = true;
                    }
                    _ => self.ids.retain(|&doc| !has_bit(term, doc)),
                }
            }
        } else {
            index.lists.decode(id - index.high, &mut self.term);
            if self.as_bits {
                match op {
                    b'A' => {
                        self.ids.clear();
                        self.ids.extend(self.term.iter().filter(|&&doc| has_bit(&self.bits, doc)));
                        self.as_bits = false;
                    }
                    b'O' => self.term.iter().for_each(|&doc| set_bit(&mut self.bits, doc)),
                    _ => self.term.iter().for_each(|&doc| clear_bit(&mut self.bits, doc)),
                }
            } else {
                match op {
                    b'A' => intersect(&self.ids, &self.term, &mut self.merged),
                    b'O' => union(&self.ids, &self.term, &mut self.merged),
                    _ => difference(&self.ids, &self.term, &mut self.merged),
                }
                std::mem::swap(&mut self.ids, &mut self.merged);
            }
        }
    }

    fn write(&self, out: &mut Vec<u8>) {
        if self.as_bits {
            write_bits(out, &self.bits);
        } else {
            write_ids(out, self.ids.iter().copied());
        }
    }
}

fn answer_chunk(text: &[u8], range: Range<usize>, mut phrase: usize, index: &Index, plan: &Plan, hits: &[(u32, u32)], matched: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(range.len());
    let mut evaluator = Evaluator::default();
    let mut pos = range.start;
    while pos < range.end {
        if text[pos] == b'B' {
            let (mut position, mut op) = (0, 0);
            let end = scan_line(unsafe { text.as_ptr().add(pos) }, |word, hash| {
                if position % 2 == 0 {
                    op = unsafe { *word };
                } else {
                    let id = (plan.table.find(hash) & ID_MASK) as usize;
                    if position == 1 {
                        evaluator.start(index, id);
                    } else {
                        evaluator.apply(index, op, id);
                    }
                }
                position += 1;
            });
            evaluator.write(&mut out);
            pos = unsafe { end.offset_from(text.as_ptr()) } as usize;
        } else {
            let found = matched[phrase] as usize..matched[phrase + 1] as usize;
            write_ids(&mut out, hits[found].iter().map(|hit| hit.1));
            phrase += 1;
            pos += text[pos..].iter().position(|&b| b == b'\n').unwrap() + 1;
        }
    }
    out
}

fn write_ids(out: &mut Vec<u8>, ids: impl ExactSizeIterator<Item = u32> + Clone) {
    let sum: u64 = ids.clone().map(u64::from).sum();
    push_int(out, ids.len() as u64);
    out.push(b' ');
    push_int(out, sum);
    for id in ids.take(10) {
        out.push(b' ');
        push_int(out, id as u64);
    }
    out.push(b'\n');
}

// The sum of a word's set bit positions is a sum of six masked population counts.
fn write_bits(out: &mut Vec<u8>, bits: &[u64]) {
    const POSITION_BITS: [u64; 6] = [
        0xAAAA_AAAA_AAAA_AAAA,
        0xCCCC_CCCC_CCCC_CCCC,
        0xF0F0_F0F0_F0F0_F0F0,
        0xFF00_FF00_FF00_FF00,
        0xFFFF_0000_FFFF_0000,
        0xFFFF_FFFF_0000_0000,
    ];
    let (mut count, mut sum) = (0u64, 0u64);
    for (i, &word) in bits.iter().enumerate() {
        let ones = word.count_ones() as u64;
        count += ones;
        sum += ones * (i as u64 * 64);
        for (k, &mask) in POSITION_BITS.iter().enumerate() {
            sum += ((word & mask).count_ones() as u64) << k;
        }
    }
    push_int(out, count);
    out.push(b' ');
    push_int(out, sum);
    let mut left = 10;
    for (i, &word) in bits.iter().enumerate() {
        let mut rest = word;
        while rest != 0 && left > 0 {
            out.push(b' ');
            push_int(out, i as u64 * 64 + rest.trailing_zeros() as u64);
            rest &= rest - 1;
            left -= 1;
        }
        if left == 0 {
            break;
        }
    }
    out.push(b'\n');
}

fn push_int(out: &mut Vec<u8>, mut value: u64) {
    let mut digits = [0u8; 20];
    let mut i = 20;
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

fn main() {
    unsafe { mallopt(M_MMAP_THRESHOLD, 1 << 16) };
    let threads = std::thread::available_parallelism().map_or(1, |c| c.get());
    let mut text = std::fs::read("/data/queries.txt").unwrap();
    let length = text.len();
    let bounds = query_chunks(&text);
    text.resize(length + 64, 0);
    let chunks = bounds.len() - 1;
    let table = Table::new(length);
    let phrase_chunks = for_each_chunk(chunks, threads, |c| read_chunk(&text, bounds[c]..bounds[c + 1], &table));

    let file = File::open("/data/docs.txt").unwrap();
    let size = file.metadata().unwrap().len() as usize;
    let input = map_input(&file, size);
    let docs = doc_count(input);
    let words = docs / 64 + 1;
    let seen = sample_counts(input, &table, docs);
    let plan = plan_index(table, phrase_chunks, &seen, docs / SAMPLE_SHARE);
    drop(seen);

    let mut bits = vec![0u64; plan.high * words];
    let workers = parse_docs(input, &plan, &mut bits, words, threads);
    let mut hits = Vec::new();
    let mut logs = Vec::with_capacity(workers.len());
    for worker in workers {
        hits.extend_from_slice(&worker.hits);
        logs.push(worker.buckets);
    }
    let lists = build_lists(logs, plan.low, docs, threads);
    let index = Index { high: plan.high, words, bits, lists };

    // A document can match a phrase more than once, but it counts once.
    hits.sort_unstable();
    hits.dedup();
    let phrase_count = plan.phrases.starts.len() - 1;
    let mut matched = vec![0u32; phrase_count + 1];
    for &(phrase, _) in &hits {
        matched[phrase as usize + 1] += 1;
    }
    for i in 1..=phrase_count {
        matched[i] += matched[i - 1];
    }

    let outputs = for_each_chunk(chunks, threads, |c| {
        answer_chunk(&text, bounds[c]..bounds[c + 1], plan.phrase_bases[c] as usize, &index, &plan, &hits, &matched)
    });
    std::fs::write("/out/results.txt", outputs.concat()).unwrap();
    unsafe { exit(0) };
}
