use std::arch::x86_64::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

const CHUNK: usize = 1 << 18;
const PARTITIONS: usize = 64;
const BLOCK: usize = 1 << 14;
const QUERY_BATCH: usize = 64;
const HIGH_BIT: u32 = 1 << 31;
const NO_TERM: u32 = u32::MAX;
const PHRASE_ONLY: u32 = 1;
const SAMPLE_SHARE: usize = 100;
// A term in more than 1/32 of documents is smaller as a bitset than as 32-bit ids.
const HIGH_SHARE: usize = 32;

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

type Key = [u64; 4];

#[inline(always)]
fn mum(a: u64, b: u64) -> u64 {
    let product = a as u128 * b as u128;
    product as u64 ^ (product >> 64) as u64
}

#[inline(always)]
fn hash_key(key: &Key) -> u64 {
    mum(key[0] ^ 0x2D35_8DCC_AA6C_78A5, key[1] ^ 0x8BB8_4B93_962E_ACC9)
        ^ mum(key[2] ^ 0x4B33_A62E_D433_D4A3, key[3] ^ 0xC2B2_AE3D_27D4_EB4F)
}

// Terms are at most 32 bytes by the query grammar, so every term and candidate token is one
// zero-padded 32-byte key. A longer token can never match a term.
fn key_of(term: &[u8]) -> Key {
    let mut bytes = [0u8; 32];
    bytes[..term.len()].copy_from_slice(term);
    std::array::from_fn(|i| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()))
}

#[derive(Default)]
struct Terms {
    slots: Vec<u64>,
    keys: Vec<Key>,
    mask: usize,
}

impl Terms {
    fn with_capacity(capacity: usize) -> Self {
        let size = (capacity * 2).next_power_of_two().max(1 << 10);
        Self { slots: vec![0; size], keys: Vec::with_capacity(capacity), mask: size - 1 }
    }

    fn intern(&mut self, key: Key) -> u32 {
        let hash = hash_key(&key);
        let mut slot = hash as usize & self.mask;
        loop {
            let entry = self.slots[slot];
            if entry == 0 {
                self.keys.push(key);
                let id = self.keys.len() as u32 - 1;
                self.slots[slot] = (hash & 0xFFFF_FFFF_0000_0000) | (id + 1) as u64;
                return id;
            }
            if self.keys[(entry as u32 - 1) as usize] == key {
                return entry as u32 - 1;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    // The tag in each slot's high bits rejects most tokens without touching the key array.
    #[inline(always)]
    fn find(&self, key: &Key) -> u32 {
        let hash = hash_key(key);
        let tag = hash >> 32;
        let mut slot = hash as usize & self.mask;
        unsafe {
            loop {
                let entry = *self.slots.get_unchecked(slot);
                if entry == 0 {
                    return NO_TERM;
                }
                if entry >> 32 == tag {
                    let id = (entry as u32) - 1;
                    if self.keys.get_unchecked(id as usize) == key {
                        return id;
                    }
                }
                slot = (slot + 1) & self.mask;
            }
        }
    }
}

// Calls `visit` with the 32-byte key of each token in the document starting at `pos`, and returns
// the position after its newline. Tokens are found by vector classification of the bytes.
#[inline(always)]
fn scan_tokens(mut pos: *const u8, mut visit: impl FnMut(&Key)) -> *const u8 {
    unsafe {
        loop {
            let bytes = _mm256_loadu_si256(pos.cast());
            let tokens = token_mask(lowered(bytes));
            let newline = _mm256_movemask_epi8(_mm256_cmpeq_epi8(bytes, _mm256_set1_epi8(b'\n' as i8))) as u32;
            let first_token = tokens.trailing_zeros();
            let first_newline = newline.trailing_zeros();
            if first_newline < first_token {
                return pos.add(first_newline as usize + 1);
            }
            if tokens == 0 {
                pos = pos.add(32);
                continue;
            }
            let start = pos.add(first_token as usize);
            let word = lowered(_mm256_loadu_si256(start.cast()));
            let len = (!token_mask(word)).trailing_zeros() as usize;
            if len == 32 && is_token_byte(*start.add(32)) {
                pos = start.add(32);
                while is_token_byte(*pos) {
                    pos = pos.add(1);
                }
                continue;
            }
            let mask = _mm256_loadu_si256(LOW_BYTES.as_ptr().add(32 - len).cast());
            let masked = _mm256_and_si256(word, mask);
            let key: Key = std::mem::transmute(masked);
            visit(&key);
            pos = start.add(len);
        }
    }
}

#[inline(always)]
fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

// Bytes of 0x80 and above compare as negative, so they fall outside every range here.
#[inline(always)]
unsafe fn lowered(bytes: __m256i) -> __m256i {
    unsafe {
        let above = _mm256_cmpgt_epi8(bytes, _mm256_set1_epi8(b'A' as i8 - 1));
        let below = _mm256_cmpgt_epi8(_mm256_set1_epi8(b'Z' as i8 + 1), bytes);
        let bit = _mm256_and_si256(_mm256_and_si256(above, below), _mm256_set1_epi8(0x20));
        _mm256_or_si256(bytes, bit)
    }
}

#[inline(always)]
unsafe fn token_mask(lower: __m256i) -> u32 {
    unsafe {
        let letters = _mm256_and_si256(
            _mm256_cmpgt_epi8(lower, _mm256_set1_epi8(b'a' as i8 - 1)),
            _mm256_cmpgt_epi8(_mm256_set1_epi8(b'z' as i8 + 1), lower),
        );
        let digits = _mm256_and_si256(
            _mm256_cmpgt_epi8(lower, _mm256_set1_epi8(b'0' as i8 - 1)),
            _mm256_cmpgt_epi8(_mm256_set1_epi8(b'9' as i8 + 1), lower),
        );
        _mm256_movemask_epi8(_mm256_or_si256(letters, digits)) as u32
    }
}

// The document id, and a pointer to the text after its tab.
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

#[derive(Clone, Copy)]
struct Hit {
    term: u32,
    doc: u32,
}

#[derive(Default)]
struct Bucket {
    blocks: Vec<Box<MaybeUninit<[Hit; BLOCK]>>>,
    used: usize,
    total: usize,
}

impl Bucket {
    #[inline(always)]
    fn push(&mut self, hit: Hit) {
        if self.used == 0 {
            self.blocks.push(Box::new_uninit());
        }
        unsafe {
            let block = self.blocks.last_mut().unwrap_unchecked().as_mut_ptr();
            block.cast::<Hit>().add(self.used).write(hit);
        }
        self.used = (self.used + 1) % BLOCK;
        self.total += 1;
    }

    fn blocks(&self) -> impl Iterator<Item = &[Hit]> {
        self.blocks.iter().enumerate().map(|(i, block)| {
            let n = if i + 1 == self.blocks.len() && self.used != 0 { self.used } else { BLOCK };
            unsafe { std::slice::from_raw_parts(block.as_ptr().cast::<Hit>(), n) }
        })
    }
}

struct Shared<T>(*mut T);
unsafe impl<T> Sync for Shared<T> {}

enum Set {
    Ids(Vec<u32>),
    Bits(Vec<u64>),
}

#[derive(Clone, Copy)]
enum Term<'a> {
    Ids(&'a [u32]),
    Bits(&'a [u64]),
}

struct Index {
    terms: Terms,
    routes: Vec<u32>,
    offsets: Vec<u32>,
    ids: Vec<u32>,
    bits: Vec<u64>,
    words: usize,
}

impl Index {
    fn term(&self, id: u32) -> Term<'_> {
        let route = self.routes[id as usize];
        if route & HIGH_BIT != 0 {
            let at = (route & !HIGH_BIT) as usize * self.words;
            Term::Bits(&self.bits[at..at + self.words])
        } else {
            let (a, b) = (self.offsets[id as usize] as usize, self.offsets[id as usize + 1] as usize);
            Term::Ids(&self.ids[a..b])
        }
    }
}

#[inline(always)]
fn has_bit(bits: &[u64], id: u32) -> bool {
    bits[id as usize / 64] >> (id % 64) & 1 != 0
}

fn intersect(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut out = Vec::with_capacity(small.len());
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
    out
}

fn union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let (x, y) = (a[i], b[j]);
        out.push(x.min(y));
        i += (x <= y) as usize;
        j += (y <= x) as usize;
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

fn difference(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len());
    let mut rest = b;
    for &id in a {
        let at = rest.partition_point(|&x| x < id);
        rest = &rest[at..];
        if rest.first() != Some(&id) {
            out.push(id);
        }
    }
    out
}

fn to_set(term: Term) -> Set {
    match term {
        Term::Ids(ids) => Set::Ids(ids.to_vec()),
        Term::Bits(bits) => Set::Bits(bits.to_vec()),
    }
}

fn apply(set: Set, op: u8, term: Term) -> Set {
    match (set, op, term) {
        (Set::Ids(a), b'A', Term::Ids(b)) => Set::Ids(intersect(&a, b)),
        (Set::Ids(mut a), b'A', Term::Bits(b)) => {
            a.retain(|&id| has_bit(b, id));
            Set::Ids(a)
        }
        (Set::Bits(a), b'A', Term::Ids(b)) => {
            Set::Ids(b.iter().copied().filter(|&id| has_bit(&a, id)).collect())
        }
        (Set::Bits(mut a), b'A', Term::Bits(b)) => {
            a.iter_mut().zip(b).for_each(|(x, y)| *x &= y);
            Set::Bits(a)
        }
        (Set::Ids(a), b'O', Term::Ids(b)) => Set::Ids(union(&a, b)),
        (Set::Ids(a), b'O', Term::Bits(b)) => {
            let mut bits = b.to_vec();
            a.iter().for_each(|&id| bits[id as usize / 64] |= 1 << (id % 64));
            Set::Bits(bits)
        }
        (Set::Bits(mut a), b'O', Term::Ids(b)) => {
            b.iter().for_each(|&id| a[id as usize / 64] |= 1 << (id % 64));
            Set::Bits(a)
        }
        (Set::Bits(mut a), b'O', Term::Bits(b)) => {
            a.iter_mut().zip(b).for_each(|(x, y)| *x |= y);
            Set::Bits(a)
        }
        (Set::Ids(a), _, Term::Ids(b)) => Set::Ids(difference(&a, b)),
        (Set::Ids(mut a), _, Term::Bits(b)) => {
            a.retain(|&id| !has_bit(b, id));
            Set::Ids(a)
        }
        (Set::Bits(mut a), _, Term::Ids(b)) => {
            b.iter().for_each(|&id| a[id as usize / 64] &= !(1 << (id % 64)));
            Set::Bits(a)
        }
        (Set::Bits(mut a), _, Term::Bits(b)) => {
            a.iter_mut().zip(b).for_each(|(x, y)| *x &= !y);
            Set::Bits(a)
        }
    }
}

fn for_each_id(set: &Set, mut visit: impl FnMut(u32)) {
    match set {
        Set::Ids(ids) => ids.iter().for_each(|&id| visit(id)),
        Set::Bits(bits) => {
            for (i, &word) in bits.iter().enumerate() {
                let mut rest = word;
                while rest != 0 {
                    visit(i as u32 * 64 + rest.trailing_zeros());
                    rest &= rest - 1;
                }
            }
        }
    }
}

fn write_ids(out: &mut Vec<u8>, ids: &[u32]) {
    let sum: u64 = ids.iter().map(|&id| id as u64).sum();
    push_int(out, ids.len() as u64);
    out.push(b' ');
    push_int(out, sum);
    for &id in ids.iter().take(10) {
        out.push(b' ');
        push_int(out, id as u64);
    }
    out.push(b'\n');
}

fn write_result(out: &mut Vec<u8>, set: &Set) {
    let (mut count, mut sum) = (0u64, 0u64);
    let mut first = [0u32; 10];
    for_each_id(set, |id| {
        if count < 10 {
            first[count as usize] = id;
        }
        count += 1;
        sum += id as u64;
    });
    push_int(out, count);
    out.push(b' ');
    push_int(out, sum);
    for &id in &first[..count.min(10) as usize] {
        out.push(b' ');
        push_int(out, id as u64);
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

// All phrases are known before any document is read, so they are matched live during tokenizing
// instead of being looked up afterwards. Each phrase is filed under its last two terms. When a
// document's latest two tokens match a filed pair, only the tokens before them need checking. No
// positions are stored, no documents are re-read, and phrase-only terms need no postings.
#[derive(Default)]
struct Phrases {
    list: Vec<Vec<u32>>,
    by_pair: HashMap<u64, Vec<u32>, BuildHasherDefault<PairHasher>>,
    filter: Vec<u64>,
}

#[derive(Default)]
struct PairHasher(u64);

impl Hasher for PairHasher {
    fn write(&mut self, _: &[u8]) {}
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

const FILTER_BITS: u32 = 22;
const RING: usize = 8;

#[inline(always)]
fn pair_key(before: u32, last: u32) -> u64 {
    mum((before as u64) << 32 | last as u64, 0x9E37_79B9_7F4A_7C15) | 1
}

impl Phrases {
    fn build(&mut self) {
        self.filter = vec![0; 1 << (FILTER_BITS - 6)];
        for (index, phrase) in self.list.iter().enumerate() {
            let key = pair_key(phrase[phrase.len() - 2], phrase[phrase.len() - 1]);
            self.by_pair.entry(key).or_default().push(index as u32);
            let bit = key >> (64 - FILTER_BITS);
            self.filter[bit as usize / 64] |= 1 << (bit % 64);
        }
    }

    // `ring` holds the document's latest tokens, and `at` is where the newest one sits.
    #[cold]
    fn check(&self, key: u64, ring: &[u32; RING], at: usize, doc: u32, hits: &mut Vec<(u32, u32)>) {
        let Some(candidates) = self.by_pair.get(&key) else { return };
        for &index in candidates {
            let phrase = &self.list[index as usize];
            let matched = phrase
                .iter()
                .rev()
                .enumerate()
                .all(|(back, &term)| ring[(at + RING - back) % RING] == term);
            if matched {
                hits.push((index, doc));
            }
        }
    }
}

enum Query {
    Bool(Vec<u32>, Vec<u8>),
    Phrase(u32),
}

fn parse_queries(text: &[u8], terms: &mut Terms, phrases: &mut Phrases, in_bool: &mut Vec<bool>) -> Vec<Query> {
    let mut known: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut queries = Vec::with_capacity(text.len() / 32);
    for line in text.split(|&b| b == b'\n').filter(|line| !line.is_empty()) {
        if line[0] == b'B' {
            let (mut ids, mut ops) = (Vec::new(), Vec::new());
            for (i, word) in line[5..].split(|&b| b == b' ').enumerate() {
                if i % 2 == 0 {
                    let id = terms.intern(key_of(word));
                    if in_bool.len() <= id as usize {
                        in_bool.resize(id as usize + 1, false);
                    }
                    in_bool[id as usize] = true;
                    ids.push(id);
                } else {
                    ops.push(word[0]);
                }
            }
            queries.push(Query::Bool(ids, ops));
        } else {
            let inner = &line[8..line.len() - 1];
            let ids: Vec<u32> = inner.split(|&b| b == b' ').map(|word| terms.intern(key_of(word))).collect();
            let index = *known.entry(ids.clone()).or_insert_with(|| {
                phrases.list.push(ids);
                phrases.list.len() as u32 - 1
            });
            queries.push(Query::Phrase(index));
        }
    }
    queries
}

// Estimates which terms are common from the first documents and gives those a bitset. A wrong
// guess only costs space, as both forms hold the same exact answer.
fn choose_routes(input: &[u8], terms: &Terms, in_bool: &[bool], doc_count: usize) -> (Vec<u32>, usize) {
    let sample = doc_count / SAMPLE_SHARE;
    let threshold = (sample / HIGH_SHARE).max(1);
    let mut seen = vec![0u32; terms.keys.len()];
    let mut last = vec![u32::MAX; terms.keys.len()];
    let mut pos = input.as_ptr();
    for _ in 0..sample {
        let (doc, text) = parse_id(pos);
        pos = scan_tokens(text, |key| {
            let term = terms.find(key);
            if term != NO_TERM && last[term as usize] != doc {
                last[term as usize] = doc;
                seen[term as usize] += 1;
            }
        });
    }
    let mut high = 0;
    let routes = seen
        .iter()
        .enumerate()
        .map(|(term, &count)| {
            if !in_bool.get(term).copied().unwrap_or(false) {
                PHRASE_ONLY
            } else if sample > 0 && count as usize > threshold {
                high += 1;
                (high - 1) as u32 | HIGH_BIT
            } else {
                0
            }
        })
        .collect();
    (routes, high)
}

fn doc_count(input: &[u8]) -> usize {
    if input.is_empty() {
        return 0;
    }
    let start = input[..input.len() - 1].iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    parse_id(input[start..].as_ptr()).0 as usize + 1
}

fn parse_docs(
    input: &'static [u8],
    terms: &Terms,
    routes: &[u32],
    bits: &mut [u64],
    words: usize,
    high: usize,
    phrases: &Phrases,
    threads: usize,
) -> (Vec<Vec<Bucket>>, Vec<(u32, u32)>) {
    let next = AtomicUsize::new(0);
    let bits_ptr = Shared(bits.as_mut_ptr());
    let partition_shift = (usize::BITS - terms.keys.len().max(1).leading_zeros()).saturating_sub(PARTITIONS.trailing_zeros());
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let (next, bits_ptr) = (&next, &bits_ptr);
                scope.spawn(move || {
                    let mut buckets: Vec<Bucket> = (0..PARTITIONS).map(|_| Bucket::default()).collect();
                    let mut last = vec![u32::MAX; terms.keys.len()];
                    let mut columns = vec![0u64; high];
                    let mut group = usize::MAX;
                    let mut hits = Vec::new();
                    // Common terms set one bit per token, so 64 documents are gathered in a small
                    // local buffer. Writing the shared bitsets directly would be a cache miss and
                    // an atomic on nearly every token.
                    let flush = |columns: &mut [u64], group: usize| {
                        for (row, column) in columns.iter_mut().enumerate() {
                            if *column != 0 {
                                unsafe {
                                    AtomicU64::from_ptr(bits_ptr.0.add(row * words + group)).fetch_or(*column, Relaxed);
                                }
                                *column = 0;
                            }
                        }
                    };
                    loop {
                        let off = next.fetch_add(CHUNK, Relaxed);
                        if off >= input.len() {
                            if group != usize::MAX {
                                flush(&mut columns, group);
                            }
                            return (buckets, hits);
                        }
                        let first = if off == 0 {
                            0
                        } else {
                            off + input[off - 1..].iter().position(|&b| b == b'\n').unwrap()
                        };
                        let stop = (off + CHUNK).min(input.len());
                        let base = input.as_ptr();
                        let mut i = first;
                        while i < stop {
                            let (doc, text) = parse_id(unsafe { base.add(i) });
                            let mut ring = [NO_TERM; RING];
                            let mut at = 0;
                            if doc as usize / 64 != group {
                                if group != usize::MAX {
                                    flush(&mut columns, group);
                                }
                                group = doc as usize / 64;
                            }
                            let end = scan_tokens(text, |key| {
                                let term = terms.find(key);
                                let before = ring[at];
                                at = (at + 1) % RING;
                                ring[at] = term;
                                if term == NO_TERM {
                                    return;
                                }
                                if before != NO_TERM {
                                    let pair = pair_key(before, term);
                                    let bit = pair >> (64 - FILTER_BITS);
                                    let word = unsafe { *phrases.filter.get_unchecked(bit as usize / 64) };
                                    if word >> (bit % 64) & 1 != 0 {
                                        phrases.check(pair, &ring, at, doc, &mut hits);
                                    }
                                }
                                let route = unsafe { *routes.get_unchecked(term as usize) };
                                if route == PHRASE_ONLY {
                                    return;
                                }
                                if route & HIGH_BIT != 0 {
                                    let at = (route & !HIGH_BIT) as usize;
                                    unsafe { *columns.get_unchecked_mut(at) |= 1u64 << (doc % 64) };
                                } else {
                                    let seen = unsafe { last.get_unchecked_mut(term as usize) };
                                    if *seen != doc {
                                        *seen = doc;
                                        let partition = (term >> partition_shift) as usize;
                                        buckets[partition].push(Hit { term, doc });
                                    }
                                }
                            });
                            i = unsafe { end.offset_from(base) } as usize;
                        }
                    }
                })
            })
            .collect();
        let mut all = Vec::new();
        let mut hits = Vec::new();
        for worker in workers {
            let (buckets, mut found) = worker.join().unwrap();
            all.push(buckets);
            hits.append(&mut found);
        }
        (all, hits)
    })
}

fn build_postings(parsed: Vec<Vec<Bucket>>, term_count: usize, threads: usize) -> (Vec<u32>, Vec<u32>) {
    let mut partitions: Vec<Vec<Bucket>> = (0..PARTITIONS).map(|_| Vec::with_capacity(threads)).collect();
    for buckets in parsed {
        for (p, bucket) in buckets.into_iter().enumerate() {
            partitions[p].push(bucket);
        }
    }
    let shift = (usize::BITS - term_count.max(1).leading_zeros()).saturating_sub(PARTITIONS.trailing_zeros());
    let mut bases = vec![0usize; PARTITIONS + 1];
    for p in 0..PARTITIONS {
        bases[p + 1] = bases[p] + partitions[p].iter().map(|b| b.total).sum::<usize>();
    }
    let tasks: Vec<Mutex<Vec<Bucket>>> = partitions.into_iter().map(Mutex::new).collect();
    let mut offsets = vec![0u32; term_count + 1];
    let mut ids = vec![0u32; bases[PARTITIONS]];
    let offsets_ptr = Shared(offsets.as_mut_ptr());
    let ids_ptr = Shared(ids.as_mut_ptr());
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (next, tasks, bases, offsets_ptr, ids_ptr) = (&next, &tasks, &bases, &offsets_ptr, &ids_ptr);
            scope.spawn(move || {
                let mut cursors = vec![0u32; 1 << shift];
                loop {
                    let p = next.fetch_add(1, Relaxed);
                    if p >= PARTITIONS {
                        return;
                    }
                    let buckets = std::mem::take(&mut *tasks[p].lock().unwrap());
                    cursors.fill(0);
                    let low = (1usize << shift) - 1;
                    for bucket in &buckets {
                        for block in bucket.blocks() {
                            for hit in block {
                                cursors[hit.term as usize & low] += 1;
                            }
                        }
                    }
                    let first = p << shift;
                    let last = ((p + 1) << shift).min(term_count);
                    let mut sum = bases[p] as u32;
                    let mut starts = Vec::with_capacity(last.saturating_sub(first) + 1);
                    // Each term range and id range has one owner: the thread that claimed p.
                    for term in first..last {
                        let count = cursors[term - first];
                        cursors[term - first] = sum;
                        unsafe { *offsets_ptr.0.add(term) = sum };
                        starts.push(sum);
                        sum += count;
                    }
                    starts.push(sum);
                    for bucket in &buckets {
                        for block in bucket.blocks() {
                            for hit in block {
                                let cursor = &mut cursors[hit.term as usize & low];
                                unsafe { *ids_ptr.0.add(*cursor as usize) = hit.doc };
                                *cursor += 1;
                            }
                        }
                    }
                    // Hits arrive in runs per chunk, so each list is nearly sorted already.
                    for pair in starts.windows(2) {
                        let list = unsafe {
                            std::slice::from_raw_parts_mut(ids_ptr.0.add(pair[0] as usize), (pair[1] - pair[0]) as usize)
                        };
                        list.sort_unstable();
                    }
                }
            });
        }
    });
    offsets[term_count] = bases[PARTITIONS] as u32;
    (offsets, ids)
}

fn main() {
    unsafe { mallopt(M_MMAP_THRESHOLD, 1 << 16) };
    let threads = std::thread::available_parallelism().map_or(1, |c| c.get());
    let text = std::fs::read("/data/queries.txt").unwrap();
    let mut terms = Terms::with_capacity(text.len() / 16 + 16);
    let mut phrases = Phrases::default();
    let mut in_bool = Vec::new();
    let queries = parse_queries(&text, &mut terms, &mut phrases, &mut in_bool);
    phrases.build();
    drop(text);

    let file = File::open("/data/docs.txt").unwrap();
    let size = file.metadata().unwrap().len() as usize;
    let input = map_input(&file, size);
    let count = doc_count(input);
    let words = count / 64 + 1;
    let (routes, high) = choose_routes(input, &terms, &in_bool, count);
    let mut bits = vec![0u64; high * words];
    let (parsed, mut hits) = parse_docs(input, &terms, &routes, &mut bits, words, high, &phrases, threads);
    let (offsets, ids) = build_postings(parsed, terms.keys.len(), threads);
    let index = Index { terms, routes, offsets, ids, bits, words };

    // A document can match a phrase more than once, but it counts once.
    hits.sort_unstable();
    hits.dedup();
    let mut matches: Vec<Vec<u32>> = vec![Vec::new(); phrases.list.len()];
    for &(phrase, doc) in &hits {
        matches[phrase as usize].push(doc);
    }

    let batches = queries.len() / QUERY_BATCH + 1;
    let outputs: Vec<Mutex<Vec<u8>>> = (0..batches).map(|_| Mutex::new(Vec::new())).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let batch = next.fetch_add(1, Relaxed);
                    if batch >= batches {
                        return;
                    }
                    let mut out = Vec::with_capacity(QUERY_BATCH * 32);
                    let first = batch * QUERY_BATCH;
                    for query in &queries[first..(first + QUERY_BATCH).min(queries.len())] {
                        match query {
                            Query::Bool(ids, ops) => {
                                let mut set = to_set(index.term(ids[0]));
                                for (&op, &id) in ops.iter().zip(&ids[1..]) {
                                    set = apply(set, op, index.term(id));
                                }
                                write_result(&mut out, &set);
                            }
                            Query::Phrase(phrase) => write_ids(&mut out, &matches[*phrase as usize]),
                        }
                    }
                    *outputs[batch].lock().unwrap() = out;
                }
            });
        }
    });
    let mut all = Vec::new();
    for output in &outputs {
        all.extend_from_slice(&output.lock().unwrap());
    }
    std::fs::write("/out/results.txt", all).unwrap();
    unsafe { exit(0) };
}
