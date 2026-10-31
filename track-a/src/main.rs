use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};

const CHUNK: usize = 1 << 20;
const MAX_LINE: usize = 256;
const HOURS: usize = 72;
const TS_LEN: usize = 27;

// Log entry: latency in bits 0..32, hour in 32..39, 5xx flag at 39, local endpoint id from 40.
const HOUR_SHIFT: u32 = 32;
const ERR_BIT: u64 = 1 << 39;
const ID_SHIFT: u32 = 40;

struct Interner {
    slots: Vec<u32>,
    entries: Vec<(u64, u32, u32)>,
    bytes: Vec<u8>,
}

impl Interner {
    fn new() -> Self {
        Self { slots: vec![0; 1 << 13], entries: Vec::new(), bytes: Vec::new() }
    }

    #[inline]
    fn intern(&mut self, key: &[u8]) -> u32 {
        let hash = hash(key);
        let mask = self.slots.len() - 1;
        let mut slot = hash as usize & mask;
        loop {
            let id = self.slots[slot];
            if id == 0 {
                return self.insert(slot, hash, key);
            }
            let (h, off, len) = self.entries[id as usize - 1];
            if h == hash && &self.bytes[off as usize..(off + len) as usize] == key {
                return id - 1;
            }
            slot = (slot + 1) & mask;
        }
    }

    #[cold]
    fn insert(&mut self, slot: usize, hash: u64, key: &[u8]) -> u32 {
        self.entries.push((hash, self.bytes.len() as u32, key.len() as u32));
        self.bytes.extend_from_slice(key);
        self.slots[slot] = self.entries.len() as u32;
        if self.entries.len() * 2 > self.slots.len() {
            self.grow();
        }
        self.entries.len() as u32 - 1
    }

    fn grow(&mut self) {
        let size = self.slots.len() * 2;
        self.slots = vec![0; size];
        for (i, entry) in self.entries.iter().enumerate() {
            let mut slot = entry.0 as usize & (size - 1);
            while self.slots[slot] != 0 {
                slot = (slot + 1) & (size - 1);
            }
            self.slots[slot] = i as u32 + 1;
        }
    }

    fn get(&self, id: usize) -> &[u8] {
        let (_, off, len) = self.entries[id];
        &self.bytes[off as usize..(off + len) as usize]
    }
}

#[inline]
fn hash(key: &[u8]) -> u64 {
    let mut head = [0u8; 8];
    let mut tail = [0u8; 8];
    let n = key.len().min(8);
    head[..n].copy_from_slice(&key[..n]);
    tail[..n].copy_from_slice(&key[key.len() - n..]);
    let mixed = (u64::from_le_bytes(head) ^ (key.len() as u64) << 56)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from_le_bytes(tail).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    mixed ^ mixed >> 29
}

// The spec fixes the span to Oct 30 18:00 .. Nov 2 18:00, so the day digits alone identify the date.
#[inline]
fn hour_index(line: &[u8]) -> u64 {
    let day = (line[8] - b'0') as usize * 10 + (line[9] - b'0') as usize;
    let hour = (line[11] - b'0') as u64 * 10 + (line[12] - b'0') as u64;
    let day_index = if day >= 30 { day - 30 } else { day + 1 } as u64;
    day_index * 24 + hour - 18
}

fn parse_chunk(buf: &[u8], mut i: usize, limit: usize, log: &mut Vec<u64>, names: &mut Interner) {
    while i < limit && i < buf.len() {
        let line = &buf[i..];
        let hour = hour_index(line);
        let mut p = TS_LEN + 1;
        while line[p] != b' ' {
            p += 1;
        }
        let start = p + 1;
        p = start;
        while line[p] != b' ' {
            p += 1;
        }
        let id = names.intern(&line[start..p]) as u64;
        let err = if line[p + 1] == b'5' { ERR_BIT } else { 0 };
        p += 5;
        let mut latency = 0u64;
        while line[p] != b'\n' {
            latency = latency * 10 + (line[p] - b'0') as u64;
            p += 1;
        }
        log.push(latency | hour << HOUR_SHIFT | err | id << ID_SHIFT);
        i += p + 1;
    }
}

fn parse_file(file: &File, size: usize, threads: usize) -> Vec<(Vec<u64>, Interner)> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut buf = vec![0u8; CHUNK + MAX_LINE + 1];
                    let mut log = Vec::with_capacity(size / 77 / threads + 1024);
                    let mut names = Interner::new();
                    loop {
                        let off = next.fetch_add(CHUNK, Relaxed);
                        if off >= size {
                            return (log, names);
                        }
                        let begin = off.saturating_sub(1);
                        let want = (off + CHUNK + MAX_LINE).min(size) - begin;
                        file.read_exact_at(&mut buf[..want], begin as u64).unwrap();
                        let first = if off == 0 {
                            0
                        } else {
                            buf[..want].iter().position(|&b| b == b'\n').unwrap() + 1
                        };
                        parse_chunk(&buf[..want], first, off + CHUNK - begin, &mut log, &mut names);
                    }
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    })
}

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

fn push_hour(out: &mut Vec<u8>, hour: usize) {
    let total = hour + 18;
    let (month, day) = match total / 24 {
        0 => (10, 30),
        1 => (10, 31),
        2 => (11, 1),
        _ => (11, 2),
    };
    let hh = total % 24;
    let two = |v: usize| [b'0' + (v / 10) as u8, b'0' + (v % 10) as u8];
    out.extend_from_slice(b"2026-");
    out.extend_from_slice(&two(month));
    out.push(b'-');
    out.extend_from_slice(&two(day));
    out.push(b'T');
    out.extend_from_slice(&two(hh));
    out.extend_from_slice(b":00:00Z ");
}

struct Shared<T>(*mut T);
unsafe impl<T> Sync for Shared<T> {}

fn main() {
    let file = File::open("/data/requests.log").unwrap();
    let size = file.metadata().unwrap().len() as usize;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let parsed = parse_file(&file, size, threads);

    let mut ranks: HashMap<&[u8], u32> = HashMap::new();
    for (_, names) in &parsed {
        for id in 0..names.entries.len() {
            ranks.entry(names.get(id)).or_insert(0);
        }
    }
    let mut endpoints: Vec<&[u8]> = ranks.keys().copied().collect();
    endpoints.sort_unstable();
    for (rank, name) in endpoints.iter().enumerate() {
        ranks.insert(name, rank as u32);
    }
    let local_ranks: Vec<Vec<u32>> = parsed
        .iter()
        .map(|(_, names)| (0..names.entries.len()).map(|id| ranks[names.get(id)]).collect())
        .collect();

    let width = endpoints.len();
    let groups = HOURS * width;
    let cursor: Vec<AtomicU32> = (0..groups + 1).map(|_| AtomicU32::new(0)).collect();
    let errors: Vec<AtomicU32> = (0..groups).map(|_| AtomicU32::new(0)).collect();
    let group_of = |entry: u64, map: &[u32]| {
        let hour = (entry >> HOUR_SHIFT & 0x7F) as usize;
        hour * width + map[(entry >> ID_SHIFT) as usize] as usize
    };

    std::thread::scope(|scope| {
        for ((log, _), map) in parsed.iter().zip(&local_ranks) {
            let (cursor, errors, group_of) = (&cursor, &errors, &group_of);
            scope.spawn(move || {
                for &entry in log {
                    let group = group_of(entry, map);
                    cursor[group].fetch_add(1, Relaxed);
                    if entry & ERR_BIT != 0 {
                        errors[group].fetch_add(1, Relaxed);
                    }
                }
            });
        }
    });

    let mut starts = vec![0u32; groups + 1];
    let mut total = 0u32;
    for group in 0..groups {
        starts[group] = total;
        total += cursor[group].swap(total, Relaxed);
    }
    starts[groups] = total;

    let mut flat = vec![0u32; total as usize];
    let flat_ptr = Shared(flat.as_mut_ptr());
    std::thread::scope(|scope| {
        for ((log, _), map) in parsed.iter().zip(&local_ranks) {
            let (cursor, group_of, flat_ptr) = (&cursor, &group_of, &flat_ptr);
            scope.spawn(move || {
                for &entry in log {
                    let slot = cursor[group_of(entry, map)].fetch_add(1, Relaxed) as usize;
                    unsafe { *flat_ptr.0.add(slot) = entry as u32 };
                }
            });
        }
    });

    let next_hour = AtomicU64::new(0);
    let mut outputs: Vec<(usize, Vec<u8>)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let (next_hour, starts, errors, endpoints, flat_ptr) =
                    (&next_hour, &starts, &errors, &endpoints, &flat_ptr);
                scope.spawn(move || {
                    let mut done = Vec::new();
                    loop {
                        let hour = next_hour.fetch_add(1, Relaxed) as usize;
                        if hour >= HOURS {
                            return done;
                        }
                        let mut out = Vec::new();
                        for rank in 0..width {
                            let group = hour * width + rank;
                            let (a, b) = (starts[group] as usize, starts[group + 1] as usize);
                            if a == b {
                                continue;
                            }
                            // Groups are disjoint ranges of the flat array, one owner per hour.
                            let values = unsafe {
                                std::slice::from_raw_parts_mut(flat_ptr.0.add(a), b - a)
                            };
                            let n = values.len();
                            let rank_of = |p: usize| (p * n + 99) / 100 - 1;
                            let (r50, r95, r99) = (rank_of(50), rank_of(95), rank_of(99));
                            let (left, &mut p99, _) = values.select_nth_unstable(r99);
                            let (left, p95) = if r95 == r99 {
                                (left, p99)
                            } else {
                                let (l, &mut v, _) = left.select_nth_unstable(r95);
                                (l, v)
                            };
                            let p50 = if r50 == r95 {
                                p95
                            } else {
                                *left.select_nth_unstable(r50).1
                            };
                            let errs = errors[group].load(Relaxed) as u64;
                            push_hour(&mut out, hour);
                            out.extend_from_slice(endpoints[rank]);
                            for value in [n as u64, p50 as u64, p95 as u64, p99 as u64, errs] {
                                out.push(b' ');
                                push_int(&mut out, value);
                            }
                            out.push(b' ');
                            push_int(&mut out, errs * 1_000_000 / n as u64);
                            out.push(b'\n');
                        }
                        done.push((hour, out));
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

unsafe extern "C" {
    #[link_name = "_exit"]
    fn exit(code: i32) -> !;
}
