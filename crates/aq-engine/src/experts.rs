//! 4-bit experts across the tiers: a pool of records on the device, page-locked host memory
//! behind it, and the checkpoint file behind that.
//!
//! A record is one expert: gate/up codes, their scales, down codes, their scales, as
//! `scripts/convert_autoround.py` writes them in `<layer>.mlp.experts.q4r`.
//!
//! - Default: every record on the device.
//! - `AQ_EXPERT_GB=<GiB>` smaller than that: the device holds the least recently used ones
//!   and host memory all of them.
//! - `AQ_EXPERT_RAM_GB=<GiB>` smaller than that too: host memory also holds the least
//!   recently used ones, read from the checkpoint with direct I/O when needed.
//!
//! Outputs do not depend on what the tiers hold.

use crate::config::TextConfig;
use crate::gpu::{Gpu, de, read_at_least};
use crate::weights::Checkpoint;
use anyhow::{Context, Result};
use cuda_core::memory;
use std::os::fd::AsRawFd;
use std::sync::{Arc, Mutex, mpsc};

const NONE: u32 = u32::MAX;
const PENDING: u32 = u32::MAX - 1;
/// Direct reads need block-aligned offsets, lengths and buffers.
const ALIGN: usize = 4096;
const READERS: usize = 8;
const GIB: f64 = (1u64 << 30) as f64;

/// Slots holding the least recently used of a larger set of records.
#[derive(Default)]
struct Lru {
    /// Record in each slot and the tick of its last use.
    owner: Vec<u32>,
    stamp: Vec<u64>,
    /// Slot of each record.
    slot: Vec<u32>,
}

impl Lru {
    fn new(slots: usize, records: usize) -> Self {
        Self { owner: vec![NONE; slots], stamp: vec![0; slots], slot: vec![NONE; records] }
    }

    /// Give each of `ids` (records without a slot) the slot of a least recently used record
    /// not used at `tick`. There must be that many.
    fn place(&mut self, ids: &[usize], tick: u64) -> Vec<u32> {
        let mut free: Vec<u32> = (0..self.owner.len() as u32).filter(|&s| self.stamp[s as usize] != tick).collect();
        if !ids.is_empty() && ids.len() < free.len() {
            free.select_nth_unstable_by_key(ids.len() - 1, |&s| self.stamp[s as usize]);
        }
        free.truncate(ids.len());
        for (&id, &s) in ids.iter().zip(&free) {
            if self.owner[s as usize] != NONE {
                self.slot[self.owner[s as usize] as usize] = NONE;
            }
            self.owner[s as usize] = id as u32;
            self.slot[id] = s;
            self.stamp[s as usize] = tick;
        }
        free
    }
}

struct Read {
    fd: std::os::fd::RawFd,
    at: u64,
    buf: usize,
    cap: usize,
    need: usize,
}

/// Host memory as a cache of records read from the checkpoint.
struct Disk {
    lru: Lru,
    /// Bytes of a host slot: a record rounded up to whole blocks, and one more for the
    /// record's offset in its first block (`head`).
    stride: usize,
    head: Vec<usize>,
    /// Per layer: its shard's direct-I/O handle and the offset of its first record.
    src: Vec<(std::fs::File, u64)>,
    jobs: Option<mpsc::Sender<Read>>,
    done: mpsc::Receiver<std::io::Result<()>>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

pub struct ExpertPool {
    /// Device records.
    pub p: u64,
    /// Byte offsets of a record's four parts, then its size.
    pub off: [usize; 5],
    used: usize,
    /// Empty when every record is on the device.
    dev: Lru,
    /// Page-locked: every record in load order, or `disk`'s slots.
    host: usize,
    disk: Option<Disk>,
    tick: u64,
    /// Since the last report: experts needed, uploaded, read from disk, and forwards.
    stats: [usize; 4],
}

impl ExpertPool {
    /// A pool for the layers (name prefixes) of `ck` that have 4-bit experts.
    pub fn new(gpu: &mut Gpu, ck: &Checkpoint, cfg: &TextConfig, prefixes: &[String]) -> Result<Self> {
        let (h, ei, ne) = (cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts);
        let parts = [ei * h, 2 * ei * (h / 128) * 4, h * ei / 2, h * (ei / 128) * 4];
        let mut off = [0; 5];
        for i in 0..4 {
            off[i + 1] = off[i] + parts[i];
        }
        let rec = off[4].max(1);
        let n = ne * prefixes.iter().filter(|p| ck.has(&format!("{p}mlp.experts.q4r"))).count();
        anyhow::ensure!(n == 0 || (h % 128 == 0 && ei % 128 == 0), "4-bit experts need widths that are multiples of 128");
        let budget = |var: &str, unit: usize| std::env::var(var).ok().and_then(|v| v.parse::<f64>().ok()).map_or(n, |g| (g * GIB) as usize / unit).min(n);
        let cap = budget("AQ_EXPERT_GB", rec);
        let mut pool = Self { p: gpu.alloc_uninit(cap * rec)?, off, used: 0, dev: Lru::default(), host: 0, disk: None, tick: 0, stats: [0; 4] };
        if cap == n {
            return Ok(pool);
        }
        // A layer's experts must all fit in a tier while the layer runs.
        let floor = |c: usize, var: &str, unit: usize| {
            anyhow::ensure!(c >= ne, "{var}: must hold at least one layer's experts ({:.2} GiB)", (ne * unit) as f64 / GIB);
            Ok(())
        };
        floor(cap, "AQ_EXPERT_GB", rec)?;
        pool.dev = Lru::new(cap, n);
        let stride = rec.next_multiple_of(ALIGN) + ALIGN;
        let ram = budget("AQ_EXPERT_RAM_GB", stride);
        let host_bytes = if ram < n {
            floor(ram, "AQ_EXPERT_RAM_GB", stride)?;
            let (jobs, jobs_rx) = mpsc::channel::<Read>();
            let jobs_rx = Arc::new(Mutex::new(jobs_rx));
            let (tx, done) = mpsc::channel();
            let threads = (0..READERS)
                .map(|_| {
                    let (jobs_rx, tx) = (jobs_rx.clone(), tx.clone());
                    std::thread::spawn(move || {
                        loop {
                            let Ok(r) = jobs_rx.lock().unwrap().recv() else { break };
                            // SAFETY: `buf` is a host slot of at least `cap` bytes that `page` gave to this read alone.
                            if tx.send(unsafe { read_at_least(r.fd, r.buf as *mut u8, r.cap, r.at, r.need) }).is_err() {
                                break;
                            }
                        }
                    })
                })
                .collect();
            pool.disk = Some(Disk { lru: Lru::new(ram, n), stride, head: vec![0; ram], src: Vec::new(), jobs: Some(jobs), done, threads });
            ram * stride
        } else {
            n * rec
        };
        pool.host = unsafe { memory::malloc_host(host_bytes) }.map_err(de)? as usize;
        eprintln!(
            "experts: {cap} of {n} on the device ({:.2} GiB), {} in page-locked host memory ({:.2} GiB){}",
            (cap * rec) as f64 / GIB,
            if ram < n { ram.to_string() } else { "all".into() },
            host_bytes as f64 / GIB,
            if ram < n { ", the rest read from the checkpoint" } else { "" }
        );
        Ok(pool)
    }

    pub fn paged(&self) -> bool {
        !self.dev.owner.is_empty()
    }

    /// Take one layer's experts (`<experts>.q4r`) as the next records; the first one's index.
    pub fn load(&mut self, gpu: &mut Gpu, ck: &Checkpoint, ne: usize, experts: &str) -> Result<u32> {
        let v = ck.get(&format!("{experts}.q4r"))?;
        let rec = self.off[4];
        anyhow::ensure!(v.dtype == safetensors::Dtype::U8 && v.bytes.len() == ne * rec, "{experts}.q4r: unexpected layout");
        if let Some(d) = self.disk.as_mut() {
            let (f, at) = v.direct.context("experts read from the checkpoint need direct I/O")?;
            d.src.push((f.try_clone()?, at));
        } else if self.paged() {
            // SAFETY: the host buffer holds every record.
            unsafe { std::ptr::copy_nonoverlapping(v.bytes.as_ptr(), (self.host + self.used * rec) as *mut u8, v.bytes.len()) };
        } else {
            // SAFETY: `Model::load` flushes the queue while the checkpoint is still open.
            unsafe { gpu.queue_upload(self.p + (self.used * rec) as u64, v.bytes, v.direct)? };
        }
        self.used += ne;
        Ok((self.used - ne) as u32)
    }

    /// Bring the experts a layer's tokens selected (`idx`, 0xffff where unused; the layer's
    /// first record is `base`) onto the device over the least recently used ones, and queue
    /// the layer's slot table. Nothing queued may still read the pool.
    pub fn page(&mut self, gpu: &Gpu, base: u32, ne: usize, idx: &[u16], table: u64) -> Result<()> {
        self.tick += 1;
        let (tick, rec) = (self.tick, self.off[4]);
        let mut miss = Vec::new();
        for id in idx.iter().filter(|&&e| e != 0xffff).map(|&e| (base + e as u32) as usize) {
            match self.dev.slot[id] {
                NONE => {
                    self.dev.slot[id] = PENDING;
                    miss.push(id);
                }
                PENDING => {}
                s if self.dev.stamp[s as usize] != tick => {
                    self.dev.stamp[s as usize] = tick;
                    self.stats[0] += 1;
                }
                _ => {}
            }
        }
        self.stats[0] += miss.len();
        self.stats[1] += miss.len();
        if let Some(d) = self.disk.as_mut() {
            // Host memory ranks by use, not by load: otherwise it would hold what the device does.
            for id in idx.iter().filter(|&&e| e != 0xffff).map(|&e| (base + e as u32) as usize) {
                if d.lru.slot[id] != NONE {
                    d.lru.stamp[d.lru.slot[id] as usize] = tick;
                }
            }
            let cold: Vec<usize> = miss.iter().copied().filter(|&id| d.lru.slot[id] == NONE).collect();
            let slots = d.lru.place(&cold, tick);
            for (&id, &s) in cold.iter().zip(&slots) {
                let (f, first) = &d.src[id / ne];
                let at = first + ((id % ne) * rec) as u64;
                let head = (at % ALIGN as u64) as usize;
                d.head[s as usize] = head;
                let read = Read { fd: f.as_raw_fd(), at: at - head as u64, buf: self.host + s as usize * d.stride, cap: (head + rec).next_multiple_of(ALIGN), need: head + rec };
                d.jobs.as_ref().unwrap().send(read).map_err(de)?;
            }
            for _ in &cold {
                d.done.recv().map_err(de)?.context("reading experts")?;
            }
            self.stats[2] += cold.len();
        }
        let slots = self.dev.place(&miss, tick);
        for (&id, &s) in miss.iter().zip(&slots) {
            let src = match &self.disk {
                Some(d) => {
                    let hs = d.lru.slot[id] as usize;
                    self.host + hs * d.stride + d.head[hs]
                }
                None => self.host + id * rec,
            };
            // SAFETY: both ends are whole records; the source is page-locked, and no host slot
            // used at this tick is replaced before the caller's next synchronization.
            unsafe { gpu.upload_raw(self.p + (s as usize * rec) as u64, src as *const u8, rec)? };
        }
        gpu.upload(table, &self.dev.slot[base as usize..][..ne])
    }

    /// Count a forward; every 512, report what the tiers had to move.
    pub fn report(&mut self) {
        self.stats[3] += 1;
        if self.stats[3] == 512 {
            let [need, up, read, n] = self.stats;
            let mib = |c: usize| (c * self.off[4]) as f64 / n as f64 / (1 << 20) as f64;
            eprintln!("experts: {} needed per forward, {:.1}% uploaded ({:.0} MiB per forward), {:.0} MiB per forward read from the checkpoint", need / n, 100.0 * up as f64 / need.max(1) as f64, mib(up), mib(read));
            self.stats = [0; 4];
        }
    }
}

impl Drop for ExpertPool {
    fn drop(&mut self) {
        if let Some(d) = self.disk.as_mut() {
            d.jobs = None;
            for t in d.threads.drain(..) {
                let _ = t.join();
            }
        }
        if self.host != 0 {
            let _ = unsafe { memory::free_host(self.host as *mut std::ffi::c_void) };
        }
    }
}
