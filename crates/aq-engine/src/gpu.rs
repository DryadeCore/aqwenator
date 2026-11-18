//! Device context: one CUDA context + stream, the kernel module, cuBLASLt, raw allocations.

use crate::cublaslt::CublasLt;
use anyhow::{Result, anyhow};
use aq_kernels::kernels::LoadedModule;
use cuda_core::{CudaContext, LaunchConfig, Stream, memory};
use std::ffi::c_void;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, mpsc};

pub fn de<E: Debug>(e: E) -> anyhow::Error {
    anyhow!("{e:?}")
}

/// How a GEMM weight matrix is stored.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum Fmt {
    #[default]
    Bf16,
    /// E4M3 with one power-of-two f32 scale per row; multiplied by E4M3 activations.
    Fp8,
    /// int8 with one bf16 magnitude per block of 32 along a row; expanded to bf16 for the GEMM.
    Int8,
    /// 4-bit codes (two per byte, value = code - 8) with one f32 scale per 128 along a row,
    /// from a pre-quantized checkpoint; expanded to bf16 for the GEMM.
    Int4,
}

/// A GEMM weight matrix `[n, k]` at `p`, with its scales at `scale` (0 for bf16).
#[derive(Clone, Copy, Default)]
pub struct Weight {
    pub p: u64,
    pub scale: u64,
    pub fmt: Fmt,
}

impl Weight {
    pub fn bf16(p: u64) -> Self {
        Self { p, scale: 0, fmt: Fmt::Bf16 }
    }
}

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    /// The active lane's stream; everything enqueued by this struct goes here.
    pub stream: Arc<Stream>,
    /// One stream + cuBLASLt workspace per lane (lane 0 = main; lane 1 = decode, created by
    /// `add_lane` for decode/prefill co-execution).
    streams: Vec<Arc<Stream>>,
    workspaces: Vec<u64>,
    pub lane: usize,
    pub k: LoadedModule,
    pub lt: CublasLt,
    /// (pointer, bytes) of every live allocation.
    allocs: Vec<(u64, usize)>,
    pub allocated_bytes: usize,
    /// The weight loader, between the first `queue_upload` and `flush_uploads`.
    loader: Option<Loader>,
    /// True between `begin_capture` and `end_capture`.
    pub capturing: std::cell::Cell<bool>,
    /// Per lane: E4M3 activations and their per-row scales for the GEMM being issued.
    x8: Vec<(u64, u64)>,
    /// Rows and columns the `x8` buffers hold.
    x8_dims: (usize, usize),
    /// Per lane: bf16 buffer an int8 weight is expanded into for its GEMM, and its size.
    wtmp: Vec<u64>,
    wtmp_bytes: usize,
    /// Per lane, for `gemm_i8w`: the two int8 digit planes of the activations (`[rows][k_max]`
    /// each), then their f32 scales per block of 32 (`[rows][k_max / 32]`), and `k_max`.
    xq: Vec<(u64, usize)>,
    /// AQ_I8_CHUNK: rows of an int8 weight expanded and multiplied at a time (0 = `i8_rows`).
    i8_chunk: usize,
    /// AQ_I8_KERNEL_M: int8 GEMMs of at most this many rows use `gemm_i8w` (0 = never); larger
    /// ones expand the weight and use the bf16 GEMM. On a B200 the kernel wins up to ~128-256
    /// rows (0.25 ms vs 0.35 ms at 128 for a 24576 x 4096 weight).
    i8_kernel_m: usize,
    /// Weights with fewer rows than this take the expansion path: the kernel launches one
    /// block per 128 weight rows, too few to fill the device, while expanding a small weight
    /// is cheap. Measured on a B200 (148 SMs): 6144 rows win, 4096 lose.
    i8_kernel_min_n: usize,
    /// AQ_QSIM diagnostic: 8-bit round trip of (weights, activations); 0 = off, 1 = E4M3,
    /// 2 = int8. Everything still runs as bf16 GEMMs.
    pub qsim: (u32, u32),
    /// Smallest GEMM M bucket (0 = none). Pinning every GEMM below this size to one plan makes
    /// a row's result independent of how many other rows share the GEMM (used by tests).
    pub gemm_min_m: usize,
}

/// One piece (at most `STAGE_BYTES`) of a checkpoint copy.
struct Piece {
    dst: u64,
    src: usize,
    len: usize,
    direct: Option<(std::os::fd::RawFd, u64)>,
}

const STAGE_BYTES: usize = 32 << 20;
const STAGE_BUFS: usize = 12;
/// Direct reads need block-aligned offsets, lengths and buffers.
const ALIGN: usize = 4096;

/// The weight loader: reader threads fill page-locked staging buffers in parallel and the
/// thread that owns the `Gpu` copies each filled buffer to the device. With direct I/O the
/// readers bypass the page cache, which reads an NVMe drive at its full speed whether or
/// not the file is cached (faulting in a cold mapping gave 1 GB/s on the DGX Spark). A
/// plain `cuMemcpyHtoD` from pageable memory stages through the driver on one thread, which
/// measured 0.35-0.5 GB/s there.
struct Loader {
    jobs: Option<mpsc::Sender<Piece>>,
    /// (buffer, piece, offset of the piece's first byte in the buffer or the read error)
    done: mpsc::Receiver<(usize, Piece, std::io::Result<usize>)>,
    /// Hands a buffer back to its reader once it has been copied out.
    back: Vec<mpsc::Sender<()>>,
    stage: Vec<usize>,
    threads: Vec<std::thread::JoinHandle<()>>,
    pending: usize,
}

impl Loader {
    fn new() -> Result<Self> {
        let (jobs, jobs_rx) = mpsc::channel::<Piece>();
        let jobs_rx = Arc::new(Mutex::new(jobs_rx));
        let (tx, done) = mpsc::channel();
        let mut l = Loader { jobs: Some(jobs), done, back: Vec::new(), stage: Vec::new(), threads: Vec::new(), pending: 0 };
        for bi in 0..STAGE_BUFS {
            let buf = unsafe { memory::malloc_host(STAGE_BYTES + ALIGN) }.map_err(de)? as usize;
            l.stage.push(buf);
            let (btx, brx) = mpsc::channel::<()>();
            l.back.push(btx);
            let (jobs_rx, tx) = (jobs_rx.clone(), tx.clone());
            l.threads.push(std::thread::spawn(move || {
                loop {
                    let Ok(p) = jobs_rx.lock().unwrap().recv() else { break };
                    let head = match p.direct {
                        Some((fd, at)) => {
                            let head = (at % ALIGN as u64) as usize;
                            // SAFETY: `buf` is this thread's own staging buffer of STAGE_BYTES + ALIGN bytes.
                            unsafe { read_at_least(fd, buf as *mut u8, (head + p.len).next_multiple_of(ALIGN), at - head as u64, head + p.len) }.map(|()| head)
                        }
                        None => {
                            // SAFETY: as above; the source is valid by `queue_upload`'s contract.
                            unsafe { std::ptr::copy_nonoverlapping(p.src as *const u8, buf as *mut u8, p.len) };
                            Ok(0)
                        }
                    };
                    // The buffer is reused only after the owner has copied it out.
                    if tx.send((bi, p, head)).is_err() || brx.recv().is_err() {
                        break;
                    }
                }
            }));
        }
        Ok(l)
    }

    /// Copy one filled staging buffer to the device and release it to its reader.
    fn copy_out(&mut self, (bi, p, head): (usize, Piece, std::io::Result<usize>)) -> Result<()> {
        let head = head.map_err(|e| anyhow::anyhow!("reading weights: {e}"))?;
        unsafe { memory::memcpy_htod_sync(p.dst, (self.stage[bi] + head) as *const u8, p.len) }.map_err(de)?;
        self.pending -= 1;
        let _ = self.back[bi].send(());
        Ok(())
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.jobs = None;
        self.back.clear();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        for p in self.stage.drain(..) {
            let _ = unsafe { memory::free_host(p as *mut c_void) };
        }
    }
}

/// Read `cap` bytes at file offset `at` into `buf`, until at least `need` of them are there
/// (a direct read at the end of the file returns fewer than the block-rounded length).
///
/// # Safety
/// `buf` must be valid for `cap` bytes.
unsafe fn read_at_least(fd: std::os::fd::RawFd, buf: *mut u8, cap: usize, at: u64, need: usize) -> std::io::Result<()> {
    let mut got = 0;
    while got < need {
        match unsafe { libc::pread(fd, buf.add(got) as *mut c_void, cap - got, (at + got as u64) as libc::off_t) } {
            n if n < 0 => return Err(std::io::Error::last_os_error()),
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => got += n as usize,
        }
    }
    Ok(())
}

pub const WORKSPACE_BYTES: usize = 64 << 20;

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).map_err(de)?;
        ctx.bind_to_thread().map_err(de)?;
        let stream = ctx.new_stream().map_err(de)?;
        let k = aq_kernels::kernels::load(&ctx).map_err(de)?;
        let ws = unsafe { memory::malloc_sync(WORKSPACE_BYTES) }.map_err(de)?;
        let lt = CublasLt::new(ws, WORKSPACE_BYTES)?;
        let me = Self {
            ctx,
            streams: vec![stream.clone()],
            stream,
            workspaces: vec![ws],
            lane: 0,
            k,
            lt,
            allocs: vec![(ws, WORKSPACE_BYTES)],
            loader: None,
            allocated_bytes: WORKSPACE_BYTES,
            capturing: Default::default(),
            x8: Vec::new(),
            x8_dims: (0, 0),
            wtmp: Vec::new(),
            wtmp_bytes: 0,
            xq: Vec::new(),
            i8_chunk: std::env::var("AQ_I8_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            i8_kernel_m: std::env::var("AQ_I8_KERNEL_M").ok().and_then(|v| v.parse().ok()).unwrap_or(96),
            i8_kernel_min_n: std::env::var("AQ_I8_KERNEL_MIN_N").ok().and_then(|v| v.parse().ok()).unwrap_or(6144),
            qsim: {
                // e.g. AQ_QSIM=w:f, a:i, wa:i (f = E4M3, i = int8 per row, b = int8 per block of 32)
                let v = std::env::var("AQ_QSIM").unwrap_or_default();
                let mode = if v.ends_with(":f") { 1 } else if v.ends_with(":i") { 2 } else if v.ends_with(":b") { 3 } else { 0 };
                let who = v.split(':').next().unwrap_or("");
                (if who.contains('w') { mode } else { 0 }, if who.contains('a') { mode } else { 0 })
            },
            // AQ_GEMM_MIN_M: debugging aid, pins small GEMMs to one plan size.
            gemm_min_m: std::env::var("AQ_GEMM_MIN_M").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
        };
        if std::env::var("AQ_KINFO").as_deref() == Ok("1") {
            me.print_kernel_info()?;
        }
        Ok(me)
    }

    /// Add a lane: a new stream at the given scheduling priority (lower = more urgent) with
    /// its own cuBLASLt workspace. Returns the lane index.
    pub fn add_lane(&mut self, priority: i32) -> Result<usize> {
        let stream = self.ctx.new_stream_with_priority(priority).map_err(de)?;
        let ws = unsafe { memory::malloc_sync(WORKSPACE_BYTES) }.map_err(de)?;
        self.allocs.push((ws, WORKSPACE_BYTES));
        self.allocated_bytes += WORKSPACE_BYTES;
        self.streams.push(stream);
        self.workspaces.push(ws);
        if self.x8_dims.0 > 0 {
            let (rows, k) = self.x8_dims;
            let x8 = (self.alloc(rows * k)?, self.alloc(rows * 4)?);
            self.x8.push(x8);
        }
        if self.wtmp_bytes > 0 {
            let w = self.alloc(self.wtmp_bytes)?;
            self.wtmp.push(w);
        }
        if let Some(&(_, k_max)) = self.xq.first() {
            let rows = self.i8_kernel_m.next_multiple_of(16);
            let p = self.alloc(2 * rows * k_max + rows * k_max / 8)?;
            self.xq.push((p, k_max));
        }
        Ok(self.streams.len() - 1)
    }

    /// Highest scheduling priority this device supports (numerically smallest).
    pub fn highest_priority(&self) -> Result<i32> {
        Ok(self.ctx.stream_priority_range().map_err(de)?.greatest())
    }

    /// Lowest scheduling priority this device supports.
    pub fn lowest_priority(&self) -> Result<i32> {
        Ok(self.ctx.stream_priority_range().map_err(de)?.least())
    }

    /// Make `lane` the target of every subsequent enqueue (kernels, copies, GEMMs, graphs).
    pub fn set_lane(&mut self, lane: usize) {
        if lane != self.lane {
            self.stream = self.streams[lane].clone();
            self.lt.set_workspace(self.workspaces[lane]);
            self.lane = lane;
        }
    }

    pub fn raw_stream(&self) -> *mut c_void {
        self.stream.cu_stream() as *mut c_void
    }

    /// Allocate `bytes` (rounded up to 256) of device memory, zeroed.
    pub fn alloc(&mut self, bytes: usize) -> Result<u64> {
        let bytes = bytes.max(1).next_multiple_of(256);
        let p = unsafe { memory::malloc_sync(bytes) }.map_err(de)?;
        unsafe { memory::memset_d8_async(p, 0, bytes, self.stream.cu_stream()) }.map_err(de)?;
        self.allocs.push((p, bytes));
        self.allocated_bytes += bytes;
        Ok(p)
    }

    /// Allocate `bytes` (rounded up to 256) of device memory without zeroing it: for buffers
    /// the caller overwrites completely (weights).
    pub fn alloc_uninit(&mut self, bytes: usize) -> Result<u64> {
        let bytes = bytes.max(1).next_multiple_of(256);
        let p = unsafe { memory::malloc_sync(bytes) }.map_err(de)?;
        self.allocs.push((p, bytes));
        self.allocated_bytes += bytes;
        Ok(p)
    }

    /// Free an allocation made by `alloc` or `alloc_uninit`. Nothing queued may still use it.
    pub fn free(&mut self, p: u64) -> Result<()> {
        let i = self.allocs.iter().position(|a| a.0 == p).ok_or_else(|| anyhow!("free of an unknown allocation"))?;
        self.sync()?;
        let (_, bytes) = self.allocs.swap_remove(i);
        self.allocated_bytes -= bytes;
        unsafe { memory::free_sync(p) }.map_err(de)
    }

    /// Size the per-lane buffers for FP8 GEMMs: activations of up to `rows` rows of `k_max`.
    pub fn enable_fp8(&mut self, rows: usize, k_max: usize) -> Result<()> {
        self.x8_dims = (rows, k_max);
        for _ in self.x8.len()..self.streams.len() {
            let x8 = (self.alloc(rows * k_max)?, self.alloc(rows * 4)?);
            self.x8.push(x8);
        }
        Ok(())
    }

    /// Our own GEMM for int8 weights (`gemm_i8w`): the activations are split into two int8
    /// digit planes, and the weights are read once, as bytes, straight into the integer
    /// tensor cores. `m` must be at most `i8_kernel_m`.
    pub fn gemm_i8w(&mut self, m: usize, n: usize, k: usize, x: u64, w: Weight, y: u64) -> Result<()> {
        let rows = self.i8_kernel_m.next_multiple_of(16);
        let (xh, kmax) = self.xq[self.lane];
        anyhow::ensure!(m <= rows && k <= kmax, "gemm_i8w operand larger than its digit planes");
        let (xl, sx) = (xh + (rows * kmax) as u64, xh + (2 * rows * kmax) as u64);
        let smem = 4 * (8704 + (m.min(128) as u32).div_ceil(16) * 2176);
        let grid = ((n as u32).div_ceil(128), (m as u32).div_ceil(128), 1);
        unsafe {
            self.k.quant_x16(&self.stream, cfg((m as u32, 1, 1), 256), x as *const u16, xh as *mut u8, xl as *mut u8, sx as *mut f32, k as u32).map_err(de)?;
            self.k
                .gemm_i8w(&self.stream, cfg_smem(grid, 256, smem), xh as *const u8, xl as *const u8, sx as *const f32, w.p as *const u8, w.scale as *const u16, y as *mut u16, m as u32, n as u32, k as u32, n as u32)
                .map_err(de)
        }
    }

    /// Rows of an int8 weight expanded and multiplied at a time. Expanding a chunk and
    /// multiplying it right away keeps the bf16 copy within the cache's reach: on the DGX
    /// Spark (9B, 16 x 16k) 1024 rows gave 192 ms between tokens, the whole weight 234 ms,
    /// 512 and 2048 rows less throughput, and uneven chunk sizes 243 ms.
    fn i8_rows(&self) -> usize {
        if self.i8_chunk > 0 { self.i8_chunk } else { 1024 }
    }

    /// Size the per-lane buffer that chunks of int8 weights (of at most `k_max` columns) are
    /// expanded into.
    pub fn enable_int8(&mut self, k_max: usize) -> Result<()> {
        self.wtmp_bytes = self.i8_rows() * k_max * 2;
        for _ in self.wtmp.len()..self.streams.len() {
            let w = self.alloc(self.wtmp_bytes)?;
            self.wtmp.push(w);
        }
        if self.i8_kernel_m > 0 {
            self.opt_in_dynamic_smem("gemm_i8w", 4 * (8704 + 8 * 2176))?;
            let rows = self.i8_kernel_m.next_multiple_of(16);
            for _ in self.xq.len()..self.streams.len() {
                let p = self.alloc(2 * rows * k_max + rows * k_max / 8)?;
                self.xq.push((p, k_max));
            }
        }
        Ok(())
    }

    /// Replace the bf16 weight matrix at `w` (`[n, k]`, fully uploaded) by its `fmt` form and
    /// free the bf16 copy unless `keep`.
    pub fn quantize_weight(&mut self, w: u64, n: usize, k: usize, fmt: Fmt, keep: bool) -> Result<Weight> {
        let q = self.alloc_uninit(n * k)?;
        let scale = match fmt {
            Fmt::Fp8 => {
                let scale = self.alloc_uninit(n * 4)?;
                unsafe { self.k.quant_e4m3(&self.stream, cfg((n as u32, 1, 1), 256), w as *const u16, q as *mut u8, scale as *mut f32, k as u32) }.map_err(de)?;
                scale
            }
            Fmt::Int8 => {
                anyhow::ensure!(k % 32 == 0, "int8 weights need K to be a multiple of 32");
                let scale = self.alloc_uninit(n * k / 32 * 2)?;
                unsafe { self.k.quant_i8b(&self.stream, cfg((n as u32, 1, 1), 256), w as *const u16, q as *mut u8, scale as *mut u16, k as u32) }.map_err(de)?;
                scale
            }
            Fmt::Bf16 | Fmt::Int4 => anyhow::bail!("quantize_weight to {fmt:?}"),
        };
        if !keep {
            self.free(w)?;
        }
        Ok(Weight { p: q, scale, fmt })
    }

    /// Start a host-to-device copy of checkpoint bytes on the weight loader; `flush_uploads`
    /// completes it. `direct` is the shard's direct-I/O handle and the offset of `src` in
    /// it; without one the bytes are copied from `src` (the mapping). Buffers that readers
    /// have already filled are copied to the device here, so reading overlaps with the
    /// caller's device allocations.
    ///
    /// Synchronous copies, not ordered with the compute stream: nothing queued there may
    /// touch `dst` before `flush_uploads`.
    ///
    /// # Safety
    /// `src` and the file must stay valid until `flush_uploads` returns.
    pub unsafe fn queue_upload(&mut self, dst: u64, src: &[u8], direct: Option<(&std::fs::File, u64)>) -> Result<()> {
        use std::os::fd::AsRawFd;
        if self.loader.is_none() {
            self.loader = Some(Loader::new()?);
        }
        let l = self.loader.as_mut().unwrap();
        for off in (0..src.len()).step_by(STAGE_BYTES) {
            let piece = Piece {
                dst: dst + off as u64,
                src: src.as_ptr() as usize + off,
                len: STAGE_BYTES.min(src.len() - off),
                direct: direct.map(|(f, at)| (f.as_raw_fd(), at + off as u64)),
            };
            l.jobs.as_ref().unwrap().send(piece).map_err(de)?;
            l.pending += 1;
        }
        while let Ok(filled) = l.done.try_recv() {
            l.copy_out(filled)?;
        }
        Ok(())
    }

    /// Wait for every checkpoint copy queued so far; the weight loader stays up.
    pub fn drain_uploads(&mut self) -> Result<()> {
        let Some(l) = self.loader.as_mut() else { return Ok(()) };
        while l.pending > 0 {
            let filled = l.done.recv().map_err(de)?;
            l.copy_out(filled)?;
        }
        Ok(())
    }

    /// Wait for every queued checkpoint copy and stop the weight loader.
    pub fn flush_uploads(&mut self) -> Result<()> {
        let Some(mut l) = self.loader.take() else { return Ok(()) };
        while l.pending > 0 {
            let filled = l.done.recv().map_err(de)?;
            l.copy_out(filled)?;
        }
        Ok(())
    }

    /// Host-to-device copy, enqueued on the compute stream so it is ordered after every kernel
    /// already queued there (a synchronous copy on the legacy default stream is not, and once
    /// overwrote a step's metadata while that step's kernels were still reading it). The
    /// source is pageable, so the driver stages it before returning and `src` may be dropped.
    pub fn upload<T>(&self, dst: u64, src: &[T]) -> Result<()> {
        unsafe { memory::memcpy_htod_async(dst, src.as_ptr(), std::mem::size_of_val(src), self.stream.cu_stream()) }.map_err(de)
    }

    pub fn alloc_upload<T>(&mut self, src: &[T]) -> Result<u64> {
        let p = self.alloc(std::mem::size_of_val(src))?;
        self.sync()?;
        self.upload(p, src)?;
        Ok(p)
    }

    pub fn download<T: Copy + Default>(&self, src: u64, n: usize) -> Result<Vec<T>> {
        let mut out = vec![T::default(); n];
        unsafe {
            memory::memcpy_dtoh_async(out.as_mut_ptr(), src, n * std::mem::size_of::<T>(), self.stream.cu_stream())
        }
        .map_err(de)?;
        self.sync()?;
        Ok(out)
    }

    pub fn zero(&self, ptr: u64, bytes: usize) -> Result<()> {
        unsafe { memory::memset_d8_async(ptr, 0, bytes, self.stream.cu_stream()) }.map_err(de)
    }

    pub fn sync(&self) -> Result<()> {
        self.stream.synchronize().map_err(de)
    }

    /// `y[m, n] = x[m, k] . w[n, k]^T` (bf16 `x`, `out_ty` out). With FP8 weights, `x` is
    /// quantized per row first and `y` (bf16 only) rescaled after: not exact.
    ///
    /// M is rounded up to a bucket (capped at `cap`, the row capacity of `x` and `y`) so a
    /// handful of tuned plans cover every batch size; the extra rows compute on scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(&mut self, m: usize, cap: usize, n: usize, k: usize, x: u64, w: Weight, y: u64, out_ty: i32) -> Result<()> {
        let mb = if m <= 256 {
            m.next_multiple_of(16)
        } else if m <= 2048 {
            m.next_multiple_of(128)
        } else {
            m.next_multiple_of(512)
        }
        .max(self.gemm_min_m)
        .min(cap)
        .max(m);
        let s = self.raw_stream();
        if w.fmt == Fmt::Int4 {
            // Same scheme as the int8 expansion below, from a quarter of the bf16 bytes.
            anyhow::ensure!(out_ty == crate::cublaslt::CUDA_R_16BF, "int4 weights write bf16");
            let tmp = self.wtmp[self.lane];
            let nc = self.i8_rows().min(n);
            anyhow::ensure!(nc * k * 2 <= self.wtmp_bytes, "int4 weight chunk larger than its expansion buffer");
            for c in (0..n).step_by(nc) {
                let rows = nc.min(n - c);
                unsafe {
                    let blocks = (rows * k / 32) as u32;
                    self.k.dequant_i4g(&self.stream, cfg((blocks.div_ceil(256), 1, 1), 256), (w.p + (c * k / 2) as u64) as *const u8, (w.scale + (c * k / 128 * 4) as u64) as *const f32, tmp as *mut u16, blocks)
                }
                .map_err(de)?;
                self.lt.gemm_rows(s, mb, rows, k, x, tmp, y + (c * 2) as u64, n)?;
            }
            return Ok(());
        }
        if w.fmt == Fmt::Int8 {
            // Expand to bf16 (exactly what the weight stands for), then the bf16 GEMM, in
            // chunks of rows small enough for the expansion to stay in cache.
            anyhow::ensure!(out_ty == crate::cublaslt::CUDA_R_16BF, "int8 weights write bf16");
            if m <= self.i8_kernel_m && n >= self.i8_kernel_min_n && k % 64 == 0 && n % 8 == 0 {
                return self.gemm_i8w(m, n, k, x, w, y);
            }
            let tmp = self.wtmp[self.lane];
            let nc = self.i8_rows().min(n);
            anyhow::ensure!(nc * k * 2 <= self.wtmp_bytes, "int8 weight chunk larger than its expansion buffer");
            for c in (0..n).step_by(nc) {
                let rows = nc.min(n - c);
                unsafe {
                    let blocks = (rows * k / 32) as u32;
                    self.k.dequant_i8b(&self.stream, cfg((blocks.div_ceil(256), 1, 1), 256), (w.p + (c * k) as u64) as *const u8, (w.scale + (c * k / 32 * 2) as u64) as *const u16, tmp as *mut u16, blocks)
                }
                .map_err(de)?;
                self.lt.gemm_rows(s, mb, rows, k, x, tmp, y + (c * 2) as u64, n)?;
            }
            return Ok(());
        }
        if w.fmt == Fmt::Bf16 {
            if self.qsim.1 > 0 {
                unsafe { self.k.qsim(&self.stream, cfg((mb as u32, 1, 1), 256), x as *mut u16, k as u32, self.qsim.1) }.map_err(de)?;
            }
            return self.lt.gemm(s, mb, n, k, x, w.p, y, out_ty);
        }
        let (w8, w_scale) = (w.p, w.scale);
        let (x8, x_scale) = self.x8[self.lane];
        unsafe { self.k.quant_e4m3(&self.stream, cfg((mb as u32, 1, 1), 256), x as *const u16, x8 as *mut u8, x_scale as *mut f32, k as u32) }.map_err(de)?;
        anyhow::ensure!(out_ty == crate::cublaslt::CUDA_R_16BF, "FP8 GEMMs write bf16");
        self.lt.gemm_fp8(s, mb, n, k, x8, w8, y, out_ty)?;
        // y[r, c] *= x_scale[r] * w_scale[c]: powers of two, so exact.
        let (xs, wsc) = (x_scale as *const f32, w_scale as *const f32);
        unsafe { self.k.scale_outer(&self.stream, cfg((mb as u32, 1, 1), 256), y as *mut u16, xs, wsc, n as u32) }.map_err(de)
    }
}

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuStreamBeginCapture_v2(stream: *mut c_void, mode: u32) -> i32;
    fn cuStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> i32;
    fn cuGraphInstantiateWithFlags(exec: *mut *mut c_void, graph: *mut c_void, flags: u64) -> i32;
    fn cuGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> i32;
    fn cuGraphDestroy(graph: *mut c_void) -> i32;
    fn cuDeviceGetAttribute(v: *mut i32, attr: i32, dev: i32) -> i32;
    fn cuModuleGetFunction(f: *mut *mut c_void, module: *mut c_void, name: *const std::ffi::c_char) -> i32;
    fn cuFuncSetAttribute(f: *mut c_void, attr: i32, value: i32) -> i32;
    fn cuFuncGetAttribute(value: *mut i32, attr: i32, f: *mut c_void) -> i32;
    fn cuModuleGetFunctionCount(count: *mut u32, module: *mut c_void) -> i32;
    fn cuModuleEnumerateFunctions(fns: *mut *mut c_void, n: u32, module: *mut c_void) -> i32;
    fn cuFuncGetName(name: *mut *const std::ffi::c_char, f: *mut c_void) -> i32;
    fn cuEventRecordWithFlags(event: *mut c_void, stream: *mut c_void, flags: u32) -> i32;
}

/// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
const MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;

fn drv(r: i32, what: &str) -> Result<()> {
    if r != 0 {
        anyhow::bail!("{what} failed: CUresult {r}");
    }
    Ok(())
}

impl Gpu {
    /// Start recording everything enqueued on the stream (relaxed mode, so cuBLASLt's
    /// internal queries are allowed).
    pub fn begin_capture(&self) -> Result<()> {
        self.capturing.set(true);
        drv(unsafe { cuStreamBeginCapture_v2(self.raw_stream(), 2) }, "cuStreamBeginCapture")
    }

    /// Stop recording and instantiate the graph; returns the executable graph handle.
    pub fn end_capture(&self) -> Result<u64> {
        self.capturing.set(false);
        let mut graph = std::ptr::null_mut();
        drv(unsafe { cuStreamEndCapture(self.raw_stream(), &mut graph) }, "cuStreamEndCapture")?;
        let mut exec = std::ptr::null_mut();
        drv(unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) }, "cuGraphInstantiate")?;
        drv(unsafe { cuGraphDestroy(graph) }, "cuGraphDestroy")?;
        Ok(exec as u64)
    }

    /// Number of streaming multiprocessors (CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).
    pub fn num_sms(&self) -> Result<usize> {
        let mut v = 0i32;
        drv(unsafe { cuDeviceGetAttribute(&mut v, 16, self.ctx.cu_device() as i32) }, "cuDeviceGetAttribute")?;
        Ok(v as usize)
    }

    /// Most dynamic shared memory one block may opt in to, in bytes
    /// (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN).
    pub fn max_smem_optin(&self) -> Result<usize> {
        let mut v = 0i32;
        drv(unsafe { cuDeviceGetAttribute(&mut v, 97, self.ctx.cu_device() as i32) }, "cuDeviceGetAttribute")?;
        Ok(v as usize)
    }

    pub fn launch_graph(&self, exec: u64) -> Result<()> {
        drv(unsafe { cuGraphLaunch(exec as *mut c_void, self.raw_stream()) }, "cuGraphLaunch")
    }

    /// `AQ_KINFO=1`: print registers / spills / static smem of every kernel in the module.
    pub fn print_kernel_info(&self) -> Result<()> {
        let module = self.k.as_cuda_module().cu_module() as *mut c_void;
        let mut n = 0u32;
        drv(unsafe { cuModuleGetFunctionCount(&mut n, module) }, "cuModuleGetFunctionCount")?;
        let mut fns = vec![std::ptr::null_mut(); n as usize];
        drv(unsafe { cuModuleEnumerateFunctions(fns.as_mut_ptr(), n, module) }, "cuModuleEnumerateFunctions")?;
        let mut rows = Vec::new();
        for f in fns {
            let mut name = std::ptr::null();
            drv(unsafe { cuFuncGetName(&mut name, f) }, "cuFuncGetName")?;
            let get = |a: i32| {
                let mut v = 0;
                unsafe { cuFuncGetAttribute(&mut v, a, f) };
                v
            };
            let name = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy().into_owned();
            rows.push((name, get(4), get(3), get(1), get(0)));
        }
        rows.sort();
        for (name, regs, local, smem, maxthr) in rows {
            eprintln!("kinfo {name:<28} regs {regs:>3}  local {local:>6} B  static smem {smem:>6} B  max threads {maxthr}");
        }
        Ok(())
    }

    pub fn opt_in_dynamic_smem(&self, kernel: &str, bytes: u32) -> Result<()> {
        const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
        let name = std::ffi::CString::new(kernel)?;
        let mut f = std::ptr::null_mut();
        let module = self.k.as_cuda_module().cu_module() as *mut c_void;
        drv(unsafe { cuModuleGetFunction(&mut f, module, name.as_ptr()) }, "cuModuleGetFunction")?;
        if std::env::var("AQ_KINFO").as_deref() == Ok("1") {
            // CU_FUNC_ATTRIBUTE_*: 1 static smem, 3 local (spill) bytes, 4 registers per thread.
            let get = |a: i32| {
                let mut v = 0;
                unsafe { cuFuncGetAttribute(&mut v, a, f) };
                v
            };
            eprintln!("kernel {kernel}: {} regs/thread, {} B local, {} B static smem, {bytes} B dynamic smem", get(4), get(3), get(1));
        }
        drv(
            unsafe { cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as i32) },
            "cuFuncSetAttribute",
        )
    }

    /// Opt a kernel into more than 48 KiB of dynamic shared memory (raw launches do not go
    /// through cuda-oxide's launch contracts, which would otherwise do this).
    pub fn set_max_dynamic_smem(&self, kernel: &str, bytes: u32) -> Result<()> {
        let module = self.k.as_cuda_module().cu_module() as *mut c_void;
        let name = std::ffi::CString::new(kernel)?;
        let mut f = std::ptr::null_mut();
        drv(unsafe { cuModuleGetFunction(&mut f, module, name.as_ptr()) }, "cuModuleGetFunction")?;
        drv(unsafe { cuFuncSetAttribute(f, MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as i32) }, "cuFuncSetAttribute")
    }
}

/// Launch config with dynamic shared memory.
pub fn cfg_smem(grid: (u32, u32, u32), block: u32, smem: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: grid, block_dim: (block, 1, 1), shared_mem_bytes: smem }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let _ = self.sync();
        for (p, _) in self.allocs.drain(..) {
            let _ = unsafe { memory::free_sync(p) };
        }
    }
}

pub fn cfg(grid: (u32, u32, u32), block: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: grid, block_dim: (block, 1, 1), shared_mem_bytes: 0 }
}

/// Per-op GPU time, enabled with `AQ_PROFILE=1|decode|prefill`. CUDA events are recorded on
/// the stream between ops (no host sync inside the step), and read back once per step, so the
/// numbers are the GPU-side duration of each op plus any launch gap before it. Graphs are
/// disabled while profiling, so small ops include their launch latency.
#[derive(Default)]
pub struct Prof {
    /// Which steps to time: 0 = all, 1 = pure-decode steps only, 2 = steps with prefill only.
    pub filter: u8,
    pub active: bool,
    events: Vec<cuda_core::Event>,
    pub names: Vec<&'static str>,
    pub acc: std::collections::BTreeMap<&'static str, (f64, u64)>,
    pub steps: u64,
}

impl Prof {
    pub fn from_env() -> Option<Self> {
        let filter = match std::env::var("AQ_PROFILE").as_deref() {
            Ok("1") => 0,
            Ok("decode") => 1,
            Ok("prefill") => 2,
            _ => return None,
        };
        Some(Self { filter, ..Self::default() })
    }
    fn record(&mut self, g: &Gpu, name: &'static str) -> Result<()> {
        let k = self.names.len();
        if self.events.len() <= k {
            self.events.push(g.ctx.new_event().map_err(de)?);
        }
        // CU_EVENT_RECORD_EXTERNAL: under stream capture this becomes a timing node of the
        // graph (a plain record would only be a dependency edge).
        drv(
            unsafe { cuEventRecordWithFlags(self.events[k].cu_event() as *mut c_void, g.raw_stream(), g.capturing.get() as u32) },
            "cuEventRecordWithFlags",
        )?;
        self.names.push(name);
        Ok(())
    }
    pub fn start(&mut self, g: &Gpu) -> Result<()> {
        self.names.clear();
        if self.active {
            self.record(g, "")?;
        }
        Ok(())
    }
    pub fn mark(&mut self, g: &Gpu, name: &'static str) -> Result<()> {
        if self.active {
            self.record(g, name)?;
        }
        Ok(())
    }
    /// Read this step's events back (one sync) and accumulate.
    pub fn finish(&mut self) -> Result<()> {
        if !self.active || self.names.len() < 2 {
            return Ok(());
        }
        self.events[self.names.len() - 1].synchronize().map_err(de)?;
        for k in 1..self.names.len() {
            let ms = self.events[k - 1].elapsed_time(&self.events[k]).map_err(de)? as f64;
            let e = self.acc.entry(self.names[k]).or_default();
            e.0 += ms / 1e3;
            e.1 += 1;
        }
        self.steps += 1;
        self.names.clear();
        Ok(())
    }
    pub fn report(&mut self) -> String {
        let total: f64 = self.acc.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = self.acc.iter().collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let steps = self.steps.max(1) as f64;
        let mut out = format!(
            "{:<18} {:>10} {:>7} {:>10} {:>12}   ({} steps, {:.3} ms/step)\n",
            "op", "total ms", "share", "us/call", "us/step", self.steps, total * 1e3 / steps
        );
        for (k, (t, n)) in rows {
            out += &format!(
                "{k:<18} {:>10.1} {:>6.1}% {:>10.1} {:>12.1}\n",
                t * 1e3,
                100.0 * t / total,
                t * 1e6 / *n as f64,
                t * 1e6 / steps
            );
        }
        self.acc.clear();
        self.steps = 0;
        out
    }
}
