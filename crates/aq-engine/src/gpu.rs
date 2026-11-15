//! Device context: one CUDA context + stream, the kernel module, cuBLASLt, raw allocations.

use crate::cublaslt::CublasLt;
use anyhow::{Result, anyhow};
use aq_kernels::kernels::LoadedModule;
use cuda_core::{CudaContext, LaunchConfig, Stream, memory};
use std::ffi::c_void;
use std::fmt::Debug;
use std::sync::Arc;

pub fn de<E: Debug>(e: E) -> anyhow::Error {
    anyhow!("{e:?}")
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
    allocs: Vec<u64>,
    pub allocated_bytes: usize,
    /// Checkpoint copies waiting for `flush_uploads`.
    uploads: Vec<Upload>,
    /// True between `begin_capture` and `end_capture`.
    pub capturing: std::cell::Cell<bool>,
    /// Smallest GEMM M bucket (0 = none). Pinning every GEMM below this size to one plan makes
    /// a row's result independent of how many other rows share the GEMM (used by tests).
    pub gemm_min_m: usize,
}

/// A queued checkpoint copy (`Gpu::queue_upload`).
struct Upload {
    dst: u64,
    src: usize,
    len: usize,
    direct: Option<(std::os::fd::RawFd, u64)>,
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
            allocs: vec![ws],
            uploads: Vec::new(),
            allocated_bytes: WORKSPACE_BYTES,
            capturing: Default::default(),
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
        self.allocs.push(ws);
        self.allocated_bytes += WORKSPACE_BYTES;
        self.streams.push(stream);
        self.workspaces.push(ws);
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
        self.allocs.push(p);
        self.allocated_bytes += bytes;
        Ok(p)
    }

    /// Allocate `bytes` (rounded up to 256) of device memory without zeroing it: for buffers
    /// the caller overwrites completely (weights).
    pub fn alloc_uninit(&mut self, bytes: usize) -> Result<u64> {
        let bytes = bytes.max(1).next_multiple_of(256);
        let p = unsafe { memory::malloc_sync(bytes) }.map_err(de)?;
        self.allocs.push(p);
        self.allocated_bytes += bytes;
        Ok(p)
    }

    /// Queue a host-to-device copy of checkpoint bytes for `flush_uploads`. `direct` is the
    /// shard's direct-I/O handle and the offset of `src` in it; without one the bytes are
    /// copied from `src` (the mapping).
    ///
    /// # Safety
    /// `src` and the file must stay valid until `flush_uploads` returns.
    pub unsafe fn queue_upload(&mut self, dst: u64, src: &[u8], direct: Option<(&std::fs::File, u64)>) {
        use std::os::fd::AsRawFd;
        self.uploads.push(Upload { dst, src: src.as_ptr() as usize, len: src.len(), direct: direct.map(|(f, at)| (f.as_raw_fd(), at)) });
    }

    /// Run the queued copies: reader threads fill page-locked staging buffers in parallel
    /// and this thread copies each filled buffer to the device. With direct I/O the readers
    /// bypass the page cache, which reads an NVMe drive at its full speed whether or not the
    /// file is cached (faulting in a cold mapping gave 1 GB/s on the DGX Spark). A plain
    /// `cuMemcpyHtoD` from pageable memory stages through the driver on one thread, which
    /// measured 0.35-0.5 GB/s there.
    ///
    /// Synchronous copies, not ordered with the compute stream: the caller must have synced
    /// after anything it queued that touches the destinations.
    pub fn flush_uploads(&mut self) -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::mpsc;
        const STAGE_BYTES: usize = 32 << 20;
        const STAGE_BUFS: usize = 12;
        /// Direct reads need block-aligned offsets, lengths and buffers.
        const ALIGN: usize = 4096;
        let uploads = std::mem::take(&mut self.uploads);
        // (upload, offset in it, length)
        let pieces: Vec<(usize, usize, usize)> = uploads
            .iter()
            .enumerate()
            .flat_map(|(u, up)| (0..up.len).step_by(STAGE_BYTES).map(move |off| (u, off, STAGE_BYTES.min(up.len - off))))
            .collect();
        let mut stage = Vec::new();
        for _ in 0..STAGE_BUFS.min(pieces.len()) {
            stage.push(unsafe { memory::malloc_host(STAGE_BYTES + ALIGN) }.map_err(de)? as usize);
        }
        let next = AtomicUsize::new(0);
        // (buffer, piece, offset of the piece's first byte in the buffer or the read error)
        let (tx, rx) = mpsc::channel::<(usize, usize, std::io::Result<usize>)>();
        let (pieces, uploads) = (&pieces, &uploads);
        let res = std::thread::scope(|sc| {
            let mut back = Vec::new();
            for (bi, &buf) in stage.iter().enumerate() {
                let (btx, brx) = mpsc::channel::<()>();
                back.push(btx);
                let tx = tx.clone();
                let next = &next;
                sc.spawn(move || {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(u, off, len)) = pieces.get(i) else { break };
                        let up = &uploads[u];
                        let head = match up.direct {
                            Some((fd, base)) => {
                                let at = base + off as u64;
                                let head = (at % ALIGN as u64) as usize;
                                // SAFETY: `buf` is this thread's own staging buffer of STAGE_BYTES + ALIGN bytes.
                                unsafe { read_at_least(fd, buf as *mut u8, (head + len).next_multiple_of(ALIGN), at - head as u64, head + len) }.map(|()| head)
                            }
                            None => {
                                // SAFETY: as above; the source is valid by `queue_upload`'s contract.
                                unsafe { std::ptr::copy_nonoverlapping((up.src + off) as *const u8, buf as *mut u8, len) };
                                Ok(0)
                            }
                        };
                        // The buffer is reused only after the main thread has copied it out.
                        if tx.send((bi, i, head)).is_err() || brx.recv().is_err() {
                            break;
                        }
                    }
                });
            }
            drop(tx);
            let mut res = Ok(());
            for (bi, i, head) in rx {
                if res.is_ok() {
                    let (u, off, len) = pieces[i];
                    res = match head {
                        Ok(head) => unsafe { memory::memcpy_htod_sync(uploads[u].dst + off as u64, (stage[bi] + head) as *const u8, len) }.map_err(de),
                        Err(e) => Err(anyhow::anyhow!("reading weights: {e}")),
                    };
                }
                let _ = back[bi].send(());
            }
            res
        });
        for p in stage {
            let _ = unsafe { memory::free_host(p as *mut c_void) };
        }
        res
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

    /// `y[m, n] = x[m, k] . w[n, k]^T` (bf16 in, `out_ty` out).
    ///
    /// M is rounded up to a bucket (capped at `cap`, the row capacity of `x` and `y`) so a
    /// handful of tuned plans cover every batch size; the extra rows compute on scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(&mut self, m: usize, cap: usize, n: usize, k: usize, x: u64, w: u64, y: u64, out_ty: i32) -> Result<()> {
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
        self.lt.gemm(s, mb, n, k, x, w, y, out_ty)
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
        for p in self.allocs.drain(..) {
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
