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
    pub stream: Arc<Stream>,
    pub k: LoadedModule,
    pub lt: CublasLt,
    allocs: Vec<u64>,
    pub allocated_bytes: usize,
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
        Ok(Self { ctx, stream, k, lt, allocs: vec![ws], allocated_bytes: WORKSPACE_BYTES })
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

    pub fn upload<T>(&self, dst: u64, src: &[T]) -> Result<()> {
        unsafe { memory::memcpy_htod_sync(dst, src.as_ptr(), std::mem::size_of_val(src)) }.map_err(de)
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
}

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
        drv(unsafe { cuStreamBeginCapture_v2(self.raw_stream(), 2) }, "cuStreamBeginCapture")
    }

    /// Stop recording and instantiate the graph; returns the executable graph handle.
    pub fn end_capture(&self) -> Result<u64> {
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

    /// Allow `kernel` to launch with up to `bytes` of dynamic shared memory (needed above 48 KiB;
    /// the raw launch path does not opt in by itself).
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
    names: Vec<&'static str>,
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
        self.events[k].record(&g.stream).map_err(de)?;
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
