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
    fn cuModuleGetFunction(f: *mut *mut c_void, module: *mut c_void, name: *const std::ffi::c_char) -> i32;
    fn cuFuncSetAttribute(f: *mut c_void, attr: i32, value: i32) -> i32;
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

    pub fn launch_graph(&self, exec: u64) -> Result<()> {
        drv(unsafe { cuGraphLaunch(exec as *mut c_void, self.raw_stream()) }, "cuGraphLaunch")
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

/// Per-op wall time, enabled with `AQ_PROFILE=1`: the stream is synchronized at every mark,
/// so this measures each op in isolation (and slows the step down). For ranking, not totals.
#[derive(Default)]
pub struct Prof {
    last: Option<std::time::Instant>,
    pub acc: std::collections::BTreeMap<&'static str, (f64, u64)>,
}

impl Prof {
    pub fn from_env() -> Option<Self> {
        (std::env::var("AQ_PROFILE").as_deref() == Ok("1")).then(Self::default)
    }
    pub fn start(&mut self, g: &Gpu) -> Result<()> {
        g.sync()?;
        self.last = Some(std::time::Instant::now());
        Ok(())
    }
    pub fn mark(&mut self, g: &Gpu, name: &'static str) -> Result<()> {
        g.sync()?;
        let now = std::time::Instant::now();
        if let Some(l) = self.last {
            let e = self.acc.entry(name).or_default();
            e.0 += (now - l).as_secs_f64();
            e.1 += 1;
        }
        self.last = Some(now);
        Ok(())
    }
    pub fn report(&mut self) -> String {
        let total: f64 = self.acc.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = self.acc.iter().collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let mut out = format!("{:<18} {:>10} {:>7} {:>10}\n", "op", "total ms", "share", "us/call");
        for (k, (t, n)) in rows {
            out += &format!("{k:<18} {:>10.1} {:>6.1}% {:>10.1}\n", t * 1e3, 100.0 * t / total, t * 1e6 / *n as f64);
        }
        self.acc.clear();
        out
    }
}
