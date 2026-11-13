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
    pub fn gemm(&mut self, m: usize, n: usize, k: usize, x: u64, w: u64, y: u64, out_ty: i32) -> Result<()> {
        let s = self.raw_stream();
        self.lt.gemm(s, m, n, k, x, w, y, out_ty)
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
