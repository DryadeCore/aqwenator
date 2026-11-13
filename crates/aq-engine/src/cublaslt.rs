//! Minimal cuBLASLt FFI: just enough for `Y[M, N] = X[M, K] . W[N, K]^T`.
//!
//! Row-major activations and HF-layout weights (`[out, in]`) map to cuBLASLt's
//! column-major "TN" problem: `C(N x M) = op(A)^T . B` with `A = W` (K x N col-major,
//! transposed) and `B = X` (K x M col-major). Plans are cached per shape.
#![allow(non_camel_case_types)]

use anyhow::{Result, bail};
use std::collections::HashMap;
use std::ffi::c_void;

type Handle = *mut c_void;
type MatmulDesc = *mut c_void;
type Layout = *mut c_void;
type Pref = *mut c_void;
type Status = i32;

pub const CUDA_R_32F: i32 = 0;
pub const CUDA_R_16BF: i32 = 14;
pub const CUDA_R_8F_E4M3: i32 = 28;
const CUBLAS_COMPUTE_32F: i32 = 68;
const DESC_TRANSA: i32 = 3;
const DESC_TRANSB: i32 = 4;
const OP_N: i32 = 0;
const OP_T: i32 = 1;
const PREF_MAX_WORKSPACE_BYTES: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct Algo {
    data: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HeuristicResult {
    algo: Algo,
    workspace_size: usize,
    state: Status,
    waves_count: f32,
    reserved: [i32; 4],
}

#[link(name = "cublasLt")]
unsafe extern "C" {
    fn cublasLtCreate(h: *mut Handle) -> Status;
    fn cublasLtMatmulDescCreate(d: *mut MatmulDesc, compute: i32, scale_type: i32) -> Status;
    fn cublasLtMatmulDescSetAttribute(d: MatmulDesc, attr: i32, buf: *const c_void, size: usize) -> Status;
    fn cublasLtMatrixLayoutCreate(l: *mut Layout, ty: i32, rows: u64, cols: u64, ld: i64) -> Status;
    fn cublasLtMatmulPreferenceCreate(p: *mut Pref) -> Status;
    fn cublasLtMatmulPreferenceSetAttribute(p: Pref, attr: i32, buf: *const c_void, size: usize) -> Status;
    fn cublasLtMatmulAlgoGetHeuristic(
        h: Handle,
        d: MatmulDesc,
        a: Layout,
        b: Layout,
        c: Layout,
        dl: Layout,
        p: Pref,
        requested: i32,
        results: *mut HeuristicResult,
        returned: *mut i32,
    ) -> Status;
    fn cublasLtMatmul(
        h: Handle,
        d: MatmulDesc,
        alpha: *const c_void,
        a: *const c_void,
        al: Layout,
        b: *const c_void,
        bl: Layout,
        beta: *const c_void,
        c: *const c_void,
        cl: Layout,
        dd: *mut c_void,
        dl: Layout,
        algo: *const Algo,
        ws: *mut c_void,
        ws_size: usize,
        stream: *mut c_void,
    ) -> Status;
}

fn check(s: Status, what: &str) -> Result<()> {
    if s != 0 {
        bail!("cuBLASLt {what} failed with status {s}");
    }
    Ok(())
}

struct Plan {
    desc: MatmulDesc,
    a: Layout,
    b: Layout,
    c: Layout,
    algo: Algo,
}

pub struct CublasLt {
    handle: Handle,
    workspace: u64,
    workspace_size: usize,
    plans: HashMap<(usize, usize, usize, i32), Plan>,
}

// The handle is only used from the engine thread.
unsafe impl Send for CublasLt {}

impl CublasLt {
    pub fn new(workspace: u64, workspace_size: usize) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        check(unsafe { cublasLtCreate(&mut handle) }, "create")?;
        Ok(Self { handle, workspace, workspace_size, plans: HashMap::new() })
    }

    fn plan(&mut self, m: usize, n: usize, k: usize, out_ty: i32) -> Result<&Plan> {
        let key = (m, n, k, out_ty);
        if !self.plans.contains_key(&key) {
            unsafe {
                let mut desc = std::ptr::null_mut();
                check(cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc")?;
                let (ta, tb) = (OP_T, OP_N);
                check(cublasLtMatmulDescSetAttribute(desc, DESC_TRANSA, (&ta as *const i32).cast(), 4), "transa")?;
                check(cublasLtMatmulDescSetAttribute(desc, DESC_TRANSB, (&tb as *const i32).cast(), 4), "transb")?;
                let (mut a, mut b, mut c) = (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
                // A = W stored as K x N col-major (ld K); B = X as K x M (ld K); C = N x M (ld N).
                check(cublasLtMatrixLayoutCreate(&mut a, CUDA_R_16BF, k as u64, n as u64, k as i64), "layout a")?;
                check(cublasLtMatrixLayoutCreate(&mut b, CUDA_R_16BF, k as u64, m as u64, k as i64), "layout b")?;
                check(cublasLtMatrixLayoutCreate(&mut c, out_ty, n as u64, m as u64, n as i64), "layout c")?;
                let mut pref = std::ptr::null_mut();
                check(cublasLtMatmulPreferenceCreate(&mut pref), "pref")?;
                let ws = self.workspace_size as u64;
                check(
                    cublasLtMatmulPreferenceSetAttribute(pref, PREF_MAX_WORKSPACE_BYTES, (&ws as *const u64).cast(), 8),
                    "pref ws",
                )?;
                let mut res = std::mem::zeroed::<HeuristicResult>();
                let mut got = 0;
                check(
                    cublasLtMatmulAlgoGetHeuristic(self.handle, desc, a, b, c, c, pref, 1, &mut res, &mut got),
                    "heuristic",
                )?;
                if got == 0 {
                    bail!("no cuBLASLt algo for m={m} n={n} k={k}");
                }
                self.plans.insert(key, Plan { desc, a, b, c, algo: res.algo });
            }
        }
        Ok(&self.plans[&key])
    }

    /// `y[m, n] = x[m, k] . w[n, k]^T`, bf16 inputs, f32 accumulate, `out_ty` output.
    pub fn gemm(&mut self, stream: *mut c_void, m: usize, n: usize, k: usize, x: u64, w: u64, y: u64, out_ty: i32) -> Result<()> {
        let (ws, ws_size, handle) = (self.workspace, self.workspace_size, self.handle);
        let p = self.plan(m, n, k, out_ty)?;
        let (alpha, beta) = (1.0f32, 0.0f32);
        let s = unsafe {
            cublasLtMatmul(
                handle,
                p.desc,
                (&alpha as *const f32).cast(),
                w as *const c_void,
                p.a,
                x as *const c_void,
                p.b,
                (&beta as *const f32).cast(),
                y as *const c_void,
                p.c,
                y as *mut c_void,
                p.c,
                &p.algo,
                ws as *mut c_void,
                ws_size,
                stream,
            )
        };
        check(s, "matmul")
    }
}
