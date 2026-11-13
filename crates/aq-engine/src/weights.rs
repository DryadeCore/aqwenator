//! Safetensors checkpoint access: mmap every shard, index tensor names once, hand out
//! zero-copy byte views. Host-side conversion only for tiny parameters.

use anyhow::{Context, Result, bail};
use half::bf16;
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct Checkpoint {
    files: Vec<Mmap>,
    /// name -> (file index, dtype, shape, absolute byte range)
    index: HashMap<String, (usize, Dtype, Vec<usize>, usize, usize)>,
}

pub struct View<'a> {
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        paths.sort();
        let mut files = Vec::new();
        let mut index = HashMap::new();
        for (fi, p) in paths.iter().enumerate() {
            let f = std::fs::File::open(p).with_context(|| format!("open {}", p.display()))?;
            let mmap = unsafe { Mmap::map(&f)? };
            let (header_len, meta) = SafeTensors::read_metadata(&mmap)?;
            let base = 8 + header_len;
            for (name, info) in meta.tensors() {
                let (s, e) = info.data_offsets;
                index.insert(name, (fi, info.dtype, info.shape.clone(), base + s, base + e));
            }
            files.push(mmap);
        }
        if files.is_empty() {
            bail!("no .safetensors files in {}", dir.display());
        }
        Ok(Self { files, index })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<View<'_>> {
        let (fi, dtype, shape, s, e) = self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        Ok(View { dtype: *dtype, shape, bytes: &self.files[*fi][*s..*e] })
    }

    /// A bf16 tensor's raw bytes; errors on any other dtype.
    pub fn bf16(&self, name: &str) -> Result<View<'_>> {
        let v = self.get(name)?;
        if v.dtype != Dtype::BF16 {
            bail!("{name}: expected BF16, found {:?} (quantized checkpoints are not wired up yet)", v.dtype);
        }
        Ok(v)
    }

    /// Small parameter converted to f32 on the host (accepts F32 or BF16).
    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>> {
        let v = self.get(name)?;
        Ok(match v.dtype {
            Dtype::F32 => v.bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            Dtype::BF16 => v.bytes.chunks_exact(2).map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
            d => bail!("{name}: unsupported dtype {d:?}"),
        })
    }

    /// Small parameter converted to bf16 bits on the host (accepts F32 or BF16).
    pub fn bf16_vec(&self, name: &str) -> Result<Vec<u16>> {
        Ok(self.f32_vec(name)?.into_iter().map(|x| bf16::from_f32(x).to_bits()).collect())
    }
}

/// Resolve `Qwen/Qwen3.5-0.8B` to its snapshot dir in the local HF cache, or pass a path through.
pub fn resolve_model(spec: &str) -> Result<PathBuf> {
    let p = PathBuf::from(spec);
    if p.join("config.json").exists() {
        return Ok(p);
    }
    let home = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface"));
    let snaps = home.join("hub").join(format!("models--{}", spec.replace('/', "--"))).join("snapshots");
    let snap = std::fs::read_dir(&snaps)
        .with_context(|| format!("{} not in HF cache ({})", spec, snaps.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
        .with_context(|| format!("no complete snapshot for {spec}"))?;
    Ok(snap)
}
