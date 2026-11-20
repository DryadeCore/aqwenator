//! The n-gram embedding of `qwen4_exp` models: each token reads `heads` rows of a very large
//! table, chosen by hashing it with the one or two tokens before it. The table never goes to
//! the device: a step gathers its tokens' rows on the host.

use crate::config::TextConfig;
use crate::weights::Checkpoint;
use anyhow::{Context, Result};

pub struct Ngram {
    /// Name prefix of the layer's embedding tensors.
    prefix: String,
    /// One multiplier per position of an n-gram, newest token first.
    mult: Vec<i64>,
    /// Per head, its share of the table: size and first row.
    sizes: Vec<i64>,
    offsets: Vec<i64>,
    eos: u32,
    heads_per_ngram: usize,
    /// Rows of one checkpoint shard of the table, or 0 with the sparse table (`ngram.rows`,
    /// `ngram.data`: the rows some prompts need, for tests without the table).
    shard_rows: usize,
    /// bf16 values of a row.
    dim: usize,
}

impl Ngram {
    pub fn open(ck: &Checkpoint, cfg: &TextConfig, prefix: String) -> Result<Self> {
        let ints = |n: &str| -> Result<Vec<i64>> {
            let v = ck.get(&format!("{prefix}{n}"))?;
            anyhow::ensure!(v.dtype == safetensors::Dtype::I64, "{prefix}{n}: expected int64");
            Ok(v.bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect())
        };
        let (mult, sizes, offsets) = (ints("layer_multipliers")?, ints("ngram_heads_vocab_sizes")?, ints("ngram_heads_offsets")?);
        let heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
        anyhow::ensure!(mult.len() == cfg.ngram_size && sizes.len() == heads && offsets.len() == heads && cfg.ngram_size <= 3, "{prefix}: unexpected n-gram layout");
        let shard = ck.get(&format!("{prefix}ngram_embedding.shard_0.weight")).or_else(|_| ck.get("ngram.data")).context("no n-gram table: neither its shards nor the sparse `ngram.data`")?;
        let dense = ck.has(&format!("{prefix}ngram_embedding.shard_0.weight"));
        let dim = shard.shape[1];
        anyhow::ensure!(shard.dtype == safetensors::Dtype::BF16 && dim * heads == cfg.hidden_size, "{prefix}: unexpected n-gram table");
        Ok(Self { prefix, mult, sizes, offsets, eos: cfg.eos(), heads_per_ngram: cfg.heads_per_ngram, shard_rows: if dense { shard.shape[0] } else { 0 }, dim })
    }

    /// The two tokens that precede a sequence's first.
    pub fn start(&self) -> [u32; 2] {
        [self.eos; 2]
    }

    /// Table rows of the token `hist[p]`: per n-gram size, the tokens are mixed (the ones
    /// before it only as far back as no end-of-sequence token lies between) and each head
    /// takes the mix modulo its own size.
    fn rows<'a>(&'a self, hist: &'a [u32], p: usize) -> impl Iterator<Item = i64> + 'a {
        let back = move |s: usize| if hist[p - s..p].contains(&self.eos) { self.eos } else { hist[p - s] };
        let mut mixed = (hist[p] as i64).wrapping_mul(self.mult[0]);
        let per = self.heads_per_ngram;
        (1..self.mult.len()).flat_map(move |s| {
            mixed ^= (back(s) as i64).wrapping_mul(self.mult[s]);
            let m = mixed;
            (0..per).map(move |h| m.rem_euclid(self.sizes[(s - 1) * per + h]) + self.offsets[(s - 1) * per + h])
        })
    }

    /// Append the embeddings of `tokens`, which follow `before`: each one its heads' rows side
    /// by side (`hidden` bf16).
    pub fn gather(&self, ck: &Checkpoint, before: [u32; 2], tokens: &[u32], out: &mut Vec<u8>) -> Result<()> {
        let hist: Vec<u32> = before.iter().chain(tokens).copied().collect();
        let rb = self.dim * 2;
        let sparse = if self.shard_rows == 0 { Some((ck.get("ngram.rows")?.bytes, ck.get("ngram.data")?.bytes)) } else { None };
        for p in 2..hist.len() {
            for row in self.rows(&hist, p) {
                let bytes = match sparse {
                    Some((rows, data)) => {
                        let key = |i: usize| i64::from_le_bytes(rows[i * 8..i * 8 + 8].try_into().unwrap());
                        let (mut lo, mut hi) = (0, rows.len() / 8);
                        while lo < hi {
                            let mid = (lo + hi) / 2;
                            if key(mid) < row { lo = mid + 1 } else { hi = mid }
                        }
                        anyhow::ensure!(lo < rows.len() / 8 && key(lo) == row, "n-gram row {row} is not in the sparse table (token {} at {})", hist[p], p - 2);
                        &data[lo * rb..(lo + 1) * rb]
                    }
                    None => {
                        let (shard, at) = (row as usize / self.shard_rows, row as usize % self.shard_rows);
                        &ck.get(&format!("{}ngram_embedding.shard_{shard}.weight", self.prefix))?.bytes[at * rb..(at + 1) * rb]
                    }
                };
                out.extend_from_slice(bytes);
            }
        }
        Ok(())
    }
}
