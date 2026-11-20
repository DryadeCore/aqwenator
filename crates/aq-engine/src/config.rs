//! `qwen3_5` text config (the `text_config` block of the HF `config.json`).

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    #[serde(default = "one")]
    pub partial_rotary_factor: f64,
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// MLP width; for a mixture-of-experts model, the shared expert's.
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rope_parameters: RopeParameters,
    /// Gate of the gated delta layers' output norm: `silu` unless "sigmoid".
    #[serde(default)]
    pub output_gate_type: Option<String>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct Root {
    text_config: TextConfig,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Linear,
    Full,
}

impl TextConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?;
        let root: Root = serde_json::from_str(&raw).context("parsing config.json")?;
        let mut c = root.text_config;
        c.tie_word_embeddings = c.tie_word_embeddings.or(root.tie_word_embeddings);
        if c.moe() {
            c.intermediate_size = c.shared_expert_intermediate_size;
        }
        c.validate()?;
        Ok(c)
    }

    fn validate(&self) -> Result<()> {
        if self.head_dim != 256 {
            bail!("attention kernel assumes head_dim 256, got {}", self.head_dim);
        }
        let group = self.num_attention_heads / self.num_key_value_heads.max(1);
        if self.num_attention_heads % self.num_key_value_heads.max(1) != 0 || 8 % group != 0 {
            bail!("decode attention kernel needs a GQA group of 1, 2, 4 or 8 query heads per KV head");
        }
        if self.linear_key_head_dim != 128 || self.linear_value_head_dim != 128 {
            bail!("gated delta rule kernel assumes 128x128 heads");
        }
        if self.linear_conv_kernel_dim != 4 {
            bail!("conv1d kernel assumes width 4");
        }
        if self.hidden_size % 8 != 0 || self.hidden_size > 8192 || self.intermediate_size % 8 != 0 || self.vocab_size % 8 != 0 {
            bail!("vectorized kernels need hidden/intermediate/vocab multiples of 8 and hidden <= 8192");
        }
        if self.moe() && (self.num_experts % 32 != 0 || self.num_experts > 512 || self.num_experts_per_tok > 16 || self.moe_intermediate_size % 64 != 0 || self.shared_expert_intermediate_size != self.moe_intermediate_size || self.hidden_size % 128 != 0) {
            bail!("expert kernels assume at most 512 experts in multiples of 32, at most 16 per token, an expert width that is a multiple of 64 and also the shared expert's, and a hidden size that is a multiple of 128");
        }
        if self.intermediate_size == 0 {
            bail!("config has no MLP width");
        }
        if self.layer_types.len() != self.num_hidden_layers {
            bail!("layer_types length mismatch");
        }
        Ok(())
    }

    pub fn kind(&self, layer: usize) -> LayerKind {
        match self.layer_types[layer].as_str() {
            "full_attention" => LayerKind::Full,
            _ => LayerKind::Linear,
        }
    }

    /// Mixture-of-experts MLPs (`qwen3_5_moe`).
    pub fn moe(&self) -> bool {
        self.num_experts > 0
    }

    pub fn tied(&self) -> bool {
        self.tie_word_embeddings.unwrap_or(false)
    }
    pub fn rot_dim(&self) -> usize {
        (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize
    }
    pub fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }
    pub fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    /// Merged GDN input projection width: `[qkv | z | b | a]`.
    pub fn gdn_proj_dim(&self) -> usize {
        self.conv_dim() + self.value_dim() + 2 * self.linear_num_value_heads
    }
    /// Row stride of the merged GDN projection: width rounded up to 256 (zero weight rows),
    /// so the GEMM's N is tile-aligned.
    pub fn gdn_proj_ld(&self) -> usize {
        self.gdn_proj_dim().next_multiple_of(256)
    }
    /// Merged attention projection width: `[q+gate | k | v]`.
    pub fn attn_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim * 2 + 2 * self.num_key_value_heads * self.head_dim
    }
    /// Bytes of recurrent state per sequence (f32 delta-rule state + bf16 conv state).
    pub fn state_bytes_per_seq(&self) -> usize {
        let n_lin = (0..self.num_hidden_layers).filter(|&l| self.kind(l) == LayerKind::Linear).count();
        n_lin
            * (self.linear_num_value_heads * self.linear_key_head_dim * self.linear_value_head_dim * 4
                + self.conv_dim() * (self.linear_conv_kernel_dim - 1) * 2)
    }
}
