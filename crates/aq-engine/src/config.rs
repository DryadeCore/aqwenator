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
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rope_parameters: RopeParameters,
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
        c.validate()?;
        Ok(c)
    }

    fn validate(&self) -> Result<()> {
        if self.head_dim != 256 {
            bail!("attention kernel assumes head_dim 256, got {}", self.head_dim);
        }
        if self.linear_key_head_dim != 128 || self.linear_value_head_dim != 128 {
            bail!("gated delta rule kernel assumes 128x128 heads");
        }
        if self.linear_conv_kernel_dim != 4 {
            bail!("conv1d kernel assumes width 4");
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
