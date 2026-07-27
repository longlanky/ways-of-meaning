//! Text embedding: the model registry, the [`Embedder`] trait, and backends.
//!
//! Everything that touches a model goes through [`Embedder`] so backends stay
//! swappable — that is what makes the bge-small/base/large comparison a config
//! change rather than a rewrite, and what leaves room for a GPU backend later.

pub mod download;
pub mod onnx;

use anyhow::{Result, bail};

/// BGE models are trained asymmetrically: queries carry an instruction prefix
/// and documents do not. Applying it to both, or to neither, measurably degrades
/// retrieval — and does so silently, which is why it lives in the registry
/// rather than at a call site.
pub const BGE_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

/// Fast enough to index a few hundred thousand files, and better than the
/// all-MiniLM-L6-v2 fallback named in the spec (62.2 vs 56.3 MTEB retrieval).
pub const DEFAULT_MODEL: &str = "bge-small-en-v1.5-int8";

/// How token vectors are reduced to one vector per text.
///
/// Getting this wrong is the classic silent failure: mean-pooling a BGE model
/// still produces plausible-looking unit vectors, just worse ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Take `last_hidden_state[:, 0, :]`. What all BGE models use, per their
    /// `1_Pooling/config.json` (`pooling_mode_cls_token: true`).
    Cls,
    /// Attention-mask-weighted mean over tokens. What sentence-transformers
    /// models such as all-MiniLM-L6-v2 use.
    Mean,
}

/// A model we know how to fetch and run.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    pub id: &'static str,
    /// Hugging Face repo, e.g. `BAAI/bge-small-en-v1.5`.
    pub repo: &'static str,
    /// Path of the ONNX graph within the repo.
    pub onnx_path: &'static str,
    /// Path of the tokenizer JSON within the repo.
    pub tokenizer_path: &'static str,
    pub dim: usize,
    /// Positional-embedding limit of the model; inputs are truncated to it.
    pub max_seq: usize,
    pub pooling: Pooling,
    /// Prefix prepended to queries only. Empty for symmetric models.
    pub query_prefix: &'static str,
    /// Approximate download size, for the progress bar and for `wom model list`.
    pub approx_mb: u64,
    /// Starting estimate of end-to-end indexing throughput, used by `wom init`
    /// before anything has been measured. This is whole-scan throughput including
    /// extraction and database writes, not raw model throughput, because that is
    /// what a first-index estimate needs to predict.
    ///
    /// Calibrated on a Ryzen 7 PRO 6850U (8c/16t, no AVX512-VNNI): bge-small-int8
    /// benchmarks at ~123 docs/sec on model alone and sustains ~60 docs/sec over a
    /// real 274k-file scan. The larger models are scaled from that by parameter
    /// count and are the least trustworthy numbers here. Any completed scan or
    /// `wom bench` run overwrites all of this with a measured value in the index
    /// metadata, which `wom init` prefers.
    pub rough_docs_per_sec: u32,
    /// Cosine below which a dense hit is treated as noise.
    ///
    /// A per-model fact for the same reason `pooling` and `query_prefix` are: get
    /// it wrong and nothing errors, results just quietly get worse. Measured on
    /// bge-small, unrelated short texts land at 0.40-0.46 and genuine matches
    /// above 0.67, so 0.55 sits in the gap.
    pub default_min_similarity: f32,
    pub note: &'static str,
}

/// Known models. `bge-*-int8` come from the Xenova mirrors, which publish
/// quantised ONNX exports; the fp32 variants come from BAAI directly.
pub const REGISTRY: &[ModelSpec] = &[
    ModelSpec {
        id: "bge-small-en-v1.5-int8",
        repo: "Xenova/bge-small-en-v1.5",
        onnx_path: "onnx/model_int8.onnx",
        tokenizer_path: "tokenizer.json",
        dim: 384,
        max_seq: 512,
        pooling: Pooling::Cls,
        query_prefix: BGE_QUERY_PREFIX,
        approx_mb: 33,
        rough_docs_per_sec: 60,
        default_min_similarity: 0.55,
        note: "default; measured 1.4x faster than fp32 on this CPU",
    },
    ModelSpec {
        id: "bge-small-en-v1.5",
        repo: "BAAI/bge-small-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        dim: 384,
        max_seq: 512,
        pooling: Pooling::Cls,
        query_prefix: BGE_QUERY_PREFIX,
        approx_mb: 127,
        rough_docs_per_sec: 43,
        default_min_similarity: 0.55,
        note: "fp32 baseline for measuring quantisation loss",
    },
    ModelSpec {
        id: "bge-base-en-v1.5",
        repo: "BAAI/bge-base-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        dim: 768,
        max_seq: 512,
        pooling: Pooling::Cls,
        query_prefix: BGE_QUERY_PREFIX,
        approx_mb: 416,
        rough_docs_per_sec: 16,
        default_min_similarity: 0.55,
        note: "middle ground; ~3x the indexing cost of bge-small",
    },
    ModelSpec {
        id: "bge-large-en-v1.5",
        repo: "BAAI/bge-large-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        dim: 1024,
        max_seq: 512,
        pooling: Pooling::Cls,
        query_prefix: BGE_QUERY_PREFIX,
        approx_mb: 1275,
        rough_docs_per_sec: 5,
        default_min_similarity: 0.55,
        note: "highest MTEB score; ~10x the indexing cost of bge-small",
    },
    ModelSpec {
        id: "all-MiniLM-L6-v2",
        repo: "Xenova/all-MiniLM-L6-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        dim: 384,
        max_seq: 256,
        pooling: Pooling::Mean,
        query_prefix: "",
        approx_mb: 90,
        rough_docs_per_sec: 75,
        default_min_similarity: 0.55,
        note: "the spec's fallback; mean pooling, no query prefix",
    },
];

pub fn lookup(id: &str) -> Result<&'static ModelSpec> {
    match REGISTRY.iter().find(|m| m.id == id) {
        Some(m) => Ok(m),
        None => {
            let known: Vec<&str> = REGISTRY.iter().map(|m| m.id).collect();
            bail!("unknown model `{id}`. Known models: {}", known.join(", "))
        }
    }
}

/// Produces unit-length vectors for documents and queries.
///
/// Implementations must L2-normalise their output: the int8 vector store, the
/// cosine-as-dot-product search, and the directory centroid averaging all assume
/// unit length.
pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    fn model_id(&self) -> &str;

    /// Embed passages. No query prefix is applied.
    fn embed_docs(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Embed a search query, applying the model's query prefix if it has one.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
}

/// Euclidean length.
pub fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Dot product, which equals cosine similarity for unit-length inputs — and
/// everything this crate stores is unit-length by construction.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// L2-normalise in place. A zero vector is left alone rather than producing NaNs
/// — it can legitimately arise from an empty document.
pub fn l2_normalize(v: &mut [f32]) {
    let n = norm(v);
    if n > 1e-12 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_is_registered() {
        assert!(lookup(DEFAULT_MODEL).is_ok());
    }

    #[test]
    fn unknown_model_error_lists_alternatives() {
        let err = lookup("gpt-9").unwrap_err().to_string();
        assert!(err.contains("bge-small-en-v1.5"), "got: {err}");
    }

    #[test]
    fn all_bge_models_use_cls_pooling_and_the_query_prefix() {
        // Guards the two silent-failure modes called out in the plan.
        for m in REGISTRY.iter().filter(|m| m.id.starts_with("bge-")) {
            assert_eq!(m.pooling, Pooling::Cls, "{} pooling", m.id);
            assert_eq!(m.query_prefix, BGE_QUERY_PREFIX, "{} prefix", m.id);
        }
    }

    #[test]
    fn minilm_uses_mean_pooling_and_no_prefix() {
        let m = lookup("all-MiniLM-L6-v2").unwrap();
        assert_eq!(m.pooling, Pooling::Mean);
        assert!(m.query_prefix.is_empty());
    }

    #[test]
    fn registry_ids_are_unique() {
        let mut ids: Vec<&str> = REGISTRY.iter().map(|m| m.id).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate model id in REGISTRY");
    }

    #[test]
    fn l2_normalize_produces_unit_length() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_leaves_zero_vector_finite() {
        let mut v = vec![0.0f32; 4];
        l2_normalize(&mut v);
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
