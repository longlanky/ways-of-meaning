//! Cross-encoder reranking: a second, more expensive opinion on the top fused
//! candidates.
//!
//! The bi-encoder (embedding model) scores query and document independently,
//! which is what makes 274k-file search fast — and why it is weak on nuance. A
//! cross-encoder sees the (query, document) pair together and is far more
//! accurate per pair, at far more cost per pair. The division of labour: RRF
//! over the five arms nominates ~100 candidates, the cross-encoder re-orders
//! them, and the directory cap picks the final list from that ordering.

use crate::config::Paths;
use crate::embed::{download, onnx};
use anyhow::{Result, anyhow, bail};
use ort::value::Tensor;
use std::sync::Mutex;
use tokenizers::{EncodeInput, Tokenizer};

/// The reranker used by `--rerank` when the config does not name one.
pub const DEFAULT_RERANKER: &str = "ms-marco-MiniLM-L-6-v2-int8";

/// Pairs per session run. Reranking happens once per committed query over ~100
/// pairs, so this only bounds activation memory, not throughput.
const MICRO_BATCH: usize = 16;

/// Characters of a document the reranker ever sees. The signal for relevance
/// is concentrated in the head — the same assumption the embedding profiles
/// make — and pair length dominates cost: measured on a Ryzen 7 PRO 6850U,
/// 80 pairs at a 2000-char cap took ~4.5 s; at 600 chars (~160 tokens per
/// pair) the same pool takes well under a second.
const DOC_CHAR_CAP: usize = 600;

/// A cross-encoder we know how to fetch and run.
#[derive(Debug, Clone, Copy)]
pub struct RerankerSpec {
    pub id: &'static str,
    /// Hugging Face repo.
    pub repo: &'static str,
    /// Path of the ONNX graph within the repo.
    pub onnx_path: &'static str,
    /// Path of the tokenizer JSON within the repo.
    pub tokenizer_path: &'static str,
    /// Pinned blake3 hex digests (see `ModelSpec::onnx_hash`).
    pub onnx_hash: Option<&'static str>,
    pub tokenizer_hash: Option<&'static str>,
    /// Pairs are truncated to this many tokens (the model's own limit).
    pub max_seq: usize,
    /// Approximate download size, for the progress bar.
    pub approx_mb: u64,
    pub note: &'static str,
}

/// Known rerankers. `bge-reranker-v2-m3` is deliberately absent: neither BAAI
/// nor a community mirror publishes a single-file ONNX export of it, and this
/// loader fetches exactly two files per model.
pub const RERANKER_REGISTRY: &[RerankerSpec] = &[
    RerankerSpec {
        id: "ms-marco-MiniLM-L-6-v2-int8",
        repo: "Xenova/ms-marco-MiniLM-L-6-v2",
        onnx_path: "onnx/model_int8.onnx",
        tokenizer_path: "tokenizer.json",
        onnx_hash: None,
        tokenizer_hash: None,
        max_seq: 512,
        approx_mb: 22,
        note: "default; tiny, English, trained on MS MARCO relevance judgements",
    },
    RerankerSpec {
        id: "jina-reranker-v2-base-multilingual-int8",
        repo: "jinaai/jina-reranker-v2-base-multilingual",
        onnx_path: "onnx/model_int8.onnx",
        tokenizer_path: "tokenizer.json",
        onnx_hash: None,
        tokenizer_hash: None,
        max_seq: 1024,
        approx_mb: 267,
        note: "multilingual; ~12x the size and per-query cost of the default",
    },
];

pub fn lookup(id: &str) -> Result<&'static RerankerSpec> {
    match RERANKER_REGISTRY.iter().find(|m| m.id == id) {
        Some(m) => Ok(m),
        None => {
            let known: Vec<&str> = RERANKER_REGISTRY.iter().map(|m| m.id).collect();
            bail!("unknown reranker `{id}`. Known rerankers: {}", known.join(", "))
        }
    }
}

/// Scores (query, document) pairs. Higher means more relevant; the values are
/// raw logits and are only comparable within the same model.
pub trait Reranker: Send + Sync {
    fn model_id(&self) -> &str;
    fn score(&self, query: &str, docs: &[String]) -> Result<Vec<f32>>;
}

pub struct OnnxReranker {
    spec: &'static RerankerSpec,
    tokenizer: Tokenizer,
    /// See `OnnxEmbedder`: one loaded graph shared across callers, serialised
    /// because ORT already parallelises inside a single run.
    session: Mutex<ort::session::Session>,
    wants_token_type_ids: bool,
}

impl OnnxReranker {
    /// Load `id`, downloading the model if this is its first use.
    pub fn load(paths: &Paths, id: &str) -> Result<Self> {
        let spec = lookup(id)?;
        let files = download::ensure_files(
            paths,
            spec.id,
            spec.repo,
            spec.onnx_path,
            spec.tokenizer_path,
            spec.approx_mb,
            spec.onnx_hash,
            spec.tokenizer_hash,
        )?;

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", files.tokenizer.display()))?;
        // Truncate the *document* side of an over-long pair (longest-first),
        // never the query. Doing it in the tokenizer keeps the final [SEP] in
        // place, which truncating the id list by hand would cut off.
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: spec.max_seq,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("configuring truncation: {e}"))?;

        // A reranker answers one small batch per query; the GPU's kernel-launch
        // overhead dominated exactly this shape of workload in the embedder
        // measurements, so it stays on the CPU.
        let (session, _) = onnx::build_session(&files.onnx, crate::config::Device::Cpu)?;
        let wants_token_type_ids = session.inputs().iter().any(|i| i.name() == "token_type_ids");

        Ok(Self {
            spec,
            tokenizer,
            session: Mutex::new(session),
            wants_token_type_ids,
        })
    }

    /// One session run over a set of pair encodings.
    fn forward_group(&self, encodings: &[&tokenizers::Encoding]) -> Result<Vec<f32>> {
        let batch = encodings.len();
        if batch == 0 {
            return Ok(Vec::new());
        }

        let longest = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(1)
            .clamp(1, self.spec.max_seq);
        let seq = onnx::bucket_seq(longest, self.spec.max_seq);

        let mut ids = vec![0i64; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        let mut types = vec![0i64; batch * seq];
        for (b, enc) in encodings.iter().enumerate() {
            let src = enc.get_ids();
            let m = enc.get_attention_mask();
            let t = enc.get_type_ids();
            let n = src.len().min(seq);
            for i in 0..n {
                ids[b * seq + i] = src[i] as i64;
                mask[b * seq + i] = m[i] as i64;
                types[b * seq + i] = t[i] as i64;
            }
        }

        let shape = [batch as i64, seq as i64];
        let mut inputs: Vec<(&str, ort::value::DynValue)> = vec![
            (
                "input_ids",
                onnx::ort_ctx(Tensor::from_array((shape, ids)), "building input_ids")?
                    .into_dyn(),
            ),
            (
                "attention_mask",
                onnx::ort_ctx(Tensor::from_array((shape, mask)), "building attention_mask")?
                    .into_dyn(),
            ),
        ];
        if self.wants_token_type_ids {
            // Unlike the document embedder's zeros, a pair's type ids carry the
            // query/document boundary — 0 for the query segment, 1 for the doc.
            inputs.push((
                "token_type_ids",
                onnx::ort_ctx(Tensor::from_array((shape, types)), "building token_type_ids")?
                    .into_dyn(),
            ));
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("ONNX session mutex was poisoned by an earlier panic"))?;
        let outputs = onnx::ort_ctx(session.run(inputs), "running reranker inference")?;

        let (out_shape, data) = onnx::ort_ctx(
            outputs[0].try_extract_tensor::<f32>(),
            "extracting reranker logits",
        )?;
        if out_shape.len() != 2 || out_shape[0] as usize != batch {
            bail!(
                "expected reranker logits of shape [{batch}, n], got {:?}. \
                 Is `{}` a cross-encoder export?",
                out_shape,
                self.spec.id
            );
        }
        let cols = out_shape[1] as usize;
        // [batch, 1]: the single relevance logit. [batch, 2]: the classifier's
        // "relevant" class, by cross-encoder convention the second column.
        let col = if cols == 1 { 0 } else { 1 };
        Ok((0..batch).map(|b| data[b * cols + col]).collect())
    }
}

impl Reranker for OnnxReranker {
    fn model_id(&self) -> &str {
        self.spec.id
    }

    fn score(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let pairs: Vec<EncodeInput> = docs
            .iter()
            .map(|d| {
                let capped = match d.char_indices().nth(DOC_CHAR_CAP) {
                    Some((i, _)) => &d[..i],
                    None => d.as_str(),
                };
                EncodeInput::Dual(query.into(), capped.into())
            })
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| anyhow!("tokenising rerank pairs: {e}"))?;

        let order: Vec<usize> = (0..encodings.len()).collect();
        let mut out = vec![0f32; encodings.len()];
        for group in order.chunks(MICRO_BATCH) {
            let refs: Vec<&tokenizers::Encoding> = group.iter().map(|i| &encodings[*i]).collect();
            let scores = self.forward_group(&refs)?;
            for (slot, s) in group.iter().zip(scores) {
                out[*slot] = s;
            }
        }
        Ok(out)
    }
}

/// Test seam: `search` is exercised with a canned-scores reranker, no ONNX
/// needed. In production code the only implementation is [`OnnxReranker`].
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Scores each document by an exact-substring rule the test controls.
    pub struct MockReranker {
        /// Documents containing this string score high; all others low.
        pub hot: &'static str,
    }

    impl Reranker for MockReranker {
        fn model_id(&self) -> &str {
            "mock-reranker"
        }

        fn score(&self, _query: &str, docs: &[String]) -> Result<Vec<f32>> {
            Ok(docs
                .iter()
                .map(|d| if d.contains(self.hot) { 10.0 } else { -10.0 })
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_support::MockReranker;

    #[test]
    fn default_reranker_is_registered() {
        assert!(lookup(DEFAULT_RERANKER).is_ok());
    }

    #[test]
    fn unknown_reranker_error_lists_alternatives() {
        let err = lookup("gpt-9-rerank").unwrap_err().to_string();
        assert!(err.contains("ms-marco"), "got: {err}");
    }

    #[test]
    fn registry_ids_are_unique() {
        let mut ids: Vec<&str> = RERANKER_REGISTRY.iter().map(|m| m.id).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate reranker id in RERANKER_REGISTRY");
    }

    #[test]
    fn mock_reranker_scores_by_substring() {
        let rr = MockReranker { hot: "tax" };
        let docs = vec!["a tax form".to_string(), "a cake recipe".to_string()];
        let s = rr.score("employer", &docs).unwrap();
        assert!(s[0] > s[1]);
    }
}
