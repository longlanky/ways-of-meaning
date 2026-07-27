//! ONNX Runtime backend, via the `ort` crate.

use crate::config::Paths;
use crate::embed::{Embedder, ModelSpec, Pooling, l2_normalize, lookup, norm};
use anyhow::{Result, anyhow, bail};
use ort::session::Session;
use ort::value::Tensor;
use std::fmt::Display;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// `ort::Error<R>` carries the partially-built object for recovery, which makes
/// it neither `Send` nor `Sync`, so `anyhow::Context` cannot attach to it. Flatten
/// it to a message instead.
fn ort_ctx<T, E: Display>(r: std::result::Result<T, E>, what: &str) -> Result<T> {
    r.map_err(|e| anyhow!("{what}: {e}"))
}

/// Texts per session run. Large enough to keep every core busy, small enough that
/// activations for the longest sequence stay modest.
const MICRO_BATCH: usize = 32;

/// Sequence lengths the model is ever asked for. Keeping this to a short ladder
/// bounds the number of distinct tensor shapes ORT has to plan allocations for.
const SEQ_BUCKETS: &[usize] = &[32, 64, 96, 128, 192, 256, 384, 512];

/// Smallest bucket that fits `len`, never above `max_seq`.
fn bucket_seq(len: usize, max_seq: usize) -> usize {
    let cap = max_seq.max(1);
    for b in SEQ_BUCKETS {
        if *b >= len {
            return (*b).min(cap);
        }
    }
    cap
}

pub struct OnnxEmbedder {
    spec: &'static ModelSpec,
    tokenizer: Tokenizer,
    /// `ort::Session::run` needs `&mut self`, and we want one loaded graph shared
    /// across the indexing threads. ORT parallelises *inside* a single run via
    /// its own intra-op pool, so serialising batches here costs nothing: the
    /// batch already saturates every core.
    session: Mutex<Session>,
    /// Whether the graph declares a `token_type_ids` input. BERT exports usually
    /// do; some quantised re-exports drop it, and passing an input the graph does
    /// not declare is a hard error in ORT.
    wants_token_type_ids: bool,
    /// Whether a GPU execution provider was actually engaged.
    on_gpu: bool,
}

/// Register the GPU execution provider, if this build has one and the config asks
/// for it. Returns whether the GPU was actually engaged.
///
/// ONNX Runtime has no usable AMD-iGPU provider on Linux through the obvious
/// routes — the ROCm EP needs a ROCm install, and DirectML is Windows-only. The
/// WebGPU provider goes through Dawn, which can target Vulkan, and a Vulkan ICD is
/// the one thing an AMD iGPU reliably has.
///
/// Measured on a Radeon 680M, this works but is not worth switching on. fp32 gains
/// ~16% (51 vs 44 docs/sec), but the default int8 model *collapses* to 20 docs/sec
/// against 64 on CPU, because quantised operators are unimplemented in the WebGPU
/// provider and fall back per-node across the bus. Single-query latency is 2-20x
/// worse, and Dawn dumps core during process teardown. Hence `Device::Cpu` by
/// default and the whole thing behind a cargo feature. See README.md.
#[cfg(feature = "gpu")]
fn register_gpu(
    builder: &mut ort::session::builder::SessionBuilder,
    device: crate::config::Device,
) -> Result<bool> {
    use ort::ep::ExecutionProvider;
    use ort::ep::webgpu::{DawnBackendType, WebGPU};

    if !device.wants_gpu() {
        return Ok(false);
    }

    let ep = WebGPU::default().with_dawn_backend_type(DawnBackendType::Vulkan);

    // `is_available` only reports whether ORT was *compiled* with the provider.
    // Registration is what actually tries to bring up a device, so both are
    // checked: a Vulkan ICD that exists but cannot create a compute device fails
    // at the second step, not the first.
    match ep.is_available() {
        Ok(true) => {}
        other => {
            let why = match other {
                Ok(false) => "this ONNX Runtime build has no WebGPU provider".to_string(),
                Err(e) => format!("could not query WebGPU availability: {e}"),
                _ => unreachable!(),
            };
            if device.may_fall_back() {
                eprintln!("wom: {why}; using the CPU");
                return Ok(false);
            }
            bail!("device = \"gpu\" was requested but {why}");
        }
    }

    match ep.register(builder) {
        Ok(()) => Ok(true),
        Err(e) => {
            if device.may_fall_back() {
                eprintln!("wom: GPU unavailable ({e}); using the CPU");
                Ok(false)
            } else {
                bail!("device = \"gpu\" was requested but registration failed: {e}");
            }
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn register_gpu(
    _builder: &mut ort::session::builder::SessionBuilder,
    device: crate::config::Device,
) -> Result<bool> {
    if device.wants_gpu() {
        let msg = "this build has no GPU support; rebuild with `--features gpu`";
        if device.may_fall_back() {
            eprintln!("wom: {msg}; using the CPU");
        } else {
            bail!("device = \"gpu\" was requested but {msg}");
        }
    }
    Ok(false)
}

impl OnnxEmbedder {
    /// Load `model_id`, honouring a device preference and downloading the model if
    /// this is its first use.
    pub fn load_on(paths: &Paths, model_id: &str, device: crate::config::Device) -> Result<Self> {
        let spec = lookup(model_id)?;
        let files = super::download::ensure_model(paths, spec)?;

        let tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", files.tokenizer.display()))?;

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Graph optimisation is redone on every process start, and `wom` is a CLI
        // that starts fresh for each query — measured at ~250 ms of a 300 ms
        // query. ORT can serialise the already-optimised graph, so do that once
        // and load the result thereafter, which skips optimisation entirely.
        let optimised = files.onnx.with_extension("opt.onnx");
        // Regenerate if the source model is newer, so a re-downloaded or replaced
        // model is never served from a stale optimised copy.
        let use_cache = match (optimised.metadata(), files.onnx.metadata()) {
            (Ok(c), Ok(m)) => match (c.modified(), m.modified()) {
                (Ok(ct), Ok(mt)) => ct >= mt,
                // Without usable timestamps, prefer correctness over speed.
                _ => false,
            },
            _ => false,
        };
        let source = if use_cache { &optimised } else { &files.onnx };

        let builder = ort_ctx(Session::builder(), "creating ORT session builder")?;
        let builder = ort_ctx(
            builder.with_optimization_level(if use_cache {
                // Already optimised; re-running the passes would cost the time
                // this cache exists to save.
                ort::session::builder::GraphOptimizationLevel::Disable
            } else {
                ort::session::builder::GraphOptimizationLevel::Level3
            }),
            "setting optimisation level",
        )?;
        let builder = if use_cache {
            builder
        } else {
            ort_ctx(
                builder.with_optimized_model_path(&optimised),
                "setting the optimised-graph cache path",
            )?
        };
        let builder = ort_ctx(
            builder.with_intra_threads(threads),
            "setting intra-op thread count",
        )?;
        // ORT's memory-pattern optimisation pre-plans allocations from the shapes
        // it has already seen, which pays off for a fixed input shape and works
        // against us here: sequence length varies per batch, so every new shape
        // adds another arena block that is never reused. Measured on a 274k-file
        // index, leaving this on plateaued at ~2.5 GB resident on a machine with
        // 3.2 GB free.
        let builder = ort_ctx(
            builder.with_memory_pattern(false),
            "disabling memory pattern optimisation",
        )?;
        let mut builder = builder;
        let on_gpu = register_gpu(&mut builder, device)?;
        let session = ort_ctx(
            builder.commit_from_file(source),
            &format!("loading ONNX graph {}", source.display()),
        )?;

        let wants_token_type_ids = session.inputs().iter().any(|i| i.name() == "token_type_ids");

        Ok(Self {
            spec,
            tokenizer,
            session: Mutex::new(session),
            wants_token_type_ids,
            on_gpu,
        })
    }

    pub fn spec(&self) -> &'static ModelSpec {
        self.spec
    }

    /// Whether inference is running on a GPU. Reported rather than assumed, since
    /// `device = "auto"` falls back silently by design.
    pub fn on_gpu(&self) -> bool {
        self.on_gpu
    }

    /// Tokenise, sort by length, run in micro-batches, restore the input order.
    ///
    /// Sorting matters because a batch is padded to its longest member: mixing a
    /// 12-token filename with a 480-token document makes the model do 40x the
    /// necessary work on the short one. Grouping similar lengths together removes
    /// most of that waste, and it also keeps the number of distinct tensor shapes
    /// small, which is what ORT's allocator wants.
    fn forward(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.iter().map(|s| s.as_str()).collect::<Vec<_>>(), true)
            .map_err(|e| anyhow!("tokenising batch: {e}"))?;

        let mut order: Vec<usize> = (0..encodings.len()).collect();
        order.sort_unstable_by_key(|i| encodings[*i].get_ids().len());

        let mut out: Vec<Vec<f32>> = vec![Vec::new(); encodings.len()];
        for group in order.chunks(MICRO_BATCH) {
            let refs: Vec<&tokenizers::Encoding> = group.iter().map(|i| &encodings[*i]).collect();
            let vecs = self.forward_group(&refs)?;
            for (slot, v) in group.iter().zip(vecs) {
                out[*slot] = v;
            }
        }
        Ok(out)
    }

    /// One session run over a set of encodings of similar length.
    fn forward_group(&self, encodings: &[&tokenizers::Encoding]) -> Result<Vec<Vec<f32>>> {
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
        // Round up to a fixed ladder so the graph sees a handful of shapes rather
        // than one per batch. The extra padding is cheap next to what ORT saves by
        // reusing an allocation plan.
        let seq = bucket_seq(longest, self.spec.max_seq);

        let mut ids = vec![0i64; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        for (b, enc) in encodings.iter().enumerate() {
            let src = enc.get_ids();
            let n = src.len().min(seq);
            for t in 0..n {
                ids[b * seq + t] = src[t] as i64;
                mask[b * seq + t] = 1;
            }
        }

        let shape = [batch as i64, seq as i64];
        let mut inputs: Vec<(&str, ort::value::DynValue)> = vec![
            (
                "input_ids",
                ort_ctx(Tensor::from_array((shape, ids)), "building input_ids")?.into_dyn(),
            ),
            (
                "attention_mask",
                ort_ctx(
                    Tensor::from_array((shape, mask.clone())),
                    "building attention_mask",
                )?
                .into_dyn(),
            ),
        ];
        if self.wants_token_type_ids {
            inputs.push((
                "token_type_ids",
                ort_ctx(
                    Tensor::from_array((shape, vec![0i64; batch * seq])),
                    "building token_type_ids",
                )?
                .into_dyn(),
            ));
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("ONNX session mutex was poisoned by an earlier panic"))?;
        let outputs = ort_ctx(session.run(inputs), "running ONNX inference")?;

        // The first output of a bare BertModel export is last_hidden_state,
        // shaped [batch, seq, hidden].
        let (out_shape, data) = ort_ctx(
            outputs[0].try_extract_tensor::<f32>(),
            "extracting last_hidden_state",
        )?;
        if out_shape.len() != 3 {
            bail!(
                "expected last_hidden_state of rank 3, got shape {:?}. \
                 Is `{}` a bare encoder export?",
                out_shape,
                self.spec.id
            );
        }
        let hidden = out_shape[2] as usize;
        if hidden != self.spec.dim {
            bail!(
                "model {} produced hidden size {hidden} but the registry says {}; \
                 the registry entry is wrong or the wrong file was downloaded",
                self.spec.id,
                self.spec.dim
            );
        }
        let out_seq = out_shape[1] as usize;

        let mut result = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut v = vec![0f32; hidden];
            match self.spec.pooling {
                // CLS: position 0 only. Verified against BGE's
                // 1_Pooling/config.json (pooling_mode_cls_token: true).
                Pooling::Cls => {
                    let off = b * out_seq * hidden;
                    v.copy_from_slice(&data[off..off + hidden]);
                }
                // Mean over unmasked tokens, so padding cannot drag the vector
                // toward the pad embedding.
                Pooling::Mean => {
                    let mut count = 0f32;
                    for t in 0..out_seq {
                        if mask[b * seq + t] == 0 {
                            continue;
                        }
                        let off = (b * out_seq + t) * hidden;
                        for h in 0..hidden {
                            v[h] += data[off + h];
                        }
                        count += 1.0;
                    }
                    if count > 0.0 {
                        for x in v.iter_mut() {
                            *x /= count;
                        }
                    }
                }
            }
            l2_normalize(&mut v);
            result.push(v);
        }
        Ok(result)
    }
}

impl Embedder for OnnxEmbedder {
    fn dim(&self) -> usize {
        self.spec.dim
    }

    fn model_id(&self) -> &str {
        self.spec.id
    }

    fn embed_docs(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.forward(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{}{}", self.spec.query_prefix, text);
        let mut out = self.forward(&[prefixed])?;
        out.pop()
            .ok_or_else(|| anyhow!("embedder returned no vector for the query"))
    }
}

// `cosine` lives in `embed` beside `l2_normalize`; it has nothing to do with ONNX.
pub use crate::embed::cosine;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_seq_rounds_up_to_the_ladder() {
        assert_eq!(bucket_seq(1, 512), 32);
        assert_eq!(bucket_seq(32, 512), 32);
        assert_eq!(bucket_seq(33, 512), 64);
        assert_eq!(bucket_seq(200, 512), 256);
        assert_eq!(bucket_seq(512, 512), 512);
    }

    #[test]
    fn bucket_seq_never_exceeds_the_models_limit() {
        // all-MiniLM-L6-v2 caps at 256; asking for more must not build a tensor
        // the graph cannot accept.
        for len in [1, 100, 255, 256, 300, 1000] {
            assert!(bucket_seq(len, 256) <= 256, "len={len}");
        }
        assert_eq!(bucket_seq(1000, 512), 512);
    }

    #[test]
    fn bucket_seq_always_covers_the_requested_length() {
        for len in 1..=512usize {
            let b = bucket_seq(len, 512);
            assert!(b >= len, "bucket {b} truncates length {len}");
        }
    }

    #[test]
    fn bucket_ladder_is_sorted_and_ends_at_the_common_limit() {
        assert!(SEQ_BUCKETS.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(*SEQ_BUCKETS.last().unwrap(), 512);
    }
}

/// Check the loaded model behaves the way the registry claims.
///
/// This exists because the two most likely mistakes in this file — mean-pooling a
/// CLS model, and forgetting the query prefix — produce perfectly well-formed
/// unit vectors and merely worse search results. Nothing crashes, so only an
/// explicit behavioural check catches them.
pub fn self_test(paths: &Paths, model_id: &str, device: crate::config::Device) -> Result<()> {
    let emb = OnnxEmbedder::load_on(paths, model_id, device)?;
    let spec = emb.spec();
    println!("model    {} ({}-dim, {:?} pooling)", spec.id, spec.dim, spec.pooling);
    println!("device   {}", if emb.on_gpu() { "GPU (WebGPU/Dawn)" } else { "CPU" });

    let mut failures = Vec::new();

    // 1. Shape and normalisation.
    let docs = emb.embed_docs(&[
        "The cat sat on the mat.".to_string(),
        "A feline rested upon the rug.".to_string(),
        "Quarterly payroll tax withholding statement.".to_string(),
    ])?;
    for (i, v) in docs.iter().enumerate() {
        if v.len() != spec.dim {
            failures.push(format!("doc {i} has {} dims, expected {}", v.len(), spec.dim));
        }
        let n = norm(v);
        if (n - 1.0).abs() > 1e-3 {
            failures.push(format!("doc {i} has norm {n:.4}, expected 1.0"));
        }
    }

    // 2. Semantics: paraphrases must beat unrelated text. A model wired up with
    //    the wrong pooling still passes the shape checks but tends to fail here.
    let para = cosine(&docs[0], &docs[1]);
    let unrel = cosine(&docs[0], &docs[2]);
    println!("cosine   paraphrase {para:.3} vs unrelated {unrel:.3}");
    if para <= unrel {
        failures.push(format!(
            "paraphrase similarity ({para:.3}) did not exceed unrelated ({unrel:.3}); \
             pooling is probably wrong"
        ));
    }
    if para < 0.6 {
        failures.push(format!(
            "paraphrase similarity {para:.3} is implausibly low for this model; \
             expected > 0.6"
        ));
    }

    // 3. The query prefix must actually be applied. Comparing embed_query against
    //    embed_docs of the same string is the only way to observe it from outside.
    let q = emb.embed_query("the cat sat on the mat")?;
    let d = emb.embed_docs(&["the cat sat on the mat".to_string()])?;
    let same = cosine(&q, &d[0]);
    if spec.query_prefix.is_empty() {
        println!("prefix   none (symmetric model), query==doc cosine {same:.4}");
        if (same - 1.0).abs() > 1e-3 {
            failures.push(format!(
                "model has no prefix, so query and doc embeddings should be identical, \
                 but cosine was {same:.4}"
            ));
        }
    } else {
        println!("prefix   applied, query-vs-doc cosine {same:.4}");
        if (same - 1.0).abs() < 1e-4 {
            failures.push(
                "embed_query matched embed_docs exactly, so the BGE query prefix is \
                 NOT being applied"
                    .to_string(),
            );
        }
    }

    // 4. Asymmetric retrieval end to end: the prefixed query should rank the
    //    relevant passage above the irrelevant one.
    let w2 = cosine(&q, &docs[2]);
    let cat = cosine(&q, &docs[0]);
    if cat <= w2 {
        failures.push(format!(
            "query ranked the unrelated passage ({w2:.3}) at or above the relevant \
             one ({cat:.3})"
        ));
    }

    if failures.is_empty() {
        println!("\nself-test passed");
        Ok(())
    } else {
        for f in &failures {
            eprintln!("FAIL: {f}");
        }
        bail!("{} self-test check(s) failed", failures.len())
    }
}
