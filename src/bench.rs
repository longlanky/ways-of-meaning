//! Measurement, so model choice is settled by evidence rather than by argument.
//!
//! `project.txt` left open "how effective the model provided is, as it is quite
//! large and may be slow to run locally". That is an empirical question about one
//! particular machine, and this is what answers it.

use crate::config::{Config, Paths};
use crate::db::Db;
use crate::embed::{Embedder, onnx::OnnxEmbedder};
use crate::rerank::Reranker;
use crate::search::{self, Request};
use anyhow::{Context, Result};
use std::time::Instant;

/// One model's measured behaviour.
pub struct ModelReport {
    pub model_id: String,
    pub dim: usize,
    pub load_secs: f64,
    /// Documents per second at a realistic mix of text lengths.
    pub docs_per_sec: f64,
    /// Time for a single query embedding, which sets the floor on search latency.
    pub query_ms: f64,
    /// Cosine of a known paraphrase pair, as a sanity check on quality.
    pub paraphrase: f32,
    /// Cosine of a known unrelated pair. The gap to `paraphrase` is what matters.
    pub unrelated: f32,
    /// Whether this measurement ran on a GPU.
    pub on_gpu: bool,
}

impl ModelReport {
    /// How well the model separates related from unrelated text. A larger margin
    /// means the similarity floor is easier to place and ranking is more robust.
    pub fn margin(&self) -> f32 {
        self.paraphrase - self.unrelated
    }

    /// Hours to embed `n` documents at the measured rate.
    pub fn hours_for(&self, n: usize) -> f64 {
        if self.docs_per_sec <= 0.0 {
            f64::INFINITY
        } else {
            n as f64 / self.docs_per_sec / 3600.0
        }
    }
}

/// Text lengths that mirror what the indexer actually sends: mostly short
/// path-and-name strings, some medium documents, a few long ones. Benchmarking
/// with uniform-length input would overstate throughput, because real batches pay
/// for padding to their longest member.
fn sample_docs(n: usize) -> Vec<String> {
    let short = "projects/esp32_epaper/src/main.cpp\nprojects esp32 epaper src main cpp";
    let medium = "Documents/tax/2024_W2.pdf\ndocuments tax 2024 w2 pdf\n\nWage and Tax \
                  Statement. Employer identification number. Wages, tips, other \
                  compensation. Federal income tax withheld. Social security wages.";
    let long_body = "The quick brown fox jumps over the lazy dog near the riverbank. ".repeat(20);

    (0..n)
        .map(|i| match i % 10 {
            // 60% short, 30% medium, 10% long.
            0..=5 => format!("{short} variant {i}"),
            6..=8 => format!("{medium} variant {i}"),
            _ => format!("Documents/notes/long_{i}.md\n\n{long_body}"),
        })
        .collect()
}

/// Measure one model end to end.
pub fn measure_model(
    paths: &Paths,
    model_id: &str,
    docs: usize,
    device: crate::config::Device,
) -> Result<ModelReport> {
    let t0 = Instant::now();
    let emb = OnnxEmbedder::load_on(paths, model_id, device)
        .with_context(|| format!("loading model {model_id}"))?;
    let load_secs = t0.elapsed().as_secs_f64();

    let texts = sample_docs(docs);

    // Warm up: the first run pays for lazy allocation and thread-pool spin-up,
    // which would otherwise be charged to the measurement.
    emb.embed_docs(&texts[..docs.min(32)])?;

    let t1 = Instant::now();
    let vecs = emb.embed_docs(&texts)?;
    let elapsed = t1.elapsed().as_secs_f64();
    anyhow::ensure!(vecs.len() == texts.len(), "embedder dropped documents");

    // Query latency, averaged over a few runs since it is small.
    let t2 = Instant::now();
    const QUERIES: usize = 10;
    for i in 0..QUERIES {
        emb.embed_query(&format!("employment documents {i}"))?;
    }
    let query_ms = t2.elapsed().as_secs_f64() * 1000.0 / QUERIES as f64;

    let probe = emb.embed_docs(&[
        "The cat sat on the mat.".to_string(),
        "A feline rested upon the rug.".to_string(),
        "Quarterly payroll tax withholding statement.".to_string(),
    ])?;

    Ok(ModelReport {
        model_id: emb.model_id().to_string(),
        dim: emb.dim(),
        load_secs,
        docs_per_sec: docs as f64 / elapsed,
        query_ms,
        paraphrase: crate::embed::onnx::cosine(&probe[0], &probe[1]),
        unrelated: crate::embed::onnx::cosine(&probe[0], &probe[2]),
        on_gpu: emb.on_gpu(),
    })
}

/// One golden retrieval case: a query and a substring the right answer contains.
pub struct GoldenCase {
    pub query: &'static str,
    pub expect_contains: &'static str,
}

/// Retrieval cases drawn from `project.txt`'s own examples plus ones exercising
/// the paths most likely to regress.
///
/// These are the real quality guard. A pooling or query-prefix mistake still
/// produces well-formed unit vectors and plausible-looking output, so only an
/// end-to-end check on known-good answers catches it.
pub const GOLDEN: &[GoldenCase] = &[
    // From the spec's examples.
    GoldenCase { query: "employment documents", expect_contains: "employment" },
    GoldenCase { query: "bosnia", expect_contains: "bosnia" },
    GoldenCase { query: "sensors monitor", expect_contains: "esp32" },
    // Semantic, not lexical: none of these words appear in the target path.
    GoldenCase { query: "tax forms from my employer", expect_contains: "employment" },
    GoldenCase { query: "balkans holiday", expect_contains: "bosnia" },
    GoldenCase { query: "embedded display firmware", expect_contains: "esp32" },
];

pub struct RecallReport {
    pub total: usize,
    pub hits: usize,
    pub misses: Vec<(&'static str, Vec<String>)>,
    pub p50_ms: f64,
    pub p95_ms: f64,
}

/// Run the golden set against the live index.
pub fn measure_recall(
    paths: &Paths,
    cfg: &Config,
    db: &Db,
    embedder: &dyn Embedder,
    reranker: Option<&dyn Reranker>,
    limit: usize,
) -> Result<RecallReport> {
    let mut hits = 0;
    let mut misses = Vec::new();
    let mut timings: Vec<f64> = Vec::new();

    for case in GOLDEN {
        let req = Request {
            text: case.query.to_string(),
            scope: Vec::new(),
            limit,
            mode: search::Mode::Hybrid,
            min_similarity: cfg.resolved_min_similarity(),
        };
        let t = Instant::now();
        let results = search::search(paths, db, Some(embedder), reranker, &req)?;
        timings.push(t.elapsed().as_secs_f64() * 1000.0);

        let found = results.iter().any(|r| {
            r.path
                .to_string_lossy()
                .to_lowercase()
                .contains(&case.expect_contains.to_lowercase())
        });
        if found {
            hits += 1;
        } else {
            misses.push((
                case.query,
                results
                    .iter()
                    .take(3)
                    .map(|r| r.display())
                    .collect::<Vec<_>>(),
            ));
        }
    }

    timings.sort_by(f64::total_cmp);
    Ok(RecallReport {
        total: GOLDEN.len(),
        hits,
        misses,
        p50_ms: percentile(&timings, 0.50),
        p95_ms: percentile(&timings, 0.95),
    })
}

/// Reranker load and scoring cost over a realistic candidate pool, so the
/// price of `--rerank` is measured rather than guessed.
pub struct RerankReport {
    pub model_id: String,
    pub load_secs: f64,
    pub pairs: usize,
    pub total_ms: f64,
}

pub fn measure_reranker(paths: &Paths, id: &str, pairs: usize) -> Result<RerankReport> {
    let t0 = Instant::now();
    let rr = crate::rerank::OnnxReranker::load(paths, id)?;
    let load_secs = t0.elapsed().as_secs_f64();
    let docs = sample_docs(pairs);
    let t = Instant::now();
    rr.score("tax forms from my employer", &docs)?;
    let total_ms = t.elapsed().as_secs_f64() * 1000.0;
    Ok(RerankReport {
        model_id: rr.model_id().to_string(),
        load_secs,
        pairs,
        total_ms,
    })
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_docs_mixes_lengths() {
        let docs = sample_docs(100);
        assert_eq!(docs.len(), 100);
        let lens: Vec<usize> = docs.iter().map(|d| d.len()).collect();
        let min = *lens.iter().min().unwrap();
        let max = *lens.iter().max().unwrap();
        // A realistic mix, not one uniform length.
        assert!(max > min * 5, "sample is too uniform: {min}..{max}");
    }

    #[test]
    fn sample_docs_handles_small_counts() {
        assert_eq!(sample_docs(0).len(), 0);
        assert_eq!(sample_docs(1).len(), 1);
    }

    #[test]
    fn percentile_picks_sensible_positions() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 0.5), 3.0);
        assert_eq!(percentile(&v, 1.0), 5.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn report_margin_and_projection() {
        let r = ModelReport {
            model_id: "m".into(),
            dim: 384,
            load_secs: 1.0,
            docs_per_sec: 100.0,
            query_ms: 5.0,
            paraphrase: 0.8,
            unrelated: 0.4,
            on_gpu: false,
        };
        assert!((r.margin() - 0.4).abs() < 1e-6);
        assert!((r.hours_for(360_000) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_rate_projects_to_infinity_rather_than_dividing_by_zero() {
        let r = ModelReport {
            model_id: "m".into(),
            dim: 1,
            load_secs: 0.0,
            docs_per_sec: 0.0,
            query_ms: 0.0,
            paraphrase: 0.0,
            unrelated: 0.0,
            on_gpu: false,
        };
        assert!(r.hours_for(1000).is_infinite());
    }

    #[test]
    fn golden_cases_are_well_formed() {
        assert!(!GOLDEN.is_empty());
        for c in GOLDEN {
            assert!(!c.query.trim().is_empty());
            assert!(!c.expect_contains.trim().is_empty());
            // The expectation must be lowercase, since matching lowercases both.
            assert_eq!(c.expect_contains, c.expect_contains.to_lowercase());
        }
    }
}
