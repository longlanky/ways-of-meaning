//! Semantic topics over the indexed file vectors.
//!
//! `wom topics build` samples the live file vectors, trains an e-means codebook
//! (see [`crate::cluster`]), assigns every embedded file to its nearest centroid,
//! and labels each topic from the paths and filenames of its members. The result
//! is a snapshot in SQLite: browsing and filtering never load the embedding
//! model, and a rebuild is the refresh mechanism.
//!
//! This module owns the database side. The math lives in `cluster`.

use crate::cluster::{self, Params};
use crate::config::deepest_root;
use crate::db::Db;
use crate::index::extract::name_tokens;
use crate::index::now_secs;
use crate::vectors::VectorStore;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use rusqlite::{OptionalExtension, params};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const DEFAULT_EPOCHS: u32 = 5;
pub const DEFAULT_SAMPLE: usize = 100_000;
pub const DEFAULT_SEED: u64 = 42;
/// The largest k the build accepts. Above this, assignments cost more than the
/// browsing feature is worth, and most clusters would hold a handful of files.
pub const MAX_K: usize = 1024;
pub const MAX_EPOCHS: u32 = 20;
/// Members considered when labelling a topic. The nearest ones are the core;
/// beyond a few hundred the tail is noise.
const LABEL_MEMBERS: usize = 500;
/// Topic ids are ordered by size, largest first, so the browser opens on the
/// most useful topic.
const DEFAULT_MAX_K: usize = 256;

pub const META_BUILT_AT: &str = "topics_built_at";
pub const META_MODEL_ID: &str = "topics_model_id";
pub const META_FILES_AT_BUILD: &str = "topics_files_at_build";
pub const META_K: &str = "topics_k";
pub const META_SAMPLE: &str = "topics_sample";
pub const META_EPOCHS: &str = "topics_epochs";
pub const META_SEED: &str = "topics_seed";
pub const META_VARIANCE: &str = "topics_variance_explained";

/// What to build. `k = None` uses [`default_k`] for the corpus size.
#[derive(Debug, Clone, Copy)]
pub struct BuildOptions {
    pub k: Option<usize>,
    pub epochs: u32,
    pub sample: usize,
    pub seed: u64,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            k: None,
            epochs: DEFAULT_EPOCHS,
            sample: DEFAULT_SAMPLE,
            seed: DEFAULT_SEED,
        }
    }
}

/// A topic as displayed: the live member count comes from `file_topics`, not
/// from a stored size, so deletions never leave a stale number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicSummary {
    pub id: i64,
    pub label: String,
    pub files: i64,
    pub example_path: String,
}

/// One parallel assignment chunk: per-file entries, the summed vectors, the
/// summed squared norms, and the summed squared distance to the assigned
/// centroid. The sums make variance explained exact without a second pass.
type AssignChunk = (Vec<(i64, u32, f32)>, Vec<f64>, f64, f64);

/// One build's outcome, for the CLI summary and tests.
#[derive(Debug, Clone)]
pub struct BuildStats {
    pub requested_k: usize,
    pub built_k: usize,
    pub sample: usize,
    pub epochs: u32,
    pub batch: usize,
    pub alpha0: f32,
    pub steps: usize,
    pub assigned: usize,
    pub variance_explained: f64,
    pub size_min: usize,
    pub size_median: usize,
    pub size_max: usize,
    pub elapsed_secs: f64,
}

/// Whether the stored topics still describe the current index.
#[derive(Debug, Clone)]
pub struct Staleness {
    pub built: bool,
    pub topic_count: i64,
    pub files: i64,
    pub built_at: Option<i64>,
    /// Files added since the build. Deletions cascade, so they never make the
    /// model wrong; additions leave files without a topic.
    pub added: i64,
    /// Present when the model that built the topics differs from the index's.
    pub model_mismatch: Option<(String, String)>,
}

/// Square-root rule, capped so the list stays browsable and at least 2.
pub fn default_k(n: usize) -> usize {
    if n < 2 {
        return 2;
    }
    ((n as f64).sqrt().round() as usize)
        .clamp(2, DEFAULT_MAX_K)
        .min(n)
}

/// Train a topic model and replace the stored one in one transaction.
///
/// `model_id` is the embedding model the vectors came from. `roots` are the
/// configured roots used to relativise paths for labels; a file outside every
/// root is tokenised from its full path.
pub fn build(
    db: &Db,
    store: &VectorStore,
    model_id: &str,
    roots: &[PathBuf],
    opts: &BuildOptions,
) -> Result<BuildStats> {
    let started = Instant::now();
    if opts.epochs == 0 || opts.epochs > MAX_EPOCHS {
        bail!("epochs must be between 1 and {MAX_EPOCHS}");
    }
    if opts.sample < 2 {
        bail!("sample must be at least 2");
    }
    let dim = store.dim();
    if dim == 0 {
        bail!("the vector store has no dimension");
    }

    // `vec_row == files.id` is the mapping the rest of the crate relies on. A
    // violation would assign vectors to the wrong paths silently, so check it.
    let mismatched: i64 = db.conn.query_row(
        "SELECT COUNT(*) FROM files WHERE vec_row >= 0 AND vec_row != id",
        [],
        |r| r.get(0),
    )?;
    if mismatched > 0 {
        bail!(
            "{mismatched} files have a vector row that is not their id. \
             Run `wom index --rebuild` before building topics."
        );
    }

    let live_ids: Vec<i64> = db
        .conn
        .prepare("SELECT id FROM files WHERE vec_row != -1 ORDER BY id")?
        .query_map([], |r| r.get(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if live_ids.len() < 2 {
        bail!("need at least 2 embedded files; run `wom index` first");
    }
    let live_count = live_ids.len();

    let k = opts.k.unwrap_or_else(|| default_k(live_count));
    if k < 2 {
        bail!("a topic model needs at least 2 topics");
    }
    if k > MAX_K {
        bail!("k is capped at {MAX_K}; asked for {k}");
    }
    if k > live_count {
        bail!("cannot make {k} topics from {live_count} embedded files");
    }
    if k * 10 > live_count {
        eprintln!(
            "wom: note: {k} topics for {live_count} files leaves very small clusters; \
             consider a smaller --k"
        );
    }
    let sample = opts.sample.max(k).min(live_count);
    if opts.sample < k {
        eprintln!(
            "wom: note: sample raised from {} to {sample} so every topic can seed",
            opts.sample
        );
    }

    // Deterministic sample of live rows. Rows can still be dead (a cleared row
    // whose file is about to be garbage-collected), so read and skip.
    let mut pool = live_ids.clone();
    let mut rng = cluster::Rng::new(opts.seed);
    rng.shuffle(&mut pool);
    let mut training = Vec::with_capacity(sample * dim);
    let mut row = vec![0f32; dim];
    let mut picked = 0usize;
    for &id in &pool {
        if store.read_into(id as usize, &mut row)? {
            training.extend_from_slice(&row);
            picked += 1;
            if picked == sample {
                break;
            }
        }
    }
    if picked < k {
        bail!(
            "only {picked} live vectors are readable; cannot make {k} topics. \
             Run `wom index --verify`."
        );
    }
    let model = cluster::train(
        &training,
        picked,
        dim,
        &Params {
            k,
            epochs: opts.epochs,
            seed: opts.seed,
        },
    )
    .context("training the topic model")?;

    // Assign every live row, in parallel chunks. The chunk-local sums give the
    // exact corpus mean and squared norm for variance explained without a
    // second pass.
    let centroids = &model.centroids;
    let chunk_size = 4096usize;
    let chunks: Vec<Result<AssignChunk>> = live_ids
        .par_chunks(chunk_size)
        .map(|ids| {
            let mut buf = vec![0f32; dim];
            let mut sum = vec![0f64; dim];
            let mut sumsq = 0f64;
            let mut sse = 0f64;
            let mut entries = Vec::with_capacity(ids.len());
            for &id in ids {
                if !store.read_into(id as usize, &mut buf)? {
                    continue;
                }
                let (topic, distance) = cluster::nearest(centroids, k, dim, &buf);
                entries.push((id, topic as u32, distance));
                for d in 0..dim {
                    sum[d] += buf[d] as f64;
                }
                for v in &buf {
                    sumsq += (*v as f64) * (*v as f64);
                }
                sse += distance as f64;
            }
            Ok((entries, sum, sumsq, sse))
        })
        .collect();

    let mut entries: Vec<(i64, u32, f32)> = Vec::with_capacity(live_count);
    let mut corpus_sum = vec![0f64; dim];
    let mut corpus_sumsq = 0f64;
    let mut corpus_sse = 0f64;
    for chunk in chunks {
        let (mut chunk_entries, sum, sumsq, sse) = chunk?;
        entries.append(&mut chunk_entries);
        for d in 0..dim {
            corpus_sum[d] += sum[d];
        }
        corpus_sumsq += sumsq;
        corpus_sse += sse;
    }
    let assigned = entries.len();
    if assigned == 0 {
        bail!("no live vectors could be assigned; run `wom index` first");
    }

    let total_variance =
        corpus_sumsq - corpus_sum.iter().map(|v| v * v).sum::<f64>() / assigned as f64;
    let variance_explained = if total_variance > 1e-9 {
        (1.0 - corpus_sse / total_variance).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Group by cluster and order topics by size, largest first.
    let mut members: Vec<Vec<(f32, i64)>> = vec![Vec::new(); k];
    for &(id, topic, distance) in &entries {
        members[topic as usize].push((distance, id));
    }
    for list in &mut members {
        list.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    }
    let mut order: Vec<usize> = (0..k).filter(|&j| !members[j].is_empty()).collect();
    order.sort_by(|&a, &b| {
        members[b]
            .len()
            .cmp(&members[a].len())
            .then_with(|| a.cmp(&b))
    });
    let built_k = order.len();
    if built_k == 0 {
        bail!("every topic came out empty; the vector index is probably unusable");
    }
    if built_k * 2 < k {
        eprintln!(
            "wom: note: only {built_k} of {k} topics received files; \
             the requested k is high for this corpus"
        );
    }
    let mut new_id = vec![0i64; k];
    for (i, &old) in order.iter().enumerate() {
        new_id[old] = i as i64 + 1;
    }

    // One read of the path map, reused for labels and examples.
    let mut path_of: HashMap<i64, String> = HashMap::with_capacity(live_count);
    {
        let mut st = db.conn.prepare("SELECT id, path FROM files")?;
        let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        for row in rows {
            let (id, path) = row?;
            path_of.insert(id, path);
        }
    }

    // Token counts per topic, then document frequency across topics for IDF.
    let mut per_topic: Vec<HashMap<String, u32>> = Vec::with_capacity(built_k);
    let mut top_counts: Vec<usize> = Vec::with_capacity(built_k);
    let mut examples: Vec<String> = Vec::with_capacity(built_k);
    for &old in &order {
        let ids: Vec<i64> = members[old]
            .iter()
            .take(LABEL_MEMBERS)
            .map(|(_, id)| *id)
            .collect();
        top_counts.push(ids.len());
        examples.push(
            ids.first()
                .and_then(|id| path_of.get(id))
                .cloned()
                .unwrap_or_default(),
        );
        per_topic.push(token_counts(&ids, &path_of, roots));
    }
    let mut document_frequency: HashMap<String, usize> = HashMap::new();
    for counts in &per_topic {
        for token in counts.keys() {
            *document_frequency.entry(token.clone()).or_insert(0) += 1;
        }
    }
    let labels: Vec<String> = per_topic
        .iter()
        .enumerate()
        .map(|(i, counts)| {
            make_label(
                counts,
                &document_frequency,
                built_k,
                top_counts[i],
                &examples[i],
            )
        })
        .collect();

    let built_at = now_secs();
    let mut sizes: Vec<usize> = order.iter().map(|&old| members[old].len()).collect();
    sizes.sort_unstable();
    let size_min = sizes.first().copied().unwrap_or(0);
    let size_max = sizes.last().copied().unwrap_or(0);
    let size_median = sizes.get(sizes.len() / 2).copied().unwrap_or(0);

    // Replace the snapshot atomically. A failure before the commit leaves the
    // previous model intact.
    let tx = db.conn.unchecked_transaction()?;
    tx.execute("DELETE FROM file_topics", [])?;
    tx.execute("DELETE FROM topics", [])?;
    {
        let mut insert_topic = tx.prepare(
            "INSERT INTO topics(id, label, example_path, built_at) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for (i, &old) in order.iter().enumerate() {
            insert_topic.execute(params![new_id[old], labels[i], examples[i], built_at])?;
        }
    }
    {
        let mut insert =
            tx.prepare("INSERT INTO file_topics(file_id, topic_id, distance) VALUES (?1, ?2, ?3)")?;
        for (id, topic, distance) in &entries {
            insert.execute(params![id, new_id[*topic as usize], *distance as f64])?;
        }
    }
    {
        let mut set_meta = tx.prepare(
            "INSERT INTO meta(k, v) VALUES (?1, ?2)
             ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        )?;
        set_meta.execute(params![META_BUILT_AT, built_at.to_string()])?;
        set_meta.execute(params![META_MODEL_ID, model_id])?;
        set_meta.execute(params![META_FILES_AT_BUILD, live_count.to_string()])?;
        set_meta.execute(params![META_K, k.to_string()])?;
        set_meta.execute(params![META_SAMPLE, picked.to_string()])?;
        set_meta.execute(params![META_EPOCHS, opts.epochs.to_string()])?;
        set_meta.execute(params![META_SEED, opts.seed.to_string()])?;
        set_meta.execute(params![META_VARIANCE, format!("{variance_explained:.6}")])?;
    }
    tx.commit()?;

    Ok(BuildStats {
        requested_k: k,
        built_k,
        sample: picked,
        epochs: opts.epochs,
        batch: model.recipe.batch,
        alpha0: model.recipe.alpha0,
        steps: model.recipe.steps,
        assigned,
        variance_explained,
        size_min,
        size_median,
        size_max,
        elapsed_secs: started.elapsed().as_secs_f64(),
    })
}

/// Every topic with its live file count, largest first.
pub fn summaries(db: &Db) -> Result<Vec<TopicSummary>> {
    let mut st = db.conn.prepare(
        "SELECT t.id, t.label, t.example_path, COUNT(ft.file_id) AS n
         FROM topics t
         LEFT JOIN file_topics ft ON ft.topic_id = t.id
         GROUP BY t.id
         ORDER BY n DESC, t.id",
    )?;
    let rows = st.query_map([], |r| {
        Ok(TopicSummary {
            id: r.get(0)?,
            label: r.get(1)?,
            example_path: r.get(2)?,
            files: r.get(3)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

/// One topic, or `None` if it does not exist.
pub fn summary(db: &Db, id: i64) -> Result<Option<TopicSummary>> {
    let mut st = db.conn.prepare(
        "SELECT t.id, t.label, t.example_path, COUNT(ft.file_id) AS n
         FROM topics t
         LEFT JOIN file_topics ft ON ft.topic_id = t.id
         WHERE t.id = ?1
         GROUP BY t.id",
    )?;
    Ok(st
        .query_row(params![id], |r| {
            Ok(TopicSummary {
                id: r.get(0)?,
                label: r.get(1)?,
                example_path: r.get(2)?,
                files: r.get(3)?,
            })
        })
        .optional()?)
}

/// Compare the stored snapshot with the current index. Cheap enough to call on
/// every list/show/filtered search.
pub fn staleness(db: &Db) -> Result<Staleness> {
    let topic_count: i64 = db
        .conn
        .query_row("SELECT COUNT(*) FROM topics", [], |r| r.get(0))?;
    if topic_count == 0 {
        return Ok(Staleness {
            built: false,
            topic_count: 0,
            files: 0,
            built_at: None,
            added: 0,
            model_mismatch: None,
        });
    }
    let files: i64 = db
        .conn
        .query_row("SELECT COUNT(*) FROM file_topics", [], |r| r.get(0))?;
    let built_at = db.get_meta_i64(META_BUILT_AT)?;
    let files_at_build = db.get_meta_i64(META_FILES_AT_BUILD)?.unwrap_or(files);
    let current = db.count_embedded()?;
    let added = (current - files_at_build).max(0);
    let topics_model = db.get_meta(META_MODEL_ID)?;
    let index_model = db.get_meta("model_id")?;
    let model_mismatch = match (topics_model, index_model) {
        (Some(built), Some(index)) if built != index => Some((built, index)),
        _ => None,
    };
    Ok(Staleness {
        built: true,
        topic_count,
        files,
        built_at,
        added,
        model_mismatch,
    })
}

/// A one-line note for stderr, or `None` when the model is current.
pub fn stale_note(state: &Staleness) -> Option<String> {
    if !state.built {
        return Some("no topics yet; run `wom topics build`".to_string());
    }
    if let Some((built, index)) = &state.model_mismatch {
        return Some(format!(
            "topics were built with {built}, the index uses {index}; run `wom topics build`"
        ));
    }
    if state.added > 0 {
        return Some(format!(
            "{} files were indexed after the topics were built; run `wom topics build` to include them",
            state.added
        ));
    }
    None
}

// ------------------------------------------------------------------ labels

/// Tokens that describe where a file lives rather than what it is. Kept small:
/// a label made of three quarters of the filesystem hierarchy is useless.
const STOPWORDS: &[&str] = &[
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "over",
    "this",
    "that",
    "home",
    "users",
    "documents",
    "document",
    "downloads",
    "desktop",
    "pictures",
    "music",
    "videos",
    "video",
    "public",
    "storage",
    "transfer",
    "misc",
    "stuff",
    "other",
    "backup",
    "backups",
    "copy",
    "temp",
    "tmp",
    "data",
    "files",
    "file",
    "new",
    "final",
    "untitled",
    "screenshot",
    "screenshots",
    "img",
    "dsc",
    "user",
];

/// Extensions and format words. A topic that is entirely `pdf` should be named
/// by its content or its directory, not by the container.
const FORMAT_WORDS: &[&str] = &[
    "pdf", "jpg", "jpeg", "png", "gif", "webp", "svg", "bmp", "tif", "tiff", "txt", "md", "rtf",
    "doc", "docx", "odt", "xls", "xlsx", "ods", "csv", "ppt", "pptx", "odp", "mp3", "wav", "flac",
    "ogg", "m4a", "aac", "wma", "au", "aup", "mp4", "mkv", "avi", "mov", "webm", "wmv", "zip",
    "tar", "gz", "bz2", "xz", "zst", "7z", "rar", "iso", "img", "exe", "dll", "so", "bin", "json",
    "xml", "yaml", "yml", "toml", "ini", "cfg", "conf", "html", "htm", "css", "js", "ts", "rs",
    "py", "c", "h", "cpp", "hpp", "java", "sh", "ps1", "bat", "url", "lnk", "desktop", "flp",
    "als", "logic", "band", "eff",
];

fn keep_token(token: &str) -> bool {
    if token.len() < 2 || token.len() > 32 {
        return false;
    }
    if token.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    // File ids and hashes: `eff05a51`, `f17d832d`. Semantic only to a filesystem.
    if token.len() >= 6 && token.chars().all(|c| c.is_ascii_hexdigit()) {
        return false;
    }
    // Two-letter tokens are mostly initials; allow the ones that carry a digit
    // (`w2`, `3d`).
    if token.len() == 2 && !token.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }
    // Backup/disc/tape partition ids: `d1f`, `d06`, `d13`, `cd2`, `cd5`. They
    // describe how a corpus was stored, not what it contains.
    if token.len() == 3
        && token.starts_with('d')
        && token[1..].chars().all(|c| c.is_ascii_hexdigit())
    {
        return false;
    }
    if token.len() == 3 && token.starts_with("cd") && token[2..].chars().all(|c| c.is_ascii_digit())
    {
        return false;
    }
    // Long digit runs are file ids or dates: `dsc0461`, `img1399`, `2024w2`.
    // A short year or track number alone is already dropped above.
    if token.chars().filter(|c| c.is_ascii_digit()).count() >= 4 && token.len() <= 8 {
        return false;
    }
    !STOPWORDS.contains(&token) && !FORMAT_WORDS.contains(&token)
}

/// Per-topic token counts, counting each token at most once per file. Presence
/// rather than raw frequency keeps one deeply nested path from dominating.
fn token_counts(
    ids: &[i64],
    path_of: &HashMap<i64, String>,
    roots: &[PathBuf],
) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    for id in ids {
        let Some(path) = path_of.get(id) else {
            continue;
        };
        let path = Path::new(path);
        let base = deepest_root(roots, path).unwrap_or_else(|| Path::new("/"));
        let tokens = name_tokens(path, base);
        let mut seen: HashSet<&str> = HashSet::new();
        for token in tokens.split_whitespace() {
            if keep_token(token) && seen.insert(token) {
                *counts.entry(token.to_string()).or_insert(0) += 1;
            }
        }
    }
    counts
}

/// Rank tokens by TF-IDF and join the best three. A topic with no usable token
/// falls back to its directory and extension, and finally to "topic".
fn make_label(
    counts: &HashMap<String, u32>,
    document_frequency: &HashMap<String, usize>,
    topic_count: usize,
    members: usize,
    example: &str,
) -> String {
    let mut scored: Vec<(f64, &str)> = counts
        .iter()
        .map(|(token, count)| {
            let tf = *count as f64 / members.max(1) as f64;
            let df = document_frequency.get(token).copied().unwrap_or(0) as f64;
            let idf = (1.0 + topic_count as f64 / (1.0 + df)).ln();
            (tf * idf, token.as_str())
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    let picked: Vec<&str> = scored.into_iter().take(3).map(|(_, token)| token).collect();
    if picked.is_empty() {
        fallback_label(example)
    } else {
        picked.join(" · ")
    }
}

fn fallback_label(example: &str) -> String {
    let path = Path::new(example);
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .filter(|e| !e.is_empty());
    let parent = path
        .parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .filter(|n| !n.is_empty());
    match (parent, extension) {
        (Some(parent), Some(ext)) => format!("{parent} · {ext}"),
        (Some(parent), None) => parent,
        (None, Some(ext)) => ext,
        (None, None) => "topic".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_store(name: &str, dim: usize) -> (PathBuf, VectorStore) {
        let path =
            std::env::temp_dir().join(format!("wom-topics-{name}-{}.i8", std::process::id()));
        std::fs::remove_file(&path).ok();
        let store = VectorStore::open(&path, dim, "test-model").unwrap();
        (path, store)
    }

    fn insert_file(db: &Db, id: i64, path: &str, vec_row: i64) {
        db.conn
            .execute(
                "INSERT INTO files(id, path, mtime_ns, size, vec_row, seen_scan)
                 VALUES (?1, ?2, 1, 1, ?3, 1)",
                params![id, path, vec_row],
            )
            .unwrap();
    }

    /// Four well-separated blobs, ten files each, in eight dimensions.
    fn fixture(name: &str) -> (PathBuf, VectorStore, Db, Vec<&'static str>) {
        let dim = 8;
        let (path, mut store) = temp_store(name, dim);
        let db = Db::open_in_memory().unwrap();
        let names = ["alpha", "beta", "gamma", "delta"];
        let mut id = 1i64;
        for (blob, name) in names.iter().enumerate() {
            for i in 0..10 {
                let mut v = vec![0f32; dim];
                v[blob] = 10.0 + i as f32 * 0.001;
                v[(blob + 1) % dim] = 0.001 * i as f32;
                store.put(id as usize, &v).unwrap();
                insert_file(&db, id, &format!("/t/{name}/{name}_{i}.txt"), id);
                id += 1;
            }
        }
        (path, store, db, names.to_vec())
    }

    #[test]
    fn build_recovers_blobs_and_labels_them() {
        let (path, store, db, names) = fixture("build");
        let stats = build(
            &db,
            &store,
            "test-model",
            &[PathBuf::from("/t")],
            &BuildOptions {
                k: Some(4),
                epochs: 4,
                sample: 100,
                seed: 7,
            },
        )
        .unwrap();

        assert_eq!(stats.requested_k, 4);
        assert_eq!(stats.built_k, 4);
        assert_eq!(stats.assigned, 40);
        assert!(stats.variance_explained > 0.99, "{:?}", stats);

        let topics = summaries(&db).unwrap();
        assert_eq!(topics.len(), 4);
        assert!(topics.iter().all(|t| t.files == 10));
        for name in &names {
            assert!(
                topics.iter().any(|t| t.label.contains(name)),
                "no label contains {name}: {:?}",
                topics.iter().map(|t| &t.label).collect::<Vec<_>>()
            );
        }
        assert!(topics.iter().all(|t| !t.example_path.is_empty()));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn build_is_repeatable_and_replaces_the_previous_snapshot() {
        let (path, store, db, _) = fixture("repeat");
        let opts = BuildOptions {
            k: Some(3),
            epochs: 3,
            sample: 100,
            seed: 11,
        };
        let first = build(&db, &store, "test-model", &[PathBuf::from("/t")], &opts).unwrap();
        let first_topics = summaries(&db).unwrap();
        let second = build(&db, &store, "test-model", &[PathBuf::from("/t")], &opts).unwrap();
        assert_eq!(first.built_k, second.built_k);
        assert_eq!(first.assigned, second.assigned);
        assert_eq!(first_topics, summaries(&db).unwrap());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn staleness_notices_added_files_and_a_model_change() {
        let (path, store, db, _) = fixture("stale");
        build(
            &db,
            &store,
            "test-model",
            &[PathBuf::from("/t")],
            &BuildOptions {
                k: Some(4),
                epochs: 2,
                sample: 100,
                seed: 3,
            },
        )
        .unwrap();

        let mut state = staleness(&db).unwrap();
        assert!(state.built);
        assert_eq!(state.topic_count, 4);
        assert_eq!(state.files, 40);
        assert_eq!(state.added, 0);

        insert_file(&db, 41, "/t/alpha/alpha_extra.txt", 41);
        state = staleness(&db).unwrap();
        assert_eq!(state.added, 1);

        db.set_meta("model_id", "other-model").unwrap();
        state = staleness(&db).unwrap();
        assert_eq!(
            state.model_mismatch,
            Some(("test-model".to_string(), "other-model".to_string()))
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn one_topic_summary_is_ordered_by_size() {
        let (path, store, db, _) = fixture("summary");
        build(
            &db,
            &store,
            "test-model",
            &[PathBuf::from("/t")],
            &BuildOptions {
                k: Some(4),
                epochs: 2,
                sample: 100,
                seed: 5,
            },
        )
        .unwrap();
        let all = summaries(&db).unwrap();
        assert!(all.windows(2).all(|w| w[0].files >= w[1].files));
        let one = summary(&db, all[0].id).unwrap().unwrap();
        assert_eq!(one, all[0]);
        assert!(summary(&db, 999).unwrap().is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn build_refuses_impossible_requests() {
        let (path, store, db, _) = fixture("refuse");
        let opts = BuildOptions {
            k: Some(99),
            epochs: 2,
            sample: 100,
            seed: 1,
        };
        assert!(build(&db, &store, "test-model", &[], &opts).is_err());

        let opts = BuildOptions {
            k: Some(2),
            epochs: 0,
            sample: 100,
            seed: 1,
        };
        assert!(build(&db, &store, "test-model", &[], &opts).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn labels_skip_hex_and_format_tokens() {
        assert!(!keep_token("eff05a51"));
        assert!(!keep_token("12345"));
        assert!(!keep_token("pdf"));
        assert!(!keep_token("a"));
        assert!(!keep_token("ab"));
        assert!(keep_token("w2"));
        assert!(keep_token("employment"));
        assert!(!keep_token("d1f"));
        assert!(!keep_token("cd2"));
        assert!(!keep_token("dsc0461"));
        assert!(!keep_token("2024w2"));
    }

    #[test]
    fn default_k_grows_with_the_corpus_and_is_bounded() {
        assert_eq!(default_k(1), 2);
        assert_eq!(default_k(4), 2);
        assert_eq!(default_k(8_258), 91);
        assert_eq!(default_k(274_379), 256);
    }
}
