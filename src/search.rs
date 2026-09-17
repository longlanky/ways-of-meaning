//! Retrieval: dense vector search, full-text search, and their fusion.
//!
//! Both halves are needed. Dense search understands that "employment documents"
//! means a W-2, but it is weak on proper nouns it never saw in training — `wom
//! bosnia` has to reach `bosnia_croatia_trip_aug2024`, which is a lexical match
//! on a directory name. Full-text search is the opposite. Reciprocal Rank Fusion
//! combines them without needing their scores to be on a comparable scale.

use crate::config::Paths;
use crate::db::{Db, NO_VEC, prefix_range};
use crate::embed::Embedder;
use crate::rerank::Reranker;
use crate::vectors::VectorStore;
use anyhow::{Context, Result};
use rusqlite::params;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Candidates drawn from each retrieval arm before fusion. Larger than the final
/// limit so fusion has room to reorder, and cheap because both arms are fast.
const CANDIDATES: usize = 200;

/// Weight on the "a directory whose children matched" arm, relative to 1.0 for
/// the two file arms.
const DIR_FROM_CHILDREN_WEIGHT: f32 = 0.6;

/// Scaling applied to every directory arm so that directories and files have the
/// same total weight available to them.
///
/// Files draw on two arms (dense, lexical) for a total of 2.0. Directories draw on
/// three (dense, lexical, children) for 2.6, which structurally advantaged them:
/// measured on a 274k-file index, `wom employment documents` returned six
/// directories and no files at all, while the spec's own example shows one
/// directory followed by four files. This equalises the budget.
const DIR_ARM_SCALE: f32 = 2.0 / (1.0 + 1.0 + DIR_FROM_CHILDREN_WEIGHT);

/// Largest share of the result list that may be directories.
///
/// Even with equal weights, a directory whose *name* matches the query well beats
/// each individual file inside it, so an unconstrained list fills up with nested
/// directories that are each defensible and collectively useless. The spec's
/// examples all show a handful of directories among mostly files.
const MAX_DIR_FRACTION: f32 = 0.34;

/// RRF damping constant. 60 is the value from the original paper and is what
/// most implementations use; it makes the top few ranks dominate without letting
/// rank 1 alone decide the outcome.
const RRF_K: f32 = 60.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Dense and lexical, fused. The default.
    Hybrid,
    /// Full-text only. Works without a model, useful for verifying the index.
    Lexical,
    /// Vector similarity only.
    Dense,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ResultKind {
    File,
    Dir,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub kind: ResultKind,
    pub path: PathBuf,
    pub score: f32,
    pub snippet: String,
    /// 1-based rank in the dense arm, if it appeared there.
    pub dense_rank: Option<usize>,
    /// Raw cosine similarity from the dense arm. Kept for display and debugging:
    /// the fused score is a rank statistic and says nothing about how similar the
    /// result actually is.
    pub cosine: Option<f32>,
    /// 1-based rank in the lexical arm, if it appeared there.
    pub lexical_rank: Option<usize>,
    /// Raw cross-encoder logit, when reranking ran. Only comparable within the
    /// same reranker model.
    pub rerank: Option<f32>,
    /// For a directory result, how many indexed files it holds. `None` for files.
    pub file_count: Option<i64>,
}

impl SearchResult {
    /// Rendered the way the spec's examples show: directories tagged, files bare.
    pub fn display(&self) -> String {
        match self.kind {
            ResultKind::Dir => format!("directory:{}", self.path.display()),
            ResultKind::File => self.path.display().to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Request {
    pub text: String,
    pub scope: Vec<PathBuf>,
    pub limit: usize,
    pub mode: Mode,
    /// Dense hits below this cosine are dropped before fusion.
    ///
    /// Without a floor, the dense arm always returns its full candidate quota, so
    /// on a small corpus every file appears in the results regardless of
    /// relevance. The value is model-dependent, hence configurable.
    pub min_similarity: f32,
}

/// Run a search.
pub fn search(
    paths: &Paths,
    db: &Db,
    embedder: Option<&dyn Embedder>,
    reranker: Option<&dyn Reranker>,
    req: &Request,
) -> Result<Vec<SearchResult>> {
    if req.text.trim().is_empty() {
        return Ok(Vec::new());
    }

    // Five ranked lists, best first. Fusion cares only about position.
    let mut file_dense: Vec<Key> = Vec::new();
    let mut dir_dense: Vec<Key> = Vec::new();
    let mut cosines: HashMap<Key, f32> = HashMap::new();

    if req.mode != Mode::Lexical {
        if let Some(e) = embedder {
            let q = e.embed_query(&req.text).context("embedding the query")?;

            for (id, cos) in dense_files(paths, db, e, &q, req)? {
                file_dense.push((ResultKind::File, id));
                cosines.insert((ResultKind::File, id), cos);
            }
            for (id, cos) in dense_dirs(paths, db, e, &q, req)? {
                dir_dense.push((ResultKind::Dir, id));
                cosines.insert((ResultKind::Dir, id), cos);
            }
        }
    }

    let mut file_lex: Vec<Key> = Vec::new();
    let mut dir_lex: Vec<Key> = Vec::new();
    if req.mode != Mode::Dense {
        file_lex = lexical_files(db, req)?
            .into_iter()
            .map(|id| (ResultKind::File, id))
            .collect();
        dir_lex = lexical_dirs(db, req)?
            .into_iter()
            .map(|id| (ResultKind::Dir, id))
            .collect();
    }

    // Dense order first, then lexical-only hits, deduplicated via a set rather
    // than repeated `Vec::contains`, which is quadratic over two 200-item lists.
    let mut seen: std::collections::HashSet<Key> = HashSet::new();
    let best_files: Vec<Key> = file_dense
        .iter()
        .chain(&file_lex)
        .copied()
        .filter(|k| seen.insert(*k))
        .collect();
    let dir_children = dirs_from_children(db, &best_files, CANDIDATES)?;

    // A directory inherits evidence from its matching children, at a lower weight
    // than the direct arms: real signal, but weaker than the directory itself
    // matching, and at full weight it would push containers above the specific
    // file the user asked for.
    let fused = fuse(&[
        Arm::new(&file_dense),
        Arm::new(&file_lex),
        Arm { ranked: &dir_dense, weight: DIR_ARM_SCALE },
        Arm { ranked: &dir_lex, weight: DIR_ARM_SCALE },
        Arm {
            ranked: &dir_children,
            weight: DIR_FROM_CHILDREN_WEIGHT * DIR_ARM_SCALE,
        },
    ]);

    // A cross-encoder re-orders the top of the fused list — deeper than the
    // final limit so it has room to promote candidates, capped so a committed
    // query stays interactive. Lexical mode is the no-model fast path and
    // skips it by design. The directory cap then runs on the new ordering.
    // When limit > pool cap, the reranked head is followed by the fused tail
    // in fused order, so a large `--limit` never silently truncates.
    let pool_size = (req.limit * 2).clamp(20, 100);
    let reranked = match reranker {
        Some(rr) if req.mode != Mode::Lexical && !fused.is_empty() => {
            Some(rerank_pool(db, rr, &req.text, &fused, pool_size)?)
        }
        _ => None,
    };
    let keys = match &reranked {
        Some((pool, _)) => {
            let mut head = select(pool, req.limit);
            if head.len() < req.limit && fused.len() > pool.len() {
                let seen: HashSet<Key> = head.iter().copied().collect();
                let mut tail: Vec<(Key, f32)> = fused[pool.len()..]
                    .iter()
                    .filter(|(k, _)| !seen.contains(k))
                    .copied()
                    .collect();
                // Reuse the directory cap over head+tail in fused order is
                // complex; simplest correct: fill remaining slots with tail in
                // fused order (dirs already capped in head).
                for (k, _) in tail.drain(..) {
                    if head.len() >= req.limit {
                        break;
                    }
                    head.push(k);
                }
            }
            head
        }
        None => select(&fused, req.limit),
    };
    let meta = load_meta(db, &keys)?;

    let dense_pos = ranks(&[&file_dense, &dir_dense]);
    let lex_pos = ranks(&[&file_lex, &dir_lex]);

    let scores: HashMap<Key, f32> = fused.into_iter().collect();
    let rerank_scores: HashMap<Key, f32> = match reranked {
        Some((_, map)) => map,
        None => HashMap::new(),
    };
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let score = scores.get(&key).copied().unwrap_or(0.0);
        let Some(m) = meta.get(&key) else {
            // The row was deleted between ranking and resolution; skip it rather
            // than show a path that is no longer in the index.
            continue;
        };
        out.push(SearchResult {
            kind: key.0,
            path: PathBuf::from(&m.path),
            score,
            snippet: m.snippet.clone(),
            dense_rank: dense_pos.get(&key).copied(),
            cosine: cosines.get(&key).copied(),
            lexical_rank: lex_pos.get(&key).copied(),
            rerank: rerank_scores.get(&key).copied(),
            file_count: m.file_count,
        });
    }
    Ok(out)
}

/// Take the top `limit` results, capping how many may be directories.
///
/// Ranking within each kind is untouched; this only stops directories from filling
/// the whole list. If there are not enough files to fill the remainder, the freed
/// slots go back to directories rather than being wasted.
fn select(fused: &[(Key, f32)], limit: usize) -> Vec<Key> {
    if limit == 0 {
        return Vec::new();
    }
    let max_dirs = ((limit as f32 * MAX_DIR_FRACTION).round() as usize).max(1);

    let mut out: Vec<Key> = Vec::with_capacity(limit);
    let mut dirs = 0usize;
    let mut deferred: Vec<Key> = Vec::new();
    for (key, _) in fused {
        if out.len() >= limit {
            break;
        }
        if key.0 == ResultKind::Dir {
            if dirs >= max_dirs {
                deferred.push(*key);
                continue;
            }
            dirs += 1;
        }
        out.push(*key);
    }
    // Backfill with the directories that were held back, in their original order.
    for key in deferred {
        if out.len() >= limit {
            break;
        }
        out.push(key);
    }
    out
}

/// 1-based position of each key across several disjoint ranked lists.
fn ranks(lists: &[&[Key]]) -> HashMap<Key, usize> {
    lists
        .iter()
        .flat_map(|l| l.iter().enumerate().map(|(i, k)| (*k, i + 1)))
        .collect()
}

// ------------------------------------------------------------------ rerank

/// Re-order the head of the fused list with a cross-encoder. Returns the pool
/// (best first by rerank score) and the raw scores for display. Ties keep the
/// fused order via the stable sort.
fn rerank_pool(
    db: &Db,
    rr: &dyn Reranker,
    query: &str,
    fused: &[(Key, f32)],
    size: usize,
) -> Result<(Vec<(Key, f32)>, HashMap<Key, f32>)> {
    let pool: Vec<Key> = fused.iter().take(size).map(|(k, _)| *k).collect();
    let texts = rerank_texts(db, &pool)?;
    let docs: Vec<String> = pool
        .iter()
        .map(|k| texts.get(k).cloned().unwrap_or_default())
        .collect();
    let scores = rr.score(query, &docs).context("reranking candidates")?;
    if scores.len() != pool.len() {
        anyhow::bail!(
            "reranker returned {} scores for {} documents",
            scores.len(),
            pool.len()
        );
    }

    let mut order: Vec<usize> = (0..pool.len()).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    let mut score_map = HashMap::with_capacity(pool.len());
    let mut out = Vec::with_capacity(pool.len());
    for i in order {
        let key = pool[i];
        score_map.insert(key, scores[i]);
        out.push((key, scores[i]));
    }
    Ok((out, score_map))
}

/// The text a cross-encoder judges: filename plus the head of the extracted
/// body for a file (bounded so a long document does not dominate the pair),
/// the path for a directory, which has no body of its own.
fn rerank_texts(db: &Db, keys: &[Key]) -> Result<HashMap<Key, String>> {
    let mut out = HashMap::new();
    for kind in [ResultKind::File, ResultKind::Dir] {
        let ids: Vec<i64> = keys
            .iter()
            .filter(|(k, _)| *k == kind)
            .map(|(_, id)| *id)
            .collect();
        if ids.is_empty() {
            continue;
        }
        // A handful of ids, so an IN list of placeholders is fine — the same
        // pattern as `load_meta`.
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = match kind {
            // files_fts.rowid == files.id by construction.
            ResultKind::File => format!(
                "SELECT rowid, name, body FROM files_fts WHERE rowid IN ({placeholders})"
            ),
            // A directory has no body of its own; the placeholder third column
            // keeps the row shape shared with the file query.
            ResultKind::Dir => {
                format!("SELECT id, path, '' FROM dirs WHERE id IN ({placeholders})")
            }
        };
        let mut st = db.conn.prepare(&sql)?;
        let refs: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|i| i as &dyn rusqlite::ToSql).collect();
        let rows = st.query_map(refs.as_slice(), |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })?;
        for row in rows {
            let (id, col1, col2) = row?;
            let text = match kind {
                ResultKind::File => {
                    let body = col2.unwrap_or_default();
                    let head = match body.char_indices().nth(2000) {
                        Some((i, _)) => &body[..i],
                        None => body.as_str(),
                    };
                    format!("{col1}\n{head}")
                }
                ResultKind::Dir => col1,
            };
            out.insert((kind, id), text);
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------ dense

/// Dense hits as (file id, cosine), best first, already filtered by the floor.
fn dense_files(
    paths: &Paths,
    db: &Db,
    embedder: &dyn Embedder,
    q: &[f32],
    req: &Request,
) -> Result<Vec<(i64, f32)>> {
    dense_arm(&paths.file_vectors(), db, embedder, q, req, ResultKind::File)
}

/// Dense hits over directory vectors.
fn dense_dirs(
    paths: &Paths,
    db: &Db,
    embedder: &dyn Embedder,
    q: &[f32],
    req: &Request,
) -> Result<Vec<(i64, f32)>> {
    dense_arm(&paths.dir_vectors(), db, embedder, q, req, ResultKind::Dir)
}

fn dense_arm(
    vec_path: &Path,
    db: &Db,
    embedder: &dyn Embedder,
    q: &[f32],
    req: &Request,
    kind: ResultKind,
) -> Result<Vec<(i64, f32)>> {
    if !vec_path.exists() {
        return Ok(Vec::new());
    }
    let store = VectorStore::open(vec_path, embedder.dim(), embedder.model_id())?;
    if store.high_water() == 0 {
        return Ok(Vec::new());
    }

    // Scope is applied as an allow-list over vector rows. Built only when a scope
    // was given, since it costs a query over the ids in range.
    let allow = if req.scope.is_empty() {
        None
    } else {
        Some(scope_allow_list(db, &req.scope, store.high_water(), kind)?)
    };

    let hits = store.search(q, CANDIDATES, allow.as_deref())?;
    // vec_row == the row id in its table, so a hit's row is already the id.
    Ok(hits
        .into_iter()
        .filter(|h| h.score >= req.min_similarity)
        .map(|h| (h.row as i64, h.score))
        .collect())
}

/// Why a hybrid search would silently return full-text-only results, or `None`
/// when the vector index is present and populated.
///
/// `write_batch` commits the full-text rows before embedding runs, so a scan
/// that dies midway leaves a fully searchable lexical index over an empty
/// vector store — and every later hybrid search quietly degrades to exact
/// matching. That is worth one warning per CLI invocation. Only the file store
/// is checked: it is written first, so it is the earliest place the gap shows.
pub fn dense_gap(paths: &Paths, embedder: &dyn Embedder) -> Option<String> {
    store_gap(&paths.file_vectors(), embedder.dim(), embedder.model_id())
}

fn store_gap(vec_path: &Path, dim: usize, model_id: &str) -> Option<String> {
    if !vec_path.exists() {
        return Some("no vector index found".to_string());
    }
    // Open errors (a model swap, a truncated file) are *not* reported here: the
    // dense arm itself fails loudly with the real cause. This check exists only
    // for the silent cases.
    match VectorStore::open(vec_path, dim, model_id) {
        Ok(store) if store.high_water() == 0 => {
            Some("the vector index is empty (indexing was likely interrupted)".to_string())
        }
        _ => None,
    }
}

/// A per-row boolean mask of vector rows that lie under one of the scope paths.
fn scope_allow_list(
    db: &Db,
    scope: &[PathBuf],
    rows: usize,
    kind: ResultKind,
) -> Result<Vec<bool>> {
    let mut allow = vec![false; rows];
    let table = table_of(kind);
    // For directories the scope directory itself is also eligible — a scoped
    // search should be able to return the directory it was scoped to. Folding that
    // into the predicate avoids a second query per scope entry.
    let sql = format!(
        "SELECT vec_row FROM {table}
         WHERE vec_row != {NO_VEC} AND ((path >= ?1 AND path < ?2) OR path = ?3)"
    );
    let mut st = db.conn.prepare(&sql)?;
    for dir in scope {
        let bare = dir.to_string_lossy().trim_end_matches('/').to_string();
        let (lo, hi) = dir_prefix_range(dir);
        // A file can never equal the scope directory, so binding it for both kinds
        // is harmless and keeps one statement.
        let own = match kind {
            ResultKind::Dir => bare,
            ResultKind::File => String::new(),
        };
        let it = st.query_map(params![lo, hi, own], |r| r.get::<_, i64>(0))?;
        for row in it {
            let row = row? as usize;
            if row < rows {
                allow[row] = true;
            }
        }
    }
    Ok(allow)
}

/// SQL table backing each result kind.
fn table_of(kind: ResultKind) -> &'static str {
    match kind {
        ResultKind::File => "files",
        ResultKind::Dir => "dirs",
    }
}

/// Half-open key range covering everything strictly inside `dir`.
///
/// The trailing separator is load-bearing: it is what stops `/p` from also
/// matching `/p-old`.
fn dir_prefix_range(dir: &Path) -> (String, String) {
    prefix_range(&format!("{}/", dir.to_string_lossy().trim_end_matches('/')))
}

// ------------------------------------------------------------------ lexical

/// Full-text hits over files, ranked by bm25.
fn lexical_files(db: &Db, req: &Request) -> Result<Vec<i64>> {
    // Column weights (path, name, body): a hit in the filename means far more
    // than one buried in the body.
    lexical_arm(db, req, "files_fts", "files", "bm25(files_fts, 1.0, 3.0, 1.0)")
}

/// Full-text hits over directory names.
fn lexical_dirs(db: &Db, req: &Request) -> Result<Vec<i64>> {
    lexical_arm(db, req, "dirs_fts", "dirs", "bm25(dirs_fts, 1.0, 3.0)")
}

fn lexical_arm(
    db: &Db,
    req: &Request,
    fts: &str,
    table: &str,
    rank: &str,
) -> Result<Vec<i64>> {
    let Some(match_expr) = fts_query(&req.text) else {
        return Ok(Vec::new());
    };

    // bm25() returns increasingly negative values for better matches, so
    // ascending order is best-first.
    // `fts`, `table` and `rank` are fixed literals from the callers above.
    let mut sql = format!(
        "SELECT t.id FROM {fts} JOIN {table} t ON t.id = {fts}.rowid WHERE {fts} MATCH ?1"
    );
    if !req.scope.is_empty() {
        sql.push_str(" AND (");
        for i in 0..req.scope.len() {
            if i > 0 {
                sql.push_str(" OR ");
            }
            // Parameters start at 2; three per scope entry: subtree range plus
            // the scope directory itself (parity with the dense allow-list, so
            // a scoped search can return the directory it was scoped to).
            // Harmless for files, which can never equal a directory path.
            sql.push_str(&format!(
                "(t.path >= ?{} AND t.path < ?{} OR t.path = ?{})",
                2 + i * 3,
                3 + i * 3,
                4 + i * 3
            ));
        }
        sql.push(')');
    }
    sql.push_str(&format!(" ORDER BY {rank} LIMIT ?"));
    sql.push_str(&(2 + req.scope.len() * 3).to_string());

    let mut binds: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(match_expr)];
    for dir in &req.scope {
        let (lo, hi) = dir_prefix_range(dir);
        let bare = dir.to_string_lossy().trim_end_matches('/').to_string();
        binds.push(Box::new(lo));
        binds.push(Box::new(hi));
        binds.push(Box::new(bare));
    }
    binds.push(Box::new(CANDIDATES as i64));

    let mut st = db.conn.prepare(&sql)?;
    let refs: Vec<&dyn rusqlite::ToSql> = binds.iter().map(|b| b.as_ref()).collect();
    let rows = st.query_map(refs.as_slice(), |r| r.get::<_, i64>(0))?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

/// Build an FTS5 MATCH expression from raw user input.
///
/// User text cannot be passed through: FTS5 has its own syntax, so a query like
/// `C++` or `a"b` or `NOT` is a syntax error rather than a search. Each word is
/// therefore extracted and emitted as a quoted literal, and the final word gets a
/// prefix wildcard so the TUI can search usefully while the user is still typing.
///
/// Returns `None` when the input contains no searchable word at all.
pub fn fts_query(text: &str) -> Option<String> {
    let words: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    if words.is_empty() {
        return None;
    }

    // OR rather than AND: a query is a description, not a conjunction of required
    // terms, and RRF will favour documents matching more of it anyway.
    let mut parts: Vec<String> = Vec::with_capacity(words.len());
    for (i, w) in words.iter().enumerate() {
        let last = i + 1 == words.len();
        // A one-character prefix wildcard matches far too much to be useful.
        if last && w.len() >= 2 {
            parts.push(format!("\"{w}\"*"));
        } else {
            parts.push(format!("\"{w}\""));
        }
    }
    Some(parts.join(" OR "))
}

// ------------------------------------------------------------------ fusion

/// Identifies one candidate. Files and directories live in separate id spaces, so
/// the kind has to travel with the id for fusion to mix them.
pub type Key = (ResultKind, i64);

/// A ranked list of candidates plus the weight its agreement carries.
pub struct Arm<'a> {
    pub ranked: &'a [Key],
    /// Multiplier on this arm's RRF contribution. 1.0 for the primary arms; less
    /// for weaker evidence such as "a directory whose children matched".
    pub weight: f32,
}

impl<'a> Arm<'a> {
    /// A primary arm, carrying full weight.
    pub fn new(ranked: &'a [Key]) -> Self {
        Self { ranked, weight: 1.0 }
    }
}

/// Reciprocal Rank Fusion over several ranked lists.
///
/// Chosen over score normalisation because cosine similarity and bm25 are not
/// comparable quantities — bm25 is unbounded and negative, cosine is in [-1, 1] —
/// and any attempt to rescale them introduces a tuning parameter per corpus. RRF
/// needs only the positions.
pub fn fuse(arms: &[Arm]) -> Vec<(Key, f32)> {
    let mut scores: HashMap<Key, f32> = HashMap::new();
    // Best rank seen per key, used only to break ties deterministically.
    let mut best: HashMap<Key, usize> = HashMap::new();
    for arm in arms {
        for (i, key) in arm.ranked.iter().enumerate() {
            let rank = i + 1;
            *scores.entry(*key).or_insert(0.0) += arm.weight / (RRF_K + rank as f32);
            let e = best.entry(*key).or_insert(rank);
            *e = (*e).min(rank);
        }
    }
    let mut out: Vec<(Key, f32)> = scores.into_iter().collect();
    out.sort_unstable_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| best[&a.0].cmp(&best[&b.0]))
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

/// Derive a ranked list of directories from a ranked list of files: each file
/// contributes reciprocal-rank credit to its parent directory.
///
/// This is the third signal behind `directory:employment_docs` — a directory
/// whose *several* children all rank well is a better answer than any one of
/// them, which neither its name nor its centroid alone expresses.
fn dirs_from_children(db: &Db, files_ranked: &[Key], limit: usize) -> Result<Vec<Key>> {
    if files_ranked.is_empty() {
        return Ok(Vec::new());
    }
    let ids: Vec<i64> = files_ranked.iter().map(|(_, id)| *id).collect();
    let placeholders = std::iter::repeat_n("?", ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let sql =
        format!("SELECT id, parent_dir_id FROM files WHERE id IN ({placeholders}) AND parent_dir_id IS NOT NULL");
    let mut st = db.conn.prepare(&sql)?;
    let refs: Vec<&dyn rusqlite::ToSql> = ids.iter().map(|i| i as &dyn rusqlite::ToSql).collect();
    let rows = st.query_map(refs.as_slice(), |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
    })?;

    let rank_of: HashMap<i64, usize> = ids.iter().enumerate().map(|(i, id)| (*id, i + 1)).collect();
    let mut credit: HashMap<i64, f32> = HashMap::new();
    for row in rows {
        let (fid, did) = row?;
        if let Some(rank) = rank_of.get(&fid) {
            *credit.entry(did).or_insert(0.0) += 1.0 / (RRF_K + *rank as f32);
        }
    }

    let mut v: Vec<(i64, f32)> = credit.into_iter().collect();
    v.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v.truncate(limit);
    Ok(v.into_iter().map(|(id, _)| (ResultKind::Dir, id)).collect())
}

// ------------------------------------------------------------------ metadata

/// Display metadata for one result.
struct Meta {
    path: String,
    snippet: String,
    file_count: Option<i64>,
}

/// Resolve display metadata for the keys that will actually be shown.
fn load_meta(db: &Db, keys: &[Key]) -> Result<HashMap<Key, Meta>> {
    let mut out = HashMap::new();
    if keys.is_empty() {
        return Ok(out);
    }

    for kind in [ResultKind::File, ResultKind::Dir] {
        let ids: Vec<i64> = keys
            .iter()
            .filter(|(k, _)| *k == kind)
            .map(|(_, id)| *id)
            .collect();
        if ids.is_empty() {
            continue;
        }
        // A handful of ids, so an IN list of placeholders is fine and avoids a
        // temp table.
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = match kind {
            ResultKind::File => {
                format!("SELECT id, path, snippet FROM files WHERE id IN ({placeholders})")
            }
            // A directory's "snippet" is a count of what it holds, which is the
            // useful thing to show next to it in the preview pane.
            // The count is returned as data; rendering it is the caller's job.
            ResultKind::Dir => {
                format!("SELECT id, path, file_count FROM dirs WHERE id IN ({placeholders})")
            }
        };
        let mut st = db.conn.prepare(&sql)?;
        let refs: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|i| i as &dyn rusqlite::ToSql).collect();
        let rows = st.query_map(refs.as_slice(), |r| {
            let id: i64 = r.get(0)?;
            let path: String = r.get(1)?;
            Ok(match kind {
                ResultKind::File => (id, Meta { path, snippet: r.get(2)?, file_count: None }),
                ResultKind::Dir => (
                    id,
                    Meta { path, snippet: String::new(), file_count: Some(r.get(2)?) },
                ),
            })
        })?;
        for row in rows {
            let (id, meta) = row?;
            out.insert((kind, id), meta);
        }
    }
    Ok(out)
}

/// Shorten a path for display by replacing the home directory with `~`.
pub fn tilde(p: &Path) -> String {
    let s = p.to_string_lossy();
    if let Some(home) = std::env::var_os("HOME") {
        let home = home.to_string_lossy().into_owned();
        if !home.is_empty() && s.starts_with(&home) {
            return format!("~{}", &s[home.len()..]);
        }
    }
    s.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_query_quotes_words_and_prefixes_the_last() {
        assert_eq!(
            fts_query("employment documents").unwrap(),
            "\"employment\" OR \"documents\"*"
        );
    }

    #[test]
    fn fts_query_neutralises_fts5_syntax() {
        // Each of these is a syntax error if passed to MATCH unescaped.
        for input in ["C++", "a\"b", "NOT bosnia", "foo AND bar", "(x OR y)", "^start", "a*b"] {
            let q = fts_query(input).expect(input);
            // Every emitted term is a quoted literal, so no operator survives
            // except the ones we add ourselves.
            let stripped = q.replace(" OR ", " ").replace('*', "");
            for term in stripped.split_whitespace() {
                assert!(
                    term.starts_with('"') && term.ends_with('"'),
                    "unquoted term {term:?} from input {input:?}"
                );
            }
        }
    }

    #[test]
    fn fts_query_on_punctuation_only_input_is_none() {
        assert!(fts_query("+++").is_none());
        assert!(fts_query("   ").is_none());
        assert!(fts_query("").is_none());
    }

    #[test]
    fn fts_query_does_not_prefix_a_single_character_last_word() {
        // `"a"*` would match a large fraction of the corpus.
        let q = fts_query("bosnia a").unwrap();
        assert!(!q.contains("\"a\"*"), "got {q}");
    }

    /// Build a file-keyed arm at weight 1.0 for the fusion tests.
    fn farm(ids: &[i64]) -> Vec<Key> {
        ids.iter().map(|i| (ResultKind::File, *i)).collect()
    }

    #[test]
    fn fuse_prefers_agreement_between_the_two_arms() {
        // 7 is ranked mid-list by both arms; 1 and 2 are top of only one each.
        let dense = farm(&[1, 7, 3]);
        let lexical = farm(&[2, 7, 4]);
        let out = fuse(&[
            Arm::new(&dense),
            Arm::new(&lexical),
        ]);
        assert_eq!(out[0].0, (ResultKind::File, 7), "agreed-on result should win");
    }

    #[test]
    fn fuse_ranks_by_position_not_list_length() {
        let a = farm(&[10]);
        let b = farm(&[20, 10]);
        let out = fuse(&[
            Arm::new(&a),
            Arm::new(&b),
        ]);
        // 10 appears at rank 1 and rank 2; 20 only at rank 1.
        assert_eq!(out[0].0, (ResultKind::File, 10));
    }

    #[test]
    fn fuse_is_deterministic_and_handles_empty_lists() {
        let empty: Vec<Key> = Vec::new();
        assert!(
            fuse(&[
                Arm::new(&empty),
                Arm::new(&empty),
            ])
            .is_empty()
        );

        let a = farm(&[1, 2, 3]);
        let b = farm(&[3, 2, 1]);
        let arms = || {
            vec![
                Arm::new(&a),
                Arm::new(&b),
            ]
        };
        assert_eq!(fuse(&arms()), fuse(&arms()));
    }

    #[test]
    fn fuse_with_a_single_arm_preserves_its_order() {
        let only = farm(&[5, 9, 1]);
        let out: Vec<i64> = fuse(&[Arm::new(&only)])
            .into_iter()
            .map(|((_, id), _)| id)
            .collect();
        assert_eq!(out, vec![5, 9, 1]);
    }

    #[test]
    fn fuse_mixes_files_and_directories_without_id_collisions() {
        // File 1 and directory 1 are different things despite sharing an id.
        let files = vec![(ResultKind::File, 1i64)];
        let dirs = vec![(ResultKind::Dir, 1i64)];
        let out = fuse(&[
            Arm::new(&files),
            Arm::new(&dirs),
        ]);
        assert_eq!(out.len(), 2, "file and dir with the same id were merged");
    }

    #[test]
    fn arm_weight_scales_its_influence() {
        // A weak arm alone must not outrank a strong arm's top hit.
        let strong = farm(&[1]);
        let weak = vec![(ResultKind::Dir, 2i64)];
        let out = fuse(&[
            Arm::new(&strong),
            Arm { ranked: &weak, weight: 0.6 },
        ]);
        assert_eq!(out[0].0, (ResultKind::File, 1));
        assert!(out[0].1 > out[1].1);
    }

    #[test]
    fn display_tags_directories_per_the_spec() {
        let d = SearchResult {
            kind: ResultKind::Dir,
            path: PathBuf::from("/h/employment_docs"),
            score: 1.0,
            snippet: String::new(),
            dense_rank: None,
            cosine: None,
            lexical_rank: None,
            rerank: None,
            file_count: None,
        };
        assert_eq!(d.display(), "directory:/h/employment_docs");

        let f = SearchResult {
            kind: ResultKind::File,
            path: PathBuf::from("/h/2024_W2.pdf"),
            ..d
        };
        assert_eq!(f.display(), "/h/2024_W2.pdf");
    }

    #[test]
    fn tilde_abbreviates_the_home_directory_only_as_a_prefix() {
        // SAFETY: single-threaded test.
        unsafe { std::env::set_var("HOME", "/home/tester") };
        assert_eq!(tilde(Path::new("/home/tester/Documents/a")), "~/Documents/a");
        assert_eq!(tilde(Path::new("/etc/passwd")), "/etc/passwd");
    }

    #[test]
    fn store_gap_flags_only_missing_and_empty_stores() {
        let p = std::env::temp_dir().join("wom-search-gap-test.i8");
        std::fs::remove_file(&p).ok();

        assert!(
            store_gap(&p, 4, "test-model").is_some(),
            "a missing store must be flagged"
        );

        let mut store = VectorStore::open(&p, 4, "test-model").unwrap();
        let gap = store_gap(&p, 4, "test-model").unwrap();
        assert!(gap.contains("empty"), "a fresh store is empty, got: {gap}");

        store.put(0, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(
            store_gap(&p, 4, "test-model").is_none(),
            "a populated store must not be flagged"
        );

        // A model mismatch is the dense arm's loud error, not this check's
        // concern — the silent-path detector stays out of it.
        assert!(store_gap(&p, 4, "other-model").is_none());
        std::fs::remove_file(&p).ok();
    }

    // ------------------------------------------------------------ rerank

    use crate::rerank::test_support::MockReranker;
    use rusqlite::params;

    fn insert_file(db: &Db, path: &str, body: &str) -> i64 {
        db.conn
            .execute(
                "INSERT INTO files(path, mtime_ns, size, seen_scan) VALUES (?1, 1, 1, 1)",
                params![path],
            )
            .unwrap();
        let id = db.conn.last_insert_rowid();
        let name = Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        db.put_fts(id, path, &name, body).unwrap();
        id
    }

    fn insert_dir(db: &Db, path: &str) -> i64 {
        db.conn
            .execute("INSERT INTO dirs(path) VALUES (?1)", params![path])
            .unwrap();
        let id = db.conn.last_insert_rowid();
        let name = Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        db.put_dir_fts(id, path, &name).unwrap();
        id
    }

    fn hybrid_req(text: &str) -> Request {
        Request {
            text: text.to_string(),
            scope: Vec::new(),
            limit: 10,
            mode: Mode::Hybrid,
            min_similarity: 0.0,
        }
    }

    #[test]
    fn reranker_reorders_fused_results() {
        let db = Db::open_in_memory().unwrap();
        // `a` repeats the query term, so BM25 favours it; the reranker favours
        // `b` by name. The reranker's say is final.
        insert_file(&db, "/t/plain_report.txt", "earnings earnings earnings summary");
        insert_file(&db, "/t/tax_return_2024.txt", "earnings");
        let rr = MockReranker { hot: "tax_return" };

        let out = search(
            &Paths::for_test(),
            &db,
            None,
            Some(&rr),
            &hybrid_req("earnings"),
        )
        .unwrap();
        assert_eq!(out.len(), 2, "both files match lexically");
        assert_eq!(out[0].path.to_string_lossy(), "/t/tax_return_2024.txt");
        assert!(
            out[0].rerank.unwrap() > out[1].rerank.unwrap(),
            "raw rerank scores are carried for display"
        );
    }

    #[test]
    fn reranker_sees_directories_by_path() {
        let db = Db::open_in_memory().unwrap();
        insert_file(&db, "/t/tax_guide.txt", "tax filing instructions");
        insert_dir(&db, "/t/tax_stuff");
        let rr = MockReranker { hot: "tax_stuff" };

        let out = search(&Paths::for_test(), &db, None, Some(&rr), &hybrid_req("tax"))
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].kind, ResultKind::Dir, "reranker promoted the directory");
    }

    #[test]
    fn lexical_mode_skips_the_reranker() {
        let db = Db::open_in_memory().unwrap();
        insert_file(&db, "/t/tax_return_2024.txt", "earnings");
        insert_file(&db, "/t/plain_report.txt", "earnings report");
        let rr = MockReranker { hot: "tax_return" };
        let req = Request {
            mode: Mode::Lexical,
            ..hybrid_req("earnings")
        };

        let out = search(&Paths::for_test(), &db, None, Some(&rr), &req).unwrap();
        assert_eq!(out.len(), 2);
        assert!(
            out.iter().all(|r| r.rerank.is_none()),
            "lexical mode is the no-model fast path"
        );
    }

    #[test]
    fn directory_cap_still_applies_after_reranking() {
        let db = Db::open_in_memory().unwrap();
        // Four dirs the reranker loves and one file it hates. Without the cap
        // the file would be squeezed out entirely; with it, one dir passes the
        // gate, the file takes its slot, and the remainder backfills.
        for d in ["d1", "d2", "d3", "d4"] {
            insert_dir(&db, &format!("/t/tax_{d}"));
        }
        insert_file(&db, "/t/tax_notes.txt", "tax");
        let rr = MockReranker { hot: "tax_d" };
        let req = Request {
            limit: 4,
            ..hybrid_req("tax")
        };

        let out = search(&Paths::for_test(), &db, None, Some(&rr), &req).unwrap();
        let dirs = out.iter().filter(|r| r.kind == ResultKind::Dir).count();
        assert_eq!(dirs, 3, "one through the cap, two backfilled, got {dirs}");
        assert!(
            out.iter().any(|r| r.kind == ResultKind::File),
            "the lone file must survive the reranked dir sweep"
        );
    }
}
