//! Building and refreshing the index.

pub mod extract;
pub mod geo;
pub mod walk;

use crate::config::{Config, Paths, Profile, deepest_root};
use crate::db::{Db, FileStat, NO_VEC, path_str};
use crate::embed::Embedder;
use crate::vectors::VectorStore;
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rusqlite::params;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Files prepared before each embed+write cycle. Bounds peak memory: at 512 files
/// with ~1 KB of body each this is ~0.5 MB in flight, rather than holding the
/// bodies of all 100k+ files at once.
const BATCH: usize = 512;

/// Directory names handed to the embedder per forward pass during the rollup.
const EMBED_GROUP: usize = BATCH;

/// Largest share a directory's content centroid can take in its vector, reached
/// only when the directory is perfectly coherent. See [`rollup_dirs`].
const MAX_CENTROID_WEIGHT: f32 = 0.5;

#[derive(Debug, Clone, Default)]
pub struct ScanOptions {
    /// Re-extract and re-embed regardless of the mtime and content-hash caches.
    pub force: bool,
    /// Walk and report without writing.
    pub dry_run: bool,
    /// Restrict to one configured root.
    pub only_root: Option<PathBuf>,
    /// Compute and store vectors. False indexes metadata and full text only,
    /// which is useful before a model is available and for `--lexical` search.
    pub embed: bool,
    /// Print a progress bar.
    pub progress: bool,
}

#[derive(Debug, Default, Clone)]
pub struct ScanStats {
    pub seen: usize,
    pub added: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub embedded: usize,
    pub deleted_files: usize,
    pub deleted_dirs: usize,
    pub dirs: usize,
    /// Entries the walker could not read, e.g. permission-denied directories.
    pub skipped: usize,
    pub elapsed_secs: f64,
}

impl ScanStats {
    pub fn embed_rate(&self) -> Option<f64> {
        if self.embedded > 0 && self.elapsed_secs > 0.0 {
            Some(self.embedded as f64 / self.elapsed_secs)
        } else {
            None
        }
    }
}

/// What one file needs, after extraction and change detection.
struct Prepared {
    path: PathBuf,
    dir: PathBuf,
    size: u64,
    mtime_ns: i64,
    name_tokens: String,
    body: String,
    snippet: String,
    kind: extract::Kind,
    hash: Vec<u8>,
    /// Text handed to the model. Empty when nothing needs embedding.
    embed_text: String,
    /// Existing row id, if the file was already known.
    existing_id: Option<i64>,
    needs_embed: bool,
    unchanged: bool,
}

/// Scan every configured root, updating the index in place.
pub fn scan(
    paths: &Paths,
    cfg: &Config,
    embedder: Option<&dyn Embedder>,
    opts: &ScanOptions,
) -> Result<ScanStats> {
    let start = Instant::now();
    paths.ensure_data_dir()?;

    // One scan at a time. A second `wom index` should wait or bail rather than
    // interleave writes into the same vector file.
    let _lock = if opts.dry_run {
        None
    } else {
        Some(ScanLock::acquire(&paths.scan_lock())?)
    };

    let db = Db::open(&paths.db_file())?;
    let caps = extract::Capabilities::detect();
    for gap in caps.describe_gaps(cfg.extract.geo) {
        eprintln!("wom: {gap}");
    }
    if cfg.extract.geo && !cfg.extract.media {
        eprintln!("wom: extract.geo is on but extract.media is off; no media is read, so no GPS data will be found");
    }

    let mut roots: Vec<&crate::config::Root> = match &opts.only_root {
        Some(r) => cfg
            .roots
            .iter()
            .filter(|x| &x.path == r)
            .collect::<Vec<_>>(),
        None => cfg.roots.iter().collect(),
    };
    // Walk each subtree once. A root nested inside another still contributes its
    // profile through `Config::profile_for`, which prefers the longest prefix, so
    // dropping it here changes nothing except the duplicated work.
    roots.sort_by_key(|r| r.path.as_os_str().len());
    let mut walk_roots: Vec<&crate::config::Root> = Vec::with_capacity(roots.len());
    for r in roots {
        if walk_roots.iter().any(|k| r.path.starts_with(&k.path)) {
            continue;
        }
        walk_roots.push(r);
    }
    let roots = walk_roots;
    if roots.is_empty() {
        anyhow::bail!(
            "no roots configured{}. Run `wom init` first.",
            opts.only_root
                .as_ref()
                .map(|r| format!(" matching {}", r.display()))
                .unwrap_or_default()
        );
    }

    let mut store = match (opts.embed, embedder) {
        (true, Some(e)) => Some(VectorStore::open(
            &paths.file_vectors(),
            e.dim(),
            e.model_id(),
        )?),
        _ => None,
    };

    let scan_gen = if opts.dry_run {
        db.get_meta_i64("scan_gen")?.unwrap_or(0)
    } else {
        db.next_scan_gen()?
    };

    let mut stats = ScanStats::default();
    let mut all_dirs: HashSet<PathBuf> = HashSet::new();

    for root in &roots {
        let known = db.file_stats_under(&format!("{}/", path_str(&root.path)))?;

        // The walk is cheap relative to extraction (~0.6s for 100k files), so
        // collecting paths up front keeps the pipeline below simple.
        let sink = std::sync::Mutex::new(Vec::new());
        stats.skipped += walk::walk_root(&root.path, cfg, |f| {
            sink.lock().expect("walk sink mutex").push(f);
        })?;
        let found = sink.into_inner().expect("walk sink mutex");
        stats.seen += found.len();

        let bar = if opts.progress {
            let b = ProgressBar::new(found.len() as u64);
            b.set_style(
                ProgressStyle::with_template(
                    "  {msg:<22} [{bar:28}] {pos}/{len} {per_sec} eta {eta}",
                )
                .expect("progress template")
                .progress_chars("=> "),
            );
            b.set_message(short_root(&root.path));
            Some(b)
        } else {
            None
        };

        for chunk in found.chunks(BATCH) {
            let prepared: Vec<Prepared> = chunk
                .par_iter()
                .filter_map(|f| {
                    prepare(f, &root.path, root.profile, cfg, &caps, &known, opts)
                })
                .collect();

            for p in &prepared {
                all_dirs.insert(p.dir.clone());
            }

            if !opts.dry_run {
                write_batch(&db, store.as_mut(), embedder, &prepared, scan_gen, &mut stats)?;
            } else {
                for p in &prepared {
                    if p.unchanged {
                        stats.unchanged += 1;
                    } else if p.existing_id.is_some() {
                        stats.updated += 1;
                    } else {
                        stats.added += 1;
                    }
                }
            }
            if let Some(b) = &bar {
                b.inc(chunk.len() as u64);
            }
        }
        if let Some(b) = bar {
            b.finish_and_clear();
        }
    }

    if !opts.dry_run {
        // Directory rows are needed for the `directory:` results and for the
        // rollup that computes their vectors.
        let root_paths: Vec<PathBuf> = roots.iter().map(|r| r.path.clone()).collect();
        stats.dirs = upsert_dirs(&db, &all_dirs, &root_paths, scan_gen)?;

        // Deleted files must have their vector rows zeroed as well as their
        // metadata removed, or a stale vector would keep matching queries.
        let doomed = db.vec_rows_not_seen(scan_gen)?;
        let doomed_dirs = db.dir_vec_rows_not_seen(scan_gen)?;
        if let Some(s) = store.as_mut() {
            for row in &doomed {
                s.clear(*row as usize)?;
            }
        }
        if !doomed_dirs.is_empty() {
            if let Some(e) = embedder {
                let mut ds = VectorStore::open(&paths.dir_vectors(), e.dim(), e.model_id())?;
                for row in &doomed_dirs {
                    ds.clear(*row as usize)?;
                }
                ds.flush()?;
            }
        }
        let (df, dd) = db.gc(scan_gen)?;
        stats.deleted_files = df;
        stats.deleted_dirs = dd;

        if let Some(s) = &store {
            s.flush()?;
        }

        // Directory vectors must be computed after the file vectors they average,
        // and after GC so deleted files do not contribute to a centroid.
        if let (Some(e), Some(fs)) = (embedder, store.as_ref()) {
            let mut ds = VectorStore::open(&paths.dir_vectors(), e.dim(), e.model_id())?;
            let root_paths: Vec<PathBuf> = roots.iter().map(|r| r.path.clone()).collect();
            rollup_dirs(&db, fs, &mut ds, e, &root_paths)?;
        }
        db.set_meta("last_scan_at", &now_secs().to_string())?;
        if let Some(e) = embedder {
            db.set_meta("model_id", e.model_id())?;
            db.set_meta("dim", &e.dim().to_string())?;
        }
    }

    stats.elapsed_secs = start.elapsed().as_secs_f64();
    if let Some(rate) = stats.embed_rate() {
        // End-to-end throughput, including extraction and database writes. This is
        // what predicts how long an index takes, and it is roughly half the
        // model-only figure `wom bench` reports — so the two are stored separately
        // rather than overwriting each other under one key.
        db.set_meta(crate::db::META_INDEX_RATE, &format!("{rate:.1}")).ok();
    }
    Ok(stats)
}

/// Extract and decide what to do with one file. Returns `None` only if the file
/// vanished between the walk and now, which is normal on a live filesystem.
fn prepare(
    f: &walk::Found,
    root: &Path,
    profile: Profile,
    cfg: &Config,
    caps: &extract::Capabilities,
    known: &HashMap<String, FileStat>,
    opts: &ScanOptions,
) -> Option<Prepared> {
    let key = path_str(&f.path);
    let prev = known.get(&key);

    // Cheap gate first: an unchanged (mtime, size) pair means skip without even
    // opening the file. This is what makes a refresh over 100k files take
    // seconds rather than minutes.
    let stat_same = prev
        .map(|p| p.mtime_ns == f.mtime_ns && p.size == f.size as i64)
        .unwrap_or(false);
    let model_same = prev
        .map(|p| p.vec_row != NO_VEC || !opts.embed)
        .unwrap_or(false);

    if stat_same && model_same && !opts.force {
        return Some(Prepared {
            path: f.path.clone(),
            dir: f.path.parent().unwrap_or(root).to_path_buf(),
            size: f.size,
            mtime_ns: f.mtime_ns,
            name_tokens: String::new(),
            body: String::new(),
            snippet: String::new(),
            kind: extract::Kind::NameOnly,
            hash: Vec::new(),
            embed_text: String::new(),
            existing_id: prev.map(|p| p.id),
            needs_embed: false,
            unchanged: true,
        });
    }

    // A nested root may set a different profile for part of this tree; the most
    // specific configured root wins. Falls back to the root being walked.
    let profile = cfg.profile_for(&f.path).unwrap_or(profile);

    let ex = extract::extract(&f.path, profile, cfg, caps);
    let name_tokens = extract::name_tokens(&f.path, root);
    let embed_text = extract::embed_text(&f.path, root, &name_tokens, &ex.body);
    let hash = extract::content_hash(&embed_text);

    // Second gate: the file was touched but its indexable text is byte-identical,
    // so the expensive step can still be skipped.
    let text_same = prev
        .and_then(|p| p.content_hash.as_ref())
        .map(|h| h == &hash)
        .unwrap_or(false);
    let needs_embed = opts.embed && (opts.force || !text_same || prev.map(|p| p.vec_row) == Some(NO_VEC));

    Some(Prepared {
        path: f.path.clone(),
        dir: f.path.parent().unwrap_or(root).to_path_buf(),
        size: f.size,
        mtime_ns: f.mtime_ns,
        name_tokens,
        body: ex.body,
        snippet: ex.snippet,
        kind: ex.kind,
        hash,
        embed_text: if needs_embed { embed_text } else { String::new() },
        existing_id: prev.map(|p| p.id),
        needs_embed,
        unchanged: false,
    })
}

/// Persist one batch: metadata, full text, and vectors, in a single transaction.
fn write_batch(
    db: &Db,
    mut store: Option<&mut VectorStore>,
    embedder: Option<&dyn Embedder>,
    prepared: &[Prepared],
    scan_gen: i64,
    stats: &mut ScanStats,
) -> Result<()> {
    let tx = db.conn.unchecked_transaction()?;

    // Assign ids first, because the vector row is the file id and the embeddings
    // are computed after this loop.
    let mut ids: Vec<i64> = Vec::with_capacity(prepared.len());
    for p in prepared {
        if p.unchanged {
            let id = p.existing_id.expect("unchanged file must already have a row");
            tx.execute(
                "UPDATE files SET seen_scan = ?1 WHERE id = ?2",
                params![scan_gen, id],
            )?;
            ids.push(id);
            stats.unchanged += 1;
            continue;
        }

        let id = match p.existing_id {
            Some(id) => {
                tx.execute(
                    "UPDATE files SET mtime_ns=?1, size=?2, content_hash=?3,
                        extract_kind=?4, snippet=?5, seen_scan=?6
                     WHERE id=?7",
                    params![
                        p.mtime_ns,
                        p.size as i64,
                        p.hash,
                        p.kind.as_str(),
                        p.snippet,
                        scan_gen,
                        id
                    ],
                )?;
                stats.updated += 1;
                id
            }
            None => {
                tx.execute(
                    "INSERT INTO files
                       (path, mtime_ns, size, content_hash, extract_kind,
                        snippet, seen_scan)
                     VALUES (?1,?2,?3,?4,?5,?6,?7)",
                    params![
                        path_str(&p.path),
                        p.mtime_ns,
                        p.size as i64,
                        p.hash,
                        p.kind.as_str(),
                        p.snippet,
                        scan_gen
                    ],
                )?;
                stats.added += 1;
                tx.last_insert_rowid()
            }
        };

        // Shares the one definition of this SQL with the rest of the codebase;
        // cached statements make the per-file call as cheap as an inline one, and
        // it runs inside `tx` because it uses the same connection.
        db.put_fts(id, &path_str(&p.path), &p.name_tokens, &p.body)?;
        ids.push(id);
    }
    tx.commit()?;

    // Embedding happens outside the transaction: it is by far the slowest step
    // and holding a write lock across it would block queries for the whole scan.
    let Some(embedder) = embedder else {
        return Ok(());
    };
    let Some(store) = store.as_mut() else {
        return Ok(());
    };

    let todo: Vec<(usize, &Prepared)> = prepared
        .iter()
        .enumerate()
        .filter(|(_, p)| p.needs_embed && !p.embed_text.is_empty())
        .collect();

    // Hand the whole batch over at once: the embedder sorts by token length
    // internally, and it can only do that across everything it is given.
    let texts: Vec<String> = todo.iter().map(|(_, p)| p.embed_text.clone()).collect();
    let vecs = embedder.embed_docs(&texts)?;

    let tx = db.conn.unchecked_transaction()?;
    for ((idx, _), v) in todo.iter().zip(&vecs) {
        let id = ids[*idx];
        // vec_row == files.id keeps the mapping trivial and self-healing.
        store.put(id as usize, v)?;
        tx.execute(
            "UPDATE files SET vec_row = ?1 WHERE id = ?2",
            params![id, id],
        )?;
        stats.embedded += 1;
    }
    tx.commit()?;
    Ok(())
}

/// Insert or refresh a row per directory, then link every file to its directory.
///
/// Intermediate directories are added even when they hold no files themselves,
/// so a directory whose children are all subdirectories still appears. The walk
/// upward stops at the configured root: ancestors above it (`/`, `/home`, ...)
/// are not part of the index and would otherwise show up as search results.
///
/// One pass, one `SELECT` of the path→id map: parent links and file links are
/// both resolved from it. Resolving parents with a point query per directory and
/// then re-reading the whole table to link files did the same work twice.
fn upsert_dirs(
    db: &Db,
    dirs: &HashSet<PathBuf>,
    roots: &[PathBuf],
    scan_gen: i64,
) -> Result<usize> {
    let mut want: HashSet<PathBuf> = HashSet::new();
    for d in dirs {
        // Ignore anything outside every root; a file's parent is always inside one.
        let Some(root) = deepest_root(roots, d) else {
            continue;
        };
        let mut cur: Option<&Path> = Some(d.as_path());
        while let Some(p) = cur {
            if !want.insert(p.to_path_buf()) {
                break;
            }
            if p == root {
                break;
            }
            cur = p.parent();
        }
    }

    let tx = db.conn.unchecked_transaction()?;
    {
        let mut ins = tx.prepare(
            "INSERT INTO dirs(path, seen_scan) VALUES (?1, ?2)
             ON CONFLICT(path) DO UPDATE SET seen_scan = excluded.seen_scan",
        )?;
        for d in &want {
            ins.execute(params![path_str(d), scan_gen])?;
        }
    }

    // The one read of the path→id map, used for both link passes below.
    let mut dir_ids: HashMap<String, i64> = HashMap::with_capacity(want.len());
    {
        let mut st = tx.prepare("SELECT id, path FROM dirs")?;
        let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        for row in rows {
            let (id, path) = row?;
            dir_ids.insert(path, id);
        }
    }

    // Parent links, resolved in Rust rather than with SQL string surgery on paths,
    // which goes subtly wrong around trailing slashes and non-ASCII names.
    {
        let mut set = tx.prepare("UPDATE dirs SET parent_id = ?1 WHERE path = ?2")?;
        for d in &want {
            if let Some(pid) = d.parent().and_then(|p| dir_ids.get(&path_str(p))) {
                set.execute(params![pid, path_str(d)])?;
            }
        }
    }

    // File links, from the same map.
    let mut pending: Vec<(i64, i64)> = Vec::new();
    {
        let mut st = tx.prepare("SELECT id, path FROM files WHERE parent_dir_id IS NULL")?;
        let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        for row in rows {
            let (id, path) = row?;
            if let Some(did) = Path::new(&path)
                .parent()
                .and_then(|p| dir_ids.get(&path_str(p)))
            {
                pending.push((id, *did));
            }
        }
    }
    {
        let mut up = tx.prepare("UPDATE files SET parent_dir_id = ?1 WHERE id = ?2")?;
        for (fid, did) in pending {
            up.execute(params![did, fid])?;
        }
    }
    tx.commit()?;

    // `file_count` is deliberately not computed here: `rollup_dirs` overwrites it
    // with a recursive count, so doing it now would be a whole-table correlated
    // subquery whose result is immediately discarded.
    Ok(want.len())
}

/// Give every directory a vector and a full-text row.
///
/// This is what makes the spec's `directory:employment_docs` result possible: a
/// directory is a searchable thing in its own right, not just a container.
///
/// The vector blends two signals:
///
/// ```text
/// coherence = ‖mean(descendant file vectors)‖          in [0, 1]
/// w         = MAX_CENTROID_WEIGHT * coherence
/// dir_vec   = normalize( (1-w) * normalize(embed(dir name words))
///                      +    w  * normalize(mean(descendant file vectors)) )
/// ```
///
/// The name half is what answers `wom bosnia` for a directory whose files never
/// say "Bosnia". The centroid half is what answers `wom employment documents` for
/// a directory called `employment_docs` full of tax forms. Either alone misses
/// half the cases.
///
/// The centroid is weighted by its own coherence rather than fixed, because a
/// centroid means less the more heterogeneous the directory is. Member vectors
/// are unit length, so `‖mean‖` is exactly the right statistic: near 1 when they
/// all point the same way, near 0 when they are scattered — and it costs nothing,
/// since the sum is already being computed.
///
/// Without this, a directory holding a bit of everything gets a centroid near the
/// corpus mean, which sits close to *every* query. Measured on the test corpus, an
/// index root scored 0.62-0.69 against three completely unrelated queries. Scaling
/// by coherence collapses that, because such a directory has a low `‖mean‖`.
///
/// Only the name embedding costs a model call — one per directory, and there are
/// orders of magnitude fewer directories than files.
pub fn rollup_dirs(
    db: &Db,
    files: &VectorStore,
    dirs_store: &mut VectorStore,
    embedder: &dyn Embedder,
    roots: &[PathBuf],
) -> Result<usize> {
    // (id, path, parent_id), deepest first, so a parent is always processed after
    // every one of its children.
    let mut rows: Vec<(i64, String, Option<i64>)> = {
        let mut st = db
            .conn
            .prepare("SELECT id, path, parent_id FROM dirs")?;
        let it = st.query_map([], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, Option<i64>>(2)?))
        })?;
        it.collect::<std::result::Result<Vec<_>, _>>()?
    };
    if rows.is_empty() {
        return Ok(0);
    }
    rows.sort_by_key(|(_, path, _)| std::cmp::Reverse(path.matches('/').count()));

    // Direct file vectors per directory.
    let mut direct: HashMap<i64, Vec<i64>> = HashMap::new();
    {
        let mut st = db.conn.prepare(
            &format!(
                "SELECT parent_dir_id, vec_row FROM files
                 WHERE parent_dir_id IS NOT NULL AND vec_row != {NO_VEC}"
            ),
        )?;
        let it = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?;
        for row in it {
            let (did, vrow) = row?;
            direct.entry(did).or_default().push(vrow);
        }
    }

    let dim = embedder.dim();
    // Running sums per directory, including everything beneath it.
    let mut sums: HashMap<i64, (Vec<f32>, usize)> = HashMap::new();

    for (id, _, parent) in &rows {
        let entry = sums.entry(*id).or_insert_with(|| (vec![0f32; dim], 0));
        if let Some(vrows) = direct.get(id) {
            for vrow in vrows {
                if let Some(v) = files.get(*vrow as usize) {
                    for (acc, x) in entry.0.iter_mut().zip(&v) {
                        *acc += x;
                    }
                    entry.1 += 1;
                }
            }
        }
        // Fold this directory's total into its parent before moving up a level.
        let (mine, count) = (entry.0.clone(), entry.1);
        if let Some(pid) = parent {
            let p = sums.entry(*pid).or_insert_with(|| (vec![0f32; dim], 0));
            for (acc, x) in p.0.iter_mut().zip(&mine) {
                *acc += x;
            }
            p.1 += count;
        }
    }

    // Embed the directory names in batches.
    let names: Vec<String> = rows
        .iter()
        .map(|(_, path, _)| {
            let p = Path::new(path);
            let base = deepest_root(roots, p).unwrap_or(Path::new("/"));
            let tokens = extract::name_tokens(p, base);
            // Fall back to the final component for a root directory, whose
            // relative path is empty.
            if tokens.trim().is_empty() {
                p.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.clone())
            } else {
                tokens
            }
        })
        .collect();

    let mut name_vecs: Vec<Vec<f32>> = Vec::with_capacity(names.len());
    for group in names.chunks(EMBED_GROUP) {
        name_vecs.extend(embedder.embed_docs(group)?);
    }

    let tx = db.conn.unchecked_transaction()?;
    let mut written = 0usize;
    {
        let mut set_vec =
            tx.prepare("UPDATE dirs SET vec_row = ?1, file_count = ?2 WHERE id = ?3")?;

        for (i, (id, path, _)) in rows.iter().enumerate() {
            let (sum, count) = sums.get(id).cloned().unwrap_or((vec![0f32; dim], 0));

            // A configured root is the container the user chose to search, never
            // an answer to a query, so it gets no vector and no full-text row.
            if roots.iter().any(|r| r.as_path() == Path::new(path)) {
                set_vec.execute(params![crate::db::NO_VEC, count as i64, id])?;
                db.delete_dir_fts(*id)?;
                continue;
            }

            let v = blend_dir_vector(&name_vecs[i], &sum, count);

            // vec_row == dirs.id, same convention as files.
            dirs_store.put(*id as usize, &v)?;
            set_vec.execute(params![id, count as i64, id])?;
            db.put_dir_fts(*id, path, names[i].as_str())?;
            written += 1;
        }
    }
    tx.commit()?;
    dirs_store.flush()?;
    Ok(written)
}

/// Combine a directory's name embedding with the centroid of its contents.
///
/// `sum` is the elementwise sum of `count` unit-length member vectors. See
/// [`rollup_dirs`] for why the centroid's weight scales with its coherence.
pub fn blend_dir_vector(name_vec: &[f32], sum: &[f32], count: usize) -> Vec<f32> {
    let mut v = name_vec.to_vec();
    crate::embed::l2_normalize(&mut v);
    if count == 0 {
        return v;
    }

    let mut centroid: Vec<f32> = sum.iter().map(|x| x / count as f32).collect();
    // Members are unit vectors, so the mean's length is their average mutual
    // agreement: 1 when identical, 0 when uniformly scattered.
    let coherence = centroid
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .clamp(0.0, 1.0);
    crate::embed::l2_normalize(&mut centroid);

    let w = MAX_CENTROID_WEIGHT * coherence;
    for (a, b) in v.iter_mut().zip(&centroid) {
        *a = (1.0 - w) * *a + w * b;
    }
    crate::embed::l2_normalize(&mut v);
    v
}

fn short_root(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string_lossy().into_owned())
}

pub fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ------------------------------------------------------------------ scheduling

/// Whether a rescan is due, and why. Returned rather than acted on so the caller
/// decides between a foreground scan and a detached one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Due {
    /// Never scanned; there is no index to serve a query from.
    Never,
    /// The configured interval has elapsed.
    Elapsed,
    /// Not yet due.
    No,
}

/// Decide whether the schedule (spec item 3) calls for a scan.
pub fn is_due(refresh: crate::config::Refresh, last_scan_at: Option<i64>, now: i64) -> Due {
    match last_scan_at {
        None => Due::Never,
        Some(last) => match refresh.interval_secs() {
            None => Due::No,
            Some(interval) => {
                // A clock that moved backwards (NTP correction, timezone-naive
                // clock reset) must not make a scan look infinitely overdue or
                // infinitely fresh; treat a future timestamp as just-scanned.
                let age = now.saturating_sub(last);
                if age >= interval as i64 {
                    Due::Elapsed
                } else {
                    Due::No
                }
            }
        },
    }
}

/// Start a scan in a detached child process, so a query can return immediately
/// from the existing index while the refresh happens behind it.
///
/// Uses a re-exec of this binary rather than a thread because a scan outliving the
/// query it was triggered by is the whole point: the user gets their results now,
/// and the index is fresher next time.
pub fn spawn_background_scan(paths: &Paths) -> Result<()> {
    // If a scan already holds the lock there is nothing to do; checking here
    // avoids spawning a process that would immediately exit.
    if scan_in_progress(&paths.scan_lock()) {
        return Ok(());
    }

    let exe = std::env::current_exe().context("locating the wom binary")?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("index").arg("--now");
    // Keep the child pointed at the same index when running under WOM_HOME.
    if let Some(home) = std::env::var_os("WOM_HOME") {
        cmd.env("WOM_HOME", home);
    }
    crate::actions::spawn_detached(&mut cmd, "background scan")
}

/// True when another process currently holds the scan lock.
pub fn scan_in_progress(lock_path: &Path) -> bool {
    match ScanLock::try_acquire(lock_path) {
        // Acquired, so nobody held it. Dropping releases it again.
        Ok(Some(_)) => false,
        Ok(None) => true,
        // If the lock file cannot even be opened, do not claim a scan is running.
        Err(_) => false,
    }
}

// ------------------------------------------------------------------ locking

/// An advisory lock held for the duration of a scan.
pub struct ScanLock {
    file: std::fs::File,
}

impl ScanLock {
    /// Take the lock, failing immediately if another scan holds it. Non-blocking
    /// on purpose: `wom index` should say what is happening rather than hang.
    pub fn acquire(path: &Path) -> Result<Self> {
        match Self::try_acquire(path)? {
            Some(lock) => Ok(lock),
            None => anyhow::bail!("another `wom index` is already running"),
        }
    }

    /// Take the lock, or return `None` if someone else holds it.
    pub fn try_acquire(path: &Path) -> Result<Option<Self>> {
        use std::os::unix::io::AsRawFd;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("opening lock file {}", path.display()))?;
        // SAFETY: `flock` on a valid fd owned by `file`; the lock is released by
        // the kernel when the fd closes, including on abnormal exit — which is why
        // a crashed scan cannot leave the index permanently locked.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::WouldBlock {
                return Ok(None);
            }
            return Err(anyhow::Error::from(err).context("locking the index for scanning"));
        }
        Ok(Some(Self { file }))
    }
}

impl Drop for ScanLock {
    fn drop(&mut self) {
        use std::os::unix::io::AsRawFd;
        // SAFETY: same fd as above, still open until this struct is dropped.
        unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn unit(v: &[f32]) -> Vec<f32> {
        let mut v = v.to_vec();
        crate::embed::l2_normalize(&mut v);
        v
    }

    #[test]
    fn blend_returns_the_name_vector_when_a_directory_has_no_files() {
        let name = unit(&[1.0, 0.0, 0.0]);
        let got = blend_dir_vector(&name, &[0.0; 3], 0);
        assert!((cos(&got, &name) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_coherent_directory_leans_on_its_contents() {
        // Every member points the same way, orthogonal to the name.
        let name = unit(&[1.0, 0.0]);
        let member = unit(&[0.0, 1.0]);
        let n = 8;
        let sum: Vec<f32> = member.iter().map(|x| x * n as f32).collect();

        let got = blend_dir_vector(&name, &sum, n);
        // coherence == 1, so w == MAX_CENTROID_WEIGHT == 0.5: an even split.
        assert!(
            (cos(&got, &member) - cos(&got, &name)).abs() < 1e-4,
            "expected an even split, got content={} name={}",
            cos(&got, &member),
            cos(&got, &name)
        );
    }

    #[test]
    fn an_incoherent_directory_falls_back_to_its_name() {
        // This is the index-root case: members cancel out, so the centroid is
        // near zero and carries no information. Regression guard for a root
        // scoring 0.6+ against every unrelated query.
        let name = unit(&[1.0, 0.0]);
        // Four members pointing in opposing directions.
        let sum = vec![0.0f32, 0.0];
        let got = blend_dir_vector(&name, &sum, 4);
        assert!(
            cos(&got, &name) > 0.999,
            "incoherent contents should leave the name dominant, got {}",
            cos(&got, &name)
        );
    }

    #[test]
    fn centroid_weight_increases_monotonically_with_coherence() {
        let name = unit(&[1.0, 0.0]);
        let member = unit(&[0.0, 1.0]);
        let n = 10;

        let mut prev = -1.0f32;
        // Scale the sum's length to simulate coherence from 0 to 1.
        for step in 0..=10 {
            let coherence = step as f32 / 10.0;
            let sum: Vec<f32> = member.iter().map(|x| x * coherence * n as f32).collect();
            let got = blend_dir_vector(&name, &sum, n);
            let lean = cos(&got, &member);
            assert!(
                lean >= prev - 1e-5,
                "content weight fell as coherence rose: {prev} -> {lean}"
            );
            prev = lean;
        }
        assert!(prev > 0.6, "a fully coherent directory should lean on content");
    }

    #[test]
    fn blended_vector_is_always_unit_length() {
        let name = unit(&[0.3, -0.7, 0.2]);
        for count in [0usize, 1, 5] {
            let sum = vec![0.1f32 * count as f32, 0.2 * count as f32, 0.0];
            let got = blend_dir_vector(&name, &sum, count);
            let norm = got.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "count={count} norm={norm}");
        }
    }

    #[test]
    fn dir_rows_stop_at_the_root_and_do_not_climb_to_slash() {
        let db = Db::open_in_memory().unwrap();
        let root = PathBuf::from("/home/n/Documents");
        let dirs: HashSet<PathBuf> = [
            PathBuf::from("/home/n/Documents/tax/2024"),
            PathBuf::from("/home/n/Documents/notes"),
        ]
        .into_iter()
        .collect();

        upsert_dirs(&db, &dirs, &[root.clone()], 1).unwrap();

        let mut got: Vec<String> = db
            .conn
            .prepare("SELECT path FROM dirs ORDER BY path")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                "/home/n/Documents",
                "/home/n/Documents/notes",
                "/home/n/Documents/tax",
                "/home/n/Documents/tax/2024",
            ],
            "ancestors above the root leaked into the index"
        );
    }

    #[test]
    fn dir_parent_links_are_resolved() {
        let db = Db::open_in_memory().unwrap();
        let root = PathBuf::from("/r");
        let dirs: HashSet<PathBuf> = [PathBuf::from("/r/a/b")].into_iter().collect();
        upsert_dirs(&db, &dirs, &[root], 1).unwrap();

        let parent_of_b: String = db
            .conn
            .query_row(
                "SELECT p.path FROM dirs d JOIN dirs p ON p.id = d.parent_id
                 WHERE d.path = '/r/a/b'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(parent_of_b, "/r/a");
    }

    #[test]
    fn never_scanned_is_always_due() {
        use crate::config::Refresh;
        for r in [
            Refresh::EveryRun,
            Refresh::Daily,
            Refresh::Weekly,
            Refresh::Manual,
        ] {
            assert_eq!(is_due(r, None, 1000), Due::Never, "{r:?}");
        }
    }

    #[test]
    fn manual_refresh_is_never_due_once_scanned() {
        use crate::config::Refresh;
        assert_eq!(is_due(Refresh::Manual, Some(0), i64::MAX / 2), Due::No);
    }

    #[test]
    fn every_run_is_always_due_once_scanned() {
        use crate::config::Refresh;
        assert_eq!(is_due(Refresh::EveryRun, Some(1000), 1000), Due::Elapsed);
    }

    #[test]
    fn daily_and_weekly_respect_their_intervals() {
        use crate::config::Refresh;
        let day = 24 * 60 * 60;
        assert_eq!(is_due(Refresh::Daily, Some(0), day - 1), Due::No);
        assert_eq!(is_due(Refresh::Daily, Some(0), day), Due::Elapsed);

        assert_eq!(is_due(Refresh::Weekly, Some(0), 6 * day), Due::No);
        assert_eq!(is_due(Refresh::Weekly, Some(0), 7 * day), Due::Elapsed);
    }

    #[test]
    fn a_backwards_clock_does_not_force_a_scan() {
        use crate::config::Refresh;
        // last_scan_at in the future, e.g. after an NTP correction. Saturating
        // subtraction keeps the age at 0 rather than wrapping to a huge number.
        assert_eq!(is_due(Refresh::Daily, Some(1_000_000), 1000), Due::No);
    }

    #[test]
    fn scan_in_progress_reflects_the_lock() {
        let p = std::env::temp_dir().join("wom-inprogress-test");
        std::fs::remove_file(&p).ok();
        assert!(!scan_in_progress(&p), "no lock held yet");
        {
            let _held = ScanLock::acquire(&p).unwrap();
            // flock is per-open-file-description, so a fresh open in this same
            // process still observes the lock.
            assert!(scan_in_progress(&p), "held lock was not observed");
        }
        assert!(!scan_in_progress(&p), "lock outlived its guard");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn scan_lock_is_exclusive_and_released_on_drop() {
        let p = std::env::temp_dir().join("wom-scanlock-test");
        std::fs::remove_file(&p).ok();
        {
            let _a = ScanLock::acquire(&p).unwrap();
            let err = match ScanLock::acquire(&p) {
                Ok(_) => panic!("second lock should have been refused"),
                Err(e) => e.to_string(),
            };
            assert!(err.contains("already running"), "got: {err}");
        }
        // Released now.
        ScanLock::acquire(&p).unwrap();
        std::fs::remove_file(&p).ok();
    }
}
