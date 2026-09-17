//! SQLite metadata store: files, directories, roots, FTS5 index, key/value meta.
//!
//! Vectors deliberately do *not* live here — they sit in a flat mmap'd file (see
//! [`crate::vectors`]) indexed by the `vec_row` columns below. SQLite is good at
//! the relational and full-text parts and bad at scanning 300k float arrays.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;

pub const SCHEMA_VERSION: i64 = 2;

/// Sentinel for "this row has no vector yet".
pub const NO_VEC: i64 = -1;

/// Measured end-to-end indexing throughput in documents per second, including
/// extraction and database work. Written by a completed scan.
pub const META_INDEX_RATE: &str = "measured_index_docs_per_sec";

/// Measured model-only throughput in documents per second. Written by `wom bench`.
/// Roughly twice [`META_INDEX_RATE`], because it excludes everything but inference.
pub const META_MODEL_RATE: &str = "measured_model_docs_per_sec";

pub struct Db {
    pub conn: Connection,
}

/// A file row as needed by the indexer's change-detection pass.
#[derive(Debug, Clone)]
pub struct FileStat {
    pub id: i64,
    pub mtime_ns: i64,
    pub size: i64,
    pub content_hash: Option<Vec<u8>>,
    pub vec_row: i64,
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("opening database {}", path.display()))?;
        Self::from_conn(conn)
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        Self::from_conn(Connection::open_in_memory()?)
    }

    fn from_conn(conn: Connection) -> Result<Self> {
        // WAL is what lets a background scan write while a foreground query
        // reads, which the refresh scheduling in `wom index` depends on.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        // Bulk inserts during a scan are the hot path; give SQLite room.
        conn.pragma_update(None, "cache_size", -64_000)?;
        conn.pragma_update(None, "temp_store", "MEMORY")?;
        let db = Db { conn };
        db.migrate()?;
        Ok(db)
    }

    fn migrate(&self) -> Result<()> {
        let found: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='meta'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);

        if found == 0 {
            self.conn
                .execute_batch(SCHEMA)
                .context("creating database schema")?;
            self.set_meta("schema_version", &SCHEMA_VERSION.to_string())?;
            self.set_meta("scan_gen", "0")?;
            return Ok(());
        }

        // Catch a database whose tables do not match this binary even though the
        // version number agrees — which happens whenever the schema is edited
        // without a version bump. Without this, the mismatch surfaces much later
        // as an opaque SQLite error like "no such column: T.name".
        let fts_sql: String = self
            .conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name = 'files_fts'",
                [],
                |r| r.get(0),
            )
            .unwrap_or_default();
        if fts_sql.contains("content=") {
            anyhow::bail!(
                "this index was built with an incompatible full-text schema. \
                 Run `wom index --rebuild` to discard and rebuild it."
            );
        }

        let have: i64 = self
            .get_meta("schema_version")?
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if have != SCHEMA_VERSION {
            // Everything here is a derived cache of the filesystem, so the
            // honest migration for a schema bump is to rebuild rather than to
            // carry forward migration code for an index nobody has yet.
            anyhow::bail!(
                "index schema is version {have}, this build expects {SCHEMA_VERSION}. \
                 Run `wom index --rebuild` to discard and rebuild the index."
            );
        }
        Ok(())
    }

    // ---- meta key/value ----

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT v FROM meta WHERE k = ?1", params![key], |r| r.get(0))
            .optional()?)
    }

    pub fn set_meta(&self, key: &str, val: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(k, v) VALUES (?1, ?2)
             ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            params![key, val],
        )?;
        Ok(())
    }

    pub fn get_meta_i64(&self, key: &str) -> Result<Option<i64>> {
        Ok(self.get_meta(key)?.and_then(|v| v.parse().ok()))
    }

    /// Bump and return the scan generation. Rows still carrying an older
    /// generation after a completed walk are gone from disk (see [`Self::gc`]).
    pub fn next_scan_gen(&self) -> Result<i64> {
        let next = self.get_meta_i64("scan_gen")?.unwrap_or(0) + 1;
        self.set_meta("scan_gen", &next.to_string())?;
        Ok(next)
    }

    // ---- lookups used by the indexer ----

    /// Every known file under `root_prefix`, keyed by path, for change detection.
    /// Loaded once per scan: one query beats 300k point lookups.
    pub fn file_stats_under(
        &self,
        root_prefix: &str,
    ) -> Result<std::collections::HashMap<String, FileStat>> {
        let mut st = self.conn.prepare(
            "SELECT id, path, mtime_ns, size, content_hash, vec_row
             FROM files WHERE path >= ?1 AND path < ?2",
        )?;
        let (lo, hi) = prefix_range(root_prefix);
        let rows = st.query_map(params![lo, hi], |r| {
            Ok((
                r.get::<_, String>(1)?,
                FileStat {
                    id: r.get(0)?,
                    mtime_ns: r.get(2)?,
                    size: r.get(3)?,
                    content_hash: r.get(4)?,
                    vec_row: r.get(5)?,
                },
            ))
        })?;
        let mut out = std::collections::HashMap::new();
        for row in rows {
            let (p, s) = row?;
            out.insert(p, s);
        }
        Ok(out)
    }

    /// Delete rows not observed by scan generation `scan_gen`, i.e. paths that have
    /// disappeared from disk. Returns (files_deleted, dirs_deleted).
    pub fn gc(&self, scan_gen: i64) -> Result<(usize, usize)> {
        let tx = self.conn.unchecked_transaction()?;
        // FTS rows share the file id as rowid but there is no foreign key to
        // cascade from, so they must be removed explicitly and first.
        tx.execute(
            "DELETE FROM files_fts
             WHERE rowid IN (SELECT id FROM files WHERE seen_scan != ?1)",
            params![scan_gen],
        )?;
        tx.execute(
            "DELETE FROM dirs_fts
             WHERE rowid IN (SELECT id FROM dirs WHERE seen_scan != ?1)",
            params![scan_gen],
        )?;
        let f = tx.execute("DELETE FROM files WHERE seen_scan != ?1", params![scan_gen])?;
        let d = tx.execute("DELETE FROM dirs WHERE seen_scan != ?1", params![scan_gen])?;
        tx.commit()?;
        Ok((f, d))
    }

    /// Per-root generation key: the last scan generation that covered `root`.
    /// A global `scan_gen` counter is still bumped once per scan (so `seen_scan`
    /// stays comparable), but scoped scans only garbage-collect inside the roots
    /// they actually walked. Without this, `wom index --root A` would mark every
    /// other root unseen and delete it.
    pub fn root_gen_key(root: &str) -> String {
        format!("scan_gen:{root}")
    }

    /// Last wall-clock scan time for one root, for per-root freshness.
    pub fn root_last_scan_key(root: &str) -> String {
        format!("last_scan_at:{root}")
    }

    pub fn set_root_gen(&self, root: &str, scan_gen: i64) -> Result<()> {
        self.set_meta(&Self::root_gen_key(root), &scan_gen.to_string())
    }

    #[allow(dead_code)]
    pub fn get_root_gen(&self, root: &str) -> Result<Option<i64>> {
        self.get_meta_i64(&Self::root_gen_key(root))
    }

    pub fn set_root_last_scan(&self, root: &str, ts: i64) -> Result<()> {
        self.set_meta(&Self::root_last_scan_key(root), &ts.to_string())
    }

    /// Build a `(path >= ? AND path < ?)` disjunction over `prefixes`, with
    /// placeholders starting at `start_idx` (1-based, for rusqlite `?N`).
    fn prefix_predicate(prefixes: &[String], start_idx: usize, col: &str) -> String {
        prefixes
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let lo = start_idx + i * 2;
                let hi = lo + 1;
                format!("({col} >= ?{lo} AND {col} < ?{hi})")
            })
            .collect::<Vec<_>>()
            .join(" OR ")
    }

    fn prefix_bounds(prefixes: &[String]) -> Vec<(String, String)> {
        prefixes.iter().map(|p| prefix_range(p)).collect()
    }

    /// Delete only rows under `prefixes` (trailing-slash directory prefixes)
    /// that this scan generation did not observe. Other roots are untouched.
    pub fn gc_under(&self, prefixes: &[String], scan_gen: i64) -> Result<(usize, usize)> {
        if prefixes.is_empty() {
            return Ok((0, 0));
        }
        let bounds = Self::prefix_bounds(prefixes);
        // seen_scan is ?1; prefix bounds start at ?2.
        let file_pred = Self::prefix_predicate(prefixes, 2, "path");
        let tx = self.conn.unchecked_transaction()?;
        let mut binds: Vec<String> = vec![scan_gen.to_string()];
        for (lo, hi) in &bounds {
            binds.push(lo.clone());
            binds.push(hi.clone());
        }
        let refs: Vec<&dyn rusqlite::ToSql> =
            binds.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
        // Note: seen_scan binds as TEXT; SQLite type affinity compares it
        // numerically against the INTEGER column, matching the unscoped `gc`.
        // FTS rows must go first (no FK cascade from files/dirs).
        tx.execute(
            &format!(
                "DELETE FROM files_fts WHERE rowid IN \
                 (SELECT id FROM files WHERE seen_scan != ?1 AND ({file_pred}))"
            ),
            refs.as_slice(),
        )?;
        let dir_pred = Self::prefix_predicate(prefixes, 2, "path");
        tx.execute(
            &format!(
                "DELETE FROM dirs_fts WHERE rowid IN \
                 (SELECT id FROM dirs WHERE seen_scan != ?1 AND ({dir_pred}))"
            ),
            refs.as_slice(),
        )?;
        let f = tx.execute(
            &format!("DELETE FROM files WHERE seen_scan != ?1 AND ({file_pred})"),
            refs.as_slice(),
        )?;
        let d = tx.execute(
            &format!("DELETE FROM dirs WHERE seen_scan != ?1 AND ({dir_pred})"),
            refs.as_slice(),
        )?;
        tx.commit()?;
        Ok((f, d))
    }

    /// Replace the full-text row for one directory. See [`Self::put_fts`].
    pub fn put_dir_fts(&self, dir_id: i64, path: &str, name_tokens: &str) -> Result<()> {
        self.delete_dir_fts(dir_id)?;
        self.conn
            .prepare_cached("INSERT INTO dirs_fts(rowid, path, name) VALUES (?1,?2,?3)")?
            .execute(params![dir_id, path, name_tokens])?;
        Ok(())
    }

    /// Remove a directory's full-text row, for roots that should never be results.
    pub fn delete_dir_fts(&self, dir_id: i64) -> Result<()> {
        self.conn
            .prepare_cached("DELETE FROM dirs_fts WHERE rowid = ?1")?
            .execute(params![dir_id])?;
        Ok(())
    }

    /// Replace the full-text row for one file. Called on insert and on update;
    /// the delete makes it idempotent, since FTS5 has no upsert.
    ///
    /// Uses cached statements so the indexer can call this per file inside its own
    /// transaction without re-preparing, which is why there is only one copy of
    /// this SQL rather than a separate bulk path that could drift from it.
    pub fn put_fts(&self, file_id: i64, path: &str, name_tokens: &str, body: &str) -> Result<()> {
        self.conn
            .prepare_cached("DELETE FROM files_fts WHERE rowid = ?1")?
            .execute(params![file_id])?;
        self.conn
            .prepare_cached("INSERT INTO files_fts(rowid, path, name, body) VALUES (?1,?2,?3,?4)")?
            .execute(params![file_id, path, name_tokens, body])?;
        Ok(())
    }

    /// True when every `files` row has exactly one matching `files_fts` row.
    /// Cheap enough to assert in tests and to expose via `wom status --verify`.
    pub fn fts_is_consistent(&self) -> Result<bool> {
        let orphans: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM files_fts
             WHERE rowid NOT IN (SELECT id FROM files)",
            [],
            |r| r.get(0),
        )?;
        let missing: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM files
             WHERE id NOT IN (SELECT rowid FROM files_fts)",
            [],
            |r| r.get(0),
        )?;
        Ok(orphans == 0 && missing == 0)
    }

    /// Vector rows belonging to files that this scan did not observe, so the
    /// caller can zero them before their metadata is deleted. A stale vector left
    /// behind would keep matching queries for a file that no longer exists.
    pub fn vec_rows_not_seen(&self, scan_gen: i64) -> Result<Vec<i64>> {
        self.vec_rows_not_seen_in("files", scan_gen)
    }

    /// Directory vector rows that this scan did not observe, for zeroing.
    pub fn dir_vec_rows_not_seen(&self, scan_gen: i64) -> Result<Vec<i64>> {
        self.vec_rows_not_seen_in("dirs", scan_gen)
    }

    /// `table` is one of two fixed literals chosen by the callers above.
    fn vec_rows_not_seen_in(&self, table: &str, scan_gen: i64) -> Result<Vec<i64>> {
        let sql =
            format!("SELECT vec_row FROM {table} WHERE seen_scan != ?1 AND vec_row != ?2");
        let mut st = self.conn.prepare(&sql)?;
        let rows = st.query_map(params![scan_gen, NO_VEC], |r| r.get::<_, i64>(0))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Vector rows under `prefixes` that this scan did not observe. Scoped twin
    /// of [`Self::vec_rows_not_seen`], so a single-root refresh zeroes only its
    /// own deletions instead of every other root's live vectors.
    pub fn vec_rows_not_seen_under(
        &self,
        table: &str,
        prefixes: &[String],
        scan_gen: i64,
    ) -> Result<Vec<i64>> {
        assert!(
            table == "files" || table == "dirs",
            "vec_rows_not_seen_under takes only files/dirs"
        );
        if prefixes.is_empty() {
            return Ok(Vec::new());
        }
        let bounds = Self::prefix_bounds(prefixes);
        let pred = Self::prefix_predicate(prefixes, 3, "path");
        let sql = format!(
            "SELECT vec_row FROM {table} \
             WHERE seen_scan != ?1 AND vec_row != ?2 AND ({pred})"
        );
        let mut binds: Vec<String> = vec![scan_gen.to_string(), NO_VEC.to_string()];
        for (lo, hi) in &bounds {
            binds.push(lo.clone());
            binds.push(hi.clone());
        }
        let refs: Vec<&dyn rusqlite::ToSql> =
            binds.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
        let mut st = self.conn.prepare(&sql)?;
        let rows = st.query_map(refs.as_slice(), |r| r.get::<_, i64>(0))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn file_vec_rows_not_seen_under(
        &self,
        prefixes: &[String],
        scan_gen: i64,
    ) -> Result<Vec<i64>> {
        self.vec_rows_not_seen_under("files", prefixes, scan_gen)
    }

    pub fn dir_vec_rows_not_seen_under(
        &self,
        prefixes: &[String],
        scan_gen: i64,
    ) -> Result<Vec<i64>> {
        self.vec_rows_not_seen_under("dirs", prefixes, scan_gen)
    }

    /// How many files each extractor produced text for, best-covered first. Backs
    /// the coverage line in `wom status`.
    pub fn extract_coverage(&self) -> Result<Vec<(String, i64)>> {
        let mut st = self
            .conn
            .prepare("SELECT extract_kind, COUNT(*) FROM files GROUP BY extract_kind ORDER BY 2 DESC")?;
        let rows = st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn count_files(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?)
    }

    pub fn count_embedded(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM files WHERE vec_row != ?1",
            params![NO_VEC],
            |r| r.get(0),
        )?)
    }

    pub fn count_dirs(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM dirs", [], |r| r.get(0))?)
    }
}

/// SQLite compares TEXT bytewise, so a prefix scan is `>= p AND < p+1`. This
/// lets `path` use its unique index instead of a full table scan, which matters
/// at 300k rows.
pub fn prefix_range(prefix: &str) -> (String, String) {
    let lo = prefix.to_string();
    let mut hi = prefix.as_bytes().to_vec();
    // Increment the final byte to get the exclusive upper bound.
    match hi.last_mut() {
        Some(b) if *b < 0xff => *b += 1,
        _ => hi.push(0xff),
    }
    (lo, String::from_utf8_lossy(&hi).into_owned())
}

/// Paths are stored as UTF-8 text. Non-UTF-8 names are rare on Linux but legal;
/// lossy conversion keeps them searchable rather than dropping them, at the cost
/// of the exact bytes.
pub fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

const SCHEMA: &str = r#"
CREATE TABLE meta (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
);

-- Roots deliberately live only in config.toml. Mirroring them here would give
-- two sources of truth for what is indexed, and the scan reads the config.

CREATE TABLE dirs (
  id         INTEGER PRIMARY KEY,
  path       TEXT UNIQUE NOT NULL,
  parent_id  INTEGER REFERENCES dirs(id) ON DELETE SET NULL,
  file_count INTEGER NOT NULL DEFAULT 0,
  vec_row    INTEGER NOT NULL DEFAULT -1,
  seen_scan  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX dirs_parent ON dirs(parent_id);
CREATE INDEX dirs_seen   ON dirs(seen_scan);

CREATE TABLE files (
  id             INTEGER PRIMARY KEY,
  path           TEXT UNIQUE NOT NULL,
  parent_dir_id  INTEGER REFERENCES dirs(id) ON DELETE SET NULL,
  mtime_ns       INTEGER NOT NULL,
  size           INTEGER NOT NULL,
  content_hash   BLOB,
  extract_kind   TEXT NOT NULL DEFAULT 'name-only',
  -- Extracted body text lives only in files_fts, never here: storing it in both
  -- would double the largest thing in the database for no gain.
  snippet        TEXT NOT NULL DEFAULT '',
  vec_row        INTEGER NOT NULL DEFAULT -1,
  embedded_model TEXT,
  seen_scan      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX files_parent  ON files(parent_dir_id);
CREATE INDEX files_seen    ON files(seen_scan);
CREATE INDEX files_vec_row ON files(vec_row);

-- FTS5 owns the body text: `rowid` is the `files.id`, and this is the only copy.
--
-- Deliberately NOT an external-content table (`content='files'`). That variant
-- needs `INSERT INTO files_fts(files_fts, ...) VALUES('delete', ...)` with column
-- values matching exactly what was indexed, and any drift between the two tables
-- corrupts the index rather than erroring. An ordinary FTS5 table supports plain
-- `DELETE ... WHERE rowid = ?`, which cannot desynchronise.
--
-- Tokenisation is left at the unicode61 default *on purpose*: it treats `_`, `-`
-- and `.` as separators, which is what makes the spec's examples work.
-- `bosnia_croatia_trip_aug2024` becomes [bosnia, croatia, trip, aug2024] so
-- `wom bosnia` reaches it, and `2024_W2.pdf` becomes [2024, w2, pdf]. Adding
-- these to `tokenchars` instead would make each filename a single token and
-- silently break exactly the queries the tool exists to answer.
CREATE VIRTUAL TABLE files_fts USING fts5(
  path, name, body,
  tokenize="unicode61 remove_diacritics 2",
  prefix="2 3 4"
);

-- Directories get their own full-text index so a query can match a directory by
-- name directly, not only through the files inside it. `wom bosnia` should reach
-- `bosnia_croatia_trip_aug2024` even if no file in it mentions Bosnia.
-- rowid is the dirs.id.
CREATE VIRTUAL TABLE dirs_fts USING fts5(
  path, name,
  tokenize="unicode61 remove_diacritics 2",
  prefix="2 3 4"
);
"#;

#[cfg(test)]
mod tests {
    use super::*;
    /// Insert a file the way the indexer does: row plus its full-text entry.
    fn insert_file(db: &Db, path: &str, scan_gen: i64) -> i64 {
        insert_file_with_body(db, path, scan_gen, "")
    }

    fn insert_file_with_body(db: &Db, path: &str, scan_gen: i64, body: &str) -> i64 {
        db.conn
            .execute(
                "INSERT INTO files(path, mtime_ns, size, seen_scan) VALUES (?1, 1, 1, ?2)",
                params![path, scan_gen],
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

    fn fts_search(db: &Db, q: &str) -> Vec<String> {
        let mut st = db
            .conn
            .prepare("SELECT path FROM files_fts WHERE files_fts MATCH ?1 ORDER BY rank")
            .unwrap();
        let rows = st
            .query_map(params![q], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        rows
    }

    #[test]
    fn migrate_creates_schema_and_version() {
        let db = Db::open_in_memory().unwrap();
        assert_eq!(
            db.get_meta_i64("schema_version").unwrap(),
            Some(SCHEMA_VERSION)
        );
        assert_eq!(db.count_files().unwrap(), 0);
    }

    #[test]
    fn scan_gen_increments_monotonically() {
        let db = Db::open_in_memory().unwrap();
        assert_eq!(db.next_scan_gen().unwrap(), 1);
        assert_eq!(db.next_scan_gen().unwrap(), 2);
        assert_eq!(db.get_meta_i64("scan_gen").unwrap(), Some(2));
    }

    #[test]
    fn gc_removes_only_rows_from_older_generations() {
        let db = Db::open_in_memory().unwrap();
        insert_file(&db, "/a/kept.txt", 7);
        insert_file(&db, "/a/stale.txt", 6);
        let (files, _dirs) = db.gc(7).unwrap();
        assert_eq!(files, 1);
        assert_eq!(db.count_files().unwrap(), 1);
        let remaining: String = db
            .conn
            .query_row("SELECT path FROM files", [], |r| r.get(0))
            .unwrap();
        assert_eq!(remaining, "/a/kept.txt");
    }

    #[test]
    fn gc_keeps_the_full_text_index_consistent() {
        // Regression: with an external-content FTS5 table this sequence reported
        // "database disk image is malformed" instead of deleting a row.
        let db = Db::open_in_memory().unwrap();
        insert_file_with_body(&db, "/a/kept.txt", 7, "sarajevo travel notes");
        insert_file_with_body(&db, "/a/stale.txt", 6, "sarajevo travel notes");

        assert_eq!(fts_search(&db, "sarajevo").len(), 2);
        db.gc(7).unwrap();

        assert!(db.fts_is_consistent().unwrap());
        assert_eq!(fts_search(&db, "sarajevo"), vec!["/a/kept.txt"]);
    }

    #[test]
    fn put_fts_is_idempotent_and_updates_in_place() {
        let db = Db::open_in_memory().unwrap();
        let id = insert_file_with_body(&db, "/a/n.txt", 1, "old text");
        db.put_fts(id, "/a/n.txt", "n.txt", "new text").unwrap();

        assert!(db.fts_is_consistent().unwrap());
        assert!(fts_search(&db, "old").is_empty());
        assert_eq!(fts_search(&db, "new"), vec!["/a/n.txt"]);
    }

    #[test]
    fn underscored_filenames_split_into_searchable_words() {
        // Directly from the spec's examples: `wom bosnia` must reach
        // `bosnia_croatia_trip_aug2024`, and the W-2 must be findable. This fails
        // if `_`/`-`/`.` are ever added to the tokenizer's `tokenchars`.
        let db = Db::open_in_memory().unwrap();
        insert_file_with_body(&db, "/t/bosnia_croatia_trip_aug2024/x.pdf", 1, "");
        insert_file_with_body(&db, "/t/employment_docs/2024_W2.pdf", 1, "");
        insert_file_with_body(&db, "/t/bosnia_croatia_trip_aug2024/dubrovnik_to_sarajevo.pdf", 1, "");

        assert_eq!(fts_search(&db, "bosnia").len(), 2, "bosnia");
        assert_eq!(fts_search(&db, "sarajevo").len(), 1, "sarajevo");
        assert_eq!(fts_search(&db, "w2").len(), 1, "w2");
        assert_eq!(fts_search(&db, "employment").len(), 1, "employment");
    }

    #[test]
    fn prefix_queries_work_for_incremental_typing() {
        // The TUI re-searches on every keystroke, so `bosn*` needs to match
        // before the word is finished. That is what prefix="2 3 4" buys.
        let db = Db::open_in_memory().unwrap();
        insert_file_with_body(&db, "/t/bosnia_trip/x.pdf", 1, "");
        assert_eq!(fts_search(&db, "bosn*").len(), 1);
    }

    /// Mirror of the `path >= ?1 AND path < ?2` predicate the queries use, so the
    /// test checks membership in the half-open range rather than one bound.
    fn in_range(s: &str, range: &(String, String)) -> bool {
        s >= range.0.as_str() && s < range.1.as_str()
    }

    #[test]
    fn prefix_range_bounds_a_directory_subtree() {
        let r = prefix_range("/home/n/projects/");
        assert!(in_range("/home/n/projects/a", &r));
        assert!(in_range("/home/n/projects/deep/nested/file.rs", &r));
        assert!(in_range("/home/n/projects/zzz", &r));

        // A sibling that merely shares the string prefix must be excluded. Note
        // it falls below `lo` rather than above `hi`, because '-' (0x2d) sorts
        // before '/' (0x2f) — which is exactly why the trailing slash matters.
        assert!(!in_range("/home/n/projects-old/x", &r));
        assert!(!in_range("/home/n/Documents/a", &r));
        assert!(!in_range("/home/n/projectz/a", &r));
    }

    #[test]
    fn prefix_range_handles_a_trailing_high_byte() {
        // 0xff cannot be incremented, so the bound is extended instead. The
        // range must still contain its own subtree.
        let p = "/tmp/\u{10ffff}/";
        let r = prefix_range(p);
        assert!(in_range(&format!("{p}child.txt"), &r));
    }

    #[test]
    fn file_stats_under_scopes_to_the_subtree() {
        let db = Db::open_in_memory().unwrap();
        insert_file(&db, "/home/n/Documents/a.txt", 1);
        insert_file(&db, "/home/n/Downloads/b.txt", 1);
        let got = db.file_stats_under("/home/n/Documents/").unwrap();
        assert_eq!(got.len(), 1);
        assert!(got.contains_key("/home/n/Documents/a.txt"));
    }

    #[test]
    fn scoped_gc_preserves_other_roots() {
        // Regression for `--only-root` wiping every other root: scoped GC must
        // only collect rows under the scanned prefixes.
        let db = Db::open_in_memory().unwrap();
        insert_file(&db, "/a/kept.txt", 2);
        insert_file(&db, "/a/stale.txt", 1);
        insert_file(&db, "/b/untouched.txt", 1);
        let (f, _) = db.gc_under(&["/a/".to_string()], 2).unwrap();
        assert_eq!(f, 1);
        assert_eq!(db.count_files().unwrap(), 2);
        let mut got: Vec<String> = db
            .conn
            .prepare("SELECT path FROM files ORDER BY path")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        got.sort();
        assert_eq!(got, vec!["/a/kept.txt", "/b/untouched.txt"]);
        assert!(db.fts_is_consistent().unwrap());
    }

    #[test]
    fn scoped_vec_rows_only_cover_the_scanned_prefix() {
        let db = Db::open_in_memory().unwrap();
        let a = insert_file(&db, "/a/x.txt", 1);
        let b = insert_file(&db, "/b/y.txt", 1);
        db.conn
            .execute("UPDATE files SET vec_row = id WHERE id IN (?1, ?2)", params![a, b])
            .unwrap();
        let got = db
            .file_vec_rows_not_seen_under(&["/a/".to_string()], 2)
            .unwrap();
        assert_eq!(got, vec![a]);
    }

    #[test]
    fn root_gen_keys_round_trip() {
        let db = Db::open_in_memory().unwrap();
        assert_eq!(db.get_root_gen("/a").unwrap(), None);
        db.set_root_gen("/a", 7).unwrap();
        db.set_root_last_scan("/a", 123).unwrap();
        assert_eq!(db.get_root_gen("/a").unwrap(), Some(7));
        assert_eq!(db.get_meta_i64("last_scan_at:/a").unwrap(), Some(123));
    }

}
