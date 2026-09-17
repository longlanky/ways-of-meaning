//! Flat mmap'd int8 vector store with brute-force top-K search.
//!
//! At this corpus size an approximate index would be all cost and no benefit:
//! ~300k 384-dim int8 vectors is ~110 MB, and a full SIMD scan of that is well
//! under 20 ms across 16 threads. So there is no HNSW/IVF here — no build step,
//! no tuning, no staleness, and exact results.
//!
//! Vectors are int8-quantised. They arrive L2-normalised, so every component is
//! in [-1, 1] and a single per-vector scale captures the range with negligible
//! loss (>0.99 recall in practice) at a quarter of the memory. That matters on a
//! machine with 3.2 GB free.

use anyhow::{Context, Result, bail};
use memmap2::MmapMut;
use rayon::prelude::*;
use std::path::Path;

const MAGIC: &[u8; 8] = b"WOMVEC01";
const HEADER: usize = 64;

/// Rows added per growth step, to keep `set_len`+remap infrequent.
const GROW_ROWS: usize = 8192;

/// Sentinel scale for "this row holds no live vector". A zeroed row scores 0 on
/// every query, so deleted files simply never rank — which is why search needs no
/// liveness filter of its own.
const DEAD: f32 = 0.0;

/// Quantise a unit vector to int8 plus a scale, or `None` if it carries no
/// direction at all.
///
/// Stored vectors and query vectors must use the *same* scheme or every cosine is
/// silently rescaled, so there is exactly one implementation of it.
fn quantise(v: &[f32]) -> Option<(Vec<i8>, f32)> {
    // Scale by the largest magnitude so the full int8 range is used. For a unit
    // vector this is <= 1, and dividing by it recovers precision a fixed 1/127
    // scale would throw away on vectors with small components.
    let peak = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    if peak <= 1e-12 {
        return None;
    }
    let inv = 127.0 / peak;
    let q = v
        .iter()
        // Round then clamp: `as i8` alone would wrap.
        .map(|x| (x * inv).round().clamp(-127.0, 127.0) as i8)
        .collect();
    Some((q, peak / 127.0))
}

pub struct VectorStore {
    path: std::path::PathBuf,
    map: MmapMut,
    dim: usize,
    /// Bytes per row: `dim` int8 components followed by an f32 scale.
    stride: usize,
    /// Rows the file has space for.
    capacity: usize,
    /// One past the highest row ever written; the scan range for search.
    high_water: usize,
    model_hash: u64,
}

impl VectorStore {
    /// Open or create the store at `path`.
    ///
    /// A dimension or model mismatch is an error rather than a silent reset:
    /// vectors from two different models are not comparable, and mixing them
    /// would degrade search with no visible symptom.
    pub fn open(path: &Path, dim: usize, model_id: &str) -> Result<Self> {
        if dim == 0 {
            bail!("vector dimension must be non-zero");
        }
        let stride = dim + 4;
        let model_hash = hash_model(model_id);

        let exists = path.exists();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("opening vector store {}", path.display()))?;

        if exists && file.metadata()?.len() >= HEADER as u64 {
            // SAFETY: the file is a private data file of this process's index; a
            // concurrent truncation by another process would be a violation, which
            // is what the scan lock prevents.
            let map = unsafe { MmapMut::map_mut(&file) }
                .with_context(|| format!("mapping {}", path.display()))?;
            let (got_dim, cap, got_hash, hw) = read_header(&map)?;
            if got_dim != dim {
                bail!(
                    "vector store {} holds {got_dim}-dim vectors but the configured model \
                     produces {dim}-dim. Run `wom index --rebuild`.",
                    path.display()
                );
            }
            if got_hash != model_hash {
                bail!(
                    "vector store {} was built with a different model. Vectors from two \
                     models are not comparable. Run `wom index --rebuild`.",
                    path.display()
                );
            }
            return Ok(Self {
                path: path.to_path_buf(),
                map,
                dim,
                stride,
                capacity: cap,
                high_water: hw,
                model_hash,
            });
        }

        let capacity = GROW_ROWS;
        file.set_len((HEADER + capacity * stride) as u64)
            .context("sizing new vector store")?;
        let mut map = unsafe { MmapMut::map_mut(&file) }
            .with_context(|| format!("mapping {}", path.display()))?;
        write_header(&mut map, dim, capacity, model_hash, 0);
        Ok(Self {
            path: path.to_path_buf(),
            map,
            dim,
            stride,
            capacity,
            high_water: 0,
            model_hash,
        })
    }

    /// One past the highest row written, i.e. the range search covers.
    pub fn high_water(&self) -> usize {
        self.high_water
    }

    fn row_offset(&self, row: usize) -> usize {
        HEADER + row * self.stride
    }

    /// Write the in-memory header fields back to the mapping.
    fn sync_header(&mut self) {
        write_header(
            &mut self.map,
            self.dim,
            self.capacity,
            self.model_hash,
            self.high_water,
        );
    }

    fn grow_to(&mut self, rows: usize) -> Result<()> {
        if rows <= self.capacity {
            return Ok(());
        }
        let new_cap = rows.next_multiple_of(GROW_ROWS);
        self.map.flush().context("flushing before growth")?;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .with_context(|| format!("reopening {} to grow", self.path.display()))?;
        file.set_len((HEADER + new_cap * self.stride) as u64)
            .context("growing vector store")?;
        self.map = unsafe { MmapMut::map_mut(&file) }.context("remapping after growth")?;
        self.capacity = new_cap;
        self.sync_header();
        Ok(())
    }

    /// Quantise and store `v` at `row`. `v` must be L2-normalised.
    pub fn put(&mut self, row: usize, v: &[f32]) -> Result<()> {
        if v.len() != self.dim {
            bail!("vector has {} dims, store expects {}", v.len(), self.dim);
        }
        self.grow_to(row + 1)?;

        let off = self.row_offset(row);
        match quantise(v) {
            Some((q, scale)) => {
                for (i, b) in q.iter().enumerate() {
                    self.map[off + i] = *b as u8;
                }
                self.map[off + self.dim..off + self.stride].copy_from_slice(&scale.to_le_bytes());
            }
            // A genuinely zero vector (an empty document) is stored as dead: it
            // carries no information and should never be returned.
            None => {
                self.map[off..off + self.dim].fill(0);
                self.map[off + self.dim..off + self.stride].copy_from_slice(&DEAD.to_le_bytes());
            }
        }

        if row + 1 > self.high_water {
            self.high_water = row + 1;
            self.sync_header();
        }
        Ok(())
    }

    /// Mark a row as holding no vector. Its score becomes 0 for every query.
    pub fn clear(&mut self, row: usize) -> Result<()> {
        if row >= self.high_water {
            return Ok(());
        }
        let off = self.row_offset(row);
        self.map[off..off + self.stride].fill(0);
        Ok(())
    }

    /// Read a row back as floats. `None` if the row is empty or out of range.
    pub fn get(&self, row: usize) -> Option<Vec<f32>> {
        if row >= self.high_water {
            return None;
        }
        let off = self.row_offset(row);
        let scale = read_f32(&self.map[off + self.dim..off + self.stride]);
        if scale == DEAD {
            return None;
        }
        Some(
            self.map[off..off + self.dim]
                .iter()
                .map(|b| (*b as i8) as f32 * scale)
                .collect(),
        )
    }

    /// Top `k` rows by cosine similarity to `query`.
    ///
    /// `allow` optionally restricts the search: `allow[row] == true` means the row
    /// is eligible. Used for directory-scoped queries.
    pub fn search(&self, query: &[f32], k: usize, allow: Option<&[bool]>) -> Result<Vec<Hit>> {
        if query.len() != self.dim {
            bail!(
                "query has {} dims, store holds {}",
                query.len(),
                self.dim
            );
        }
        if let Some(a) = allow {
            if a.len() < self.high_water {
                bail!(
                    "scope allow-list has {} entries but the vector store holds {}; \
                     refusing to silently exclude rows",
                    a.len(),
                    self.high_water
                );
            }
        }
        if k == 0 || self.high_water == 0 {
            return Ok(Vec::new());
        }

        // Quantise the query with the same function the stored rows used, so the
        // inner loop is integer-only and the two scales are guaranteed to agree.
        let Some((q, qscale)) = quantise(query) else {
            return Ok(Vec::new());
        };

        let dim = self.dim;
        let stride = self.stride;
        let body = &self.map[HEADER..];

        // Chunk over rows, keep a bounded heap per chunk, merge at the end. Each
        // chunk touches a contiguous slice, which is what makes this
        // memory-bandwidth-bound rather than latency-bound.
        let chunk = (self.high_water / rayon::current_num_threads().max(1)).max(4096);
        let mut merged: Vec<Hit> = (0..self.high_water)
            .into_par_iter()
            .chunks(chunk)
            .map(|rows| {
                let mut local = TopK::new(k);
                for row in rows {
                    if let Some(a) = allow {
                        if !a.get(row).copied().unwrap_or(false) {
                            continue;
                        }
                    }
                    let off = row * stride;
                    let scale = read_f32(&body[off + dim..off + stride]);
                    if scale == DEAD {
                        continue;
                    }
                    // SAFETY-free reinterpretation: i8 and u8 have the same layout,
                    // and the slice length is checked by the bounds above.
                    let v: &[i8] =
                        unsafe { std::slice::from_raw_parts(body[off..].as_ptr() as *const i8, dim) };
                    let raw = dot_i8(&q, v);
                    local.push(Hit {
                        row: row as u32,
                        score: raw as f32 * scale * qscale,
                    });
                }
                local.into_vec()
            })
            .reduce(Vec::new, |mut a, b| {
                a.extend(b);
                a
            });

        merged.sort_unstable_by(|x, y| y.score.total_cmp(&x.score).then(x.row.cmp(&y.row)));
        merged.truncate(k);
        Ok(merged)
    }

    pub fn flush(&self) -> Result<()> {
        self.map.flush().context("flushing vector store")
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub row: u32,
    pub score: f32,
}

/// Integer dot product. Written as a plain fold so LLVM can pick the widest
/// available SIMD (VPMADDUBSW/VPDPBUSD on x86-64) without an intrinsics
/// dependency or a target-feature gate.
#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: i32 = 0;
    for i in 0..a.len() {
        acc += (a[i] as i32) * (b[i] as i32);
    }
    acc
}

/// Bounded min-heap of the best `k` hits. Avoids sorting every row.
struct TopK {
    k: usize,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<Ordered>>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: std::collections::BinaryHeap::with_capacity(k + 1),
        }
    }

    fn push(&mut self, h: Hit) {
        if self.heap.len() < self.k {
            self.heap.push(std::cmp::Reverse(Ordered(h)));
        } else if let Some(std::cmp::Reverse(Ordered(worst))) = self.heap.peek() {
            if h.score > worst.score {
                self.heap.pop();
                self.heap.push(std::cmp::Reverse(Ordered(h)));
            }
        }
    }

    fn into_vec(self) -> Vec<Hit> {
        self.heap.into_iter().map(|r| r.0.0).collect()
    }
}

/// Total ordering over hits so they can live in a heap. Scores are finite by
/// construction (finite inputs, integer dot product), and the row id breaks ties
/// so results are deterministic.
#[derive(PartialEq)]
struct Ordered(Hit);

impl Eq for Ordered {}

impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .score
            .total_cmp(&other.0.score)
            .then(other.0.row.cmp(&self.0.row))
    }
}

// ------------------------------------------------------------------ header

fn write_header(map: &mut MmapMut, dim: usize, cap: usize, model_hash: u64, high_water: usize) {
    map[0..8].copy_from_slice(MAGIC);
    map[8..12].copy_from_slice(&(dim as u32).to_le_bytes());
    map[12..16].copy_from_slice(&(cap as u32).to_le_bytes());
    map[16..24].copy_from_slice(&model_hash.to_le_bytes());
    map[24..32].copy_from_slice(&(high_water as u64).to_le_bytes());
}

fn read_header(map: &[u8]) -> Result<(usize, usize, u64, usize)> {
    if &map[0..8] != MAGIC {
        bail!("not a wom vector store (bad magic). Run `wom index --rebuild`.");
    }
    let dim = u32::from_le_bytes(map[8..12].try_into().unwrap()) as usize;
    let cap = u32::from_le_bytes(map[12..16].try_into().unwrap()) as usize;
    let hash = u64::from_le_bytes(map[16..24].try_into().unwrap());
    let hw = u64::from_le_bytes(map[24..32].try_into().unwrap()) as usize;
    if hw > cap {
        bail!("vector store header is inconsistent (high water {hw} > capacity {cap})");
    }
    Ok((dim, cap, hash, hw))
}

fn read_f32(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Model identity folded into 64 bits, so a model swap is detected on open.
fn hash_model(model_id: &str) -> u64 {
    let h = blake3::hash(model_id.as_bytes());
    u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("wom-vec-{name}.i8"));
        std::fs::remove_file(&p).ok();
        p
    }

    fn unit(v: &[f32]) -> Vec<f32> {
        let mut v = v.to_vec();
        crate::embed::l2_normalize(&mut v);
        v
    }

    use crate::embed::{cosine, norm};

    #[test]
    fn roundtrip_preserves_direction_within_quantisation_error() {
        let p = tmp("roundtrip");
        let mut s = VectorStore::open(&p, 4, "m").unwrap();
        let v = unit(&[0.5, -0.25, 0.8, 0.1]);
        s.put(0, &v).unwrap();
        let got = s.get(0).unwrap();

        let cos = cosine(&v, &got) / norm(&got);
        std::fs::remove_file(&p).ok();
        assert!(cos > 0.9999, "cosine after quantisation was {cos}");
    }

    #[test]
    fn quantisation_error_stays_small_across_many_random_vectors() {
        let p = tmp("qerror");
        let dim = 384;
        let mut s = VectorStore::open(&p, dim, "m").unwrap();
        // Deterministic pseudo-random directions; no RNG dependency.
        let mut worst = 1.0f32;
        for i in 0..64 {
            let v = unit(
                &(0..dim)
                    .map(|j| (((i * 7919 + j * 104729) % 2003) as f32 / 1000.0) - 1.0)
                    .collect::<Vec<f32>>(),
            );
            s.put(i, &v).unwrap();
            let got = s.get(i).unwrap();
            let cos = cosine(&v, &got) / norm(&got);
            worst = worst.min(cos);
        }
        std::fs::remove_file(&p).ok();
        // The plan claims >0.99 recall from this scheme; the per-vector cosine
        // needs to be far tighter than that for the claim to hold.
        assert!(worst > 0.999, "worst cosine was {worst}");
    }

    #[test]
    fn search_ranks_the_nearest_vector_first() {
        let p = tmp("rank");
        let mut s = VectorStore::open(&p, 3, "m").unwrap();
        s.put(0, &unit(&[1.0, 0.0, 0.0])).unwrap();
        s.put(1, &unit(&[0.0, 1.0, 0.0])).unwrap();
        s.put(2, &unit(&[0.9, 0.1, 0.0])).unwrap();

        let hits = s.search(&unit(&[1.0, 0.0, 0.0]), 3, None).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!(hits[0].row, 0);
        assert_eq!(hits[1].row, 2);
        assert!(hits[0].score > hits[1].score);
        assert!((hits[0].score - 1.0).abs() < 0.01, "self-cosine {}", hits[0].score);
    }

    #[test]
    fn cleared_rows_never_rank() {
        let p = tmp("cleared");
        let mut s = VectorStore::open(&p, 3, "m").unwrap();
        s.put(0, &unit(&[1.0, 0.0, 0.0])).unwrap();
        s.put(1, &unit(&[1.0, 0.0, 0.0])).unwrap();
        s.clear(0).unwrap();

        let hits = s.search(&unit(&[1.0, 0.0, 0.0]), 5, None).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!(hits.len(), 1, "a cleared row was returned");
        assert_eq!(hits[0].row, 1);
        assert!(s.get(0).is_none(), "cleared row still readable");
    }

    #[test]
    fn allow_list_restricts_the_search() {
        let p = tmp("allow");
        let mut s = VectorStore::open(&p, 3, "m").unwrap();
        s.put(0, &unit(&[1.0, 0.0, 0.0])).unwrap();
        s.put(1, &unit(&[0.9, 0.1, 0.0])).unwrap();

        let allow = vec![false, true];
        let hits = s.search(&unit(&[1.0, 0.0, 0.0]), 5, Some(&allow)).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].row, 1);
    }

    #[test]
    fn growth_past_the_initial_capacity_preserves_earlier_rows() {
        let p = tmp("grow");
        let mut s = VectorStore::open(&p, 3, "m").unwrap();
        s.put(0, &unit(&[1.0, 0.0, 0.0])).unwrap();
        // Force several remaps.
        let far = GROW_ROWS * 2 + 5;
        s.put(far, &unit(&[0.0, 0.0, 1.0])).unwrap();

        assert_eq!(s.high_water(), far + 1);
        let first = s.get(0).unwrap();
        let last = s.get(far).unwrap();
        std::fs::remove_file(&p).ok();
        assert!(first[0] > 0.9, "row 0 was corrupted by growth: {first:?}");
        assert!(last[2] > 0.9, "far row wrong: {last:?}");
    }

    #[test]
    fn reopening_recovers_dim_high_water_and_contents() {
        let p = tmp("reopen");
        {
            let mut s = VectorStore::open(&p, 3, "m").unwrap();
            s.put(5, &unit(&[0.0, 1.0, 0.0])).unwrap();
            s.flush().unwrap();
        }
        let s = VectorStore::open(&p, 3, "m").unwrap();
        assert_eq!(s.high_water(), 6);
        let v = s.get(5).unwrap();
        std::fs::remove_file(&p).ok();
        assert!(v[1] > 0.9);
    }

    #[test]
    fn reopening_with_a_different_dim_is_refused() {
        let p = tmp("dimswap");
        {
            VectorStore::open(&p, 384, "m").unwrap();
        }
        let err = match VectorStore::open(&p, 1024, "m") {
            Ok(_) => panic!("opening with a mismatched dim should have failed"),
            Err(e) => e.to_string(),
        };
        std::fs::remove_file(&p).ok();
        assert!(err.contains("384"), "unhelpful error: {err}");
        assert!(err.contains("rebuild"), "error should say how to fix: {err}");
    }

    #[test]
    fn reopening_with_a_different_model_is_refused() {
        // The silent-corruption case the plan calls out: same dim, different model.
        let p = tmp("modelswap");
        {
            VectorStore::open(&p, 384, "bge-small-en-v1.5").unwrap();
        }
        let err = match VectorStore::open(&p, 384, "all-MiniLM-L6-v2") {
            Ok(_) => panic!("opening with a different model should have failed"),
            Err(e) => e.to_string(),
        };
        std::fs::remove_file(&p).ok();
        assert!(err.contains("different model"), "unhelpful error: {err}");
    }

    #[test]
    fn dimension_mismatch_on_put_and_search_is_an_error() {
        let p = tmp("mismatch");
        let mut s = VectorStore::open(&p, 3, "m").unwrap();
        assert!(s.put(0, &[1.0, 0.0]).is_err());
        assert!(s.search(&[1.0, 0.0], 1, None).is_err());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn empty_store_and_zero_k_return_nothing() {
        let p = tmp("empty");
        let s = VectorStore::open(&p, 3, "m").unwrap();
        assert!(s.search(&unit(&[1.0, 0.0, 0.0]), 5, None).unwrap().is_empty());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn zero_vectors_are_stored_dead_rather_than_producing_nan() {
        let p = tmp("zero");
        let mut s = VectorStore::open(&p, 3, "m").unwrap();
        s.put(0, &[0.0, 0.0, 0.0]).unwrap();
        assert!(s.get(0).is_none());
        let hits = s.search(&unit(&[1.0, 0.0, 0.0]), 5, None).unwrap();
        std::fs::remove_file(&p).ok();
        assert!(hits.is_empty());
    }

    #[test]
    fn topk_keeps_the_highest_scores() {
        let mut t = TopK::new(2);
        for (row, score) in [(0, 0.1), (1, 0.9), (2, 0.5), (3, 0.3)] {
            t.push(Hit { row, score });
        }
        let mut v = t.into_vec();
        v.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].row, 1);
        assert_eq!(v[1].row, 2);
    }

    #[test]
    fn search_is_deterministic_for_tied_scores() {
        let p = tmp("ties");
        let mut s = VectorStore::open(&p, 3, "m").unwrap();
        for row in 0..10 {
            s.put(row, &unit(&[1.0, 0.0, 0.0])).unwrap();
        }
        let a = s.search(&unit(&[1.0, 0.0, 0.0]), 3, None).unwrap();
        let b = s.search(&unit(&[1.0, 0.0, 0.0]), 3, None).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!(
            a.iter().map(|h| h.row).collect::<Vec<_>>(),
            b.iter().map(|h| h.row).collect::<Vec<_>>()
        );
    }

    #[test]
    fn dot_i8_matches_a_scalar_reference() {
        let a: Vec<i8> = (0..64).map(|i| (i as i8) - 32).collect();
        let b: Vec<i8> = (0..64).map(|i| 31 - (i as i8)).collect();
        let want: i32 = a.iter().zip(&b).map(|(x, y)| *x as i32 * *y as i32).sum();
        assert_eq!(dot_i8(&a, &b), want);
    }
}
