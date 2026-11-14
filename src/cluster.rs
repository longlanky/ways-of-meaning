//! e-means clustering (Bessa et al., arXiv:2611.09501), CPU reference.
//!
//! `wom topics` needs a fast, high-quality k-means over the static set of file
//! vectors. Lloyd's algorithm spends 25 full passes; e-means reaches the same
//! loss in a handful of mini-batch passes with an exponential moving average of
//! each cluster's count and sum:
//!
//! ```text
//! N_j <- (1 - alpha) N_j + alpha n_j
//! S_j <- (1 - alpha) S_j + alpha s_j
//! c_j  = S_j / N_j
//! ```
//!
//! The batch size and initial step come from the paper's fitted scaling laws,
//! the step is cosine-annealed to a tenth of its peak, dead clusters are revived
//! by splitting a busy donor, and the last pass's assignments give each centroid
//! its exact mean. Counts and sums are f64; vectors and centroids are f32.
//!
//! This is deliberately Euclidean, not spherical. The stored vectors are unit
//! length, so Euclidean k-means is invariant to the global mean direction that
//! dominates cosine comparisons between unrelated embeddings; it clusters the
//! residual structure that makes topics distinguishable. Spherical mode belongs
//! with an IVF index for inner-product search, which `wom` does not have.

use anyhow::{Result, bail};
use rayon::prelude::*;

/// Initial EMA mass. The paper uses 1e-10; the reference implementation uses
/// 2^-16 so the first updates are not dominated by a vanishing denominator.
const INITIAL_COUNT: f64 = 1.0 / 65536.0;

/// A floor that keeps an empty cluster's count finite when alpha is 1. The
/// centroid is preserved by setting its sum from the position.
const MIN_COUNT: f64 = 1e-300;

/// Clusters below this fraction of the mean mass are revived (paper §4).
const REVIVAL_THRESHOLD: f64 = 0.1;

/// Alternating coordinate perturbation when a donor is split. The paper's
/// Appendix J uses 1e-4; large enough to separate f32 positions, small enough
/// not to move either child materially.
const SPLIT_EPSILON: f32 = 1e-4;

/// The cosine schedule's floor, as a fraction of the peak step.
const ALPHA_MIN_FACTOR: f32 = 0.1;

/// Logical updates between revival checks.
const CHECK_REVIVAL_EVERY: usize = 5;

/// One run's knobs. `seed` makes the sample order, initialization, and revival
/// reproducible.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub k: usize,
    pub epochs: u32,
    pub seed: u64,
}

/// The resolved scaling-law recipe, exposed for the build summary and tests.
#[derive(Debug, Clone, Copy)]
pub struct Recipe {
    pub batch: usize,
    pub alpha0: f32,
    pub steps: usize,
}

/// A trained codebook plus the last pass's labels for the training rows.
#[derive(Debug, Clone)]
pub struct Model {
    /// Row-major, `k * dim`.
    pub centroids: Vec<f32>,
    pub recipe: Recipe,
}

/// Nearest power of two, ties upward. The reference's exact rule, so the
/// resolved batch sizes match the paper's tables.
fn round_pow2(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    let low = 2f64.powf(value.log2().floor());
    let high = 2.0 * low;
    if value / low < high / value {
        low as usize
    } else {
        high as usize
    }
}

/// Resolve the paper's throughput batch law (Eq. 7) and step law (Eq. 6) for
/// `n` training rows, `k` clusters, and `epochs` passes.
pub fn recipe(n: usize, k: usize, epochs: u32) -> Recipe {
    let t = epochs as f64;
    let raw = 0.035134 * (k as f64).powf(0.134749) * (n as f64).powf(0.876473) * t.powf(0.813344);
    // The published recipe caps the throughput batch at about n/2 and then at n.
    let batch = round_pow2(raw).min(round_pow2(n as f64 / 2.0)).clamp(1, n);
    let alpha0 = (0.8244
        * (k as f64).powf(-0.4617)
        * (batch as f64).powf(0.9003)
        * (n as f64).powf(-0.4498)
        * t.powf(-0.1057))
    .clamp(1e-4, 1.0) as f32;
    Recipe {
        batch,
        alpha0,
        steps: n.div_ceil(batch) * epochs as usize,
    }
}

/// Equation 5: cosine-anneal the step from `alpha0` towards 0.1 * `alpha0`.
fn scheduled_alpha(alpha0: f32, step: usize, total_steps: usize) -> f32 {
    let fraction = step.min(total_steps) as f32 / total_steps.max(1) as f32;
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

/// Squared L2 distance to the nearest centroid. Four independent accumulators
/// so the loop vectorizes without `-ffast-math`; ties keep the lowest index.
#[inline]
pub(crate) fn nearest(centroids: &[f32], k: usize, dim: usize, row: &[f32]) -> (usize, f32) {
    let mut best_distance = f32::INFINITY;
    let mut best_index = 0usize;
    let quads = dim / 4;
    for j in 0..k {
        let centroid = &centroids[j * dim..(j + 1) * dim];
        let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for q in 0..quads {
            let i = q * 4;
            let d0 = row[i] - centroid[i];
            let d1 = row[i + 1] - centroid[i + 1];
            let d2 = row[i + 2] - centroid[i + 2];
            let d3 = row[i + 3] - centroid[i + 3];
            a0 += d0 * d0;
            a1 += d1 * d1;
            a2 += d2 * d2;
            a3 += d3 * d3;
        }
        let mut distance = (a0 + a1) + (a2 + a3);
        for i in quads * 4..dim {
            let d = row[i] - centroid[i];
            distance += d * d;
        }
        if distance < best_distance {
            best_distance = distance;
            best_index = j;
        }
    }
    (best_index, best_distance)
}

/// Split dead clusters by halving donors sampled proportionally to their mass.
/// Returns how many splits happened.
fn revive(
    centroids: &mut [f32],
    sums: &mut [f64],
    counts: &mut [f64],
    k: usize,
    dim: usize,
    rng: &mut Rng,
) -> usize {
    let total: f64 = counts.iter().sum();
    let threshold = REVIVAL_THRESHOLD * total / k as f64;
    let dead: Vec<usize> = (0..k).filter(|&j| counts[j] < threshold).collect();
    if dead.is_empty() {
        return 0;
    }
    let donors: Vec<usize> = (0..k)
        .filter(|&j| counts[j] >= threshold && counts[j] > 0.0)
        .collect();
    let splits = dead.len().min(donors.len());
    if splits == 0 {
        return 0;
    }

    // Efraimidis-Spirakis: a key of ln(u) / mass, sampled without replacement.
    // A larger mass shifts the key towards zero, so descending order favours it.
    let mut keyed: Vec<(f64, usize)> = donors
        .iter()
        .map(|&j| {
            let u = rng.next_f64().max(f64::MIN_POSITIVE);
            (u.ln() / counts[j], j)
        })
        .collect();
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0));

    for (i, &(_, donor)) in keyed.iter().take(splits).enumerate() {
        let target = dead[i];
        let half = counts[donor] / 2.0;
        counts[donor] = half;
        counts[target] = half;
        for d in 0..dim {
            let sign = if d % 2 == 0 { 1.0 } else { -1.0 };
            let base = centroids[donor * dim + d];
            let donor_position = base * (1.0 - SPLIT_EPSILON * sign);
            let target_position = base * (1.0 + SPLIT_EPSILON * sign);
            centroids[donor * dim + d] = donor_position;
            centroids[target * dim + d] = target_position;
            sums[donor * dim + d] = donor_position as f64 * half;
            sums[target * dim + d] = target_position as f64 * half;
        }
        // A subnormal donor can round both positions back together; one ULP is
        // enough to make the two clusters distinct again.
        if centroids[donor * dim] == centroids[target * dim] {
            let bits = centroids[target * dim].to_bits().wrapping_add(1);
            centroids[target * dim] = f32::from_bits(bits);
        }
    }
    splits
}

/// A seeded SplitMix64 stream. Small, dependency-free, and stable across
/// platforms, which is all clustering needs.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub(crate) fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            values.swap(i, j);
        }
    }
}

/// Train on `n` rows of `dim` f32 values, row-major.
pub fn train(x: &[f32], n: usize, dim: usize, params: &Params) -> Result<Model> {
    if dim == 0 {
        bail!("clustering needs a non-zero dimension");
    }
    if n < 2 {
        bail!("clustering needs at least 2 vectors, got {n}");
    }
    if params.k < 2 {
        bail!("clustering needs at least 2 clusters, got {}", params.k);
    }
    if params.k > n {
        bail!("cannot make {} clusters from {n} vectors", params.k);
    }
    if params.epochs == 0 {
        bail!("clustering needs at least one epoch");
    }
    if x.len() != n * dim {
        bail!(
            "vector matrix has {} values, expected {n} x {dim} = {}",
            x.len(),
            n * dim
        );
    }

    let k = params.k;
    let resolved = recipe(n, k, params.epochs);
    let mut rng = Rng::new(params.seed);

    let mut order: Vec<u32> = (0..n as u32).collect();
    rng.shuffle(&mut order);
    let mut centroids = vec![0f32; k * dim];
    for j in 0..k {
        let row = order[j] as usize;
        centroids[j * dim..(j + 1) * dim].copy_from_slice(&x[row * dim..(row + 1) * dim]);
    }

    let mut counts = vec![INITIAL_COUNT; k];
    let mut sums = vec![0f64; k * dim];
    for j in 0..k {
        for d in 0..dim {
            sums[j * dim + d] = centroids[j * dim + d] as f64 * INITIAL_COUNT;
        }
    }
    let mut batch_sums = vec![0f64; k * dim];
    let mut batch_counts = vec![0u32; k];
    let mut pass_sums = vec![0f64; k * dim];
    let mut pass_counts = vec![0u64; k];
    let mut step = 0usize;

    for epoch in 0..params.epochs {
        rng.shuffle(&mut order);
        let last = epoch + 1 == params.epochs;
        if last {
            pass_sums.fill(0.0);
            pass_counts.fill(0);
        }
        for chunk in order.chunks(resolved.batch) {
            batch_sums.fill(0.0);
            batch_counts.fill(0);

            // Indexed parallel map preserves order, and assignment is a pure
            // function of the centroids, so the result is deterministic.
            let assigned: Vec<u32> = chunk
                .par_iter()
                .map(|&row| {
                    let row = row as usize;
                    nearest(&centroids, k, dim, &x[row * dim..(row + 1) * dim]).0 as u32
                })
                .collect();

            for (position, &row) in chunk.iter().enumerate() {
                let row = row as usize;
                let cluster = assigned[position] as usize;
                batch_counts[cluster] += 1;
                let source = &x[row * dim..(row + 1) * dim];
                let target = &mut batch_sums[cluster * dim..(cluster + 1) * dim];
                for d in 0..dim {
                    target[d] += source[d] as f64;
                }
                if last {
                    pass_counts[cluster] += 1;
                    let target = &mut pass_sums[cluster * dim..(cluster + 1) * dim];
                    for d in 0..dim {
                        target[d] += source[d] as f64;
                    }
                }
            }

            let alpha = scheduled_alpha(resolved.alpha0, step, resolved.steps) as f64;
            for j in 0..k {
                counts[j] = (1.0 - alpha) * counts[j] + alpha * batch_counts[j] as f64;
                if counts[j] < MIN_COUNT {
                    // Preserve the position of a cluster that received nothing.
                    counts[j] = MIN_COUNT;
                    for d in 0..dim {
                        sums[j * dim + d] = centroids[j * dim + d] as f64 * MIN_COUNT;
                    }
                } else {
                    let inverse = 1.0 / counts[j];
                    for d in 0..dim {
                        let sum =
                            (1.0 - alpha) * sums[j * dim + d] + alpha * batch_sums[j * dim + d];
                        sums[j * dim + d] = sum;
                        centroids[j * dim + d] = (sum * inverse) as f32;
                    }
                }
            }

            step += 1;
            if step.is_multiple_of(CHECK_REVIVAL_EVERY) {
                revive(&mut centroids, &mut sums, &mut counts, k, dim, &mut rng);
            }
        }
    }

    // Exact closing means from the final pass's assignments. A cluster that got
    // no rows keeps its last EMA position.
    for j in 0..k {
        if pass_counts[j] > 0 {
            let count = pass_counts[j] as f64;
            for d in 0..dim {
                centroids[j * dim + d] = (pass_sums[j * dim + d] / count) as f32;
            }
        }
    }

    Ok(Model {
        centroids,
        recipe: resolved,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_matches_the_published_laws() {
        // Values resolved by the reference implementation for this corpus size.
        let r = recipe(8_013, 32, 5);
        assert_eq!(r.batch, 512);
        assert_eq!(r.steps, 80);
        assert!((r.alpha0 - 0.67695).abs() < 1e-3, "alpha0 {}", r.alpha0);

        let r = recipe(8_013, 512, 5);
        assert_eq!(r.batch, 1024);
        assert_eq!(r.steps, 40);
        assert!((r.alpha0 - 0.35126).abs() < 1e-3, "alpha0 {}", r.alpha0);
    }

    #[test]
    fn round_pow2_matches_the_reference_rule() {
        assert_eq!(round_pow2(3.0), 4);
        assert_eq!(round_pow2(2.1), 2);
        assert_eq!(round_pow2(1.0), 1);
        assert_eq!(round_pow2(0.2), 1);
        // A tie goes up, as in the reference's `<` comparison: 2 and 4 are
        // equidistant in log space from 2*sqrt(2).
        assert_eq!(round_pow2(2.0 * 2.0f64.sqrt()), 4);
    }

    #[test]
    fn nearest_breaks_ties_towards_the_lowest_index() {
        let centroids = vec![0.0, 0.0, 0.0, 0.0];
        let (index, distance) = nearest(&centroids, 2, 2, &[1.0, 1.0]);
        assert_eq!(index, 0);
        assert!((distance - 2.0).abs() < 1e-6);
    }

    #[test]
    fn separated_blobs_are_recovered() {
        let dim = 8;
        let blobs = 4;
        let per_blob = 40;
        let n = blobs * per_blob;
        let mut x = vec![0f32; n * dim];
        for blob in 0..blobs {
            for i in 0..per_blob {
                let row = blob * per_blob + i;
                for d in 0..dim {
                    // A strong unique coordinate plus deterministic jitter.
                    let base = if d == blob { 10.0 } else { 0.0 };
                    let jitter = ((row * 31 + d * 17) % 7) as f32 * 0.001;
                    x[row * dim + d] = base + jitter;
                }
            }
        }

        let model = train(
            &x,
            n,
            dim,
            &Params {
                k: blobs,
                epochs: 5,
                seed: 7,
            },
        )
        .unwrap();

        // Every row's nearest final centroid must agree with its blob, and each
        // blob must map to a distinct centroid.
        let mut blob_to_cluster = vec![usize::MAX; blobs];
        for (blob, slot) in blob_to_cluster.iter_mut().enumerate() {
            for i in 0..per_blob {
                let row = blob * per_blob + i;
                let (cluster, _) =
                    nearest(&model.centroids, blobs, dim, &x[row * dim..(row + 1) * dim]);
                if *slot == usize::MAX {
                    *slot = cluster;
                }
                assert_eq!(*slot, cluster, "blob {blob} split");
            }
        }
        let mut seen = blob_to_cluster.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), blobs, "two blobs shared a centroid");

        // The closing means should sit almost exactly on the blob coordinates.
        for (blob, &cluster) in blob_to_cluster.iter().enumerate() {
            let centroid = &model.centroids[cluster * dim..(cluster + 1) * dim];
            for (d, &value) in centroid.iter().enumerate() {
                if d == blob {
                    assert!((value - 10.0).abs() < 0.01, "centroid {blob} moved");
                } else {
                    assert!(value.abs() < 0.01, "centroid {blob} leaked into {d}");
                }
            }
        }
    }

    #[test]
    fn training_is_deterministic_for_a_seed() {
        let dim = 4;
        let n = 24;
        let mut x = vec![0f32; n * dim];
        for row in 0..n {
            for d in 0..dim {
                x[row * dim + d] = ((row * 13 + d * 7) % 11) as f32 / 11.0;
            }
        }
        let params = Params {
            k: 4,
            epochs: 3,
            seed: 99,
        };
        let a = train(&x, n, dim, &params).unwrap();
        let b = train(&x, n, dim, &params).unwrap();
        assert_eq!(a.centroids, b.centroids);
    }

    #[test]
    fn revival_halves_mass_and_separates_the_children() {
        let dim = 2;
        let k = 2;
        let mut centroids = vec![10.0, 10.0, 0.0, 0.0];
        let mut counts = vec![10.0, 0.0];
        let mut sums = vec![0f64; k * dim];
        for j in 0..k {
            for d in 0..dim {
                sums[j * dim + d] = centroids[j * dim + d] as f64 * counts[j];
            }
        }
        let splits = revive(
            &mut centroids,
            &mut sums,
            &mut counts,
            k,
            dim,
            &mut Rng::new(3),
        );
        assert_eq!(splits, 1);
        assert!((counts[0] - 5.0).abs() < 1e-9);
        assert!((counts[1] - 5.0).abs() < 1e-9);
        assert_ne!(centroids[0..dim], centroids[dim..]);
        let mass: f64 = sums.iter().sum();
        assert!((mass - 200.0).abs() < 1.0, "mass drifted: {mass}");
    }

    #[test]
    fn empty_clusters_are_kept_finite() {
        // k = n with strongly repeated rows gives the algorithm every chance to
        // leave a cluster empty; it must still return finite centroids.
        let dim = 3;
        let n = 8;
        let mut x = vec![0f32; n * dim];
        for row in 0..n {
            x[row * dim] = row as f32;
        }
        let model = train(
            &x,
            n,
            dim,
            &Params {
                k: 8,
                epochs: 2,
                seed: 1,
            },
        )
        .unwrap();
        assert!(model.centroids.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_impossible_shapes() {
        let params = Params {
            k: 2,
            epochs: 1,
            seed: 0,
        };
        assert!(train(&[], 0, 4, &params).is_err());
        assert!(train(&[0f32; 4], 1, 4, &params).is_err());
        assert!(train(&[0f32; 8], 2, 4, &Params { k: 3, ..params }).is_err());
        assert!(train(&[0f32; 8], 2, 4, &Params { k: 1, ..params }).is_err());
        assert!(
            train(
                &[0f32; 8],
                2,
                4,
                &Params {
                    epochs: 0,
                    ..params
                }
            )
            .is_err()
        );
        assert!(train(&[0f32; 7], 2, 4, &params).is_err());
    }
}
