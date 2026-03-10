use anyhow::{bail, Result};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::Serialize;

// ─── public types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct MoranRecord {
    pub feature: String,
    pub moran_i: f64,
    pub expected_i: f64,
    pub variance_i: f64,
    pub z_score: f64,
    pub group: String,
}

#[derive(Serialize)]
pub struct LocalMoranRecord {
    pub cell_i: String,
    pub feature: String,
    pub local_i: f64,
    pub z_score: f64,
    pub group: String,
}

#[derive(Serialize)]
pub struct GearyCRecord {
    pub feature: String,
    pub geary_c: f64,
    pub expected_c: f64, // always 1.0 under CSR
    pub variance_c: f64,
    pub z_score: f64,
    pub group: String,
}

#[derive(Serialize)]
pub struct MoranPermRecord {
    pub feature: String,
    pub p_value_perm: f64,
    pub group: String,
}

#[derive(Serialize)]
pub struct BivariateMoranRecord {
    pub feature_a: String,
    pub feature_b: String,
    pub bivariate_i: f64,
    pub z_score: f64,
    pub group: String,
}

// ─── rstar integration ────────────────────────────────────────────────────────

#[derive(Clone)]
struct IndexedPoint {
    coords: [f64; 2],
    index: usize,
}

impl RTreeObject for IndexedPoint {
    type Envelope = AABB<[f64; 2]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.coords)
    }
}

impl PointDistance for IndexedPoint {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let dx = self.coords[0] - point[0];
        let dy = self.coords[1] - point[1];
        dx * dx + dy * dy
    }
}

// ─── main entry point ─────────────────────────────────────────────────────────

/// Compute global Moran's I for each column of `values` (N × F matrix).
///
/// Spatial weights are binary (1 = within `radius`, 0 = beyond).
/// Significance is assessed via the analytical approximation under normality:
///   E[I] = -1/(N-1)
///   Var[I] = (N²·S1 − N·S2 + 3·S0²) / ((N²-1)·S0²) − E[I]²
/// where for binary symmetric weights: S0 = 2E, S1 = 4E, S2 = 4·Σᵢ degᵢ²
/// and E is the number of undirected edges.
pub fn compute_morans_i(
    coords: &[[f64; 2]],
    values: &Array2<f64>,
    feature_names: &[String],
    radius: f64,
    group: &str,
) -> Result<Vec<MoranRecord>> {
    let n = coords.len();

    if n < 2 {
        bail!("need at least 2 cells for Moran's I, got {n}");
    }
    if !radius.is_finite() || radius <= 0.0 {
        bail!("radius must be a finite value > 0");
    }
    if values.nrows() != n {
        bail!("values rows ({}) != coords length ({n})", values.nrows());
    }
    if values.ncols() != feature_names.len() {
        bail!(
            "values cols ({}) != feature_names length ({})",
            values.ncols(),
            feature_names.len()
        );
    }

    // Build upper-triangle edge index pairs
    let points: Vec<IndexedPoint> = coords
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint {
            coords: c,
            index: i,
        })
        .collect();
    let tree = RTree::bulk_load(points);
    let r2 = radius * radius;

    let edge_pairs: Vec<(usize, usize)> = coords
        .par_iter()
        .enumerate()
        .flat_map(|(i, c)| {
            tree.locate_within_distance(*c, r2)
                .filter(|p| p.index > i)
                .map(|p| (i, p.index))
                .collect::<Vec<_>>()
        })
        .collect();

    // Graph statistics (topology only — independent of feature values)
    //   S0 = Σ w_ij (over all ordered pairs) = 2 * |edges|  (binary symmetric)
    //   S1 = 0.5 * Σ (w_ij + w_ji)² = 4 * |edges| = 2 * S0
    //   S2 = Σᵢ (row_sum_i + col_sum_i)² = 4 * Σᵢ degree_i²
    let n_edges = edge_pairs.len();
    let s0 = 2.0 * n_edges as f64;
    let s1 = 2.0 * s0; // = 4 * n_edges

    if s0 == 0.0 {
        bail!("no neighbour edges found within radius {radius} — Moran's I is undefined");
    }

    let mut degrees = vec![0usize; n];
    for &(i, j) in &edge_pairs {
        degrees[i] += 1;
        degrees[j] += 1;
    }
    let s2: f64 = 4.0 * degrees.iter().map(|&d| (d as f64).powi(2)).sum::<f64>();

    let n_f = n as f64;
    let e_i = -1.0 / (n_f - 1.0);
    let denom = (n_f * n_f - 1.0) * s0 * s0;
    let var_i_base = if denom.abs() < 1e-14 {
        0.0
    } else {
        (n_f * n_f * s1 - n_f * s2 + 3.0 * s0 * s0) / denom - e_i * e_i
    };

    // Compute Moran's I per feature in parallel
    let records: Vec<MoranRecord> = (0..feature_names.len())
        .into_par_iter()
        .map(|f| {
            let col = values.column(f);
            let mean_x = col.iter().sum::<f64>() / n_f;
            let devs: Vec<f64> = col.iter().map(|&x| x - mean_x).collect();

            // Numerator = Σ_{ij} w_ij·dᵢ·dⱼ  (counted twice for symmetry)
            let numerator: f64 = edge_pairs
                .iter()
                .map(|&(i, j)| devs[i] * devs[j])
                .sum::<f64>()
                * 2.0;
            let denominator: f64 = devs.iter().map(|&d| d * d).sum::<f64>();

            let moran_i = if denominator.abs() < 1e-14 {
                0.0
            } else {
                (n_f / s0) * (numerator / denominator)
            };

            let variance_i = var_i_base.max(0.0);
            let z_score = if variance_i < 1e-14 {
                0.0
            } else {
                (moran_i - e_i) / variance_i.sqrt()
            };

            MoranRecord {
                feature: feature_names[f].clone(),
                moran_i,
                expected_i: e_i,
                variance_i,
                z_score,
                group: group.to_string(),
            }
        })
        .collect();

    Ok(records)
}

// ─── Geary's C ────────────────────────────────────────────────────────────────

/// Compute Geary's C for each column of `values` (N × F matrix).
///
/// For binary symmetric weights (radius graph):
///   C = ((n−1) / S₀) × Σ_{edge(i,j)} (zᵢ − zⱼ)² / Σᵢ zᵢ²
///
/// where S₀ = 2 × |edges| (sum of all ordered weights) and zᵢ = xᵢ − x̄.
///
/// Under complete spatial randomness (CSR): E[C] = 1.
/// C < 1 → positive spatial autocorrelation (similar values cluster together).
/// C > 1 → negative spatial autocorrelation (dissimilar values are neighbours).
///
/// Significance uses the analytical normality variance (Cliff & Ord 1981):
///   Var[C] = [(2S₁ + S₂)(n−1) − 4S₀²] / [2(n+1)S₀²]
pub fn compute_gearys_c(
    coords: &[[f64; 2]],
    values: &Array2<f64>,
    feature_names: &[String],
    radius: f64,
    group: &str,
) -> Result<Vec<GearyCRecord>> {
    let n = coords.len();
    if n < 2 {
        bail!("need at least 2 cells for Geary's C, got {n}");
    }
    if !radius.is_finite() || radius <= 0.0 {
        bail!("radius must be a finite value > 0");
    }
    if values.nrows() != n {
        bail!("values rows ({}) != coords length ({n})", values.nrows());
    }
    if values.ncols() != feature_names.len() {
        bail!(
            "values cols ({}) != feature_names length ({})",
            values.ncols(),
            feature_names.len()
        );
    }

    // Build upper-triangle edge pairs (same graph as Moran's I)
    let points: Vec<IndexedPoint> = coords
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint { coords: c, index: i })
        .collect();
    let tree = RTree::bulk_load(points);
    let r2 = radius * radius;

    let edge_pairs: Vec<(usize, usize)> = coords
        .par_iter()
        .enumerate()
        .flat_map(|(i, c)| {
            tree.locate_within_distance(*c, r2)
                .filter(|p| p.index > i)
                .map(|p| (i, p.index))
                .collect::<Vec<_>>()
        })
        .collect();

    let n_edges = edge_pairs.len();
    let s0 = 2.0 * n_edges as f64; // = Σᵢⱼ wᵢⱼ
    if s0 == 0.0 {
        bail!("no neighbour edges found within radius {radius} — Geary's C is undefined");
    }

    // For binary symmetric weights:
    //   S₁ = (1/2) Σ (wᵢⱼ + wⱼᵢ)² = 2 * S₀
    //   S₂ = Σᵢ (row_sum_i + col_sum_i)² = 4 * Σᵢ degᵢ²
    let mut degrees = vec![0usize; n];
    for &(i, j) in &edge_pairs {
        degrees[i] += 1;
        degrees[j] += 1;
    }
    let s1 = 2.0 * s0;
    let s2: f64 = 4.0 * degrees.iter().map(|&d| (d as f64).powi(2)).sum::<f64>();

    let n_f = n as f64;
    let variance_c = {
        let numerator = (2.0 * s1 + s2) * (n_f - 1.0) - 4.0 * s0 * s0;
        let denominator = 2.0 * (n_f + 1.0) * s0 * s0;
        if denominator.abs() < 1e-14 {
            0.0
        } else {
            (numerator / denominator).max(0.0)
        }
    };

    let records: Vec<GearyCRecord> = (0..feature_names.len())
        .into_par_iter()
        .map(|f| {
            let col = values.column(f);
            let mean_x = col.iter().sum::<f64>() / n_f;
            let devs: Vec<f64> = col.iter().map(|&x| x - mean_x).collect();
            let sum_sq: f64 = devs.iter().map(|&d| d * d).sum();

            // Cross-term: Σ_{edge(i,j)} (zᵢ − zⱼ)² (one per undirected edge)
            let cross_sq: f64 = edge_pairs.iter().map(|&(i, j)| (devs[i] - devs[j]).powi(2)).sum();

            let geary_c = if sum_sq.abs() < 1e-14 {
                1.0 // constant feature → C undefined; return expected value
            } else {
                ((n_f - 1.0) / s0) * cross_sq / sum_sq
            };

            let z_score = if variance_c < 1e-14 {
                0.0
            } else {
                (geary_c - 1.0) / variance_c.sqrt()
            };

            GearyCRecord {
                feature: feature_names[f].clone(),
                geary_c,
                expected_c: 1.0,
                variance_c,
                z_score,
                group: group.to_string(),
            }
        })
        .collect();

    Ok(records)
}

// ─── Moran's I permutation test ───────────────────────────────────────────────

/// Compute permutation-based p-values for Moran's I.
///
/// Shuffles cell positions `n_perms` times and re-computes Moran's I for each
/// feature to build an empirical null distribution.  Returns a conservative
/// two-tailed p-value using the (n_exceeding + 1) / (n_perms + 1) estimator.
///
/// Call alongside `compute_morans_i` — this function only returns p-values,
/// not the statistic itself.
pub fn compute_morans_i_perm(
    coords: &[[f64; 2]],
    values: &Array2<f64>,
    feature_names: &[String],
    radius: f64,
    n_perms: usize,
    seed: u64,
    group: &str,
) -> Result<Vec<MoranPermRecord>> {
    let n = coords.len();
    if n < 2 {
        bail!("need at least 2 cells for Moran's I permutation test, got {n}");
    }
    if !radius.is_finite() || radius <= 0.0 {
        bail!("radius must be a finite value > 0");
    }
    if n_perms == 0 {
        bail!("n_perms must be > 0");
    }
    if values.nrows() != n {
        bail!("values rows ({}) != coords length ({n})", values.nrows());
    }
    if values.ncols() != feature_names.len() {
        bail!(
            "values cols ({}) != feature_names length ({})",
            values.ncols(),
            feature_names.len()
        );
    }

    // Build edge pairs once — shared across all permutations
    let points: Vec<IndexedPoint> = coords
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint { coords: c, index: i })
        .collect();
    let tree = RTree::bulk_load(points);
    let r2 = radius * radius;

    let edge_pairs: Vec<(usize, usize)> = coords
        .par_iter()
        .enumerate()
        .flat_map(|(i, c)| {
            tree.locate_within_distance(*c, r2)
                .filter(|p| p.index > i)
                .map(|p| (i, p.index))
                .collect::<Vec<_>>()
        })
        .collect();

    let n_edges = edge_pairs.len();
    let s0 = 2.0 * n_edges as f64;
    if s0 == 0.0 {
        bail!("no neighbour edges found within radius {radius}");
    }
    let n_f = n as f64;
    let n_features = feature_names.len();

    // Precompute per-feature means (invariant under permutation)
    let means: Vec<f64> = (0..n_features)
        .map(|f| values.column(f).iter().sum::<f64>() / n_f)
        .collect();

    // Inner function: compute Moran's I for given row-permutation indices
    let moran_i = |perm: &[usize], f: usize| -> f64 {
        let col = values.column(f);
        let mean = means[f];
        let devs: Vec<f64> = perm.iter().map(|&orig| col[orig] - mean).collect();
        let numerator: f64 = edge_pairs.iter().map(|&(i, j)| devs[i] * devs[j]).sum::<f64>() * 2.0;
        let denominator: f64 = devs.iter().map(|&d| d * d).sum::<f64>();
        if denominator.abs() < 1e-14 {
            return 0.0;
        }
        (n_f / s0) * (numerator / denominator)
    };

    // Observed Moran's I (identity permutation)
    let identity: Vec<usize> = (0..n).collect();
    let observed: Vec<f64> = (0..n_features)
        .into_par_iter()
        .map(|f| moran_i(&identity, f))
        .collect();

    // Per-permutation boolean matrix: perm_exceeded[p][f] = true if |I_perm| ≥ |I_obs|
    let perm_exceeded: Vec<Vec<bool>> = (0..n_perms)
        .into_par_iter()
        .map(|p| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(p as u64 + 1));
            let mut perm: Vec<usize> = (0..n).collect();
            perm.shuffle(&mut rng);
            (0..n_features)
                .map(|f| moran_i(&perm, f).abs() >= observed[f].abs())
                .collect()
        })
        .collect();

    let feature_exceed: Vec<usize> = (0..n_features)
        .map(|f| perm_exceeded.iter().filter(|v| v[f]).count())
        .collect();

    let n_perms_f = n_perms as f64;
    let records: Vec<MoranPermRecord> = (0..n_features)
        .map(|f| MoranPermRecord {
            feature: feature_names[f].clone(),
            p_value_perm: (feature_exceed[f] as f64 + 1.0) / (n_perms_f + 1.0),
            group: group.to_string(),
        })
        .collect();

    Ok(records)
}

// ─── Bivariate Moran's I ──────────────────────────────────────────────────────

/// Compute bivariate Moran's I for all pairs of features (f_a, f_b) with f_a < f_b.
///
/// For each pair:
///   I_AB = (n/S₀) × Σ_{edge(i,j)} (devᵢᴬ·devⱼᴮ + devᵢᴮ·devⱼᴬ) / √(Σdevᴬ² · Σdevᴮ²)
///
/// This tests whether cells with a high value of feature A tend to have
/// neighbours with a high value of feature B.  The z-score is computed
/// using the topology-only variance from global Moran's I (E[I_AB] = 0
/// under the null of no bivariate spatial association).
pub fn compute_bivariate_morans_i(
    coords: &[[f64; 2]],
    values: &Array2<f64>,
    feature_names: &[String],
    radius: f64,
    group: &str,
) -> Result<Vec<BivariateMoranRecord>> {
    let n = coords.len();
    if n < 2 {
        bail!("need at least 2 cells for bivariate Moran's I, got {n}");
    }
    if !radius.is_finite() || radius <= 0.0 {
        bail!("radius must be a finite value > 0");
    }
    if values.nrows() != n {
        bail!("values rows ({}) != coords length ({n})", values.nrows());
    }
    if values.ncols() != feature_names.len() {
        bail!(
            "values cols ({}) != feature_names length ({})",
            values.ncols(),
            feature_names.len()
        );
    }
    let n_features = feature_names.len();
    if n_features < 2 {
        bail!("need at least 2 features for bivariate Moran's I");
    }

    // Build edge pairs (upper triangle)
    let points: Vec<IndexedPoint> = coords
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint { coords: c, index: i })
        .collect();
    let tree = RTree::bulk_load(points);
    let r2 = radius * radius;
    let edge_pairs: Vec<(usize, usize)> = coords
        .par_iter()
        .enumerate()
        .flat_map(|(i, c)| {
            tree.locate_within_distance(*c, r2)
                .filter(|p| p.index > i)
                .map(|p| (i, p.index))
                .collect::<Vec<_>>()
        })
        .collect();

    let n_edges = edge_pairs.len();
    let s0 = 2.0 * n_edges as f64;
    if s0 == 0.0 {
        bail!("no neighbour edges found within radius {radius}");
    }

    // Topology-only variance (E[I_AB] = 0 under null)
    let mut degrees = vec![0usize; n];
    for &(i, j) in &edge_pairs {
        degrees[i] += 1;
        degrees[j] += 1;
    }
    let s1 = 2.0 * s0; // = 4 * n_edges for binary symmetric weights
    let s2: f64 = 4.0 * degrees.iter().map(|&d| (d as f64).powi(2)).sum::<f64>();
    let n_f = n as f64;
    let var_base = {
        let denom = (n_f * n_f - 1.0) * s0 * s0;
        if denom.abs() < 1e-14 {
            0.0
        } else {
            (n_f * n_f * s1 - n_f * s2 + 3.0 * s0 * s0) / denom
        }
    };

    // Precompute deviations and sum-of-squares for each feature
    let devs: Vec<Vec<f64>> = (0..n_features)
        .map(|f| {
            let col = values.column(f);
            let mean = col.iter().sum::<f64>() / n_f;
            col.iter().map(|&x| x - mean).collect()
        })
        .collect();
    let ss: Vec<f64> = devs
        .iter()
        .map(|d| d.iter().map(|&v| v * v).sum::<f64>())
        .collect();

    // Compute all upper-triangle pairs in parallel
    let pairs: Vec<(usize, usize)> = (0..n_features)
        .flat_map(|a| (a + 1..n_features).map(move |b| (a, b)))
        .collect();

    let records: Vec<BivariateMoranRecord> = pairs
        .into_par_iter()
        .map(|(fa, fb)| {
            let da = &devs[fa];
            let db = &devs[fb];
            let denom_sq = ss[fa] * ss[fb];

            let numerator: f64 = edge_pairs
                .iter()
                .map(|&(i, j)| da[i] * db[j] + da[j] * db[i])
                .sum();

            let bivariate_i = if denom_sq < 1e-28 {
                0.0
            } else {
                (n_f / s0) * numerator / denom_sq.sqrt()
            };

            let z_score = if var_base < 1e-14 {
                0.0
            } else {
                bivariate_i / var_base.sqrt()
            };

            BivariateMoranRecord {
                feature_a: feature_names[fa].clone(),
                feature_b: feature_names[fb].clone(),
                bivariate_i,
                z_score,
                group: group.to_string(),
            }
        })
        .collect();

    Ok(records)
}

// ─── Local Moran's I (LISA) ───────────────────────────────────────────────────

/// Compute Local Moran's I (LISA) for each cell and feature.
///
/// For cell i and feature f:
///   I_i = (z_i / m₂) × Σⱼ∈N(i) zⱼ
///
/// where z_i = x_i − x̄, m₂ = (1/n) × Σ z_j², and N(i) are the spatial
/// neighbours within `radius`.
///
/// The z-score uses the analytical conditional-randomisation variance from
/// Anselin (1995):
///   E[I_i] = −dᵢ / (n−1)
///   Var[I_i] = dᵢ(n−b₂)/(n−1) + dᵢ(dᵢ−1)(2b₂−n)/((n−1)(n−2)) − E[I_i]²
/// where b₂ = m₄/m₂² (global kurtosis) and dᵢ = |N(i)|.
///
/// Returns one record per (cell, feature) combination.
pub fn compute_local_morans_i(
    coords: &[[f64; 2]],
    barcodes: &[String],
    values: &Array2<f64>,
    feature_names: &[String],
    radius: f64,
    group: &str,
) -> Result<Vec<LocalMoranRecord>> {
    let n = coords.len();

    if n < 4 {
        bail!("need at least 4 cells for Local Moran's I, got {n}");
    }
    if !radius.is_finite() || radius <= 0.0 {
        bail!("radius must be a finite value > 0");
    }
    if barcodes.len() != n {
        bail!("barcodes length ({}) != coords length ({n})", barcodes.len());
    }
    if values.nrows() != n {
        bail!("values rows ({}) != coords length ({n})", values.nrows());
    }
    if values.ncols() != feature_names.len() {
        bail!(
            "values cols ({}) != feature_names length ({})",
            values.ncols(),
            feature_names.len()
        );
    }

    // Build adjacency list from the same R*-tree as global Moran's I
    let points: Vec<IndexedPoint> = coords
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint {
            coords: c,
            index: i,
        })
        .collect();
    let tree = RTree::bulk_load(points);
    let r2 = radius * radius;

    let neighbors: Vec<Vec<usize>> = coords
        .iter()
        .enumerate()
        .map(|(i, c)| {
            tree.locate_within_distance(*c, r2)
                .filter(|p| p.index != i)
                .map(|p| p.index)
                .collect()
        })
        .collect();

    let n_f = n as f64;

    let records: Vec<LocalMoranRecord> = (0..feature_names.len())
        .into_par_iter()
        .flat_map_iter(|f| {
            let col = values.column(f);
            let mean_x = col.iter().sum::<f64>() / n_f;
            let devs: Vec<f64> = col.iter().map(|&x| x - mean_x).collect();

            // m₂ = (1/n) Σ z_j²   m₄ = (1/n) Σ z_j⁴
            let m2 = devs.iter().map(|&d| d * d).sum::<f64>() / n_f;
            let m4 = devs.iter().map(|&d| d.powi(4)).sum::<f64>() / n_f;
            let b2 = if m2 > 1e-14 { m4 / (m2 * m2) } else { 0.0 };

            // Collect into Vec so that all borrows of `devs`/`neighbors` remain
            // within the lifetime of this `Fn` closure invocation.
            (0..n)
                .map(|i| {
                    let d_i = neighbors[i].len();
                    let spatial_lag: f64 = neighbors[i].iter().map(|&j| devs[j]).sum();

                    let local_i = if m2.abs() < 1e-14 {
                        0.0
                    } else {
                        (devs[i] / m2) * spatial_lag
                    };

                    let e_i = -(d_i as f64) / (n_f - 1.0);

                    let var_i = if n_f > 2.0 && d_i > 0 {
                        let d = d_i as f64;
                        let term1 = d * (n_f - b2) / (n_f - 1.0);
                        let term2 = d * (d - 1.0) * (2.0 * b2 - n_f)
                            / ((n_f - 1.0) * (n_f - 2.0));
                        (term1 + term2 - e_i * e_i).max(0.0)
                    } else {
                        0.0
                    };

                    let z_score = if var_i < 1e-14 {
                        0.0
                    } else {
                        (local_i - e_i) / var_i.sqrt()
                    };

                    LocalMoranRecord {
                        cell_i: barcodes[i].clone(),
                        feature: feature_names[f].clone(),
                        local_i,
                        z_score,
                        group: group.to_string(),
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(records)
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        compute_bivariate_morans_i, compute_gearys_c, compute_local_morans_i, compute_morans_i,
        compute_morans_i_perm, LocalMoranRecord,
    };
    use ndarray::arr2;

    #[test]
    fn morans_i_perfectly_clustered_is_one() {
        // Two clusters of 2 cells each, far apart.
        // Feature value = cluster membership (0 or 1).
        // With radius=2 only within-cluster edges exist.
        //
        //   deviations: [-0.5, -0.5, 0.5, 0.5]
        //   edges: (0,1) and (2,3)
        //   numerator = 2 * (0.25 + 0.25) = 1.0
        //   denominator = 4 * 0.25 = 1.0
        //   S0 = 4, I = (4/4)*(1/1) = 1.0
        let coords = [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
        let values = arr2(&[[0.0], [0.0], [1.0], [1.0]]);
        let names = vec!["feat".to_string()];

        let records = compute_morans_i(&coords, &values, &names, 2.0, "test").unwrap();
        assert_eq!(records.len(), 1);
        let i = records[0].moran_i;
        assert!((i - 1.0).abs() < 1e-10, "expected I=1.0, got {i}");
    }

    #[test]
    fn morans_i_no_spatial_structure_near_expected() {
        // Uniform random-ish values; I should be near E[I] = -1/(N-1).
        // We just check it doesn't blow up and E[I] formula is correct.
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let values = arr2(&[[1.0], [2.0], [3.0], [4.0]]);
        let names = vec!["feat".to_string()];

        let records = compute_morans_i(&coords, &values, &names, 2.0, "test").unwrap();
        assert_eq!(records.len(), 1);
        let expected = -1.0 / 3.0;
        assert!((records[0].expected_i - expected).abs() < 1e-10);
    }

    #[test]
    fn morans_i_rejects_non_positive_radius() {
        let coords = [[0.0, 0.0], [1.0, 0.0]];
        let values = arr2(&[[1.0], [2.0]]);
        let names = vec!["f".to_string()];
        assert!(compute_morans_i(&coords, &values, &names, 0.0, "g").is_err());
        assert!(compute_morans_i(&coords, &values, &names, -1.0, "g").is_err());
    }

    #[test]
    fn local_morans_i_hotspot_has_positive_local_i() {
        // Two clusters of 4 cells with high/low values respectively.
        // The high-value cluster cells should have positive local I.
        let coords = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [10.0, 0.0],
            [11.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
        ];
        let barcodes: Vec<String> = (0..8).map(|i| format!("c{i}")).collect();
        let values = arr2(&[
            [1.0f64],
            [1.0],
            [1.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ]);
        let names = vec!["feat".to_string()];

        let records =
            compute_local_morans_i(&coords, &barcodes, &values, &names, 2.0, "test").unwrap();
        assert_eq!(records.len(), 8);

        // All cells with value 1.0 (first cluster) should have positive local I
        let high_cells: Vec<&LocalMoranRecord> =
            records.iter().filter(|r| r.cell_i.starts_with('c') && {
                let idx: usize = r.cell_i[1..].parse().unwrap();
                idx < 4
            }).collect();
        for r in &high_cells {
            assert!(r.local_i > 0.0, "expected positive local I for cell {}", r.cell_i);
        }
    }

    #[test]
    fn bivariate_morans_i_correlated_features_positive() {
        // Two features with identical values → I_AB ≈ I_A ≈ 1 (perfectly correlated spatial pattern)
        let coords = [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
        let values = arr2(&[[0.0f64, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]);
        let names = vec!["a".to_string(), "b".to_string()];
        let records =
            compute_bivariate_morans_i(&coords, &values, &names, 2.0, "test").unwrap();
        assert_eq!(records.len(), 1);
        assert!(
            records[0].bivariate_i > 0.5,
            "expected positive bivariate I for correlated features, got {}",
            records[0].bivariate_i
        );
    }

    #[test]
    fn bivariate_morans_i_anticorrelated_features_negative() {
        // Feature A high where B is low and vice versa across two clusters
        let coords = [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
        let values = arr2(&[[1.0f64, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]);
        let names = vec!["a".to_string(), "b".to_string()];
        let records =
            compute_bivariate_morans_i(&coords, &values, &names, 2.0, "test").unwrap();
        assert_eq!(records.len(), 1);
        assert!(
            records[0].bivariate_i < 0.0,
            "expected negative bivariate I for anticorrelated features, got {}",
            records[0].bivariate_i
        );
    }

    #[test]
    fn gearys_c_clustered_is_below_one() {
        // Two tight clusters: values match spatial clusters → C < 1 (positive autocorrelation)
        let coords = [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
        let values = arr2(&[[0.0], [0.0], [1.0], [1.0]]);
        let names = vec!["feat".to_string()];
        let records = compute_gearys_c(&coords, &values, &names, 2.0, "test").unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].expected_c, 1.0);
        assert!(
            records[0].geary_c < 1.0,
            "expected C < 1 for clustered data, got {}",
            records[0].geary_c
        );
        // z_score should be negative (clustered = low C = below expected)
        assert!(records[0].z_score < 0.0);
    }

    #[test]
    fn gearys_c_dispersed_is_above_one() {
        // Alternating values in a line: high negative autocorrelation → C > 1
        let coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let values = arr2(&[[0.0], [1.0], [0.0], [1.0]]);
        let names = vec!["feat".to_string()];
        let records = compute_gearys_c(&coords, &values, &names, 1.5, "test").unwrap();
        assert_eq!(records.len(), 1);
        assert!(
            records[0].geary_c > 1.0,
            "expected C > 1 for dispersed data, got {}",
            records[0].geary_c
        );
    }

    #[test]
    fn gearys_c_rejects_non_positive_radius() {
        let coords = [[0.0, 0.0], [1.0, 0.0]];
        let values = arr2(&[[1.0], [2.0]]);
        let names = vec!["f".to_string()];
        assert!(compute_gearys_c(&coords, &values, &names, 0.0, "g").is_err());
    }

    #[test]
    fn morans_i_perm_clustered_has_small_p() {
        // Two clusters of 4 cells each, far apart; values match cluster membership.
        // Observed I = 1.0; only ~2 of 70 value assignments achieve |I|=1, so p ≈ 0.03.
        let coords = [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],    // cluster 1
            [10.0, 0.0], [11.0, 0.0], [10.0, 1.0], [11.0, 1.0], // cluster 2
        ];
        let values = arr2(&[
            [0.0f64], [0.0], [0.0], [0.0],
            [1.0],    [1.0], [1.0], [1.0],
        ]);
        let names = vec!["feat".to_string()];
        let records =
            compute_morans_i_perm(&coords, &values, &names, 2.0, 999, 42, "test").unwrap();
        assert_eq!(records.len(), 1);
        assert!(
            records[0].p_value_perm <= 0.05,
            "expected small p-value for strongly clustered data, got {}",
            records[0].p_value_perm
        );
    }

    #[test]
    fn local_morans_i_rejects_too_few_cells() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let barcodes: Vec<String> = (0..3).map(|i| format!("c{i}")).collect();
        let values = arr2(&[[1.0f64], [2.0], [3.0]]);
        let names = vec!["f".to_string()];
        assert!(compute_local_morans_i(&coords, &barcodes, &values, &names, 1.5, "g").is_err());
    }
}
