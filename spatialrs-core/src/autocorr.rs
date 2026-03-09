use anyhow::{bail, Result};
use ndarray::Array2;
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::Serialize;

// ─── public types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct MoranRecord {
    pub feature:    String,
    pub moran_i:    f64,
    pub expected_i: f64,
    pub variance_i: f64,
    pub z_score:    f64,
    pub group:      String,
}

// ─── rstar integration ────────────────────────────────────────────────────────

#[derive(Clone)]
struct IndexedPoint {
    coords: [f64; 2],
    index:  usize,
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
    coords:        &[[f64; 2]],
    values:        &Array2<f64>,
    feature_names: &[String],
    radius:        f64,
    group:         &str,
) -> Result<Vec<MoranRecord>> {
    let n = coords.len();

    if n < 2 {
        bail!("need at least 2 cells for Moran's I, got {n}");
    }
    if !radius.is_finite() || radius <= 0.0 {
        bail!("radius must be a finite value > 0");
    }
    if values.nrows() != n {
        bail!(
            "values rows ({}) != coords length ({n})",
            values.nrows()
        );
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

    let n_f   = n as f64;
    let e_i   = -1.0 / (n_f - 1.0);
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
            let col     = values.column(f);
            let mean_x  = col.iter().sum::<f64>() / n_f;
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
                feature:    feature_names[f].clone(),
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

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::compute_morans_i;
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
        let names  = vec!["feat".to_string()];

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
        let names  = vec!["feat".to_string()];

        let records = compute_morans_i(&coords, &values, &names, 2.0, "test").unwrap();
        assert_eq!(records.len(), 1);
        let expected = -1.0 / 3.0;
        assert!((records[0].expected_i - expected).abs() < 1e-10);
    }

    #[test]
    fn morans_i_rejects_non_positive_radius() {
        let coords = [[0.0, 0.0], [1.0, 0.0]];
        let values = arr2(&[[1.0], [2.0]]);
        let names  = vec!["f".to_string()];
        assert!(compute_morans_i(&coords, &values, &names, 0.0, "g").is_err());
        assert!(compute_morans_i(&coords, &values, &names, -1.0, "g").is_err());
    }
}
