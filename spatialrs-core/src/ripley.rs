use anyhow::{bail, Result};
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::Serialize;
use std::f64::consts::PI;

// ─── public types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct RipleyRecord {
    pub cell_type: String,
    pub r: f64,
    pub k_r: f64,
    pub l_r: f64,
    pub group: String,
}

#[derive(Serialize)]
pub struct CrossRipleyRecord {
    pub type_a: String,
    pub type_b: String,
    pub r: f64,
    pub k_cross: f64,
    pub l_cross: f64,
    pub group: String,
}

// ─── rstar integration ────────────────────────────────────────────────────────

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

/// Estimate Ripley's K and L functions for a target cell type.
///
/// For each radius r in `radii`:
///   K(r) = (A / n²) × Σ_{i≠j} 1(d(i,j) ≤ r)
///   L(r) = √(K(r) / π) − r
///
/// where n is the number of target-type cells and A is their bounding-box area.
/// Under complete spatial randomness (CSR), L(r) ≈ 0; positive values
/// indicate clustering at scale r, negative values indicate regularity.
pub fn compute_ripley(
    coords: &[[f64; 2]],
    cell_types: &[String],
    target_type: &str,
    radii: &[f64],
    group: &str,
) -> Result<Vec<RipleyRecord>> {
    if coords.len() != cell_types.len() {
        bail!(
            "coords length ({}) does not match cell_types length ({})",
            coords.len(),
            cell_types.len()
        );
    }
    if radii.is_empty() {
        bail!("radii must not be empty");
    }
    for &r in radii {
        if !r.is_finite() || r <= 0.0 {
            bail!("all radii must be finite and > 0, got {r}");
        }
    }

    // Filter to target cell type
    let target_coords: Vec<[f64; 2]> = coords
        .iter()
        .zip(cell_types.iter())
        .filter(|(_, t)| t.as_str() == target_type)
        .map(|(c, _)| *c)
        .collect();

    let n = target_coords.len();
    if n < 2 {
        bail!(
            "need at least 2 cells of type '{}' for Ripley's K, got {n}",
            target_type
        );
    }

    // Bounding box area (floor at 1.0 to avoid division-by-zero for collinear points)
    let x_min = target_coords
        .iter()
        .map(|c| c[0])
        .fold(f64::INFINITY, f64::min);
    let x_max = target_coords
        .iter()
        .map(|c| c[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = target_coords
        .iter()
        .map(|c| c[1])
        .fold(f64::INFINITY, f64::min);
    let y_max = target_coords
        .iter()
        .map(|c| c[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let area = ((x_max - x_min) * (y_max - y_min)).max(1.0);

    // Build R*-tree for fast range queries
    let points: Vec<IndexedPoint> = target_coords
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint { coords: c, index: i })
        .collect();
    let tree = RTree::bulk_load(points);

    let n_f = n as f64;

    let records: Vec<RipleyRecord> = radii
        .par_iter()
        .map(|&r| {
            let r2 = r * r;
            // Count ordered pairs (i, j) with i ≠ j and d(i,j) ≤ r
            let pair_count: usize = target_coords
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    tree.locate_within_distance(*c, r2)
                        .filter(|p| p.index != i)
                        .count()
                })
                .sum();

            let k_r = area * pair_count as f64 / (n_f * n_f);
            let l_r = (k_r / PI).sqrt() - r;

            RipleyRecord {
                cell_type: target_type.to_string(),
                r,
                k_r,
                l_r,
                group: group.to_string(),
            }
        })
        .collect();

    Ok(records)
}

// ─── Cross-K / Cross-L function ──────────────────────────────────────────────

/// Estimate the bivariate (cross) Ripley's K function between two cell types.
///
///   K_AB(r) = (A / (nᴬ · nᴮ)) × Σᵢ∈A Σⱼ∈B 1(d(i,j) ≤ r)
///   L_AB(r) = √(K_AB(r) / π) − r
///
/// Under CSR with no interaction between A and B, L_AB(r) ≈ 0.
/// Positive values indicate that type B is more clustered around type A
/// than expected by chance at scale r.
///
/// If `type_a == type_b` this reduces to the standard single-type K
/// (pairs are still i≠j).
pub fn compute_cross_ripley(
    coords: &[[f64; 2]],
    cell_types: &[String],
    type_a: &str,
    type_b: &str,
    radii: &[f64],
    group: &str,
) -> Result<Vec<CrossRipleyRecord>> {
    if coords.len() != cell_types.len() {
        bail!(
            "coords length ({}) does not match cell_types length ({})",
            coords.len(),
            cell_types.len()
        );
    }
    if radii.is_empty() {
        bail!("radii must not be empty");
    }
    for &r in radii {
        if !r.is_finite() || r <= 0.0 {
            bail!("all radii must be finite and > 0, got {r}");
        }
    }

    let coords_a: Vec<[f64; 2]> = coords
        .iter()
        .zip(cell_types.iter())
        .filter(|(_, t)| t.as_str() == type_a)
        .map(|(c, _)| *c)
        .collect();

    let coords_b: Vec<[f64; 2]> = coords
        .iter()
        .zip(cell_types.iter())
        .filter(|(_, t)| t.as_str() == type_b)
        .map(|(c, _)| *c)
        .collect();

    let na = coords_a.len();
    let nb = coords_b.len();

    if na < 1 {
        bail!("no cells of type '{}' found", type_a);
    }
    if nb < 1 {
        bail!("no cells of type '{}' found", type_b);
    }

    // Bounding box over both types combined
    let all_x = coords_a
        .iter()
        .chain(coords_b.iter())
        .map(|c| c[0]);
    let all_y = coords_a
        .iter()
        .chain(coords_b.iter())
        .map(|c| c[1]);
    let x_min = all_x.clone().fold(f64::INFINITY, f64::min);
    let x_max = all_x.fold(f64::NEG_INFINITY, f64::max);
    let y_min = all_y.clone().fold(f64::INFINITY, f64::min);
    let y_max = all_y.fold(f64::NEG_INFINITY, f64::max);
    let area = ((x_max - x_min) * (y_max - y_min)).max(1.0);

    // Build R*-tree over type-B cells for range queries
    let points_b: Vec<IndexedPoint> = coords_b
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint { coords: c, index: i })
        .collect();
    let tree_b = RTree::bulk_load(points_b);

    let na_f = na as f64;
    let nb_f = nb as f64;
    let same_type = type_a == type_b;

    let records: Vec<CrossRipleyRecord> = radii
        .par_iter()
        .map(|&r| {
            let r2 = r * r;
            let pair_count: usize = coords_a
                .iter()
                .enumerate()
                .map(|(ia, ca)| {
                    tree_b
                        .locate_within_distance(*ca, r2)
                        .filter(|p| {
                            // exclude self when A == B
                            if same_type {
                                let dx = p.coords[0] - ca[0];
                                let dy = p.coords[1] - ca[1];
                                dx * dx + dy * dy > 0.0
                            } else {
                                let _ = ia;
                                true
                            }
                        })
                        .count()
                })
                .sum();

            let k_cross = area * pair_count as f64 / (na_f * nb_f);
            let l_cross = (k_cross / PI).sqrt() - r;

            CrossRipleyRecord {
                type_a: type_a.to_string(),
                type_b: type_b.to_string(),
                r,
                k_cross,
                l_cross,
                group: group.to_string(),
            }
        })
        .collect();

    Ok(records)
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{compute_cross_ripley, compute_ripley};

    #[test]
    fn ripley_returns_one_record_per_radius() {
        let coords: Vec<[f64; 2]> = (0..3)
            .flat_map(|x| (0..3).map(move |y| [x as f64 * 10.0, y as f64 * 10.0]))
            .collect();
        let types = vec!["T".to_string(); 9];
        let radii = vec![5.0, 15.0, 25.0];
        let records =
            compute_ripley(&coords, &types, "T", &radii, "test").unwrap();
        assert_eq!(records.len(), 3);
        // L(r) at all scales, just check it doesn't blow up
        for r in &records {
            assert!(r.k_r.is_finite());
            assert!(r.l_r.is_finite());
        }
    }

    #[test]
    fn ripley_clustered_has_positive_l() {
        // Two tight clusters of 3 cells each, far apart
        // At r=2 only within-cluster pairs counted → high density relative to bbox → L > 0
        let coords = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [50.0, 50.0],
            [51.0, 50.0],
            [50.5, 51.0],
        ];
        let types = vec!["T".to_string(); 6];
        let records = compute_ripley(&coords, &types, "T", &[2.0], "test").unwrap();
        assert_eq!(records.len(), 1);
        // With clustering, L(r) should be positive at short range
        assert!(
            records[0].l_r > 0.0,
            "expected L > 0 for clustered cells, got {}",
            records[0].l_r
        );
    }

    #[test]
    fn ripley_rejects_empty_radii() {
        let coords = vec![[0.0, 0.0], [1.0, 0.0]];
        let types = vec!["T".to_string(); 2];
        assert!(compute_ripley(&coords, &types, "T", &[], "g").is_err());
    }

    #[test]
    fn ripley_rejects_too_few_target_cells() {
        let coords = vec![[0.0, 0.0], [1.0, 0.0]];
        let types = vec!["T".to_string(), "U".to_string()];
        assert!(compute_ripley(&coords, &types, "T", &[1.0], "g").is_err());
    }

    #[test]
    fn cross_ripley_co_localised_has_positive_l() {
        // Type A cells are spread over a large domain; B cells are clustered
        // near A[0].  The large bounding box means K_AB >> π·r², so L_cross > 0.
        let coords = vec![
            [0.0, 0.0],   [100.0, 0.0], [0.0, 100.0], // A — spread out
            [0.5, 0.5],   [1.0, 0.5],   [0.5, 1.0],   // B — clustered near A[0]
        ];
        let types = vec![
            "A".to_string(), "A".to_string(), "A".to_string(),
            "B".to_string(), "B".to_string(), "B".to_string(),
        ];
        let records =
            compute_cross_ripley(&coords, &types, "A", "B", &[2.0], "test").unwrap();
        assert_eq!(records.len(), 1);
        assert!(
            records[0].l_cross > 0.0,
            "expected L_cross > 0 for co-localised types, got {}",
            records[0].l_cross
        );
    }

    #[test]
    fn cross_ripley_rejects_missing_type() {
        let coords = vec![[0.0, 0.0], [1.0, 0.0]];
        let types = vec!["A".to_string(), "A".to_string()];
        assert!(compute_cross_ripley(&coords, &types, "A", "B", &[1.0], "g").is_err());
    }
}
