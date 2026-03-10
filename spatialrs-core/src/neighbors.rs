use anyhow::{bail, Result};
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::Serialize;

#[derive(Serialize)]
pub struct GraphStatsRecord {
    pub cell_i: String,
    pub n_neighbors: usize,
    pub group: String,
}

#[derive(Serialize)]
pub struct EdgeRecord {
    pub cell_i: String,
    pub cell_j: String,
    pub distance: f64,
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

// ─── graph statistics ─────────────────────────────────────────────────────────

/// Compute per-cell neighbour count (degree) for a radius graph.
///
/// Returns one record per cell with `n_neighbors` = number of cells within
/// `radius` (self excluded).  Useful as a QC metric for choosing `--radius`.
pub fn compute_graph_stats(
    coords: &[[f64; 2]],
    barcodes: &[String],
    radius: f64,
    group: &str,
) -> Result<Vec<GraphStatsRecord>> {
    validate_input_lengths(coords.len(), barcodes.len())?;
    validate_positive_radius(radius)?;

    let points: Vec<IndexedPoint> = coords
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint { coords: c, index: i })
        .collect();
    let tree = RTree::bulk_load(points);
    let r2 = radius * radius;

    let records: Vec<GraphStatsRecord> = coords
        .par_iter()
        .enumerate()
        .map(|(i, c)| {
            let n_neighbors = tree
                .locate_within_distance(*c, r2)
                .filter(|p| p.index != i)
                .count();
            GraphStatsRecord {
                cell_i: barcodes[i].clone(),
                n_neighbors,
                group: group.to_string(),
            }
        })
        .collect();

    Ok(records)
}

// ─── radius graph ─────────────────────────────────────────────────────────────

/// Build a radius graph from `coords` (N×2 slice of [x, y]) and `barcodes`.
/// Returns edges in **both** directions (i→j and j→i) so that per-cell
/// neighbour lookups work without a second pass.
pub fn radius_graph(
    coords: &[[f64; 2]],
    barcodes: &[String],
    radius: f64,
    group: &str,
) -> Result<Vec<EdgeRecord>> {
    validate_input_lengths(coords.len(), barcodes.len())?;
    validate_positive_radius(radius)?;

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

    // Collect upper-triangle pairs in parallel, then emit both directions.
    let pairs: Vec<(usize, usize, f64)> = coords
        .par_iter()
        .enumerate()
        .flat_map(|(i, c)| {
            tree.locate_within_distance(*c, r2)
                .filter(|p| p.index > i) // upper triangle only
                .map(|p| {
                    let d = (p.distance_2(c) as f64).sqrt();
                    (i, p.index, d)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut records = Vec::with_capacity(pairs.len() * 2);
    for (i, j, d) in pairs {
        records.push(EdgeRecord {
            cell_i: barcodes[i].clone(),
            cell_j: barcodes[j].clone(),
            distance: d,
            group: group.to_string(),
        });
        records.push(EdgeRecord {
            cell_i: barcodes[j].clone(),
            cell_j: barcodes[i].clone(),
            distance: d,
            group: group.to_string(),
        });
    }

    Ok(records)
}

/// Upper-triangle index pairs only, no barcodes.  Used by `interactions` permutation test.
pub(crate) fn radius_graph_index_pairs(
    coords: &[[f64; 2]],
    radius: f64,
) -> Result<Vec<(usize, usize)>> {
    validate_positive_radius(radius)?;

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

    let pairs: Vec<(usize, usize)> = coords
        .par_iter()
        .enumerate()
        .flat_map(|(i, c)| {
            tree.locate_within_distance(*c, r2)
                .filter(|p| p.index > i)
                .map(|p| (i, p.index))
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(pairs)
}

/// Upper-triangle-only radius graph (one edge per pair, no duplicates).
/// Used internally by `interactions`.
pub(crate) fn radius_graph_dedup(
    coords: &[[f64; 2]],
    barcodes: &[String],
    radius: f64,
    group: &str,
) -> Result<Vec<EdgeRecord>> {
    validate_input_lengths(coords.len(), barcodes.len())?;
    validate_positive_radius(radius)?;

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

    let records: Vec<EdgeRecord> = coords
        .par_iter()
        .enumerate()
        .flat_map(|(i, c)| {
            tree.locate_within_distance(*c, r2)
                .filter(|p| p.index > i)
                .map(|p| {
                    let d = (p.distance_2(c) as f64).sqrt();
                    EdgeRecord {
                        cell_i: barcodes[i].clone(),
                        cell_j: barcodes[p.index].clone(),
                        distance: d,
                        group: group.to_string(),
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(records)
}

// ─── kNN graph ────────────────────────────────────────────────────────────────

/// Build a kNN graph.  For each cell the `k` nearest neighbours are returned
/// (self is excluded).  Edges are NOT deduplicated — both directions may appear.
pub fn knn_graph(
    coords: &[[f64; 2]],
    barcodes: &[String],
    k: usize,
    group: &str,
) -> Result<Vec<EdgeRecord>> {
    validate_input_lengths(coords.len(), barcodes.len())?;

    let points: Vec<IndexedPoint> = coords
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint {
            coords: c,
            index: i,
        })
        .collect();

    let tree = RTree::bulk_load(points);

    let records: Vec<EdgeRecord> = coords
        .par_iter()
        .enumerate()
        .flat_map(|(i, c)| {
            tree.nearest_neighbor_iter_with_distance_2(c)
                .filter(|(p, _)| p.index != i) // skip self
                .take(k)
                .map(|(p, d2)| EdgeRecord {
                    cell_i: barcodes[i].clone(),
                    cell_j: barcodes[p.index].clone(),
                    distance: d2.sqrt(),
                    group: group.to_string(),
                })
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(records)
}

fn validate_input_lengths(n_coords: usize, n_barcodes: usize) -> Result<()> {
    if n_coords != n_barcodes {
        bail!("coords length ({n_coords}) does not match barcodes length ({n_barcodes})");
    }
    Ok(())
}

fn validate_positive_radius(radius: f64) -> Result<()> {
    if !radius.is_finite() || radius <= 0.0 {
        bail!("radius must be a finite value > 0");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{compute_graph_stats, radius_graph};

    #[test]
    fn graph_stats_counts_neighbours_correctly() {
        // 4 cells in a line; radius 1.5 → endpoints have 1 neighbour, middle cells have 2
        let coords = [[0.0f64, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let barcodes: Vec<String> = (0..4).map(|i| format!("c{i}")).collect();
        let records = compute_graph_stats(&coords, &barcodes, 1.5, "g").unwrap();
        assert_eq!(records.len(), 4);
        let get = |id: &str| records.iter().find(|r| r.cell_i == id).unwrap().n_neighbors;
        assert_eq!(get("c0"), 1);
        assert_eq!(get("c1"), 2);
        assert_eq!(get("c2"), 2);
        assert_eq!(get("c3"), 1);
    }

    #[test]
    fn graph_stats_rejects_non_positive_radius() {
        let coords = [[0.0f64, 0.0], [1.0, 0.0]];
        let barcodes = vec!["a".to_string(), "b".to_string()];
        assert!(compute_graph_stats(&coords, &barcodes, 0.0, "g").is_err());
    }

    #[test]
    fn radius_graph_rejects_non_positive_radius() {
        let coords = [[0.0, 0.0], [1.0, 1.0]];
        let barcodes = vec!["a".to_string(), "b".to_string()];

        let err = match radius_graph(&coords, &barcodes, 0.0, "g") {
            Ok(_) => panic!("expected invalid radius error"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("radius must be a finite value > 0"));

        let err = match radius_graph(&coords, &barcodes, -1.0, "g") {
            Ok(_) => panic!("expected invalid radius error"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("radius must be a finite value > 0"));
    }
}
