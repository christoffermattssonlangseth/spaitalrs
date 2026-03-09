use crate::neighbors::{radius_graph_dedup, radius_graph_index_pairs};
use anyhow::{bail, Result};
use rand::seq::SliceRandom;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Serialize)]
pub struct InteractionRecord {
    pub group: String,
    pub cell_type_a: String,
    pub cell_type_b: String,
    pub count: usize,
}

#[derive(Serialize)]
pub struct InteractionStatsRecord {
    pub group:         String,
    pub cell_type_a:   String,
    pub cell_type_b:   String,
    pub observed:      usize,
    pub expected_mean: f64,
    pub expected_std:  f64,
    pub z_score:       f64,
    pub p_value:       f64,
}

/// Count cell-type pair interactions within `radius`.
/// Each pair is counted once (canonical ordering: a ≤ b alphabetically).
pub fn count_interactions(
    coords: &[[f64; 2]],
    barcodes: &[String],
    cell_types: &[String],
    radius: f64,
    group: &str,
) -> Result<Vec<InteractionRecord>> {
    if coords.len() != barcodes.len() {
        bail!(
            "coords length ({}) does not match barcodes length ({})",
            coords.len(),
            barcodes.len()
        );
    }
    if barcodes.len() != cell_types.len() {
        bail!(
            "barcodes length ({}) does not match cell_types length ({})",
            barcodes.len(),
            cell_types.len()
        );
    }

    let edges = radius_graph_dedup(coords, barcodes, radius, group)?;

    // Build barcode → cell_type lookup
    let type_map: HashMap<&str, &str> = barcodes
        .iter()
        .zip(cell_types.iter())
        .map(|(b, t)| (b.as_str(), t.as_str()))
        .collect();

    let mut counts: HashMap<(String, String), usize> = HashMap::new();

    for edge in &edges {
        let ta = type_map[edge.cell_i.as_str()];
        let tb = type_map[edge.cell_j.as_str()];
        // Canonical order
        let key = if ta <= tb {
            (ta.to_string(), tb.to_string())
        } else {
            (tb.to_string(), ta.to_string())
        };
        *counts.entry(key).or_insert(0) += 1;
    }

    let records = counts
        .into_iter()
        .map(|((a, b), count)| InteractionRecord {
            group: group.to_string(),
            cell_type_a: a,
            cell_type_b: b,
            count,
        })
        .collect();

    Ok(records)
}

/// Permutation-based interaction enrichment.
///
/// Builds the spatial graph once, then shuffles cell-type labels `n_perms` times
/// to generate a null distribution.  Returns per-pair statistics including
/// z-score and empirical p-value.
pub fn permute_interactions(
    coords:     &[[f64; 2]],
    barcodes:   &[String],
    cell_types: &[String],
    radius:     f64,
    n_perms:    usize,
    seed:       u64,
    group:      &str,
) -> Result<Vec<InteractionStatsRecord>> {
    if coords.len() != barcodes.len() {
        bail!(
            "coords length ({}) does not match barcodes length ({})",
            coords.len(),
            barcodes.len()
        );
    }
    if barcodes.len() != cell_types.len() {
        bail!(
            "barcodes length ({}) does not match cell_types length ({})",
            barcodes.len(),
            cell_types.len()
        );
    }
    if n_perms == 0 {
        bail!("n_perms must be > 0");
    }

    // Build edge index pairs once (upper triangle)
    let edge_pairs = radius_graph_index_pairs(coords, radius)?;

    // Observed counts
    let observed = count_pairs_by_type(&edge_pairs, cell_types);

    // Null distribution via parallel permutations
    let perm_counts: Vec<HashMap<(String, String), usize>> = (0..n_perms)
        .into_par_iter()
        .map(|perm_idx| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(perm_idx as u64 + 1));
            let mut shuffled = cell_types.to_vec();
            shuffled.shuffle(&mut rng);
            count_pairs_by_type(&edge_pairs, &shuffled)
        })
        .collect();

    // Collect all pair keys across observed and permutations
    let all_pairs: HashSet<(String, String)> = observed
        .keys()
        .cloned()
        .chain(perm_counts.iter().flat_map(|m| m.keys().cloned()))
        .collect();

    let n_perms_f = n_perms as f64;
    let records: Vec<InteractionStatsRecord> = all_pairs
        .into_iter()
        .map(|(a, b)| {
            let obs = *observed.get(&(a.clone(), b.clone())).unwrap_or(&0);
            let null: Vec<f64> = perm_counts
                .iter()
                .map(|m| *m.get(&(a.clone(), b.clone())).unwrap_or(&0) as f64)
                .collect();
            let mean = null.iter().sum::<f64>() / n_perms_f;
            let std = (null.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_perms_f).sqrt();
            let z_score = if std > 1e-14 { (obs as f64 - mean) / std } else { 0.0 };
            // Conservative empirical p-value (+1 in both numerator and denominator)
            let n_exceeding = null.iter().filter(|&&x| x >= obs as f64).count();
            let p_value = (n_exceeding as f64 + 1.0) / (n_perms_f + 1.0);
            InteractionStatsRecord {
                group: group.to_string(),
                cell_type_a: a,
                cell_type_b: b,
                observed: obs,
                expected_mean: mean,
                expected_std: std,
                z_score,
                p_value,
            }
        })
        .collect();

    Ok(records)
}

fn count_pairs_by_type(
    edge_pairs: &[(usize, usize)],
    cell_types: &[String],
) -> HashMap<(String, String), usize> {
    let mut counts: HashMap<(String, String), usize> = HashMap::new();
    for &(i, j) in edge_pairs {
        let ta = &cell_types[i];
        let tb = &cell_types[j];
        let key = if ta <= tb {
            (ta.clone(), tb.clone())
        } else {
            (tb.clone(), ta.clone())
        };
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::count_interactions;

    #[test]
    fn interactions_reject_non_positive_radius() {
        let coords = [[0.0, 0.0], [1.0, 1.0]];
        let barcodes = vec!["a".to_string(), "b".to_string()];
        let cell_types = vec!["t1".to_string(), "t2".to_string()];

        let err = match count_interactions(&coords, &barcodes, &cell_types, -1.0, "g") {
            Ok(_) => panic!("expected invalid radius error"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("radius must be a finite value > 0"));
    }
}
