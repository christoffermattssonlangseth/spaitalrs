use crate::neighbors::radius_graph_index_pairs;
use anyhow::{bail, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::Serialize;

// ─── public types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct TransitionRecord {
    pub niche_a: usize,
    pub niche_b: usize,
    pub count: usize,
    pub fraction: f64,
    pub group: String,
}

#[derive(Serialize)]
pub struct TransitionStatsRecord {
    pub niche_a: usize,
    pub niche_b: usize,
    pub observed: usize,
    pub expected_mean: f64,
    pub expected_std: f64,
    pub z_score: f64,
    pub p_value: f64,
    pub group: String,
}

// ─── main entry point ─────────────────────────────────────────────────────────

/// Compute the niche spatial co-occurrence (transition) matrix.
///
/// For every pair of cells connected by a spatial edge within `radius`,
/// counts how often each ordered niche pair (a, b) with a ≤ b is adjacent.
/// Returns one record per upper-triangle entry (including diagonal).
///
/// `fraction` = count / total_edges so the matrix sums to 1.
pub fn compute_transitions(
    coords: &[[f64; 2]],
    niche_labels: &[usize],
    radius: f64,
    n_niches: usize,
    group: &str,
) -> Result<Vec<TransitionRecord>> {
    if coords.len() != niche_labels.len() {
        bail!(
            "coords length ({}) does not match niche_labels length ({})",
            coords.len(),
            niche_labels.len()
        );
    }
    if n_niches == 0 {
        bail!("n_niches must be > 0");
    }

    let edge_pairs = radius_graph_index_pairs(coords, radius)?;

    let mut counts = vec![vec![0usize; n_niches]; n_niches];
    for &(i, j) in &edge_pairs {
        let a = niche_labels[i];
        let b = niche_labels[j];
        if a >= n_niches || b >= n_niches {
            bail!(
                "niche label {} out of range [0, {})",
                a.max(b),
                n_niches
            );
        }
        // Store in upper triangle (a ≤ b)
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        counts[lo][hi] += 1;
    }

    let total: usize = counts.iter().flat_map(|r| r.iter()).sum();
    let total_f = total as f64;

    let mut records = Vec::with_capacity(n_niches * (n_niches + 1) / 2);
    for a in 0..n_niches {
        for b in a..n_niches {
            let count = counts[a][b];
            let fraction = if total_f > 0.0 {
                count as f64 / total_f
            } else {
                0.0
            };
            records.push(TransitionRecord {
                niche_a: a,
                niche_b: b,
                count,
                fraction,
                group: group.to_string(),
            });
        }
    }

    Ok(records)
}

// ─── permutation test ─────────────────────────────────────────────────────────

/// Permutation-based enrichment test for the niche transition matrix.
///
/// Builds the spatial graph once, then shuffles niche labels `n_perms` times to
/// generate a null distribution for each niche pair count.  Returns z-scores
/// and empirical p-values (conservative +1 correction).
pub fn permute_transitions(
    coords: &[[f64; 2]],
    niche_labels: &[usize],
    radius: f64,
    n_niches: usize,
    n_perms: usize,
    seed: u64,
    group: &str,
) -> Result<Vec<TransitionStatsRecord>> {
    if coords.len() != niche_labels.len() {
        bail!(
            "coords length ({}) does not match niche_labels length ({})",
            coords.len(),
            niche_labels.len()
        );
    }
    if n_niches == 0 {
        bail!("n_niches must be > 0");
    }
    if n_perms == 0 {
        bail!("n_perms must be > 0");
    }

    let edge_pairs = radius_graph_index_pairs(coords, radius)?;

    // Observed upper-triangle counts
    let observed = count_transitions_flat(&edge_pairs, niche_labels, n_niches);

    // Null distribution via parallel permutations
    let perm_counts: Vec<Vec<usize>> = (0..n_perms)
        .into_par_iter()
        .map(|p| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(p as u64 + 1));
            let mut shuffled = niche_labels.to_vec();
            shuffled.shuffle(&mut rng);
            count_transitions_flat(&edge_pairs, &shuffled, n_niches)
        })
        .collect();

    let n_pairs = n_niches * (n_niches + 1) / 2;
    let n_perms_f = n_perms as f64;

    let records: Vec<TransitionStatsRecord> = (0..n_pairs)
        .map(|idx| {
            // Decode upper-triangle linear index → (a, b)
            let (a, b) = flat_idx_to_pair(idx, n_niches);
            let obs = observed[idx];
            let null: Vec<f64> = perm_counts.iter().map(|v| v[idx] as f64).collect();
            let mean = null.iter().sum::<f64>() / n_perms_f;
            let std =
                (null.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_perms_f).sqrt();
            let z_score = if std > 1e-14 {
                (obs as f64 - mean) / std
            } else {
                0.0
            };
            let n_exceeding = null.iter().filter(|&&x| x >= obs as f64).count();
            let p_value = (n_exceeding as f64 + 1.0) / (n_perms_f + 1.0);
            TransitionStatsRecord {
                niche_a: a,
                niche_b: b,
                observed: obs,
                expected_mean: mean,
                expected_std: std,
                z_score,
                p_value,
                group: group.to_string(),
            }
        })
        .collect();

    Ok(records)
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Count upper-triangle niche pair co-occurrences into a flat Vec of length
/// n_niches*(n_niches+1)/2 in row-major upper-triangle order.
fn count_transitions_flat(
    edge_pairs: &[(usize, usize)],
    niche_labels: &[usize],
    n_niches: usize,
) -> Vec<usize> {
    let n_pairs = n_niches * (n_niches + 1) / 2;
    let mut counts = vec![0usize; n_pairs];
    for &(i, j) in edge_pairs {
        let a = niche_labels[i];
        let b = niche_labels[j];
        if a < n_niches && b < n_niches {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            counts[pair_to_flat_idx(lo, hi, n_niches)] += 1;
        }
    }
    counts
}

fn pair_to_flat_idx(a: usize, b: usize, n: usize) -> usize {
    // a ≤ b; row-major upper triangle.
    // Equivalent to a*n - a*(a-1)/2 + (b-a) but avoids usize underflow when a=0.
    a * (2 * n - a + 1) / 2 + (b - a)
}

fn flat_idx_to_pair(idx: usize, n: usize) -> (usize, usize) {
    let mut remaining = idx;
    for a in 0..n {
        let row_len = n - a;
        if remaining < row_len {
            return (a, a + remaining);
        }
        remaining -= row_len;
    }
    unreachable!()
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{compute_transitions, permute_transitions};

    #[test]
    fn transitions_two_niches_line() {
        // 4 cells in a line: niche 0 0 1 1
        // radius=1.5 → edges (0,1), (1,2), (2,3)
        // pair types: (0,0)=1, (0,1)=1, (1,1)=1  total=3
        let coords = [[0.0f64, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let niches = vec![0usize, 0, 1, 1];
        let records = compute_transitions(&coords, &niches, 1.5, 2, "test").unwrap();

        let find = |a: usize, b: usize| {
            records
                .iter()
                .find(|r| r.niche_a == a && r.niche_b == b)
                .map(|r| r.count)
        };
        assert_eq!(find(0, 0), Some(1));
        assert_eq!(find(0, 1), Some(1));
        assert_eq!(find(1, 1), Some(1));

        let total_frac: f64 = records.iter().map(|r| r.fraction).sum();
        assert!((total_frac - 1.0).abs() < 1e-10, "fractions should sum to 1");
    }

    #[test]
    fn transitions_rejects_bad_n_niches() {
        let coords = [[0.0f64, 0.0], [1.0, 0.0]];
        let niches = vec![0usize, 1];
        assert!(compute_transitions(&coords, &niches, 1.5, 0, "g").is_err());
    }

    #[test]
    fn transitions_out_of_range_niche_is_error() {
        let coords = [[0.0f64, 0.0], [1.0, 0.0]];
        let niches = vec![0usize, 5]; // n_niches=2 but label=5
        assert!(compute_transitions(&coords, &niches, 2.0, 2, "g").is_err());
    }

    #[test]
    fn permute_transitions_segregated_has_high_z_for_same_niche() {
        // 6 cells: first 3 niche 0 (clustered), last 3 niche 1 (clustered)
        // Same-niche edges should be enriched relative to null
        let coords = [
            [0.0f64, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [20.0, 0.0],
            [21.0, 0.0],
            [22.0, 0.0],
        ];
        let niches = vec![0usize, 0, 0, 1, 1, 1];
        let records =
            permute_transitions(&coords, &niches, 1.5, 2, 200, 42, "test").unwrap();
        assert_eq!(records.len(), 3); // (0,0), (0,1), (1,1)

        let same0 = records.iter().find(|r| r.niche_a == 0 && r.niche_b == 0).unwrap();
        let cross = records.iter().find(|r| r.niche_a == 0 && r.niche_b == 1).unwrap();
        assert!(same0.z_score > 0.0, "same-niche 0 edges should be enriched");
        assert!(cross.z_score < 0.0, "cross-niche edges should be depleted");
    }
}
