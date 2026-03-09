use anyhow::{bail, Result};
use ndarray::Array2;
use rayon::prelude::*;
use serde::Serialize;

// ─── public types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct MarkerRecord {
    pub niche:      usize,
    pub gene:       String,
    pub mean_niche: f64,
    pub mean_rest:  f64,
    pub log2fc:     f64,
    pub z_score:    f64,
    pub p_value:    f64,
    pub q_value_bh: f64,
}

// ─── main entry point ─────────────────────────────────────────────────────────

/// Identify marker genes for each niche using a one-vs-rest Wilcoxon rank-sum test.
///
/// For each niche k, cells in that niche are compared against all other cells.
/// The Mann-Whitney U statistic is standardised to a z-score under the normal
/// approximation.  Benjamini-Hochberg FDR correction is applied per niche.
pub fn find_niche_markers(
    expression:   &Array2<f32>,
    gene_names:   &[String],
    niche_labels: &[usize],
    n_niches:     usize,
) -> Result<Vec<MarkerRecord>> {
    let (n_cells, n_genes) = expression.dim();

    if n_cells != niche_labels.len() {
        bail!(
            "expression rows ({n_cells}) != niche_labels length ({})",
            niche_labels.len()
        );
    }
    if n_genes != gene_names.len() {
        bail!(
            "expression cols ({n_genes}) != gene_names length ({})",
            gene_names.len()
        );
    }
    if n_niches == 0 {
        bail!("n_niches must be > 0");
    }

    let mut all_records: Vec<MarkerRecord> = Vec::new();

    for niche_k in 0..n_niches {
        let niche_idx: Vec<usize> = (0..n_cells)
            .filter(|&i| niche_labels[i] == niche_k)
            .collect();
        let rest_idx: Vec<usize> = (0..n_cells)
            .filter(|&i| niche_labels[i] != niche_k)
            .collect();

        if niche_idx.is_empty() || rest_idx.is_empty() {
            continue;
        }

        // Per-gene stats in parallel
        let gene_stats: Vec<(f64, f64, f64, f64, f64)> = (0..n_genes)
            .into_par_iter()
            .map(|g| {
                let niche_vals: Vec<f64> = niche_idx
                    .iter()
                    .map(|&i| expression[[i, g]] as f64)
                    .collect();
                let rest_vals: Vec<f64> = rest_idx
                    .iter()
                    .map(|&i| expression[[i, g]] as f64)
                    .collect();

                let mean_niche = niche_vals.iter().sum::<f64>() / niche_vals.len() as f64;
                let mean_rest  = rest_vals.iter().sum::<f64>() / rest_vals.len() as f64;
                // Pseudo-count to avoid log(0)
                let log2fc = ((mean_niche + 1e-9) / (mean_rest + 1e-9)).log2();
                let z      = wilcoxon_z(&niche_vals, &rest_vals);
                let p      = two_tailed_p(z);

                (mean_niche, mean_rest, log2fc, z, p)
            })
            .collect();

        // Benjamini-Hochberg FDR per niche
        let p_values: Vec<f64> = gene_stats.iter().map(|&(_, _, _, _, p)| p).collect();
        let q_values = bh_correction(&p_values);

        for (g, ((mean_niche, mean_rest, log2fc, z_score, p_value), q_value_bh)) in
            gene_stats.iter().zip(q_values.iter()).enumerate()
        {
            all_records.push(MarkerRecord {
                niche: niche_k,
                gene: gene_names[g].clone(),
                mean_niche: *mean_niche,
                mean_rest: *mean_rest,
                log2fc: *log2fc,
                z_score: *z_score,
                p_value: *p_value,
                q_value_bh: *q_value_bh,
            });
        }
    }

    Ok(all_records)
}

// ─── statistics helpers ───────────────────────────────────────────────────────

/// Standardised Mann-Whitney U statistic (signed z-score).
/// Average ranks are used to handle ties correctly.
fn wilcoxon_z(niche_vals: &[f64], rest_vals: &[f64]) -> f64 {
    let n1 = niche_vals.len();
    let n2 = rest_vals.len();
    if n1 == 0 || n2 == 0 {
        return 0.0;
    }

    // Combine with group flag; true = niche group
    let mut combined: Vec<(f64, bool)> = niche_vals
        .iter()
        .map(|&v| (v, true))
        .chain(rest_vals.iter().map(|&v| (v, false)))
        .collect();

    combined.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = combined.len();
    let mut r1 = 0.0f64;
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Extend tie block
        while j < n && combined[j].0 == combined[i].0 {
            j += 1;
        }
        // 1-based average rank for the tie block
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            if combined[k].1 {
                r1 += avg_rank;
            }
        }
        i = j;
    }

    let u1          = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let expected_u  = (n1 * n2) as f64 / 2.0;
    let variance_u  = (n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0;
    let std_u       = variance_u.sqrt();

    if std_u < 1e-14 { 0.0 } else { (u1 - expected_u) / std_u }
}

/// Two-tailed p-value for a standard normal z-score.
/// Uses the Abramowitz and Stegun 26.2.17 polynomial approximation.
fn two_tailed_p(z: f64) -> f64 {
    let z_abs = z.abs();
    if z_abs > 40.0 {
        return 0.0;
    }
    if z_abs < 1e-14 {
        return 1.0;
    }
    let t    = 1.0 / (1.0 + 0.231_641_9 * z_abs);
    let poly = t * (0.319_381_530
        + t * (-0.356_563_782
        + t * (1.781_477_937
        + t * (-1.821_255_978
        + t * 1.330_274_429))));
    let pdf       = (-0.5 * z_abs * z_abs).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let upper_tail = pdf * poly;
    (2.0 * upper_tail).min(1.0)
}

/// Benjamini-Hochberg FDR correction.
/// Returns adjusted p-values (q-values) in the same order as the input.
fn bh_correction(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len();
    if n == 0 {
        return Vec::new();
    }

    // Sort indices by ascending p-value
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        p_values[a]
            .partial_cmp(&p_values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // q_i = p_i * m / rank_i  (rank is 1-based position after sorting)
    let mut q = vec![1.0f64; n];
    for (rank, &orig_idx) in idx.iter().enumerate() {
        q[orig_idx] = (p_values[orig_idx] * n as f64 / (rank + 1) as f64).min(1.0);
    }

    // Step-down: enforce q[rank] ≤ q[rank + 1] (right to left in sorted order)
    let mut min_q = 1.0f64;
    for &orig_idx in idx.iter().rev() {
        q[orig_idx] = q[orig_idx].min(min_q);
        min_q = q[orig_idx];
    }

    q
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{bh_correction, find_niche_markers, two_tailed_p, wilcoxon_z};
    use ndarray::arr2;

    #[test]
    fn wilcoxon_z_detects_upregulation() {
        // Niche values clearly higher than rest → positive z
        // n1=3, n2=3: U1=9, E[U]=4.5, SD≈2.29 → z≈1.96
        let niche = vec![10.0, 11.0, 12.0];
        let rest  = vec![1.0, 2.0, 3.0];
        let z = wilcoxon_z(&niche, &rest);
        assert!(z > 1.5, "expected positive z > 1.5, got {z}");
    }

    #[test]
    fn wilcoxon_z_identical_distributions_is_zero() {
        let v = vec![1.0, 2.0, 3.0];
        let z = wilcoxon_z(&v, &v);
        assert!(z.abs() < 1e-10, "expected z ≈ 0, got {z}");
    }

    #[test]
    fn two_tailed_p_large_z_is_small() {
        let p = two_tailed_p(5.0);
        assert!(p < 1e-4, "expected very small p for z=5, got {p}");
    }

    #[test]
    fn two_tailed_p_zero_z_is_one() {
        let p = two_tailed_p(0.0);
        assert!((p - 1.0).abs() < 1e-6, "expected p≈1 for z=0, got {p}");
    }

    #[test]
    fn bh_correction_orders_correctly() {
        // p-values: [0.01, 0.04, 0.03, 0.20]
        // sorted order: 0.01(rank1), 0.03(rank2), 0.04(rank3), 0.20(rank4)
        // raw q:        0.01*4/1=0.04,  0.03*4/2=0.06, 0.04*4/3≈0.053, 0.20*4/4=0.20
        // step-down:    0.04, 0.053, 0.053, 0.20 → after clamp from right:
        //               idx 0: min(0.04,0.053)=0.04, idx 2: min(0.06,0.053)=0.053, ...
        let p = vec![0.01, 0.04, 0.03, 0.20];
        let q = bh_correction(&p);
        // The smallest raw p should have smallest q
        assert!(q[0] <= q[1]);
        assert!(q[0] <= q[2]);
        assert!(q[3] == 0.20); // 0.20 * 4/4 = 0.20
        for &qi in &q {
            assert!(qi <= 1.0 && qi >= 0.0);
        }
    }

    #[test]
    fn find_niche_markers_detects_signal() {
        // 4 cells (2 per niche), 2 genes
        // Gene 0: niche 0 cells have value 10, niche 1 cells have value 0
        // Gene 1: uniform (no signal)
        let expr = arr2(&[
            [10.0f32, 1.0],
            [10.0f32, 1.0],
            [0.0f32,  1.0],
            [0.0f32,  1.0],
        ]);
        let genes       = vec!["signal".to_string(), "noise".to_string()];
        let niche_labels = vec![0, 0, 1, 1];

        let records = find_niche_markers(&expr, &genes, &niche_labels, 2).unwrap();
        // 2 niches × 2 genes = 4 records
        assert_eq!(records.len(), 4);

        // For niche 0, gene "signal" should have the highest z_score
        let niche0: Vec<&super::MarkerRecord> = records
            .iter()
            .filter(|r| r.niche == 0 && r.gene == "signal")
            .collect();
        assert_eq!(niche0.len(), 1);
        // n1=2, n2=2, perfect separation: U1=4, E[U]=2, SD≈1.29 → z≈1.55
        assert!(niche0[0].z_score > 1.0, "expected positive z > 1.0 for niche 0 signal gene");
        assert!(niche0[0].log2fc > 0.0, "expected positive log2fc for niche 0 signal gene");
    }
}
