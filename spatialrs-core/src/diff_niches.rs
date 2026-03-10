use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::Serialize;
use std::f64::consts::PI;

// ─── public types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct DiffNicheRecord {
    pub niche: usize,
    pub group_a: String,
    pub group_b: String,
    pub n_samples_a: usize,
    pub n_samples_b: usize,
    pub mean_fraction_a: f64,
    pub mean_fraction_b: f64,
    pub log2fc: f64,
    pub z_score: f64,
    pub p_value: f64,
    pub q_value_bh: f64,
}

// ─── main entry point ─────────────────────────────────────────────────────────

/// Compare per-niche abundance fractions between two conditions at the *sample* level.
///
/// `fractions_a[s][k]` = fraction of cells in sample s (condition A) that belong to niche k.
/// `fractions_b[s][k]` = same for condition B samples.
///
/// For each niche a Mann–Whitney U test is run across per-sample fractions, with
/// Benjamini–Hochberg FDR correction applied across all niches.
pub fn diff_niches(
    fractions_a: &[Vec<f64>],
    fractions_b: &[Vec<f64>],
    group_a: &str,
    group_b: &str,
    n_niches: usize,
) -> Result<Vec<DiffNicheRecord>> {
    if fractions_a.is_empty() {
        bail!("no samples found for group_a='{group_a}'");
    }
    if fractions_b.is_empty() {
        bail!("no samples found for group_b='{group_b}'");
    }
    if n_niches == 0 {
        bail!("n_niches must be > 0");
    }

    let n_a = fractions_a.len();
    let n_b = fractions_b.len();

    let stats: Vec<(f64, f64, f64, f64, f64)> = (0..n_niches)
        .into_par_iter()
        .map(|k| {
            let a_vals: Vec<f64> = fractions_a
                .iter()
                .map(|fracs| fracs.get(k).copied().unwrap_or(0.0))
                .collect();
            let b_vals: Vec<f64> = fractions_b
                .iter()
                .map(|fracs| fracs.get(k).copied().unwrap_or(0.0))
                .collect();

            let mean_a = a_vals.iter().sum::<f64>() / n_a as f64;
            let mean_b = b_vals.iter().sum::<f64>() / n_b as f64;
            let log2fc = ((mean_a + 1e-9) / (mean_b + 1e-9)).log2();
            let z = wilcoxon_z(&a_vals, &b_vals);
            let p = two_tailed_p(z);

            (mean_a, mean_b, log2fc, z, p)
        })
        .collect();

    let p_values: Vec<f64> = stats.iter().map(|s| s.4).collect();
    let q_values = bh_correction(&p_values);

    let records = (0..n_niches)
        .zip(stats)
        .zip(q_values)
        .map(|((niche, (mean_a, mean_b, log2fc, z_score, p_value)), q_value_bh)| {
            DiffNicheRecord {
                niche,
                group_a: group_a.to_string(),
                group_b: group_b.to_string(),
                n_samples_a: n_a,
                n_samples_b: n_b,
                mean_fraction_a: mean_a,
                mean_fraction_b: mean_b,
                log2fc,
                z_score,
                p_value,
                q_value_bh,
            }
        })
        .collect();

    Ok(records)
}

// ─── statistics helpers ───────────────────────────────────────────────────────

fn wilcoxon_z(a_vals: &[f64], b_vals: &[f64]) -> f64 {
    let n1 = a_vals.len();
    let n2 = b_vals.len();
    if n1 == 0 || n2 == 0 {
        return 0.0;
    }
    let mut combined: Vec<(f64, bool)> = a_vals
        .iter()
        .map(|&v| (v, true))
        .chain(b_vals.iter().map(|&v| (v, false)))
        .collect();
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n = combined.len();
    let mut r1 = 0.0f64;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && combined[j].0 == combined[i].0 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            if combined[k].1 {
                r1 += avg_rank;
            }
        }
        i = j;
    }
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let expected_u = (n1 * n2) as f64 / 2.0;
    let std_u = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();
    if std_u < 1e-14 {
        0.0
    } else {
        (u1 - expected_u) / std_u
    }
}

fn two_tailed_p(z: f64) -> f64 {
    let z_abs = z.abs();
    if z_abs > 40.0 {
        return 0.0;
    }
    if z_abs < 1e-14 {
        return 1.0;
    }
    let t = 1.0 / (1.0 + 0.231_641_9 * z_abs);
    let poly = t * (0.319_381_530
        + t * (-0.356_563_782
            + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-0.5 * z_abs * z_abs).exp() / (2.0 * PI).sqrt();
    (2.0 * pdf * poly).min(1.0)
}

fn bh_correction(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        p_values[a]
            .partial_cmp(&p_values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut q = vec![1.0f64; n];
    for (rank, &orig) in idx.iter().enumerate() {
        q[orig] = (p_values[orig] * n as f64 / (rank + 1) as f64).min(1.0);
    }
    let mut min_q = 1.0f64;
    for &orig in idx.iter().rev() {
        q[orig] = q[orig].min(min_q);
        min_q = q[orig];
    }
    q
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::diff_niches;

    #[test]
    fn diff_niches_detects_enriched_niche() {
        // 5 samples per group, 2 niches
        // Group A: niche 0 fraction ≈ 0.9, niche 1 ≈ 0.1
        // Group B: niche 0 fraction ≈ 0.1, niche 1 ≈ 0.9
        let fracs_a: Vec<Vec<f64>> = (0..5).map(|_| vec![0.9, 0.1]).collect();
        let fracs_b: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1, 0.9]).collect();

        let records = diff_niches(&fracs_a, &fracs_b, "A", "B", 2).unwrap();
        assert_eq!(records.len(), 2);

        let niche0 = records.iter().find(|r| r.niche == 0).unwrap();
        assert!(niche0.log2fc > 0.0, "niche 0 enriched in A");
        assert!(niche0.z_score > 1.0, "expected positive z for niche 0 in A");
    }

    #[test]
    fn diff_niches_empty_group_is_error() {
        assert!(diff_niches(&[], &[vec![0.5, 0.5]], "A", "B", 2).is_err());
        assert!(diff_niches(&[vec![0.5, 0.5]], &[], "A", "B", 2).is_err());
    }
}
