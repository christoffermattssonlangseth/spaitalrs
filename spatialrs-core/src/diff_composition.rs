use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

// ─── public types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct DiffCompositionRecord {
    pub cell_type: String,
    pub group_a: String,
    pub group_b: String,
    pub n_a: usize,
    pub n_b: usize,
    pub mean_a: f64,
    pub mean_b: f64,
    pub log2fc: f64,
    pub z_score: f64,
    pub p_value: f64,
    pub q_value_bh: f64,
}

// ─── main entry point ─────────────────────────────────────────────────────────

/// Compare neighbourhood composition fractions between two conditions.
///
/// `fractions` is a flat list of `(barcode, cell_type, fraction)` tuples (the
/// rows of a `spatialrs composition` CSV).  `condition_map` maps each barcode
/// to its condition label.  For each cell type, a Mann–Whitney U test is run
/// comparing the per-cell fractions in `group_a` vs `group_b`, and the p-values
/// are Benjamini–Hochberg corrected.
pub fn diff_composition(
    fractions: &[(String, String, f64)],
    condition_map: &HashMap<String, String>,
    group_a: &str,
    group_b: &str,
) -> Result<Vec<DiffCompositionRecord>> {
    if fractions.is_empty() {
        bail!("fractions list is empty");
    }

    // Partition per-cell fractions into the two groups
    let mut type_fracs_a: HashMap<String, Vec<f64>> = HashMap::new();
    let mut type_fracs_b: HashMap<String, Vec<f64>> = HashMap::new();

    for (barcode, cell_type, fraction) in fractions {
        let Some(cond) = condition_map.get(barcode.as_str()) else {
            continue;
        };
        if cond == group_a {
            type_fracs_a
                .entry(cell_type.clone())
                .or_default()
                .push(*fraction);
        } else if cond == group_b {
            type_fracs_b
                .entry(cell_type.clone())
                .or_default()
                .push(*fraction);
        }
    }

    let all_types: HashSet<String> = type_fracs_a
        .keys()
        .chain(type_fracs_b.keys())
        .cloned()
        .collect();

    if all_types.is_empty() {
        bail!(
            "no cells matched group_a='{}' or group_b='{}'",
            group_a,
            group_b
        );
    }

    let mut cell_types: Vec<String> = all_types.into_iter().collect();
    cell_types.sort();

    let empty: Vec<f64> = Vec::new();
    let stats: Vec<(f64, f64, f64, f64, f64, usize, usize)> = cell_types
        .par_iter()
        .map(|ct| {
            let a_vals = type_fracs_a.get(ct).unwrap_or(&empty);
            let b_vals = type_fracs_b.get(ct).unwrap_or(&empty);

            let n_a = a_vals.len();
            let n_b = b_vals.len();

            let mean_a = if n_a > 0 {
                a_vals.iter().sum::<f64>() / n_a as f64
            } else {
                0.0
            };
            let mean_b = if n_b > 0 {
                b_vals.iter().sum::<f64>() / n_b as f64
            } else {
                0.0
            };
            let log2fc = ((mean_a + 1e-9) / (mean_b + 1e-9)).log2();

            let z = if n_a > 0 && n_b > 0 {
                wilcoxon_z(a_vals, b_vals)
            } else {
                0.0
            };
            let p = two_tailed_p(z);

            (mean_a, mean_b, log2fc, z, p, n_a, n_b)
        })
        .collect();

    let p_values: Vec<f64> = stats.iter().map(|s| s.4).collect();
    let q_values = bh_correction(&p_values);

    let records = cell_types
        .into_iter()
        .zip(stats)
        .zip(q_values)
        .map(
            |((ct, (mean_a, mean_b, log2fc, z_score, p_value, n_a, n_b)), q_value_bh)| {
                DiffCompositionRecord {
                    cell_type: ct,
                    group_a: group_a.to_string(),
                    group_b: group_b.to_string(),
                    n_a,
                    n_b,
                    mean_a,
                    mean_b,
                    log2fc,
                    z_score,
                    p_value,
                    q_value_bh,
                }
            },
        )
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
    let variance_u = (n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0;
    let std_u = variance_u.sqrt();

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
    let upper_tail = pdf * poly;
    (2.0 * upper_tail).min(1.0)
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
    for (rank, &orig_idx) in idx.iter().enumerate() {
        q[orig_idx] = (p_values[orig_idx] * n as f64 / (rank + 1) as f64).min(1.0);
    }

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
    use super::diff_composition;
    use std::collections::HashMap;

    #[test]
    fn diff_composition_detects_enriched_type() {
        // Group A: TypeA fraction always 0.9, TypeB always 0.1
        // Group B: TypeA fraction always 0.1, TypeB always 0.9
        let mut cmap: HashMap<String, String> = HashMap::new();
        for i in 0..10 {
            cmap.insert(format!("ca{i}"), "A".to_string());
            cmap.insert(format!("cb{i}"), "B".to_string());
        }

        let mut fracs: Vec<(String, String, f64)> = Vec::new();
        for i in 0..10 {
            fracs.push((format!("ca{i}"), "TypeA".to_string(), 0.9));
            fracs.push((format!("ca{i}"), "TypeB".to_string(), 0.1));
            fracs.push((format!("cb{i}"), "TypeA".to_string(), 0.1));
            fracs.push((format!("cb{i}"), "TypeB".to_string(), 0.9));
        }

        let records = diff_composition(&fracs, &cmap, "A", "B").unwrap();
        assert_eq!(records.len(), 2);

        let type_a_rec = records.iter().find(|r| r.cell_type == "TypeA").unwrap();
        assert!(
            type_a_rec.log2fc > 0.0,
            "TypeA should be enriched in group A"
        );
        assert!(
            type_a_rec.z_score > 1.0,
            "expected positive z for TypeA in group A"
        );
    }

    #[test]
    fn diff_composition_empty_is_error() {
        let cmap: HashMap<String, String> = HashMap::new();
        assert!(diff_composition(&[], &cmap, "A", "B").is_err());
    }
}
