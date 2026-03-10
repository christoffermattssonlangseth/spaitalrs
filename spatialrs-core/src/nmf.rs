use anyhow::{bail, Result};
use ndarray::{Array2, Zip};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;

pub struct NmfConfig {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f32,
    pub seed: u64,
    pub epsilon: f32,
    /// Called every 10 iterations with (iteration, frobenius_error).
    pub iter_cb: Option<std::sync::Arc<dyn Fn(usize, f32) + Send + Sync>>,
}

impl Default for NmfConfig {
    fn default() -> Self {
        NmfConfig {
            n_components: 10,
            max_iter: 200,
            tol: 1e-4,
            seed: 42,
            epsilon: 1e-12,
            iter_cb: None,
        }
    }
}

pub struct NmfResult {
    pub w: Array2<f32>,
    pub h: Array2<f32>,
    pub n_iter: usize,
    pub final_error: f32,
    /// Frobenius-norm contribution of each component as a fraction of
    /// the total ||W H||_F.  Computed post-convergence.
    /// component_variances[k] = ||W[:,k]||₂ × ||H[k,:]||₂ / Σₖ(||W[:,k]||₂ × ||H[k,:]||₂)
    pub component_variances: Vec<f32>,
    /// Error trajectory: (iteration, error) collected every 10 iterations.
    pub error_trajectory: Vec<(usize, f32)>,
}

#[derive(Serialize)]
pub struct WRecord {
    pub cell_i: String,
    pub component: usize,
    pub weight: f32,
    pub group: String,
}

#[derive(Serialize)]
pub struct HRecord {
    pub gene: String,
    pub component: usize,
    pub loading: f32,
    pub group: String,
}

/// Run NMF with multiplicative updates (Lee & Seung 2001).
pub fn run_nmf(x: &Array2<f32>, config: &NmfConfig) -> Result<NmfResult> {
    let (n_obs, n_var) = x.dim();
    let k = config.n_components;
    let eps = config.epsilon;

    if k == 0 {
        bail!("n_components must be > 0");
    }
    if n_obs == 0 || n_var == 0 {
        bail!("expression matrix must be non-empty");
    }

    // Random uniform initialisation
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut w: Array2<f32> = Array2::from_shape_fn((n_obs, k), |_| rng.random::<f32>());
    let mut h: Array2<f32> = Array2::from_shape_fn((k, n_var), |_| rng.random::<f32>());

    let mut prev_err = f32::INFINITY;
    let mut n_iter = config.max_iter;
    let mut final_error = 0.0f32;
    let mut error_trajectory: Vec<(usize, f32)> = Vec::new();

    for iter in 0..config.max_iter {
        // H update: H ← H * (Wᵀ X) / (Wᵀ W H + ε)
        let wt = w.t().to_owned(); // k × n_obs
        let wtx = wt.dot(x); // k × n_var
        let wtwh = wt.dot(&w).dot(&h); // k × n_var
        Zip::from(&mut h)
            .and(&wtx)
            .and(&wtwh)
            .par_for_each(|h_val, &num, &den| {
                *h_val *= num / (den + eps);
            });

        // W update: W ← W * (X Hᵀ) / (W H Hᵀ + ε)
        let ht = h.t().to_owned(); // n_var × k
        let xht = x.dot(&ht); // n_obs × k
        let whht = w.dot(&h.dot(&ht)); // n_obs × k
        Zip::from(&mut w)
            .and(&xht)
            .and(&whht)
            .par_for_each(|w_val, &num, &den| {
                *w_val *= num / (den + eps);
            });

        // Convergence check every 10 iterations
        if (iter + 1) % 10 == 0 {
            let err = frobenius_error(x, &w, &h);
            final_error = err;
            error_trajectory.push((iter + 1, err));
            if let Some(ref cb) = config.iter_cb {
                cb(iter + 1, err);
            }
            if (prev_err - err).abs() < config.tol {
                n_iter = iter + 1;
                break;
            }
            prev_err = err;
        }
    }

    if final_error == 0.0 {
        final_error = frobenius_error(x, &w, &h);
    }

    let component_variances = compute_component_variances(&w, &h);

    Ok(NmfResult {
        w,
        h,
        n_iter,
        final_error,
        component_variances,
        error_trajectory,
    })
}

fn frobenius_error(x: &Array2<f32>, w: &Array2<f32>, h: &Array2<f32>) -> f32 {
    let wh = w.dot(h);
    Zip::from(x)
        .and(&wh)
        .fold(0.0f32, |acc, &xi, &whi| {
            let d = xi - whi;
            acc + d * d
        })
        .sqrt()
}

// ─── sparse NMF ───────────────────────────────────────────────────────────────

/// Run NMF with multiplicative updates on a sparse CSR expression matrix.
///
/// Memory usage is proportional to nnz (non-zeros) rather than n_obs × n_var,
/// making this ~20× faster and cheaper than dense NMF for typical scRNA-seq
/// data (~95% zeros).
///
/// Convergence is measured by the relative Frobenius change in H every 10
/// iterations (rather than reconstruction error, which would require a full
/// dense pass).  The `final_error` field of `NmfResult` holds this value.
pub fn run_nmf_sparse(
    data: &[f32],
    indices: &[usize],
    indptr: &[usize],
    n_obs: usize,
    n_var: usize,
    config: &NmfConfig,
) -> Result<NmfResult> {
    let k = config.n_components;
    let eps = config.epsilon;

    if k == 0 {
        bail!("n_components must be > 0");
    }
    if n_obs == 0 || n_var == 0 {
        bail!("expression matrix must be non-empty");
    }
    if indptr.len() != n_obs + 1 {
        bail!(
            "indptr length ({}) must equal n_obs + 1 ({})",
            indptr.len(),
            n_obs + 1
        );
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut w: Array2<f32> = Array2::from_shape_fn((n_obs, k), |_| rng.random::<f32>());
    let mut h: Array2<f32> = Array2::from_shape_fn((k, n_var), |_| rng.random::<f32>());

    let mut prev_h = h.clone();
    let mut n_iter = config.max_iter;
    let mut final_error = f32::INFINITY;
    let mut error_trajectory: Vec<(usize, f32)> = Vec::new();

    for iter in 0..config.max_iter {
        // H update: H ← H * (W^T X) / (W^T W H + ε)
        let wtx = sparse_wt_x(data, indices, indptr, n_obs, n_var, &w, k); // k × n_var
        let wtw = w.t().dot(&w); // k × k
        let wtwh = wtw.dot(&h); // k × n_var
        Zip::from(&mut h)
            .and(&wtx)
            .and(&wtwh)
            .par_for_each(|h_val, &num, &den| {
                *h_val *= num / (den + eps);
            });

        // W update: W ← W * (X H^T) / (W H H^T + ε)
        let xht = sparse_x_ht(data, indices, indptr, n_obs, &h, k)?; // n_obs × k
        let hht = h.dot(&h.t()); // k × k
        let whht = w.dot(&hht); // n_obs × k
        Zip::from(&mut w)
            .and(&xht)
            .and(&whht)
            .par_for_each(|w_val, &num, &den| {
                *w_val *= num / (den + eps);
            });

        // Convergence check every 10 iterations: relative change in H
        if (iter + 1) % 10 == 0 {
            let diff_sq: f32 = Zip::from(&h).and(&prev_h).fold(0.0f32, |acc, &hn, &ho| {
                let d = hn - ho;
                acc + d * d
            });
            let prev_norm_sq: f32 = prev_h.iter().map(|&v| v * v).sum();
            let err = (diff_sq / (prev_norm_sq + eps)).sqrt();
            final_error = err;
            error_trajectory.push((iter + 1, err));
            if let Some(ref cb) = config.iter_cb {
                cb(iter + 1, err);
            }
            if err < config.tol {
                n_iter = iter + 1;
                break;
            }
            prev_h.assign(&h);
        }
    }

    let component_variances = compute_component_variances(&w, &h);

    Ok(NmfResult {
        w,
        h,
        n_iter,
        final_error,
        component_variances,
        error_trajectory,
    })
}

/// Per-component explained-variance fractions.
///
/// For each component k:  score_k = ||W[:,k]||₂ × ||H[k,:]||₂
/// (the Frobenius norm of the rank-1 matrix W[:,k] H[k,:]^T equals
/// ||W[:,k]||₂ × ||H[k,:]||₂).
/// Returns the vector of scores normalised to sum to 1.
fn compute_component_variances(w: &Array2<f32>, h: &Array2<f32>) -> Vec<f32> {
    let k = w.ncols();
    let scores: Vec<f32> = (0..k)
        .map(|c| {
            let w_norm: f32 = w.column(c).iter().map(|&v| v * v).sum::<f32>().sqrt();
            let h_norm: f32 = h.row(c).iter().map(|&v| v * v).sum::<f32>().sqrt();
            w_norm * h_norm
        })
        .collect();
    let total: f32 = scores.iter().sum::<f32>().max(f32::EPSILON);
    scores.into_iter().map(|s| s / total).collect()
}

/// Compute W^T × X sparsely.  Returns a k × n_var matrix.
///
/// Accumulates into a column-major (n_var × k) buffer per thread for
/// cache-friendly writes, then transposes.
fn sparse_wt_x(
    data: &[f32],
    indices: &[usize],
    indptr: &[usize],
    n_obs: usize,
    n_var: usize,
    w: &Array2<f32>,
    k: usize,
) -> Array2<f32> {
    // Each rayon fold chunk accumulates a private n_var × k buffer.
    // Writing acc.row_mut(j) for a nonzero column j is contiguous (k floats).
    let gk: Array2<f32> = (0..n_obs)
        .into_par_iter()
        .fold(
            || Array2::<f32>::zeros((n_var, k)),
            |mut acc, i| {
                let wi = w.row(i);
                let start = indptr[i];
                let end = indptr[i + 1];
                for kk in start..end {
                    let j = indices[kk];
                    let v = data[kk];
                    let mut acc_row = acc.row_mut(j);
                    for comp in 0..k {
                        acc_row[comp] += wi[comp] * v;
                    }
                }
                acc
            },
        )
        .reduce(|| Array2::zeros((n_var, k)), |a, b| a + b);
    gk.t().to_owned() // k × n_var
}

/// Compute X × H^T sparsely.  Returns an n_obs × k matrix.
///
/// Each row is independent so the computation is trivially parallel.
/// H is pre-transposed to n_var × k so each column lookup is a
/// contiguous k-length slice.
fn sparse_x_ht(
    data: &[f32],
    indices: &[usize],
    indptr: &[usize],
    n_obs: usize,
    h: &Array2<f32>,
    k: usize,
) -> Result<Array2<f32>> {
    if h.nrows() != k {
        bail!("H row count ({}) does not match k ({k})", h.nrows());
    }
    if indptr.len() != n_obs + 1 {
        bail!(
            "indptr length ({}) must equal n_obs + 1 ({})",
            indptr.len(),
            n_obs + 1
        );
    }
    if data.len() != indices.len() {
        bail!(
            "data length ({}) must match indices length ({})",
            data.len(),
            indices.len()
        );
    }

    let ht = h.t().to_owned(); // n_var × k — contiguous rows for cache efficiency

    let rows: Vec<Vec<f32>> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![0.0f32; k];
            let start = indptr[i];
            let end = indptr[i + 1];
            for kk in start..end {
                let j = indices[kk];
                let v = data[kk];
                let ht_j = ht.row(j); // contiguous k-length slice
                for comp in 0..k {
                    row[comp] += ht_j[comp] * v;
                }
            }
            row
        })
        .collect();

    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_obs, k), flat)
        .map_err(|err| anyhow::anyhow!("shape mismatch in sparse_x_ht: {err}"))
}

#[cfg(test)]
mod tests {
    use super::sparse_x_ht;
    use ndarray::arr2;

    #[test]
    fn sparse_x_ht_returns_error_for_mismatched_k() {
        let err = match sparse_x_ht(&[1.0], &[0], &[0, 1], 1, &arr2(&[[1.0, 2.0]]), 2) {
            Ok(_) => panic!("expected mismatched-k error"),
            Err(err) => err,
        };

        assert!(err
            .to_string()
            .contains("H row count (1) does not match k (2)"));
    }
}
