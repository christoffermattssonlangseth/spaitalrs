use anyhow::{bail, Result};
use ndarray::{Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;
use std::f64::consts::PI;

// ─── public types ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub enum CovarianceType {
    /// One variance per (component, dimension)
    Diagonal,
    /// One variance per component, shared across dimensions
    Spherical,
}

pub struct GmmConfig {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f64, // convergence tolerance on log-likelihood change
    pub seed: u64,
    pub covariance: CovarianceType,
    pub reg_covar: f64, // regularisation added to variance floor
    /// Called each EM iteration with (iteration, log_likelihood).
    pub iter_cb: Option<std::sync::Arc<dyn Fn(usize, f64) + Send + Sync>>,
}

impl Default for GmmConfig {
    fn default() -> Self {
        GmmConfig {
            n_components: 10,
            max_iter: 200,
            tol: 1e-6,
            seed: 42,
            covariance: CovarianceType::Diagonal,
            reg_covar: 1e-6,
            iter_cb: None,
        }
    }
}

pub struct GmmResult {
    /// Soft assignments, shape N × K
    pub responsibilities: Array2<f64>,
    /// Component means, shape K × D
    pub means: Array2<f64>,
    /// Hard assignment per cell (argmax of responsibilities)
    pub labels: Vec<usize>,
    pub n_iter: usize,
    pub log_likelihood: f64,
    pub bic: f64,
    pub aic: f64,
}

#[derive(Serialize)]
pub struct NicheRecord {
    pub cell_i: String,
    pub niche: usize,
    pub group: String,
}

#[derive(Serialize)]
pub struct NicheProbRecord {
    pub cell_i: String,
    pub component: usize,
    pub probability: f64,
    pub group: String,
}

// ─── main entry point ─────────────────────────────────────────────────────────

pub fn run_gmm(x: &Array2<f64>, config: &GmmConfig) -> Result<GmmResult> {
    let (n, d) = x.dim();
    let k = config.n_components;

    if k == 0 {
        bail!("n_components must be > 0");
    }
    if n < k {
        bail!("fewer observations ({n}) than GMM components ({k})");
    }
    if d == 0 {
        bail!("embedding has zero dimensions");
    }

    let mut rng = StdRng::seed_from_u64(config.seed);

    // K-means++ initialisation for means
    let mut means = kmeans_plus_plus(x, k, &mut rng);

    // Variances stored as K × D regardless of type:
    //   Diagonal  → each element is the per-(k,d) variance
    //   Spherical → all elements in a row are the same scalar
    let init_var = overall_variance(x).max(config.reg_covar);
    let mut variances = Array2::<f64>::from_elem((k, d), init_var);

    // Mixing weights
    let mut weights = vec![1.0 / k as f64; k];

    let mut resp = Array2::<f64>::zeros((n, k));
    let mut log_likelihood = f64::NEG_INFINITY;
    let mut n_iter = config.max_iter;

    for iter in 0..config.max_iter {
        let new_ll;
        (resp, new_ll) = e_step(x, &means, &variances, &weights);

        m_step(
            x,
            &resp,
            &mut means,
            &mut variances,
            &mut weights,
            config.reg_covar,
            &config.covariance,
        );

        if let Some(ref cb) = config.iter_cb {
            cb(iter + 1, new_ll);
        }

        if (new_ll - log_likelihood).abs() < config.tol {
            n_iter = iter + 1;
            log_likelihood = new_ll;
            break;
        }
        log_likelihood = new_ll;
    }

    let labels = argmax_rows(&resp);

    let n_params = match config.covariance {
        CovarianceType::Diagonal => k * (2 * d + 1) - 1,
        CovarianceType::Spherical => k * (d + 2) - 1,
    };
    let bic = n_params as f64 * (n as f64).ln() - 2.0 * log_likelihood;
    let aic = 2.0 * n_params as f64 - 2.0 * log_likelihood;

    Ok(GmmResult {
        responsibilities: resp,
        means,
        labels,
        n_iter,
        log_likelihood,
        bic,
        aic,
    })
}

// ─── EM steps ─────────────────────────────────────────────────────────────────

/// Returns (responsibilities N×K, log-likelihood scalar).
/// Uses log-space arithmetic for numerical stability.
///
/// Processes cells in chunks so that only one small `log_probs` buffer is
/// allocated per chunk rather than per cell, eliminating ~2N heap allocations
/// per iteration.  `resp` is filled in-place using ndarray's parallel chunk
/// iterator, avoiding a second copy pass.
fn e_step(
    x: &Array2<f64>,
    means: &Array2<f64>,
    variances: &Array2<f64>,
    weights: &[f64],
) -> (Array2<f64>, f64) {
    let (n, d) = x.dim();
    let k = weights.len();
    let log_2pi = (2.0 * PI).ln();

    let log_norm: Vec<f64> = (0..k)
        .map(|ki| {
            -0.5 * (0..d)
                .map(|di| log_2pi + variances[[ki, di]].ln())
                .sum::<f64>()
        })
        .collect();
    let log_weights: Vec<f64> = weights.iter().map(|&w| w.ln()).collect();

    const CHUNK: usize = 1024;
    let mut resp = Array2::<f64>::zeros((n, k));

    // Each parallel chunk gets a mutable view of its rows in resp and fills
    // them directly — no per-cell Vec allocation.
    let ll_chunks: Vec<f64> = resp
        .axis_chunks_iter_mut(Axis(0), CHUNK)
        .into_par_iter()
        .enumerate()
        .map(|(ci, mut resp_chunk)| {
            let row_start = ci * CHUNK;
            let chunk_rows = resp_chunk.nrows();
            let mut ll = 0.0f64;
            // One stack-local buffer reused across all cells in the chunk.
            let mut log_probs = vec![0.0f64; k];

            for local_i in 0..chunk_rows {
                let xi = x.row(row_start + local_i);

                for ki in 0..k {
                    let mu_k = means.row(ki);
                    let var_k = variances.row(ki);
                    let mahal: f64 = xi
                        .iter()
                        .zip(mu_k.iter())
                        .zip(var_k.iter())
                        .map(|((&xd, &md), &vd)| {
                            let diff = xd - md;
                            diff * diff / vd
                        })
                        .sum();
                    log_probs[ki] = log_weights[ki] + log_norm[ki] - 0.5 * mahal;
                }

                let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = log_probs.iter().map(|&v| (v - max_lp).exp()).sum();
                let logsumexp = max_lp + sum_exp.ln();
                ll += logsumexp;

                let mut resp_row = resp_chunk.row_mut(local_i);
                for ki in 0..k {
                    resp_row[ki] = (log_probs[ki] - logsumexp).exp();
                }
            }
            ll
        })
        .collect();

    (resp, ll_chunks.iter().sum())
}

fn m_step(
    x: &Array2<f64>,
    resp: &Array2<f64>,
    means: &mut Array2<f64>,
    variances: &mut Array2<f64>,
    weights: &mut Vec<f64>,
    reg_covar: f64,
    cov_type: &CovarianceType,
) {
    let (n, d) = x.dim();
    let k = weights.len();
    let n_f = n as f64;

    // Parallelise over components.  Each component does two row-order passes
    // through x and resp (cache-friendly) and writes to independent outputs.
    let results: Vec<(f64, Vec<f64>, Vec<f64>)> = (0..k)
        .into_par_iter()
        .map(|ki| {
            let mut nk = 0.0f64;
            let mut mean = vec![0.0f64; d];
            let mut var = vec![0.0f64; d];

            // Pass 1: weighted sum for mean (row-order → cache-friendly)
            for i in 0..n {
                let r = resp[[i, ki]];
                if r == 0.0 {
                    continue;
                }
                nk += r;
                let xi = x.row(i);
                for di in 0..d {
                    mean[di] += r * xi[di];
                }
            }
            let nk_safe = nk.max(1e-10);
            for di in 0..d {
                mean[di] /= nk_safe;
            }

            // Pass 2: weighted sum for variance
            for i in 0..n {
                let r = resp[[i, ki]];
                if r == 0.0 {
                    continue;
                }
                let xi = x.row(i);
                for di in 0..d {
                    let diff = xi[di] - mean[di];
                    var[di] += r * diff * diff;
                }
            }
            for di in 0..d {
                var[di] = var[di] / nk_safe + reg_covar;
            }

            (nk, mean, var)
        })
        .collect();

    for (ki, (nk, mean_k, var_k)) in results.into_iter().enumerate() {
        weights[ki] = nk / n_f;
        for di in 0..d {
            means[[ki, di]] = mean_k[di];
        }
        match cov_type {
            CovarianceType::Diagonal => {
                for di in 0..d {
                    variances[[ki, di]] = var_k[di];
                }
            }
            CovarianceType::Spherical => {
                let s = var_k.iter().sum::<f64>() / d as f64;
                variances.row_mut(ki).fill(s);
            }
        }
    }
}

// ─── helpers ──────────────────────────────────────────────────────────────────

fn kmeans_plus_plus(x: &Array2<f64>, k: usize, rng: &mut StdRng) -> Array2<f64> {
    let (n, d) = x.dim();
    let mut centers = Array2::<f64>::zeros((k, d));

    let first = rng.random_range(0..n);
    centers.row_mut(0).assign(&x.row(first));

    for c in 1..k {
        // Squared distance from each point to its nearest current center
        let dists: Vec<f64> = (0..n)
            .map(|i| {
                (0..c)
                    .map(|prev| {
                        x.row(i)
                            .iter()
                            .zip(centers.row(prev).iter())
                            .map(|(&xi, &ci)| (xi - ci) * (xi - ci))
                            .sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let total: f64 = dists.iter().sum();
        let mut threshold = rng.random::<f64>() * total;
        let mut chosen = n - 1;
        for (i, &d2) in dists.iter().enumerate() {
            threshold -= d2;
            if threshold <= 0.0 {
                chosen = i;
                break;
            }
        }
        centers.row_mut(c).assign(&x.row(chosen));
    }

    centers
}

/// Overall mean variance across all dimensions — used to initialise component variances.
fn overall_variance(x: &Array2<f64>) -> f64 {
    let n = x.nrows() as f64;
    let d = x.ncols() as f64;
    let mean = x.mean_axis(Axis(0)).unwrap();
    x.rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .zip(mean.iter())
                .map(|(&xi, &mi)| (xi - mi) * (xi - mi))
                .sum::<f64>()
        })
        .sum::<f64>()
        / (n * d)
}

fn argmax_rows(resp: &Array2<f64>) -> Vec<usize> {
    resp.rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}
