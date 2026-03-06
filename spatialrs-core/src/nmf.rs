use anyhow::{bail, Result};
use ndarray::{Array2, Zip};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::Serialize;

pub struct NmfConfig {
    pub n_components: usize,
    pub max_iter:     usize,
    pub tol:          f32,
    pub seed:         u64,
    pub epsilon:      f32,
}

impl Default for NmfConfig {
    fn default() -> Self {
        NmfConfig {
            n_components: 10,
            max_iter:     200,
            tol:          1e-4,
            seed:         42,
            epsilon:      1e-12,
        }
    }
}

pub struct NmfResult {
    pub w:           Array2<f32>,
    pub h:           Array2<f32>,
    pub n_iter:      usize,
    pub final_error: f32,
}

#[derive(Serialize)]
pub struct WRecord {
    pub cell_i:    String,
    pub component: usize,
    pub weight:    f32,
    pub group:     String,
}

#[derive(Serialize)]
pub struct HRecord {
    pub gene:      String,
    pub component: usize,
    pub loading:   f32,
    pub group:     String,
}

/// Run NMF with multiplicative updates (Lee & Seung 2001).
pub fn run_nmf(x: &Array2<f32>, config: &NmfConfig) -> Result<NmfResult> {
    let (n_obs, n_var) = x.dim();
    let k   = config.n_components;
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

    let mut prev_err    = f32::INFINITY;
    let mut n_iter      = config.max_iter;
    let mut final_error = 0.0f32;

    for iter in 0..config.max_iter {
        // H update: H ← H * (Wᵀ X) / (Wᵀ W H + ε)
        let wt   = w.t().to_owned();        // k × n_obs
        let wtx  = wt.dot(x);              // k × n_var
        let wtwh = wt.dot(&w).dot(&h);     // k × n_var
        Zip::from(&mut h)
            .and(&wtx)
            .and(&wtwh)
            .par_for_each(|h_val, &num, &den| {
                *h_val *= num / (den + eps);
            });

        // W update: W ← W * (X Hᵀ) / (W H Hᵀ + ε)
        let ht   = h.t().to_owned();        // n_var × k
        let xht  = x.dot(&ht);             // n_obs × k
        let whht = w.dot(&h.dot(&ht));      // n_obs × k
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

    Ok(NmfResult { w, h, n_iter, final_error })
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
