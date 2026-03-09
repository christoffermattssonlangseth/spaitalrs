use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{MultiProgress, ParallelProgressIterator, ProgressBar, ProgressStyle};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use spatialrs_core::{
    aggregation::{aggregate_neighbors, AggregationRecord, GraphMode, WeightingMode},
    autocorr::{compute_morans_i, MoranRecord},
    composition::compute_composition,
    gmm::{run_gmm, CovarianceType, GmmConfig, NicheProbRecord, NicheRecord},
    interactions::{count_interactions, permute_interactions, InteractionStatsRecord},
    markers::{find_niche_markers, MarkerRecord},
    neighbors::{knn_graph, radius_graph, EdgeRecord},
    nmf::{run_nmf, HRecord, NmfConfig, WRecord},
};
use spatialrs_io::{read_h5ad, AnnData};
use std::collections::HashMap;
use std::path::PathBuf;

// ─── CLI definition ───────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "spatialrs", about = "Spatial transcriptomics graph tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, ValueEnum)]
enum WeightingArg {
    Uniform,
    Gaussian,
    Exponential,
    InverseDistance,
}

#[derive(Clone, ValueEnum)]
enum CovarianceArg {
    Diagonal,
    Spherical,
}

#[derive(Deserialize)]
struct WInputRecord {
    cell_i: String,
    component: usize,
    weight: f32,
    group: String,
}

#[derive(Deserialize)]
struct AggInputRecord {
    cell_i: String,
    dim:    usize,
    value:  f64,
    group:  String,
}

#[derive(Deserialize)]
struct NicheInputRecord {
    cell_i: String,
    niche:  usize,
    #[allow(dead_code)]
    group:  String,
}

#[derive(Serialize)]
struct ModelStatsRecord {
    k:               usize,
    log_likelihood:  f64,
    bic:             f64,
    aic:             f64,
    n_iter:          usize,
    covariance_type: String,
    group:           String,
}

#[derive(Subcommand)]
enum Command {
    /// Compute radius-based neighbour graph
    Radius {
        input: PathBuf,
        #[arg(long)]
        radius: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compute k-nearest-neighbour graph
    Knn {
        input: PathBuf,
        #[arg(long)]
        k: usize,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Count cell-type pair interactions within a radius
    Interactions {
        input: PathBuf,
        #[arg(long)]
        cell_type: String,
        #[arg(long)]
        radius: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        /// Output CSV for raw interaction counts (group, cell_type_a, cell_type_b, count)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Number of permutations for null model (requires --output-stats)
        #[arg(long, default_value_t = 1000)]
        n_permutations: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Output CSV for enrichment statistics (observed, expected_mean, z_score, p_value)
        #[arg(long)]
        output_stats: Option<PathBuf>,
    },
    /// Compute per-cell neighbourhood composition
    Composition {
        input: PathBuf,
        #[arg(long)]
        cell_type: String,
        #[arg(long)]
        radius: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Factorize the gene expression matrix X using NMF
    Nmf {
        input: PathBuf,
        #[arg(long, default_value_t = 10)]
        n_components: usize,
        #[arg(long, default_value_t = 200)]
        max_iter: usize,
        #[arg(long, default_value_t = 1e-4)]
        tol: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Optional obs column to partition cells before factorizing
        #[arg(long)]
        groupby: Option<String>,
        /// Output path for W (cell factor) matrix CSV
        #[arg(long)]
        output_w: Option<PathBuf>,
        /// Output path for H (gene loading) matrix CSV
        #[arg(long)]
        output_h: Option<PathBuf>,
    },
    /// Fit a Gaussian Mixture Model to identify spatial niches / compartments
    Gmm {
        input: PathBuf,
        /// obsm key to use as input embedding (mutually exclusive with --nmf-w / --agg)
        #[arg(long, conflicts_with_all = ["nmf_w", "agg"])]
        embedding: Option<String>,
        /// NMF W factors CSV (mutually exclusive with --embedding / --agg)
        #[arg(long, conflicts_with_all = ["embedding", "agg"])]
        nmf_w: Option<PathBuf>,
        /// Aggregation CSV from `spatialrs aggregate` (mutually exclusive with --embedding / --nmf-w)
        #[arg(long, conflicts_with_all = ["embedding", "nmf_w"])]
        agg: Option<PathBuf>,
        /// Number of mixture components (niches)
        #[arg(long, short = 'k')]
        k: usize,
        #[arg(long, default_value = "diagonal")]
        covariance: CovarianceArg,
        #[arg(long, default_value_t = 200)]
        max_iter: usize,
        #[arg(long, default_value_t = 1e-6)]
        tol: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Regularisation added to variance to prevent singularity
        #[arg(long, default_value_t = 1e-6)]
        reg_covar: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        /// Output CSV for hard niche assignments (cell_i, niche, group)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Output CSV for soft probabilities (cell_i, component, probability, group)
        #[arg(long)]
        output_probs: Option<PathBuf>,
        /// Output CSV for model fit statistics (k, log_likelihood, bic, aic, n_iter, covariance_type)
        #[arg(long)]
        output_model_stats: Option<PathBuf>,
    },
    /// Compute global Moran's I spatial autocorrelation for NMF components or embedding dims
    Morans {
        input: PathBuf,
        /// obsm key for the embedding (mutually exclusive with --nmf-w)
        #[arg(long, conflicts_with = "nmf_w")]
        embedding: Option<String>,
        /// NMF W factors CSV from `spatialrs nmf` (mutually exclusive with --embedding)
        #[arg(long, conflicts_with = "embedding")]
        nmf_w: Option<PathBuf>,
        /// Radius for spatial neighbour search
        #[arg(long)]
        radius: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Identify niche marker genes using one-vs-rest Wilcoxon rank-sum tests
    Markers {
        input: PathBuf,
        /// Niche assignments CSV produced by `spatialrs gmm`
        #[arg(long)]
        niche_csv: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Aggregate neighbour embeddings using distance-weighted averaging
    Aggregate {
        input: PathBuf,
        /// obsm key for the embedding to aggregate (mutually exclusive with --nmf-w)
        #[arg(long, conflicts_with = "nmf_w")]
        embedding: Option<String>,
        /// Path to a W factors CSV produced by `spatialrs nmf` (mutually exclusive with --embedding)
        #[arg(long, conflicts_with = "embedding")]
        nmf_w: Option<PathBuf>,
        /// Radius for neighbour search (mutually exclusive with --k)
        #[arg(long)]
        radius: Option<f64>,
        /// Number of nearest neighbours (mutually exclusive with --radius)
        #[arg(long)]
        k: Option<usize>,
        #[arg(long, default_value = "uniform")]
        weighting: WeightingArg,
        /// Gaussian sigma (required when --weighting gaussian)
        #[arg(long)]
        sigma: Option<f64>,
        /// Exponential decay (required when --weighting exponential)
        #[arg(long)]
        decay: Option<f64>,
        /// Inverse-distance epsilon floor (default 1e-9)
        #[arg(long)]
        epsilon: Option<f64>,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

// ─── entry point ──────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Radius {
            input,
            radius,
            groupby,
            output,
        } => {
            let adata = read_h5ad(&input, &[&groupby], &[], false)?;
            let groups = partition_by_group(&adata, &groupby)?;

            let pb = group_bar(groups.len(), "radius");
            let records: Vec<EdgeRecord> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    radius_graph(&coords, &barcodes, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::Knn {
            input,
            k,
            groupby,
            output,
        } => {
            let adata = read_h5ad(&input, &[&groupby], &[], false)?;
            let groups = partition_by_group(&adata, &groupby)?;

            let pb = group_bar(groups.len(), "knn");
            let records: Vec<EdgeRecord> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    knn_graph(&coords, &barcodes, k, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::Interactions {
            input,
            cell_type,
            radius,
            groupby,
            output,
            n_permutations,
            seed,
            output_stats,
        } => {
            let adata = read_h5ad(&input, &[&groupby, &cell_type], &[], false)?;
            let groups = partition_by_group(&adata, &groupby)?;

            let pb = group_bar(groups.len(), "interactions");
            let records: Vec<_> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    let types = extract_strings_subset(&adata, &cell_type, indices);
                    count_interactions(&coords, &barcodes, &types, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;

            if let Some(ref stats_path) = output_stats {
                let pb2 = group_bar(groups.len(), "permutations");
                let stats: Vec<InteractionStatsRecord> = groups
                    .par_iter()
                    .progress_with(pb2)
                    .map(|(label, indices)| {
                        let coords = extract_coords_subset(&adata, indices);
                        let barcodes = extract_barcodes_subset(&adata, indices);
                        let types = extract_strings_subset(&adata, &cell_type, indices);
                        permute_interactions(
                            &coords, &barcodes, &types, radius,
                            n_permutations, seed, label,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                write_csv(&stats, Some(stats_path.as_path()))?;
            }
        }

        Command::Composition {
            input,
            cell_type,
            radius,
            groupby,
            output,
        } => {
            let adata = read_h5ad(&input, &[&groupby, &cell_type], &[], false)?;
            let groups = partition_by_group(&adata, &groupby)?;

            let pb = group_bar(groups.len(), "composition");
            let records: Vec<_> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    let types = extract_strings_subset(&adata, &cell_type, indices);
                    compute_composition(&coords, &barcodes, &types, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::Nmf {
            input,
            n_components,
            max_iter,
            tol,
            seed,
            groupby,
            output_w,
            output_h,
        } => {
            let obs_cols: Vec<&str> = groupby.as_deref().into_iter().collect();
            let adata = read_h5ad(&input, &obs_cols, &[], true)?;

            let x_full = adata
                .expression
                .as_ref()
                .context("expression matrix not loaded")?;

            // Run NMF once on ALL cells pooled — factors are then comparable across samples.
            // Group label is attached to output records from obs afterwards.
            let n_cells = adata.obs_names.len();
            let multi = MultiProgress::new();
            let pb = iter_bar(&multi, max_iter, &format!("NMF ({n_cells} cells)"));
            let pb2 = pb.clone();
            let config = NmfConfig {
                n_components,
                max_iter,
                tol,
                seed,
                epsilon: 1e-12,
                iter_cb: Some(Arc::new(move |iter: usize, err: f32| {
                    pb2.set_position(iter as u64);
                    pb2.set_message(format!("err={err:.2}"));
                })),
            };
            let result = run_nmf(x_full, &config)?;
            pb.finish_with_message(format!("err={:.2}", result.final_error));

            let group_col_opt = groupby.as_deref().and_then(|col| adata.obs.get(col));

            let mut w_records: Vec<WRecord> = Vec::new();
            let mut h_records: Vec<HRecord> = Vec::new();

            for (i, barcode) in adata.obs_names.iter().enumerate() {
                let group = group_col_opt
                    .and_then(|col| col.get(i))
                    .map(|s| s.as_str())
                    .unwrap_or("all")
                    .to_string();
                for comp in 0..n_components {
                    w_records.push(WRecord {
                        cell_i: barcode.clone(),
                        component: comp,
                        weight: result.w[[i, comp]],
                        group: group.clone(),
                    });
                }
            }

            for (gene_idx, gene) in adata.var_names.iter().enumerate() {
                for comp in 0..n_components {
                    h_records.push(HRecord {
                        gene: gene.clone(),
                        component: comp,
                        loading: result.h[[comp, gene_idx]],
                        group: "all".to_string(),
                    });
                }
            }

            write_csv(&w_records, output_w.as_deref())?;

            if output_h.is_some() {
                write_csv(&h_records, output_h.as_deref())?;
            }
        }

        Command::Gmm {
            input,
            embedding,
            nmf_w,
            agg,
            k,
            covariance,
            max_iter,
            tol,
            seed,
            reg_covar,
            groupby,
            output,
            output_probs,
            output_model_stats,
        } => {
            let obsm_keys: Vec<&str> = embedding.as_deref().into_iter().collect();
            let adata = read_h5ad(&input, &[&groupby], &obsm_keys, false)?;

            // Build full embedding matrix (N × D) from one of three sources
            let emb_owned: Array2<f64>;
            let emb_full: &Array2<f64> = match (&embedding, &nmf_w, &agg) {
                (Some(key), None, None) => {
                    adata.embeddings
                        .get(key)
                        .ok_or_else(|| anyhow::anyhow!("embedding '{key}' not loaded"))?
                }
                (None, Some(path), None) => {
                    emb_owned = read_nmf_w_embedding(path, &adata, &groupby)?;
                    &emb_owned
                }
                (None, None, Some(path)) => {
                    emb_owned = read_agg_embedding(path, &adata, &groupby)?;
                    &emb_owned
                }
                (None, None, None) => bail!("one of --embedding, --nmf-w, or --agg is required"),
                _ => unreachable!("clap conflicts_with_all prevents this"),
            };

            let cov_type = match covariance {
                CovarianceArg::Diagonal  => CovarianceType::Diagonal,
                CovarianceArg::Spherical => CovarianceType::Spherical,
            };

            // Run GMM once on ALL cells pooled — this gives globally consistent niche IDs
            // across samples. The group label is attached to output records afterwards.
            let n_cells = emb_full.nrows();
            let multi = MultiProgress::new();
            let pb = iter_bar(&multi, max_iter, &format!("GMM ({n_cells} cells)"));
            let pb2 = pb.clone();
            let config = GmmConfig {
                n_components: k,
                max_iter,
                tol,
                seed,
                covariance: cov_type,
                reg_covar,
                iter_cb: Some(Arc::new(move |iter: usize, ll: f64| {
                    pb2.set_position(iter as u64);
                    pb2.set_message(format!("ll={ll:.1}"));
                })),
            };
            let result = run_gmm(emb_full, &config)?;
            pb.finish_with_message(format!("ll={:.1}", result.log_likelihood));

            let group_col = adata.obs.get(&groupby)
                .ok_or_else(|| anyhow::anyhow!("groupby column '{groupby}' not found in obs"))?;

            let mut niche_records: Vec<NicheRecord>     = Vec::new();
            let mut prob_records:  Vec<NicheProbRecord> = Vec::new();

            for (i, barcode) in adata.obs_names.iter().enumerate() {
                let group = group_col.get(i).map(|s| s.as_str()).unwrap_or("").to_string();
                niche_records.push(NicheRecord {
                    cell_i: barcode.clone(),
                    niche:  result.labels[i],
                    group,
                });
                if output_probs.is_some() {
                    let group2 = group_col.get(i).map(|s| s.as_str()).unwrap_or("").to_string();
                    for comp in 0..k {
                        prob_records.push(NicheProbRecord {
                            cell_i:      barcode.clone(),
                            component:   comp,
                            probability: result.responsibilities[[i, comp]],
                            group:       group2.clone(),
                        });
                    }
                }
            }

            write_csv(&niche_records, output.as_deref())?;
            if output_probs.is_some() {
                write_csv(&prob_records, output_probs.as_deref())?;
            }
            if let Some(ref stats_path) = output_model_stats {
                let cov_str = match covariance {
                    CovarianceArg::Diagonal  => "diagonal",
                    CovarianceArg::Spherical => "spherical",
                };
                let stats = vec![ModelStatsRecord {
                    k,
                    log_likelihood: result.log_likelihood,
                    bic:            result.bic,
                    aic:            result.aic,
                    n_iter:         result.n_iter,
                    covariance_type: cov_str.to_string(),
                    group:          "all".to_string(),
                }];
                write_csv(&stats, Some(stats_path.as_path()))?;
            }
        }

        Command::Morans {
            input,
            embedding,
            nmf_w,
            radius,
            groupby,
            output,
        } => {
            let obsm_keys: Vec<&str> = embedding.as_deref().into_iter().collect();
            let adata = read_h5ad(&input, &[&groupby], &obsm_keys, false)?;

            let emb_owned: Array2<f64>;
            let emb_full: &Array2<f64> = match (&embedding, &nmf_w) {
                (Some(key), None) => adata
                    .embeddings
                    .get(key)
                    .ok_or_else(|| anyhow::anyhow!("embedding '{key}' not loaded"))?,
                (None, Some(path)) => {
                    emb_owned = read_nmf_w_embedding(path, &adata, &groupby)?;
                    &emb_owned
                }
                (None, None) => bail!("one of --embedding or --nmf-w is required"),
                _ => unreachable!("clap conflicts_with prevents this"),
            };

            let feature_names: Vec<String> = match (&embedding, &nmf_w) {
                (Some(_), _) => (0..emb_full.ncols()).map(|i| format!("dim_{i}")).collect(),
                _            => (0..emb_full.ncols()).map(|i| format!("component_{i}")).collect(),
            };

            let groups = partition_by_group(&adata, &groupby)?;

            let pb = group_bar(groups.len(), "morans");
            let records: Vec<MoranRecord> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords  = extract_coords_subset(&adata, indices);
                    let emb_sub = emb_full.select(ndarray::Axis(0), indices);
                    compute_morans_i(&coords, &emb_sub, &feature_names, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::Markers {
            input,
            niche_csv,
            output,
        } => {
            let adata = read_h5ad(&input, &[], &[], true)?;

            // Read niche assignments CSV
            let mut rdr = csv::Reader::from_path(&niche_csv)
                .with_context(|| format!("cannot open {:?}", niche_csv))?;
            let mut niche_map: HashMap<String, usize> = HashMap::new();
            let mut max_niche = 0usize;
            for record in rdr.deserialize() {
                let row: NicheInputRecord =
                    record.with_context(|| format!("reading {:?}", niche_csv))?;
                max_niche = max_niche.max(row.niche);
                niche_map.insert(row.cell_i, row.niche);
            }
            let n_niches = max_niche + 1;

            // Build per-cell niche label vector in obs order
            let niche_labels: Vec<usize> = adata
                .obs_names
                .iter()
                .map(|b| {
                    niche_map
                        .get(b)
                        .copied()
                        .with_context(|| format!("barcode '{b}' not found in niche CSV"))
                })
                .collect::<Result<Vec<_>>>()?;

            let expression = adata
                .expression
                .context("expression matrix not loaded")?;

            let n_cells = expression.nrows();
            let n_genes = expression.ncols();
            let pb = group_bar(n_niches, "markers");
            let _ = (n_cells, n_genes, pb); // progress shown inside find_niche_markers via Rayon

            let records: Vec<MarkerRecord> = find_niche_markers(
                &expression,
                &adata.var_names,
                &niche_labels,
                n_niches,
            )?;

            write_csv(&records, output.as_deref())?;
        }

        Command::Aggregate {
            input,
            embedding,
            nmf_w,
            radius,
            k,
            weighting,
            sigma,
            decay,
            epsilon,
            groupby,
            output,
        } => {
            let obsm_keys: Vec<&str> = embedding.as_deref().into_iter().collect();
            let adata = read_h5ad(&input, &[&groupby], &obsm_keys, false)?;

            let graph = match (radius, k) {
                (Some(r), None) => GraphMode::Radius(r),
                (None, Some(k)) => GraphMode::Knn(k),
                (Some(_), Some(_)) => bail!("specify --radius or --k, not both"),
                (None, None) => bail!("one of --radius or --k is required"),
            };

            let weight_mode = build_weighting_mode(&weighting, sigma, decay, epsilon)?;

            let groups = partition_by_group(&adata, &groupby)?;

            let records = match (embedding.as_deref(), nmf_w.as_deref()) {
                (Some(key), None) => {
                    let emb_full = adata
                        .embeddings
                        .get(key)
                        .with_context(|| format!("embedding '{key}' not loaded"))?;
                    aggregate_group_embedding(&adata, &groups, emb_full, &graph, &weight_mode)?
                }
                (None, Some(path)) => {
                    let emb_full = read_nmf_w_embedding(path, &adata, &groupby)?;
                    aggregate_group_embedding(&adata, &groups, &emb_full, &graph, &weight_mode)?
                }
                (Some(_), Some(_)) => bail!("specify --embedding or --nmf-w, not both"),
                (None, None) => bail!("one of --embedding or --nmf-w is required"),
            };

            write_csv(&records, output.as_deref())?;
        }
    }

    Ok(())
}

// ─── shared helpers ───────────────────────────────────────────────────────────

fn build_weighting_mode(
    arg: &WeightingArg,
    sigma: Option<f64>,
    decay: Option<f64>,
    epsilon: Option<f64>,
) -> Result<WeightingMode> {
    Ok(match arg {
        WeightingArg::Uniform => WeightingMode::Uniform,
        WeightingArg::Gaussian => {
            let s = sigma.context("--sigma is required for gaussian weighting")?;
            WeightingMode::Gaussian { sigma: s }
        }
        WeightingArg::Exponential => {
            let d = decay.context("--decay is required for exponential weighting")?;
            WeightingMode::Exponential { decay: d }
        }
        WeightingArg::InverseDistance => WeightingMode::InverseDistance {
            epsilon: epsilon.unwrap_or(1e-9),
        },
    })
}

/// Group cell indices by the value of the specified obs column.
fn partition_by_group(adata: &AnnData, groupby: &str) -> Result<Vec<(String, Vec<usize>)>> {
    let col = adata
        .obs
        .get(groupby)
        .with_context(|| format!("obs column '{groupby}' not loaded"))?;

    let mut map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    for (i, label) in col.iter().enumerate() {
        map.entry(label.clone()).or_default().push(i);
    }

    let mut groups: Vec<(String, Vec<usize>)> = map.into_iter().collect();
    groups.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(groups)
}

fn extract_coords_subset(adata: &AnnData, indices: &[usize]) -> Vec<[f64; 2]> {
    indices
        .iter()
        .map(|&i| [adata.coordinates[[i, 0]], adata.coordinates[[i, 1]]])
        .collect()
}

fn extract_barcodes_subset(adata: &AnnData, indices: &[usize]) -> Vec<String> {
    indices
        .iter()
        .map(|&i| adata.obs_names[i].clone())
        .collect()
}

fn extract_strings_subset(adata: &AnnData, col: &str, indices: &[usize]) -> Vec<String> {
    let values = &adata.obs[col];
    indices.iter().map(|&i| values[i].clone()).collect()
}

fn aggregate_group_embedding(
    adata: &AnnData,
    groups: &[(String, Vec<usize>)],
    emb_full: &Array2<f64>,
    graph: &GraphMode,
    weight_mode: &WeightingMode,
) -> Result<Vec<AggregationRecord>> {
    let pb = group_bar(groups.len(), "aggregate");
    groups
        .par_iter()
        .progress_with(pb)
        .map(|(label, indices)| {
            let coords = extract_coords_subset(adata, indices);
            let barcodes = extract_barcodes_subset(adata, indices);
            let emb_sub = emb_full.select(ndarray::Axis(0), indices);
            aggregate_neighbors(&coords, &barcodes, &emb_sub, graph, weight_mode, label)
        })
        .collect::<Result<Vec<_>>>()
        .map(|group_records| group_records.into_iter().flatten().collect())
}

fn read_nmf_w_embedding(
    path: &std::path::Path,
    adata: &AnnData,
    groupby: &str,
) -> Result<Array2<f64>> {
    let groups = adata
        .obs
        .get(groupby)
        .with_context(|| format!("obs column '{groupby}' not loaded"))?;

    let mut rdr =
        csv::Reader::from_path(path).with_context(|| format!("cannot open {:?}", path))?;

    let mut rows = Vec::<WInputRecord>::new();
    let mut max_component = None::<usize>;
    for record in rdr.deserialize() {
        let row: WInputRecord = record.with_context(|| format!("reading {:?}", path))?;
        max_component = Some(match max_component {
            Some(current) => current.max(row.component),
            None => row.component,
        });
        rows.push(row);
    }

    let n_components = max_component
        .map(|component| component + 1)
        .context("NMF W CSV is empty")?;

    let mut values_by_key: HashMap<(String, String), Vec<Option<f64>>> = HashMap::new();
    for row in rows {
        let key = (row.group, row.cell_i);
        let entry = values_by_key
            .entry(key.clone())
            .or_insert_with(|| vec![None; n_components]);
        if entry.len() != n_components {
            bail!("inconsistent component count while reading {:?}", path);
        }
        if entry[row.component].replace(row.weight as f64).is_some() {
            bail!(
                "duplicate W value for cell '{}' group '{}' component {}",
                key.1,
                key.0,
                row.component
            );
        }
    }

    let mut embedding = Array2::<f64>::zeros((adata.obs_names.len(), n_components));
    for (row_idx, barcode) in adata.obs_names.iter().enumerate() {
        let expected_group = &groups[row_idx];

        let component_values = values_by_key
            .get(&(expected_group.clone(), barcode.clone()))
            .or_else(|| values_by_key.get(&("all".to_string(), barcode.clone())))
            .with_context(|| {
                format!(
                    "missing W factors for cell '{}' in group '{}' from {:?}",
                    barcode, expected_group, path
                )
            })?;

        for (component, value) in component_values.iter().enumerate() {
            embedding[[row_idx, component]] = value.with_context(|| {
                format!(
                    "missing W factor for cell '{}' group '{}' component {}",
                    barcode, expected_group, component
                )
            })?;
        }
    }

    Ok(embedding)
}

/// Load an aggregation CSV (output of `spatialrs aggregate`) into a dense N × D matrix.
/// Rows are ordered by `adata.obs_names`; the group for each cell is looked up from `groupby`.
fn read_agg_embedding(
    path:    &std::path::Path,
    adata:   &AnnData,
    groupby: &str,
) -> Result<Array2<f64>> {
    let groups = adata
        .obs
        .get(groupby)
        .with_context(|| format!("obs column '{groupby}' not loaded"))?;

    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("cannot open {:?}", path))?;

    // key: (group, cell_i)  value: per-dim values
    let mut map: HashMap<(String, String), Vec<Option<f64>>> = HashMap::new();
    let mut max_dim = None::<usize>;

    for record in rdr.deserialize() {
        let row: AggInputRecord = record.with_context(|| format!("reading {:?}", path))?;
        max_dim = Some(match max_dim {
            Some(m) => m.max(row.dim),
            None    => row.dim,
        });
        let entry = map
            .entry((row.group, row.cell_i))
            .or_default();
        // Extend the vec if needed
        if entry.len() <= row.dim {
            entry.resize(row.dim + 1, None);
        }
        if entry[row.dim].replace(row.value).is_some() {
            bail!("duplicate dim {} in {:?}", row.dim, path);
        }
    }

    let n_dims = max_dim
        .map(|m| m + 1)
        .context("aggregation CSV is empty")?;

    let mut embedding = Array2::<f64>::zeros((adata.obs_names.len(), n_dims));
    for (row_idx, barcode) in adata.obs_names.iter().enumerate() {
        let cell_group = &groups[row_idx];
        let values = map
            .get(&(cell_group.clone(), barcode.clone()))
            .with_context(|| format!(
                "missing aggregation row for cell '{}' group '{}' in {:?}",
                barcode, cell_group, path
            ))?;

        for (d, v_opt) in values.iter().enumerate() {
            embedding[[row_idx, d]] = v_opt.with_context(|| format!(
                "missing dim {d} for cell '{}' group '{}' in {:?}",
                barcode, cell_group, path
            ))?;
        }
    }

    Ok(embedding)
}

/// Progress bar shown over a small number of groups (e.g. 2 samples).
fn group_bar(n_groups: usize, label: &str) -> ProgressBar {
    let pb = ProgressBar::new(n_groups as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "  {prefix:.bold.cyan} [{bar:40.cyan/blue}] {pos}/{len} groups  {elapsed_precise}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_prefix(label.to_string());
    pb
}

/// Per-iteration progress bar for NMF / GMM (one per group, managed by MultiProgress).
fn iter_bar(multi: &MultiProgress, n_iter: usize, prefix: &str) -> ProgressBar {
    let pb = multi.add(ProgressBar::new(n_iter as u64));
    pb.set_style(
        ProgressStyle::with_template(
            "  {prefix:.bold.cyan} [{bar:40.cyan/blue}] {pos}/{len} iter  {msg}  {elapsed_precise}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_prefix(prefix.to_string());
    pb.set_message("starting");
    pb
}

fn write_csv<T: Serialize>(records: &[T], output: Option<&std::path::Path>) -> Result<()> {
    let writer: Box<dyn std::io::Write> = match output {
        Some(path) => Box::new(
            std::fs::File::create(path).with_context(|| format!("cannot create {:?}", path))?,
        ),
        None => Box::new(std::io::stdout()),
    };
    let mut wtr = csv::Writer::from_writer(writer);

    for record in records {
        wtr.serialize(record).context("serializing record")?;
    }
    wtr.flush().context("flushing CSV writer")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::read_nmf_w_embedding;
    use ndarray::{arr2, Array2};
    use spatialrs_io::AnnData;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn read_nmf_w_embedding_supports_global_group_rows() {
        let adata = sample_adata();

        with_temp_csv(
            "nmf-w-global",
            "cell_i,component,weight,group\ncell1,0,1.0,all\ncell1,1,2.0,all\ncell2,0,3.0,all\ncell2,1,4.0,all\n",
            |path| {
                let matrix = read_nmf_w_embedding(path, &adata, "sample").unwrap();
                assert_eq!(matrix, arr2(&[[1.0, 2.0], [3.0, 4.0]]));
            },
        );
    }

    #[test]
    fn read_nmf_w_embedding_rejects_missing_components() {
        let adata = sample_adata();

        with_temp_csv(
            "nmf-w-missing-component",
            "cell_i,component,weight,group\ncell1,0,1.0,s1\ncell1,1,2.0,s1\ncell2,0,3.0,s2\n",
            |path| {
                let err = match read_nmf_w_embedding(path, &adata, "sample") {
                    Ok(_) => panic!("expected missing component error"),
                    Err(err) => err,
                };
                assert!(err
                    .to_string()
                    .contains("missing W factor for cell 'cell2' group 's2' component 1"));
            },
        );
    }

    fn sample_adata() -> AnnData {
        AnnData {
            obs_names: vec!["cell1".to_string(), "cell2".to_string()],
            var_names: Vec::new(),
            coordinates: Array2::zeros((2, 2)),
            obs: HashMap::from([(
                "sample".to_string(),
                vec!["s1".to_string(), "s2".to_string()],
            )]),
            expression: None,
            embeddings: HashMap::new(),
        }
    }

    fn with_temp_csv(name: &str, contents: &str, f: impl FnOnce(&std::path::Path)) {
        let path = temp_path(name);
        fs::write(&path, contents).unwrap();
        f(&path);
        let _ = fs::remove_file(&path);
    }

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "spatialrs-cli-{name}-{}-{nanos}.csv",
            process::id()
        ))
    }
}
