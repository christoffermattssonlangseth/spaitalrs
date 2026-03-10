use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{MultiProgress, ParallelProgressIterator, ProgressBar, ProgressStyle};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use spatialrs_core::{
    aggregation::{aggregate_neighbors, AggregationRecord, GraphMode, WeightingMode},
    autocorr::{
        compute_bivariate_morans_i, compute_gearys_c, compute_local_morans_i, compute_morans_i,
        compute_morans_i_perm, BivariateMoranRecord, GearyCRecord, LocalMoranRecord, MoranPermRecord,
        MoranRecord,
    },
    composition::{compute_composition, compute_entropy, EntropyRecord},
    diff_composition::{diff_composition, DiffCompositionRecord},
    diff_niches::diff_niches,
    gmm::{run_gmm, CovarianceType, GmmConfig, NicheProbRecord, NicheRecord},
    interactions::{count_interactions, permute_interactions, InteractionStatsRecord},
    markers::{find_niche_markers, MarkerRecord},
    neighbors::{compute_graph_stats, knn_graph, radius_graph, EdgeRecord, GraphStatsRecord},
    nmf::{run_nmf, run_nmf_sparse, HRecord, NmfConfig, WRecord},
    ripley::{compute_cross_ripley, compute_ripley, CrossRipleyRecord, RipleyRecord},
    transitions::{compute_transitions, permute_transitions, TransitionRecord, TransitionStatsRecord},
};
use spatialrs_io::{read_h5ad, AnnData};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

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
    dim: usize,
    value: f64,
    group: String,
}

#[derive(Deserialize)]
struct NicheInputRecord {
    cell_i: String,
    niche: usize,
    #[allow(dead_code)]
    group: String,
}

#[derive(Serialize)]
struct ModelStatsRecord {
    k: usize,
    log_likelihood: f64,
    bic: f64,
    aic: f64,
    n_iter: usize,
    covariance_type: String,
    group: String,
}

#[derive(Deserialize)]
struct CompInputRecord {
    cell_i: String,
    cell_type: String,
    fraction: f64,
    #[allow(dead_code)]
    group: String,
}

#[derive(Serialize)]
struct NmfDiagRecord {
    metric: String,
    index: usize,
    value: f32,
}

#[derive(Serialize)]
struct NicheSummaryRecord {
    group: String,
    niche: usize,
    n_cells: usize,
    fraction: f64,
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
        /// Also compute and write per-cell Shannon entropy of the composition
        #[arg(long)]
        output_entropy: Option<PathBuf>,
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
        /// var column name to filter genes (e.g. "highly_variable"); keeps only True rows
        #[arg(long)]
        var_filter: Option<String>,
        /// Read from layers/<name> instead of X (e.g. "counts" for raw counts)
        #[arg(long)]
        layer: Option<String>,
        /// Use sparse NMF: keep X in CSR format instead of densifying.
        /// Requires X stored as a sparse CSR group in the h5ad (standard for
        /// AnnData written by scanpy/scran).  Recommended for datasets with
        /// > 200 k cells — reduces memory from O(N×G) to O(nnz) and is
        /// typically 10-20× faster for typical scRNA-seq sparsity (~95% zeros).
        #[arg(long, default_value_t = false)]
        sparse: bool,
        /// Output CSV for per-component explained-variance fractions and the
        /// error trajectory (metric, index, value).
        #[arg(long)]
        output_diagnostics: Option<PathBuf>,
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
        /// Number of permutations for empirical p-value (0 = skip)
        #[arg(long, default_value_t = 0)]
        n_permutations: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Output CSV for permutation-based p-values (feature, p_value_perm, group)
        #[arg(long)]
        output_perm: Option<PathBuf>,
    },
    /// Compute Geary's C spatial autocorrelation (complement to Moran's I)
    Geary {
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
    /// Identify spatially variable genes (SVG) via per-gene Moran's I
    Svg {
        input: PathBuf,
        /// Radius for spatial neighbour search
        #[arg(long)]
        radius: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        /// var column to filter genes (keeps rows where value is True); e.g. "highly_variable"
        #[arg(long)]
        var_filter: Option<String>,
        /// Load from layers/<name> instead of X
        #[arg(long)]
        layer: Option<String>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compute per-cell neighbour count (degree) for a radius graph
    GraphStats {
        input: PathBuf,
        /// Radius for spatial neighbour search
        #[arg(long)]
        radius: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Identify pooled niche marker genes using one-vs-rest Wilcoxon rank-sum tests
    Markers {
        input: PathBuf,
        /// Niche assignments CSV produced by `spatialrs gmm`
        #[arg(long)]
        niche_csv: PathBuf,
        /// Read from layers/<name> instead of X (e.g. "counts" for raw counts)
        #[arg(long)]
        layer: Option<String>,
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
    /// Sweep GMM over a range of k values to produce a BIC/AIC curve
    GmmSweep {
        input: PathBuf,
        /// obsm key to use as input embedding (mutually exclusive with --nmf-w / --agg)
        #[arg(long, conflicts_with_all = ["nmf_w", "agg"])]
        embedding: Option<String>,
        /// NMF W factors CSV (mutually exclusive with --embedding / --agg)
        #[arg(long, conflicts_with_all = ["embedding", "agg"])]
        nmf_w: Option<PathBuf>,
        /// Aggregation CSV (mutually exclusive with --embedding / --nmf-w)
        #[arg(long, conflicts_with_all = ["embedding", "nmf_w"])]
        agg: Option<PathBuf>,
        /// Minimum number of components (inclusive)
        #[arg(long, default_value_t = 2)]
        k_min: usize,
        /// Maximum number of components (inclusive)
        #[arg(long)]
        k_max: usize,
        /// Step between k values
        #[arg(long, default_value_t = 1)]
        k_step: usize,
        #[arg(long, default_value = "diagonal")]
        covariance: CovarianceArg,
        #[arg(long, default_value_t = 200)]
        max_iter: usize,
        #[arg(long, default_value_t = 1e-6)]
        tol: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 1e-6)]
        reg_covar: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        /// Output CSV for BIC/AIC per k (k, bic, aic, log_likelihood, n_iter, covariance_type, group)
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compute the niche spatial transition (co-occurrence) matrix
    Transitions {
        input: PathBuf,
        /// Niche assignments CSV produced by `spatialrs gmm`
        #[arg(long)]
        niche_csv: PathBuf,
        /// Radius for spatial neighbour search
        #[arg(long)]
        radius: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
        /// Run a permutation test and write enrichment statistics
        /// (observed, expected_mean, z_score, p_value) per niche pair
        #[arg(long)]
        output_stats: Option<PathBuf>,
        /// Number of permutations for the transition enrichment test
        #[arg(long, default_value_t = 1000)]
        n_permutations: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Compute per-cell Local Moran's I (LISA) spatial autocorrelation
    Lisa {
        input: PathBuf,
        /// obsm key for the embedding (mutually exclusive with --nmf-w)
        #[arg(long, conflicts_with = "nmf_w")]
        embedding: Option<String>,
        /// NMF W factors CSV (mutually exclusive with --embedding)
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
    /// Test for differential neighbourhood composition between two conditions
    DiffComposition {
        input: PathBuf,
        /// Composition CSV produced by `spatialrs composition`
        #[arg(long)]
        composition_csv: PathBuf,
        /// obs column that identifies the condition (e.g. "condition")
        #[arg(long)]
        condition: String,
        /// First condition label (group A)
        #[arg(long)]
        group_a: String,
        /// Second condition label (group B)
        #[arg(long)]
        group_b: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Estimate Ripley's K / L spatial clustering function
    Ripley {
        input: PathBuf,
        /// obs column containing cell type labels
        #[arg(long)]
        cell_type: String,
        /// Target cell type to analyse
        #[arg(long)]
        target_type: String,
        /// Minimum radius
        #[arg(long)]
        r_min: f64,
        /// Maximum radius
        #[arg(long)]
        r_max: f64,
        /// Number of evenly-spaced radii between r_min and r_max (inclusive)
        #[arg(long, default_value_t = 20)]
        n_radii: usize,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Estimate the bivariate (cross) Ripley's K / L function between two cell types
    CrossRipley {
        input: PathBuf,
        /// obs column containing cell type labels
        #[arg(long)]
        cell_type: String,
        /// First cell type (A)
        #[arg(long)]
        type_a: String,
        /// Second cell type (B)
        #[arg(long)]
        type_b: String,
        #[arg(long)]
        r_min: f64,
        #[arg(long)]
        r_max: f64,
        #[arg(long, default_value_t = 20)]
        n_radii: usize,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compute bivariate Moran's I for all pairs of embedding dimensions or NMF components
    BivariateMorans {
        input: PathBuf,
        /// obsm key for the embedding (mutually exclusive with --nmf-w)
        #[arg(long, conflicts_with = "nmf_w")]
        embedding: Option<String>,
        /// NMF W factors CSV (mutually exclusive with --embedding)
        #[arg(long, conflicts_with = "embedding")]
        nmf_w: Option<PathBuf>,
        #[arg(long)]
        radius: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare per-niche abundance between two conditions at the sample level
    DiffNiches {
        input: PathBuf,
        /// Niche assignments CSV produced by `spatialrs gmm`
        #[arg(long)]
        niche_csv: PathBuf,
        /// obs column identifying the condition (e.g. "condition")
        #[arg(long)]
        condition: String,
        /// obs column identifying the sample (must match the `group` column in the niche CSV)
        #[arg(long, default_value = "sample")]
        groupby: String,
        /// First condition label (group A)
        #[arg(long)]
        group_a: String,
        /// Second condition label (group B)
        #[arg(long)]
        group_b: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Summarize niche abundances per sample from a niche assignment CSV
    Summarize {
        /// Niche assignments CSV produced by `spatialrs gmm`
        niche_csv: PathBuf,
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
            let adata = read_h5ad(&input, &[&groupby], &[], false, false, None, None)?;
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
            let adata = read_h5ad(&input, &[&groupby], &[], false, false, None, None)?;
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
            let adata = read_h5ad(
                &input,
                &[&groupby, &cell_type],
                &[],
                false,
                false,
                None,
                None,
            )?;
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
                            &coords,
                            &barcodes,
                            &types,
                            radius,
                            n_permutations,
                            seed,
                            label,
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
            output_entropy,
        } => {
            let adata = read_h5ad(
                &input,
                &[&groupby, &cell_type],
                &[],
                false,
                false,
                None,
                None,
            )?;
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

            if let Some(ref entropy_path) = output_entropy {
                let entropy: Vec<EntropyRecord> = compute_entropy(&records);
                write_csv(&entropy, Some(entropy_path.as_path()))?;
            }
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
            var_filter,
            layer,
            sparse,
            output_diagnostics,
        } => {
            let obs_cols: Vec<&str> = groupby.as_deref().into_iter().collect();
            let adata = read_h5ad(
                &input,
                &obs_cols,
                &[],
                !sparse, // load dense unless --sparse
                sparse,  // load sparse when --sparse
                var_filter.as_deref(),
                layer.as_deref(),
            )?;

            // Run NMF once on ALL cells pooled — factors are then comparable across samples.
            // Group label is attached to output records from obs afterwards.
            let n_cells = adata.obs_names.len();
            let multi = MultiProgress::new();
            let mode_label = if sparse { "sparse" } else { "dense" };
            let pb = iter_bar(
                &multi,
                max_iter,
                &format!("NMF ({n_cells} cells, {mode_label})"),
            );
            let pb2 = pb.clone();
            let config = NmfConfig {
                n_components,
                max_iter,
                tol,
                seed,
                epsilon: 1e-12,
                iter_cb: Some(Arc::new(move |iter: usize, err: f32| {
                    pb2.set_position(iter as u64);
                    pb2.set_message(format!("err={err:.2e}"));
                })),
            };

            let result = if sparse {
                let csr = adata
                    .sparse_expression
                    .as_ref()
                    .context("sparse expression matrix not loaded")?;
                run_nmf_sparse(
                    &csr.data,
                    &csr.indices,
                    &csr.indptr,
                    csr.nrows,
                    csr.ncols,
                    &config,
                )?
            } else {
                let x_full = adata
                    .expression
                    .as_ref()
                    .context("expression matrix not loaded")?;
                run_nmf(x_full, &config)?
            };
            pb.finish_with_message(format!("err={:.2e}", result.final_error));

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

            if let Some(ref diag_path) = output_diagnostics {
                let mut diag_records: Vec<NmfDiagRecord> = Vec::new();
                for (comp, &var) in result.component_variances.iter().enumerate() {
                    diag_records.push(NmfDiagRecord {
                        metric: "component_variance".to_string(),
                        index: comp,
                        value: var,
                    });
                }
                for &(iter, err) in &result.error_trajectory {
                    diag_records.push(NmfDiagRecord {
                        metric: "iteration_error".to_string(),
                        index: iter,
                        value: err,
                    });
                }
                write_csv(&diag_records, Some(diag_path.as_path()))?;
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
            let adata = read_h5ad(&input, &[&groupby], &obsm_keys, false, false, None, None)?;

            // Build full embedding matrix (N × D) from one of three sources
            let emb_owned: Array2<f64>;
            let emb_full: &Array2<f64> = match (&embedding, &nmf_w, &agg) {
                (Some(key), None, None) => adata
                    .embeddings
                    .get(key)
                    .ok_or_else(|| anyhow::anyhow!("embedding '{key}' not loaded"))?,
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
                CovarianceArg::Diagonal => CovarianceType::Diagonal,
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

            let group_col = adata
                .obs
                .get(&groupby)
                .ok_or_else(|| anyhow::anyhow!("groupby column '{groupby}' not found in obs"))?;

            let mut niche_records: Vec<NicheRecord> = Vec::new();
            let mut prob_records: Vec<NicheProbRecord> = Vec::new();

            for (i, barcode) in adata.obs_names.iter().enumerate() {
                let group = group_col
                    .get(i)
                    .map(|s| s.as_str())
                    .unwrap_or("")
                    .to_string();
                niche_records.push(NicheRecord {
                    cell_i: barcode.clone(),
                    niche: result.labels[i],
                    group,
                });
                if output_probs.is_some() {
                    let group2 = group_col
                        .get(i)
                        .map(|s| s.as_str())
                        .unwrap_or("")
                        .to_string();
                    for comp in 0..k {
                        prob_records.push(NicheProbRecord {
                            cell_i: barcode.clone(),
                            component: comp,
                            probability: result.responsibilities[[i, comp]],
                            group: group2.clone(),
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
                    CovarianceArg::Diagonal => "diagonal",
                    CovarianceArg::Spherical => "spherical",
                };
                let stats = vec![ModelStatsRecord {
                    k,
                    log_likelihood: result.log_likelihood,
                    bic: result.bic,
                    aic: result.aic,
                    n_iter: result.n_iter,
                    covariance_type: cov_str.to_string(),
                    group: "all".to_string(),
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
            n_permutations,
            seed,
            output_perm,
        } => {
            let obsm_keys: Vec<&str> = embedding.as_deref().into_iter().collect();
            let adata = read_h5ad(&input, &[&groupby], &obsm_keys, false, false, None, None)?;

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
                _ => (0..emb_full.ncols())
                    .map(|i| format!("component_{i}"))
                    .collect(),
            };

            let groups = partition_by_group(&adata, &groupby)?;

            let pb = group_bar(groups.len(), "morans");
            let records: Vec<MoranRecord> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let emb_sub = emb_full.select(ndarray::Axis(0), indices);
                    compute_morans_i(&coords, &emb_sub, &feature_names, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;

            if n_permutations > 0 {
                if let Some(ref perm_path) = output_perm {
                    let pb2 = group_bar(groups.len(), "morans-perm");
                    let perm_records: Vec<MoranPermRecord> = groups
                        .par_iter()
                        .progress_with(pb2)
                        .map(|(label, indices)| {
                            let coords = extract_coords_subset(&adata, indices);
                            let emb_sub = emb_full.select(ndarray::Axis(0), indices);
                            compute_morans_i_perm(
                                &coords,
                                &emb_sub,
                                &feature_names,
                                radius,
                                n_permutations,
                                seed,
                                label,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .flatten()
                        .collect();
                    write_csv(&perm_records, Some(perm_path.as_path()))?;
                } else {
                    bail!("--output-perm is required when --n-permutations > 0");
                }
            }
        }

        Command::Geary {
            input,
            embedding,
            nmf_w,
            radius,
            groupby,
            output,
        } => {
            let obsm_keys: Vec<&str> = embedding.as_deref().into_iter().collect();
            let adata = read_h5ad(&input, &[&groupby], &obsm_keys, false, false, None, None)?;

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
                _ => (0..emb_full.ncols())
                    .map(|i| format!("component_{i}"))
                    .collect(),
            };

            let groups = partition_by_group(&adata, &groupby)?;
            let pb = group_bar(groups.len(), "geary");
            let records: Vec<GearyCRecord> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let emb_sub = emb_full.select(ndarray::Axis(0), indices);
                    compute_gearys_c(&coords, &emb_sub, &feature_names, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::Svg {
            input,
            radius,
            groupby,
            var_filter,
            layer,
            output,
        } => {
            let adata = read_h5ad(
                &input,
                &[&groupby],
                &[],
                true, // load dense expression
                false,
                var_filter.as_deref(),
                layer.as_deref(),
            )?;

            let expression = adata
                .expression
                .as_ref()
                .context("expression matrix not loaded")?;

            // Convert f32 expression to f64 for autocorr functions
            let values: Array2<f64> = expression.mapv(|v| v as f64);
            let feature_names = adata.var_names.clone();

            let groups = partition_by_group(&adata, &groupby)?;
            let pb = group_bar(groups.len(), "svg");
            let records: Vec<MoranRecord> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let vals_sub = values.select(ndarray::Axis(0), indices);
                    compute_morans_i(&coords, &vals_sub, &feature_names, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::GraphStats {
            input,
            radius,
            groupby,
            output,
        } => {
            let adata = read_h5ad(&input, &[&groupby], &[], false, false, None, None)?;
            let groups = partition_by_group(&adata, &groupby)?;

            let pb = group_bar(groups.len(), "graph-stats");
            let records: Vec<GraphStatsRecord> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    compute_graph_stats(&coords, &barcodes, radius, label)
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
            layer,
            output,
        } => {
            let adata = read_h5ad(&input, &[], &[], true, false, None, layer.as_deref())?;

            // Marker testing is pooled across all cells; the niche CSV only
            // provides the global niche label for each barcode in obs order.
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

            let expression = adata.expression.context("expression matrix not loaded")?;

            let n_cells = expression.nrows();
            let n_genes = expression.ncols();
            let pb = group_bar(n_niches, "markers");
            let _ = (n_cells, n_genes, pb); // progress shown inside find_niche_markers via Rayon

            let records: Vec<MarkerRecord> =
                find_niche_markers(&expression, &adata.var_names, &niche_labels, n_niches)?;

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
            let adata = read_h5ad(&input, &[&groupby], &obsm_keys, false, false, None, None)?;

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

        Command::GmmSweep {
            input,
            embedding,
            nmf_w,
            agg,
            k_min,
            k_max,
            k_step,
            covariance,
            max_iter,
            tol,
            seed,
            reg_covar,
            groupby,
            output,
        } => {
            if k_min == 0 {
                bail!("--k-min must be >= 1");
            }
            if k_max < k_min {
                bail!("--k-max must be >= --k-min");
            }
            if k_step == 0 {
                bail!("--k-step must be >= 1");
            }

            let obsm_keys: Vec<&str> = embedding.as_deref().into_iter().collect();
            let adata = read_h5ad(&input, &[&groupby], &obsm_keys, false, false, None, None)?;

            let emb_owned: Array2<f64>;
            let emb_full: &Array2<f64> = match (&embedding, &nmf_w, &agg) {
                (Some(key), None, None) => adata
                    .embeddings
                    .get(key)
                    .ok_or_else(|| anyhow::anyhow!("embedding '{key}' not loaded"))?,
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
                CovarianceArg::Diagonal => CovarianceType::Diagonal,
                CovarianceArg::Spherical => CovarianceType::Spherical,
            };
            let cov_str = match covariance {
                CovarianceArg::Diagonal => "diagonal",
                CovarianceArg::Spherical => "spherical",
            };

            let k_values: Vec<usize> = (k_min..=k_max).step_by(k_step).collect();
            let n_cells = emb_full.nrows();
            let pb = group_bar(k_values.len(), "gmm-sweep");

            let stats: Vec<ModelStatsRecord> = k_values
                .iter()
                .map(|&k| {
                    pb.inc(1);
                    let config = GmmConfig {
                        n_components: k,
                        max_iter,
                        tol,
                        seed,
                        covariance: cov_type,
                        reg_covar,
                        iter_cb: None,
                    };
                    let result = run_gmm(emb_full, &config)
                        .with_context(|| format!("GMM failed for k={k}"))?;
                    Ok(ModelStatsRecord {
                        k,
                        log_likelihood: result.log_likelihood,
                        bic: result.bic,
                        aic: result.aic,
                        n_iter: result.n_iter,
                        covariance_type: cov_str.to_string(),
                        group: "all".to_string(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            pb.finish_with_message(format!("{n_cells} cells, k={k_min}..{k_max}"));

            write_csv(&stats, output.as_deref())?;
        }

        Command::Transitions {
            input,
            niche_csv,
            radius,
            groupby,
            output,
            output_stats,
            n_permutations,
            seed,
        } => {
            let adata = read_h5ad(&input, &[&groupby], &[], false, false, None, None)?;

            // Load niche assignments
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

            let groups = partition_by_group(&adata, &groupby)?;
            let pb = group_bar(groups.len(), "transitions");
            let records: Vec<TransitionRecord> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let niches: Vec<usize> = indices.iter().map(|&i| niche_labels[i]).collect();
                    compute_transitions(&coords, &niches, radius, n_niches, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;

            if let Some(ref stats_path) = output_stats {
                let pb2 = group_bar(groups.len(), "transitions-perm");
                let stats: Vec<TransitionStatsRecord> = groups
                    .par_iter()
                    .progress_with(pb2)
                    .map(|(label, indices)| {
                        let coords = extract_coords_subset(&adata, indices);
                        let niches: Vec<usize> =
                            indices.iter().map(|&i| niche_labels[i]).collect();
                        permute_transitions(
                            &coords,
                            &niches,
                            radius,
                            n_niches,
                            n_permutations,
                            seed,
                            label,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                write_csv(&stats, Some(stats_path.as_path()))?;
            }
        }

        Command::Lisa {
            input,
            embedding,
            nmf_w,
            radius,
            groupby,
            output,
        } => {
            let obsm_keys: Vec<&str> = embedding.as_deref().into_iter().collect();
            let adata = read_h5ad(&input, &[&groupby], &obsm_keys, false, false, None, None)?;

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
                _ => (0..emb_full.ncols())
                    .map(|i| format!("component_{i}"))
                    .collect(),
            };

            let groups = partition_by_group(&adata, &groupby)?;
            let pb = group_bar(groups.len(), "lisa");
            let records: Vec<LocalMoranRecord> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    let emb_sub = emb_full.select(ndarray::Axis(0), indices);
                    compute_local_morans_i(&coords, &barcodes, &emb_sub, &feature_names, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::DiffComposition {
            input,
            composition_csv,
            condition,
            group_a,
            group_b,
            output,
        } => {
            let adata = read_h5ad(&input, &[&condition], &[], false, false, None, None)?;

            // Build condition map: barcode → condition label
            let cond_col = adata
                .obs
                .get(&condition)
                .with_context(|| format!("obs column '{condition}' not found"))?;
            let condition_map: HashMap<String, String> = adata
                .obs_names
                .iter()
                .zip(cond_col.iter())
                .map(|(b, c)| (b.clone(), c.clone()))
                .collect();

            // Load composition CSV
            let mut rdr = csv::Reader::from_path(&composition_csv)
                .with_context(|| format!("cannot open {:?}", composition_csv))?;
            let fractions: Vec<(String, String, f64)> = rdr
                .deserialize()
                .map(|r| {
                    let row: CompInputRecord =
                        r.with_context(|| format!("reading {:?}", composition_csv))?;
                    Ok((row.cell_i, row.cell_type, row.fraction))
                })
                .collect::<Result<Vec<_>>>()?;

            let records: Vec<DiffCompositionRecord> =
                diff_composition(&fractions, &condition_map, &group_a, &group_b)?;

            write_csv(&records, output.as_deref())?;
        }

        Command::Ripley {
            input,
            cell_type,
            target_type,
            r_min,
            r_max,
            n_radii,
            groupby,
            output,
        } => {
            if r_min <= 0.0 || !r_min.is_finite() {
                bail!("--r-min must be > 0");
            }
            if r_max <= r_min {
                bail!("--r-max must be > --r-min");
            }
            if n_radii < 2 {
                bail!("--n-radii must be >= 2");
            }

            let adata =
                read_h5ad(&input, &[&groupby, &cell_type], &[], false, false, None, None)?;

            // Build linearly-spaced radius grid
            let step = (r_max - r_min) / (n_radii - 1) as f64;
            let radii: Vec<f64> = (0..n_radii).map(|i| r_min + i as f64 * step).collect();

            let groups = partition_by_group(&adata, &groupby)?;
            let pb = group_bar(groups.len(), "ripley");
            let records: Vec<RipleyRecord> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let types = extract_strings_subset(&adata, &cell_type, indices);
                    compute_ripley(&coords, &types, &target_type, &radii, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::CrossRipley {
            input,
            cell_type,
            type_a,
            type_b,
            r_min,
            r_max,
            n_radii,
            groupby,
            output,
        } => {
            if r_min <= 0.0 || !r_min.is_finite() {
                bail!("--r-min must be > 0");
            }
            if r_max <= r_min {
                bail!("--r-max must be > --r-min");
            }
            if n_radii < 2 {
                bail!("--n-radii must be >= 2");
            }

            let adata =
                read_h5ad(&input, &[&groupby, &cell_type], &[], false, false, None, None)?;

            let step = (r_max - r_min) / (n_radii - 1) as f64;
            let radii: Vec<f64> = (0..n_radii).map(|i| r_min + i as f64 * step).collect();

            let groups = partition_by_group(&adata, &groupby)?;
            let pb = group_bar(groups.len(), "cross-ripley");
            let records: Vec<CrossRipleyRecord> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let types = extract_strings_subset(&adata, &cell_type, indices);
                    compute_cross_ripley(&coords, &types, &type_a, &type_b, &radii, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::BivariateMorans {
            input,
            embedding,
            nmf_w,
            radius,
            groupby,
            output,
        } => {
            let obsm_keys: Vec<&str> = embedding.as_deref().into_iter().collect();
            let adata = read_h5ad(&input, &[&groupby], &obsm_keys, false, false, None, None)?;

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
                _ => (0..emb_full.ncols())
                    .map(|i| format!("component_{i}"))
                    .collect(),
            };

            let groups = partition_by_group(&adata, &groupby)?;
            let pb = group_bar(groups.len(), "bivariate-morans");
            let records: Vec<BivariateMoranRecord> = groups
                .par_iter()
                .progress_with(pb)
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let emb_sub = emb_full.select(ndarray::Axis(0), indices);
                    compute_bivariate_morans_i(&coords, &emb_sub, &feature_names, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::DiffNiches {
            input,
            niche_csv,
            condition,
            groupby,
            group_a,
            group_b,
            output,
        } => {
            let adata =
                read_h5ad(&input, &[&groupby, &condition], &[], false, false, None, None)?;

            // Build sample → condition map from adata obs
            let groupby_col = adata
                .obs
                .get(&groupby)
                .with_context(|| format!("obs column '{groupby}' not found"))?;
            let condition_col = adata
                .obs
                .get(&condition)
                .with_context(|| format!("obs column '{condition}' not found"))?;
            let mut sample_condition: HashMap<String, String> = HashMap::new();
            for (sample, cond) in groupby_col.iter().zip(condition_col.iter()) {
                sample_condition.entry(sample.clone()).or_insert_with(|| cond.clone());
            }

            // Load niche CSV; group column = sample name
            let mut rdr = csv::Reader::from_path(&niche_csv)
                .with_context(|| format!("cannot open {:?}", niche_csv))?;
            let mut niche_rows: Vec<NicheInputRecord> = Vec::new();
            let mut max_niche = 0usize;
            for record in rdr.deserialize() {
                let row: NicheInputRecord =
                    record.with_context(|| format!("reading {:?}", niche_csv))?;
                max_niche = max_niche.max(row.niche);
                niche_rows.push(row);
            }
            let n_niches = max_niche + 1;

            // Accumulate per-sample niche counts
            let mut sample_niche_counts: HashMap<String, Vec<usize>> = HashMap::new();
            let mut sample_totals: HashMap<String, usize> = HashMap::new();
            for row in &niche_rows {
                let counts = sample_niche_counts
                    .entry(row.group.clone())
                    .or_insert_with(|| vec![0usize; n_niches]);
                counts[row.niche] += 1;
                *sample_totals.entry(row.group.clone()).or_insert(0) += 1;
            }

            // Build per-condition fraction vectors
            let mut fracs_a: Vec<Vec<f64>> = Vec::new();
            let mut fracs_b: Vec<Vec<f64>> = Vec::new();
            for (sample, counts) in &sample_niche_counts {
                let cond = match sample_condition.get(sample) {
                    Some(c) => c,
                    None => continue,
                };
                let total = sample_totals[sample];
                let fracs: Vec<f64> =
                    counts.iter().map(|&c| c as f64 / total as f64).collect();
                if cond == &group_a {
                    fracs_a.push(fracs);
                } else if cond == &group_b {
                    fracs_b.push(fracs);
                }
            }

            let records = diff_niches(&fracs_a, &fracs_b, &group_a, &group_b, n_niches)?;
            write_csv(&records, output.as_deref())?;
        }

        Command::Summarize { niche_csv, output } => {
            let mut rdr = csv::Reader::from_path(&niche_csv)
                .with_context(|| format!("cannot open {:?}", niche_csv))?;

            let mut group_niche_counts: HashMap<(String, usize), usize> = HashMap::new();
            let mut group_totals: HashMap<String, usize> = HashMap::new();
            for record in rdr.deserialize() {
                let row: NicheInputRecord =
                    record.with_context(|| format!("reading {:?}", niche_csv))?;
                *group_niche_counts.entry((row.group.clone(), row.niche)).or_insert(0) += 1;
                *group_totals.entry(row.group).or_insert(0) += 1;
            }

            let mut records: Vec<NicheSummaryRecord> = group_niche_counts
                .into_iter()
                .map(|((group, niche), n_cells)| {
                    let total = group_totals[&group];
                    NicheSummaryRecord {
                        group,
                        niche,
                        n_cells,
                        fraction: n_cells as f64 / total as f64,
                    }
                })
                .collect();
            records.sort_by(|a, b| a.group.cmp(&b.group).then(a.niche.cmp(&b.niche)));

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

    // key: (group, cell_i)  value: per-dim values
    let mut map: HashMap<(String, String), Vec<Option<f64>>> = HashMap::new();
    let mut max_dim = None::<usize>;

    for record in rdr.deserialize() {
        let row: AggInputRecord = record.with_context(|| format!("reading {:?}", path))?;
        max_dim = Some(match max_dim {
            Some(m) => m.max(row.dim),
            None => row.dim,
        });
        let entry = map.entry((row.group, row.cell_i)).or_default();
        // Extend the vec if needed
        if entry.len() <= row.dim {
            entry.resize(row.dim + 1, None);
        }
        if entry[row.dim].replace(row.value).is_some() {
            bail!("duplicate dim {} in {:?}", row.dim, path);
        }
    }

    let n_dims = max_dim.map(|m| m + 1).context("aggregation CSV is empty")?;

    let mut embedding = Array2::<f64>::zeros((adata.obs_names.len(), n_dims));
    for (row_idx, barcode) in adata.obs_names.iter().enumerate() {
        let cell_group = &groups[row_idx];
        let values = map
            .get(&(cell_group.clone(), barcode.clone()))
            .with_context(|| {
                format!(
                    "missing aggregation row for cell '{}' group '{}' in {:?}",
                    barcode, cell_group, path
                )
            })?;

        for (d, v_opt) in values.iter().enumerate() {
            embedding[[row_idx, d]] = v_opt.with_context(|| {
                format!(
                    "missing dim {d} for cell '{}' group '{}' in {:?}",
                    barcode, cell_group, path
                )
            })?;
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
    use super::{read_nmf_w_embedding, write_csv};
    use csv::Reader;
    use ndarray::{arr2, Array2};
    use spatialrs_core::markers::MarkerRecord;
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

    #[test]
    fn marker_csv_schema_is_pooled_and_has_no_group_column() {
        let path = temp_path("markers-schema");
        let records = vec![MarkerRecord {
            niche: 0,
            gene: "gene_a".to_string(),
            mean_niche: 1.0,
            mean_rest: 0.5,
            log2fc: 1.0,
            z_score: 2.0,
            p_value: 0.01,
            q_value_bh: 0.02,
        }];

        write_csv(&records, Some(path.as_path())).unwrap();

        let mut rdr = Reader::from_path(&path).unwrap();
        let headers = rdr
            .headers()
            .unwrap()
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();

        assert_eq!(
            headers,
            vec![
                "niche",
                "gene",
                "mean_niche",
                "mean_rest",
                "log2fc",
                "z_score",
                "p_value",
                "q_value_bh",
            ]
        );

        let _ = fs::remove_file(&path);
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
            sparse_expression: None,
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
