use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;
use serde::Serialize;
use spatialrs_core::{
    aggregation::{aggregate_neighbors, AggregationRecord, GraphMode, WeightingMode},
    composition::compute_composition,
    interactions::count_interactions,
    neighbors::{knn_graph, radius_graph, EdgeRecord},
    nmf::{run_nmf, HRecord, NmfConfig, WRecord},
};
use spatialrs_io::{read_h5ad, AnnData};
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
        #[arg(long)]
        output: Option<PathBuf>,
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
    /// Aggregate neighbour embeddings using distance-weighted averaging
    Aggregate {
        input: PathBuf,
        /// obsm key for the embedding to aggregate
        #[arg(long)]
        embedding: String,
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
        Command::Radius { input, radius, groupby, output } => {
            let adata = read_h5ad(&input, &[&groupby], &[], false)?;
            let groups = partition_by_group(&adata, &groupby)?;

            let records: Vec<EdgeRecord> = groups
                .par_iter()
                .map(|(label, indices)| {
                    let coords   = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    radius_graph(&coords, &barcodes, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::Knn { input, k, groupby, output } => {
            let adata = read_h5ad(&input, &[&groupby], &[], false)?;
            let groups = partition_by_group(&adata, &groupby)?;

            let records: Vec<EdgeRecord> = groups
                .par_iter()
                .map(|(label, indices)| {
                    let coords   = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    knn_graph(&coords, &barcodes, k, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::Interactions { input, cell_type, radius, groupby, output } => {
            let adata = read_h5ad(&input, &[&groupby, &cell_type], &[], false)?;
            let groups = partition_by_group(&adata, &groupby)?;

            let records: Vec<_> = groups
                .par_iter()
                .map(|(label, indices)| {
                    let coords   = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    let types    = extract_strings_subset(&adata, &cell_type, indices);
                    count_interactions(&coords, &barcodes, &types, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::Composition { input, cell_type, radius, groupby, output } => {
            let adata = read_h5ad(&input, &[&groupby, &cell_type], &[], false)?;
            let groups = partition_by_group(&adata, &groupby)?;

            let records: Vec<_> = groups
                .par_iter()
                .map(|(label, indices)| {
                    let coords   = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    let types    = extract_strings_subset(&adata, &cell_type, indices);
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

            let config = NmfConfig { n_components, max_iter, tol, seed, epsilon: 1e-12 };

            // Build groups: either partition by obs column or treat all as one group
            let groups: Vec<(String, Vec<usize>)> = if let Some(ref col) = groupby {
                partition_by_group(&adata, col)?
            } else {
                vec![("all".to_string(), (0..adata.obs_names.len()).collect())]
            };

            // Run NMF per group (parallel)
            let results: Vec<(String, Vec<usize>, spatialrs_core::nmf::NmfResult)> = groups
                .into_par_iter()
                .map(|(label, indices)| {
                    let x_sub = x_full.select(ndarray::Axis(0), &indices);
                    let result = run_nmf(&x_sub, &config)?;
                    Ok((label, indices, result))
                })
                .collect::<Result<Vec<_>>>()?;

            // Build W records
            let mut w_records: Vec<WRecord> = Vec::new();
            let mut h_records: Vec<HRecord> = Vec::new();

            for (label, indices, result) in &results {
                for (local_row, &global_i) in indices.iter().enumerate() {
                    let barcode = &adata.obs_names[global_i];
                    for comp in 0..n_components {
                        w_records.push(WRecord {
                            cell_i:    barcode.clone(),
                            component: comp,
                            weight:    result.w[[local_row, comp]],
                            group:     label.clone(),
                        });
                    }
                }

                for (gene_idx, gene) in adata.var_names.iter().enumerate() {
                    for comp in 0..n_components {
                        h_records.push(HRecord {
                            gene:      gene.clone(),
                            component: comp,
                            loading:   result.h[[comp, gene_idx]],
                            group:     label.clone(),
                        });
                    }
                }

                eprintln!(
                    "[nmf] group='{}' cells={} iter={} error={:.4}",
                    label,
                    indices.len(),
                    result.n_iter,
                    result.final_error,
                );
            }

            write_csv(&w_records, output_w.as_deref())?;

            if output_h.is_some() {
                write_csv(&h_records, output_h.as_deref())?;
            }
        }

        Command::Aggregate {
            input,
            embedding,
            radius,
            k,
            weighting,
            sigma,
            decay,
            epsilon,
            groupby,
            output,
        } => {
            let adata = read_h5ad(&input, &[&groupby], &[&embedding], false)?;

            let emb_full = adata
                .embeddings
                .get(&embedding)
                .with_context(|| format!("embedding '{embedding}' not loaded"))?;

            let graph = match (radius, k) {
                (Some(r), None) => GraphMode::Radius(r),
                (None, Some(k)) => GraphMode::Knn(k),
                (Some(_), Some(_)) => bail!("specify --radius or --k, not both"),
                (None, None) => bail!("one of --radius or --k is required"),
            };

            let weight_mode = build_weighting_mode(&weighting, sigma, decay, epsilon)?;

            let groups = partition_by_group(&adata, &groupby)?;

            let records: Vec<AggregationRecord> = groups
                .par_iter()
                .map(|(label, indices)| {
                    let coords   = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    let emb_sub  = emb_full.select(ndarray::Axis(0), indices);
                    aggregate_neighbors(
                        &coords, &barcodes, &emb_sub,
                        &graph, &weight_mode, label,
                    )
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }
    }

    Ok(())
}

// ─── shared helpers ───────────────────────────────────────────────────────────

fn build_weighting_mode(
    arg:     &WeightingArg,
    sigma:   Option<f64>,
    decay:   Option<f64>,
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

    let mut map: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
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
    indices.iter().map(|&i| adata.obs_names[i].clone()).collect()
}

fn extract_strings_subset(adata: &AnnData, col: &str, indices: &[usize]) -> Vec<String> {
    let values = &adata.obs[col];
    indices.iter().map(|&i| values[i].clone()).collect()
}

fn write_csv<T: Serialize>(records: &[T], output: Option<&std::path::Path>) -> Result<()> {
    let writer: Box<dyn std::io::Write> = match output {
        Some(path) => Box::new(
            std::fs::File::create(path)
                .with_context(|| format!("cannot create {:?}", path))?,
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
