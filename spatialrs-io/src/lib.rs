use anyhow::{bail, Context, Result};
use hdf5_metno as hdf5;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;

/// In-memory representation of the data we need from an .h5ad file.
pub struct AnnData {
    /// Cell barcodes / obs index, length N.
    pub obs_names: Vec<String>,
    /// Gene names / var index, length G.
    pub var_names: Vec<String>,
    /// Spatial coordinates, shape N×2 (x, y).
    pub coordinates: Array2<f64>,
    /// Obs columns requested by the caller (column name → per-cell string values).
    pub obs: HashMap<String, Vec<String>>,
    /// X matrix (N × G), loaded only when `load_expression` is true.
    pub expression: Option<Array2<f32>>,
    /// obsm embeddings keyed by obsm key name.
    pub embeddings: HashMap<String, Array2<f64>>,
}

/// Read an .h5ad file and return an `AnnData` containing the obs index,
/// spatial coordinates, requested obs columns, optional expression matrix,
/// and requested obsm embeddings.
pub fn read_h5ad(
    path: &Path,
    obs_cols: &[&str],
    obsm_keys: &[&str],
    load_expression: bool,
) -> Result<AnnData> {
    let file = hdf5::File::open(path)
        .with_context(|| format!("cannot open {:?}", path))?;

    let obs_names   = read_obs_names(&file)?;
    let coordinates = read_spatial(&file)?;

    if obs_names.len() != coordinates.nrows() {
        bail!(
            "obs_names length ({}) != coordinate rows ({})",
            obs_names.len(),
            coordinates.nrows()
        );
    }

    let mut obs = HashMap::new();
    for col in obs_cols {
        let values = read_obs_column(&file, col)
            .with_context(|| format!("reading obs column '{col}'"))?;
        obs.insert(col.to_string(), values);
    }

    let var_names = if load_expression || !obsm_keys.is_empty() {
        read_var_names(&file).unwrap_or_default()
    } else {
        Vec::new()
    };

    let expression = if load_expression {
        Some(
            read_expression_matrix(&file)
                .context("reading expression matrix X")?,
        )
    } else {
        None
    };

    let mut embeddings = HashMap::new();
    for key in obsm_keys {
        let emb = read_obsm_embedding(&file, key)
            .with_context(|| format!("reading obsm/{key}"))?;
        embeddings.insert(key.to_string(), emb);
    }

    Ok(AnnData { obs_names, var_names, coordinates, obs, expression, embeddings })
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn read_obs_names(file: &hdf5::File) -> Result<Vec<String>> {
    let obs = file.group("obs").context("no 'obs' group")?;

    // Strategy 1: dataset at obs/_index
    if let Ok(ds) = obs.dataset("_index") {
        return ds.read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .context("reading obs/_index");
    }

    // Strategy 2: dataset at obs/index
    if let Ok(ds) = obs.dataset("index") {
        return ds.read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .context("reading obs/index");
    }

    // Strategy 3: check _index attribute on the obs group for the real dataset name
    if let Ok(attr) = obs.attr("_index") {
        let name: hdf5::types::VarLenUnicode = attr.read_scalar()
            .context("reading obs _index attribute")?;
        let ds = obs.dataset(name.as_str())
            .with_context(|| format!("obs dataset '{}' (from _index attr)", name))?;
        return ds.read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .with_context(|| format!("reading obs/{}", name));
    }

    bail!("could not locate obs index in the h5ad file")
}

fn read_var_names(file: &hdf5::File) -> Result<Vec<String>> {
    let var = file.group("var").context("no 'var' group")?;

    if let Ok(ds) = var.dataset("_index") {
        return ds.read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .context("reading var/_index");
    }

    if let Ok(ds) = var.dataset("index") {
        return ds.read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .context("reading var/index");
    }

    if let Ok(attr) = var.attr("_index") {
        let name: hdf5::types::VarLenUnicode = attr.read_scalar()
            .context("reading var _index attribute")?;
        let ds = var.dataset(name.as_str())
            .with_context(|| format!("var dataset '{}' (from _index attr)", name))?;
        return ds.read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .with_context(|| format!("reading var/{}", name));
    }

    bail!("could not locate var index in the h5ad file")
}

fn read_spatial(file: &hdf5::File) -> Result<Array2<f64>> {
    let obsm = file.group("obsm").context("no 'obsm' group")?;

    for key in &["spatial", "X_spatial"] {
        if let Ok(ds) = obsm.dataset(key) {
            if let Ok(arr) = ds.read_2d::<f64>() {
                return slice_two_cols_f64(arr);
            }
            if let Ok(arr) = ds.read_2d::<f32>() {
                let arr_f64 = arr.mapv(|v| v as f64);
                return slice_two_cols_f64(arr_f64);
            }
            bail!("obsm/{key} is not f32 or f64");
        }
    }

    bail!("no spatial coordinates found under obsm/spatial or obsm/X_spatial")
}

fn slice_two_cols_f64(arr: Array2<f64>) -> Result<Array2<f64>> {
    if arr.ncols() < 2 {
        bail!("spatial array has fewer than 2 columns");
    }
    Ok(arr.slice(ndarray::s![.., 0..2]).to_owned())
}

/// Read a full obsm embedding (all columns) as f64.
fn read_obsm_embedding(file: &hdf5::File, key: &str) -> Result<Array2<f64>> {
    let obsm = file.group("obsm").context("no 'obsm' group")?;
    let ds   = obsm.dataset(key)
        .with_context(|| format!("obsm/{key} not found"))?;

    if let Ok(arr) = ds.read_2d::<f64>() {
        return Ok(arr);
    }
    if let Ok(arr) = ds.read_2d::<f32>() {
        return Ok(arr.mapv(|v| v as f64));
    }
    bail!("obsm/{key} is not f32 or f64")
}

/// Read the expression matrix X (N × G) as f32.
/// Handles both sparse CSR groups and dense datasets.
fn read_expression_matrix(file: &hdf5::File) -> Result<Array2<f32>> {
    // Try sparse CSR group first
    if let Ok(grp) = file.group("X") {
        if let (Ok(data_ds), Ok(indices_ds), Ok(indptr_ds)) = (
            grp.dataset("data"),
            grp.dataset("indices"),
            grp.dataset("indptr"),
        ) {
            let data: Vec<f32> = if let Ok(d) = data_ds.read_1d::<f32>() {
                d.to_vec()
            } else if let Ok(d) = data_ds.read_1d::<f64>() {
                d.iter().map(|&v| v as f32).collect()
            } else {
                bail!("X/data is not f32 or f64");
            };

            let indices = read_usize_vec(&indices_ds)
                .context("reading X/indices")?;
            let indptr  = read_usize_vec(&indptr_ds)
                .context("reading X/indptr")?;

            // Read shape from attribute
            let (n_obs, n_var) = read_csr_shape(&grp)?;

            let mut dense = Array2::<f32>::zeros((n_obs, n_var));
            for row in 0..n_obs {
                let start = indptr[row];
                let end   = indptr[row + 1];
                for k in start..end {
                    let col = indices[k];
                    dense[[row, col]] = data[k];
                }
            }
            return Ok(dense);
        }
    }

    // Fallback: dense dataset at X
    let ds = file.dataset("X").context("no 'X' dataset or group")?;
    if let Ok(arr) = ds.read_2d::<f32>() {
        return Ok(arr);
    }
    if let Ok(arr) = ds.read_2d::<f64>() {
        return Ok(arr.mapv(|v| v as f32));
    }
    bail!("X dataset is not f32 or f64")
}

fn read_usize_vec(ds: &hdf5::Dataset) -> Result<Vec<usize>> {
    if let Ok(a) = ds.read_1d::<i32>()  { return Ok(a.iter().map(|&v| v as usize).collect()); }
    if let Ok(a) = ds.read_1d::<i64>()  { return Ok(a.iter().map(|&v| v as usize).collect()); }
    if let Ok(a) = ds.read_1d::<u32>()  { return Ok(a.iter().map(|&v| v as usize).collect()); }
    if let Ok(a) = ds.read_1d::<u64>()  { return Ok(a.iter().map(|&v| v as usize).collect()); }
    bail!("integer dataset could not be read as i32, i64, u32, or u64")
}

fn read_csr_shape(grp: &hdf5::Group) -> Result<(usize, usize)> {
    let attr = grp.attr("shape").context("X group has no 'shape' attribute")?;

    if let Ok(s) = attr.read_1d::<i64>() {
        let v = s.to_vec();
        return Ok((v[0] as usize, v[1] as usize));
    }
    if let Ok(s) = attr.read_1d::<i32>() {
        let v = s.to_vec();
        return Ok((v[0] as usize, v[1] as usize));
    }
    if let Ok(s) = attr.read_1d::<u64>() {
        let v = s.to_vec();
        return Ok((v[0] as usize, v[1] as usize));
    }
    bail!("X/shape attribute could not be read as integer")
}

fn read_obs_column(file: &hdf5::File, col: &str) -> Result<Vec<String>> {
    let obs = file.group("obs")?;

    // Try categorical encoding first: obs/{col}/codes + obs/{col}/categories
    if let Ok(grp) = obs.group(col) {
        if let (Ok(codes_ds), Ok(cats_ds)) = (grp.dataset("codes"), grp.dataset("categories")) {
            let categories: Vec<String> = cats_ds
                .read_1d::<hdf5::types::VarLenUnicode>()
                .context("reading categories")?
                .iter()
                .map(|s| s.to_string())
                .collect();

            let n = codes_ds.size();
            let codes: Vec<i32> = if let Ok(c) = codes_ds.read_1d::<i8>() {
                c.iter().map(|&v| v as i32).collect()
            } else if let Ok(c) = codes_ds.read_1d::<i16>() {
                c.iter().map(|&v| v as i32).collect()
            } else {
                codes_ds.read_1d::<i32>()
                    .with_context(|| format!("reading codes for obs/{col}"))?
                    .iter()
                    .map(|&v| v)
                    .collect()
            };

            if codes.len() != n {
                bail!("code length mismatch for obs/{col}");
            }

            return codes
                .iter()
                .map(|&c| {
                    categories
                        .get(c as usize)
                        .cloned()
                        .with_context(|| format!("code {c} out of range for obs/{col}"))
                })
                .collect();
        }
    }

    // Fallback: direct VarLenUnicode dataset at obs/{col}
    let ds = obs.dataset(col)
        .with_context(|| format!("obs/{col} not found"))?;
    ds.read_1d::<hdf5::types::VarLenUnicode>()
        .map(|a| a.iter().map(|s| s.to_string()).collect())
        .with_context(|| format!("reading obs/{col} as VarLenUnicode"))
}
