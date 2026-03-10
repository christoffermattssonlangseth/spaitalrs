use anyhow::{bail, Context, Result};
use hdf5_metno as hdf5;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;

/// Sparse CSR matrix of f32 values.
pub struct CsrMatrix {
    /// Non-zero values.
    pub data: Vec<f32>,
    /// Column index for each non-zero value.
    pub indices: Vec<usize>,
    /// Row-start pointers, length nrows + 1.
    pub indptr: Vec<usize>,
    pub nrows: usize,
    pub ncols: usize,
}

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
    /// X matrix (N × G) as dense f32, loaded only when `load_expression` is true.
    pub expression: Option<Array2<f32>>,
    /// X matrix in sparse CSR format, loaded only when `load_sparse` is true.
    pub sparse_expression: Option<CsrMatrix>,
    /// obsm embeddings keyed by obsm key name.
    pub embeddings: HashMap<String, Array2<f64>>,
}

/// Read an .h5ad file and return an `AnnData` containing the obs index,
/// spatial coordinates, requested obs columns, optional expression matrix,
/// and requested obsm embeddings.
///
/// `load_expression` – densify the X matrix into `AnnData.expression`.
/// `load_sparse`     – keep X in CSR format in `AnnData.sparse_expression`
///                     (much lower memory for typical sparse scRNA-seq data).
/// `layer`           – read from `layers/<name>` instead of `X` when provided.
/// Both load flags may be false; at most one should be true for NMF.
pub fn read_h5ad(
    path: &Path,
    obs_cols: &[&str],
    obsm_keys: &[&str],
    load_expression: bool,
    load_sparse: bool,
    var_filter: Option<&str>,
    layer: Option<&str>,
) -> Result<AnnData> {
    let file = hdf5::File::open(path).with_context(|| format!("cannot open {:?}", path))?;

    let obs_names = read_obs_names(&file)?;
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
        let values =
            read_obs_column(&file, col).with_context(|| format!("reading obs column '{col}'"))?;
        ensure_row_count(
            &format!("obs column '{col}'"),
            values.len(),
            obs_names.len(),
        )?;
        obs.insert(col.to_string(), values);
    }

    // Resolve var names and gene mask once (used by both dense and sparse paths)
    let (var_names_all, gene_mask): (Vec<String>, Option<Vec<bool>>) =
        if load_expression || load_sparse {
            let all_var_names = read_var_names(&file).unwrap_or_default();
            let mask = if let Some(col) = var_filter {
                let m = read_var_bool_column(&file, col)
                    .with_context(|| format!("reading var column '{col}'"))?;
                if m.len() != all_var_names.len() {
                    bail!(
                        "var column '{col}' length ({}) != var_names length ({})",
                        m.len(),
                        all_var_names.len()
                    );
                }
                let n_kept = m.iter().filter(|&&v| v).count();
                eprintln!(
                    "var_filter '{col}': keeping {n_kept}/{} genes",
                    all_var_names.len()
                );
                Some(m)
            } else {
                None
            };
            (all_var_names, mask)
        } else {
            let var_names = if !obsm_keys.is_empty() {
                read_var_names(&file).unwrap_or_default()
            } else {
                Vec::new()
            };
            (var_names, None)
        };

    let filtered_var_names: Vec<String> = match &gene_mask {
        Some(mask) => var_names_all
            .iter()
            .zip(mask.iter())
            .filter_map(|(n, &keep)| if keep { Some(n.clone()) } else { None })
            .collect(),
        None => var_names_all,
    };

    let expression = if load_expression {
        let expr = read_expression_matrix(&file, gene_mask.as_deref(), layer)
            .context("reading expression matrix X")?;
        ensure_row_count("expression matrix X", expr.nrows(), obs_names.len())?;
        Some(expr)
    } else {
        None
    };

    let sparse_expression = if load_sparse {
        let csr = read_expression_sparse(&file, gene_mask.as_deref(), layer)
            .context("reading sparse expression matrix X")?;
        if csr.nrows != obs_names.len() {
            bail!(
                "sparse expression matrix row count ({}) does not match obs count ({})",
                csr.nrows,
                obs_names.len()
            );
        }
        Some(csr)
    } else {
        None
    };

    let var_names = filtered_var_names;

    let mut embeddings = HashMap::new();
    for key in obsm_keys {
        let emb = read_obsm_embedding(&file, key).with_context(|| format!("reading obsm/{key}"))?;
        ensure_row_count(&format!("obsm/{key}"), emb.nrows(), obs_names.len())?;
        embeddings.insert(key.to_string(), emb);
    }

    Ok(AnnData {
        obs_names,
        var_names,
        coordinates,
        obs,
        expression,
        sparse_expression,
        embeddings,
    })
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn read_obs_names(file: &hdf5::File) -> Result<Vec<String>> {
    let obs = file.group("obs").context("no 'obs' group")?;

    // Strategy 1: dataset at obs/_index
    if let Ok(ds) = obs.dataset("_index") {
        return ds
            .read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .context("reading obs/_index");
    }

    // Strategy 2: dataset at obs/index
    if let Ok(ds) = obs.dataset("index") {
        return ds
            .read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .context("reading obs/index");
    }

    // Strategy 3: check _index attribute on the obs group for the real dataset name
    if let Ok(attr) = obs.attr("_index") {
        let name: hdf5::types::VarLenUnicode =
            attr.read_scalar().context("reading obs _index attribute")?;
        let ds = obs
            .dataset(name.as_str())
            .with_context(|| format!("obs dataset '{}' (from _index attr)", name))?;
        return ds
            .read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .with_context(|| format!("reading obs/{}", name));
    }

    bail!("could not locate obs index in the h5ad file")
}

fn read_var_names(file: &hdf5::File) -> Result<Vec<String>> {
    let var = file.group("var").context("no 'var' group")?;

    if let Ok(ds) = var.dataset("_index") {
        return ds
            .read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .context("reading var/_index");
    }

    if let Ok(ds) = var.dataset("index") {
        return ds
            .read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .context("reading var/index");
    }

    if let Ok(attr) = var.attr("_index") {
        let name: hdf5::types::VarLenUnicode =
            attr.read_scalar().context("reading var _index attribute")?;
        let ds = var
            .dataset(name.as_str())
            .with_context(|| format!("var dataset '{}' (from _index attr)", name))?;
        return ds
            .read_1d::<hdf5::types::VarLenUnicode>()
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
    let ds = obsm
        .dataset(key)
        .with_context(|| format!("obsm/{key} not found"))?;

    if let Ok(arr) = ds.read_2d::<f64>() {
        return Ok(arr);
    }
    if let Ok(arr) = ds.read_2d::<f32>() {
        return Ok(arr.mapv(|v| v as f64));
    }
    bail!("obsm/{key} is not f32 or f64")
}

/// Read the expression matrix (N × G) as f32.
/// `layer` selects `layers/<name>` instead of `X` when provided.
/// If `gene_mask` is provided, only columns where mask[col] is true are included.
fn read_expression_matrix(
    file: &hdf5::File,
    gene_mask: Option<&[bool]>,
    layer: Option<&str>,
) -> Result<Array2<f32>> {
    let layer_path = layer.map(|l| format!("layers/{l}"));
    let group_key = layer_path.as_deref().unwrap_or("X");

    // Try sparse CSR group first
    if let Ok(grp) = file.group(group_key) {
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

            let indices = read_usize_vec(&indices_ds).context("reading X/indices")?;
            let indptr = read_usize_vec(&indptr_ds).context("reading X/indptr")?;

            let (n_obs, n_var) = read_csr_shape(&grp)?;
            validate_csr_layout(n_obs, n_var, data.len(), &indices, &indptr)?;

            // Build column remapping: old_col → new_col (None = skip)
            let (n_out, col_remap): (usize, Vec<Option<usize>>) = match gene_mask {
                None => (n_var, (0..n_var).map(Some).collect()),
                Some(mask) => {
                    let mut remap = vec![None; n_var];
                    let mut new_col = 0usize;
                    for (i, &keep) in mask.iter().enumerate() {
                        if keep {
                            remap[i] = Some(new_col);
                            new_col += 1;
                        }
                    }
                    (new_col, remap)
                }
            };

            let mut dense = Array2::<f32>::zeros((n_obs, n_out));
            for row in 0..n_obs {
                let start = indptr[row];
                let end = indptr[row + 1];
                for k in start..end {
                    let col = indices[k];
                    if let Some(new_col) = col_remap[col] {
                        dense[[row, new_col]] = data[k];
                    }
                }
            }
            return Ok(dense);
        }
    }

    // Fallback: dense dataset
    let ds = file
        .dataset(group_key)
        .with_context(|| format!("no '{}' dataset or group", group_key))?;
    let arr = if let Ok(a) = ds.read_2d::<f32>() {
        a
    } else if let Ok(a) = ds.read_2d::<f64>() {
        a.mapv(|v| v as f32)
    } else {
        bail!("{} dataset is not f32 or f64", group_key);
    };

    // Apply column filter to dense matrix
    match gene_mask {
        None => Ok(arr),
        Some(mask) => {
            let kept_cols: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter_map(|(i, &k)| if k { Some(i) } else { None })
                .collect();
            Ok(arr.select(ndarray::Axis(1), &kept_cols))
        }
    }
}

/// Read the expression matrix as a sparse CSR matrix of f32.
/// `layer` selects `layers/<name>` instead of `X` when provided.
/// If `gene_mask` is provided, only columns where mask[col] is true are included
/// and the column indices are remapped to the filtered space.
/// Requires the target to be stored as a CSR sparse group (data/indices/indptr).
fn read_expression_sparse(
    file: &hdf5::File,
    gene_mask: Option<&[bool]>,
    layer: Option<&str>,
) -> Result<CsrMatrix> {
    let group_key = match layer {
        Some(l) => format!("layers/{l}"),
        None => "X".to_string(),
    };
    let label = layer.unwrap_or("X");
    let grp = file.group(&group_key).with_context(|| {
        format!("sparse mode requires '{label}' stored as a CSR group (data/indices/indptr)")
    })?;

    let (data_ds, indices_ds, indptr_ds) = (
        grp.dataset("data").context("X/data not found")?,
        grp.dataset("indices").context("X/indices not found")?,
        grp.dataset("indptr").context("X/indptr not found")?,
    );

    let data: Vec<f32> = if let Ok(d) = data_ds.read_1d::<f32>() {
        d.to_vec()
    } else if let Ok(d) = data_ds.read_1d::<f64>() {
        d.iter().map(|&v| v as f32).collect()
    } else {
        bail!("X/data is not f32 or f64");
    };

    let indices = read_usize_vec(&indices_ds).context("reading X/indices")?;
    let indptr = read_usize_vec(&indptr_ds).context("reading X/indptr")?;
    let (n_obs, n_var) = read_csr_shape(&grp)?;
    validate_csr_layout(n_obs, n_var, data.len(), &indices, &indptr)?;

    // Apply gene mask: filter columns and remap indices.
    match gene_mask {
        None => Ok(CsrMatrix {
            data,
            indices,
            indptr,
            nrows: n_obs,
            ncols: n_var,
        }),
        Some(mask) => {
            let mut col_remap = vec![None::<usize>; n_var];
            let mut new_col = 0usize;
            for (i, &keep) in mask.iter().enumerate() {
                if keep {
                    col_remap[i] = Some(new_col);
                    new_col += 1;
                }
            }
            let n_out = new_col;

            let mut new_data: Vec<f32> = Vec::new();
            let mut new_indices: Vec<usize> = Vec::new();
            let mut new_indptr: Vec<usize> = vec![0; n_obs + 1];

            for row in 0..n_obs {
                let start = indptr[row];
                let end = indptr[row + 1];
                for k in start..end {
                    if let Some(nc) = col_remap[indices[k]] {
                        new_data.push(data[k]);
                        new_indices.push(nc);
                    }
                }
                new_indptr[row + 1] = new_data.len();
            }

            Ok(CsrMatrix {
                data: new_data,
                indices: new_indices,
                indptr: new_indptr,
                nrows: n_obs,
                ncols: n_out,
            })
        }
    }
}

/// Read a boolean var column (e.g. "highly_variable") as a Vec<bool>.
/// Handles uint8 (0/1), int8, and native bool HDF5 types.
fn read_var_bool_column(file: &hdf5::File, col: &str) -> Result<Vec<bool>> {
    let var = file.group("var").context("no 'var' group")?;
    let ds = var
        .dataset(col)
        .with_context(|| format!("var/{col} not found"))?;

    if let Ok(arr) = ds.read_1d::<u8>() {
        return Ok(arr.iter().map(|&v| v != 0).collect());
    }
    if let Ok(arr) = ds.read_1d::<i8>() {
        return Ok(arr.iter().map(|&v| v != 0).collect());
    }
    if let Ok(arr) = ds.read_1d::<bool>() {
        return Ok(arr.to_vec());
    }
    bail!("var/{col} could not be read as a boolean column (tried u8, i8, bool)")
}

fn read_usize_vec(ds: &hdf5::Dataset) -> Result<Vec<usize>> {
    if let Ok(a) = ds.read_1d::<i32>() {
        return a.iter().map(|&v| checked_i32_to_usize(v)).collect();
    }
    if let Ok(a) = ds.read_1d::<i64>() {
        return a.iter().map(|&v| checked_i64_to_usize(v)).collect();
    }
    if let Ok(a) = ds.read_1d::<u32>() {
        return Ok(a.iter().map(|&v| v as usize).collect());
    }
    if let Ok(a) = ds.read_1d::<u64>() {
        return Ok(a.iter().map(|&v| v as usize).collect());
    }
    bail!("integer dataset could not be read as i32, i64, u32, or u64")
}

fn read_csr_shape(grp: &hdf5::Group) -> Result<(usize, usize)> {
    let attr = grp
        .attr("shape")
        .context("X group has no 'shape' attribute")?;

    if let Ok(s) = attr.read_1d::<i64>() {
        let v = s.to_vec();
        return parse_shape_pair_i64(&v);
    }
    if let Ok(s) = attr.read_1d::<i32>() {
        let v = s.to_vec();
        return parse_shape_pair_i32(&v);
    }
    if let Ok(s) = attr.read_1d::<u64>() {
        let v = s.to_vec();
        return parse_shape_pair_u64(&v);
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

            let codes = read_category_codes(&codes_ds)
                .with_context(|| format!("reading codes for obs/{col}"))?;
            return decode_categorical_codes(&categories, &codes, col);
        }
    }

    // Fallback: direct VarLenUnicode dataset at obs/{col}
    let ds = obs
        .dataset(col)
        .with_context(|| format!("obs/{col} not found"))?;
    ds.read_1d::<hdf5::types::VarLenUnicode>()
        .map(|a| a.iter().map(|s| s.to_string()).collect())
        .with_context(|| format!("reading obs/{col} as VarLenUnicode"))
}

fn ensure_row_count(label: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        bail!("{label} row count ({actual}) does not match obs count ({expected})");
    }
    Ok(())
}

fn validate_csr_layout(
    n_obs: usize,
    n_var: usize,
    data_len: usize,
    indices: &[usize],
    indptr: &[usize],
) -> Result<()> {
    if indptr.len() != n_obs + 1 {
        bail!(
            "X/indptr length ({}) does not match n_obs + 1 ({})",
            indptr.len(),
            n_obs + 1
        );
    }
    if indptr.first().copied() != Some(0) {
        bail!("X/indptr must start at 0");
    }
    if indices.len() != data_len {
        bail!(
            "X/indices length ({}) does not match X/data length ({data_len})",
            indices.len()
        );
    }
    if indptr[indptr.len() - 1] != data_len {
        bail!(
            "X/indptr terminal value ({}) does not match X/data length ({data_len})",
            indptr[indptr.len() - 1]
        );
    }
    for window in indptr.windows(2) {
        if window[0] > window[1] {
            bail!("X/indptr must be monotonically non-decreasing");
        }
        if window[1] > data_len {
            bail!(
                "X/indptr contains value {} beyond X/data length ({data_len})",
                window[1]
            );
        }
    }
    for &col in indices {
        if col >= n_var {
            bail!("X/indices contains column {col} outside matrix width ({n_var})");
        }
    }
    Ok(())
}

fn checked_i32_to_usize(value: i32) -> Result<usize> {
    if value < 0 {
        bail!("integer dataset contains negative value {value}");
    }
    Ok(value as usize)
}

fn checked_i64_to_usize(value: i64) -> Result<usize> {
    if value < 0 {
        bail!("integer dataset contains negative value {value}");
    }
    Ok(value as usize)
}

fn parse_shape_pair_i64(values: &[i64]) -> Result<(usize, usize)> {
    if values.len() != 2 {
        bail!("X/shape attribute must have exactly 2 elements");
    }
    Ok((
        checked_i64_to_usize(values[0])?,
        checked_i64_to_usize(values[1])?,
    ))
}

fn parse_shape_pair_i32(values: &[i32]) -> Result<(usize, usize)> {
    if values.len() != 2 {
        bail!("X/shape attribute must have exactly 2 elements");
    }
    Ok((
        checked_i32_to_usize(values[0])?,
        checked_i32_to_usize(values[1])?,
    ))
}

fn parse_shape_pair_u64(values: &[u64]) -> Result<(usize, usize)> {
    if values.len() != 2 {
        bail!("X/shape attribute must have exactly 2 elements");
    }
    Ok((values[0] as usize, values[1] as usize))
}

fn read_category_codes(ds: &hdf5::Dataset) -> Result<Vec<i64>> {
    if let Ok(c) = ds.read_1d::<i8>() {
        return Ok(c.iter().map(|&v| v as i64).collect());
    }
    if let Ok(c) = ds.read_1d::<i16>() {
        return Ok(c.iter().map(|&v| v as i64).collect());
    }
    if let Ok(c) = ds.read_1d::<i32>() {
        return Ok(c.iter().map(|&v| v as i64).collect());
    }
    if let Ok(c) = ds.read_1d::<i64>() {
        return Ok(c.to_vec());
    }
    bail!("categorical codes are not i8, i16, i32, or i64")
}

fn decode_categorical_codes(
    categories: &[String],
    codes: &[i64],
    col: &str,
) -> Result<Vec<String>> {
    codes
        .iter()
        .map(|&code| match code {
            -1 => Ok(String::new()),
            c if c < -1 => bail!("code {c} out of range for obs/{col}"),
            c => categories
                .get(c as usize)
                .cloned()
                .with_context(|| format!("code {c} out of range for obs/{col}")),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{decode_categorical_codes, read_h5ad, validate_csr_layout};
    use hdf5::types::VarLenUnicode;
    use hdf5_metno as hdf5;
    use ndarray::arr2;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::str::FromStr;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn decode_categorical_codes_maps_missing_values_to_empty_strings() {
        let values = decode_categorical_codes(
            &["tumor".to_string(), "stroma".to_string()],
            &[0, -1, 1],
            "sample",
        )
        .unwrap();

        assert_eq!(values, vec!["tumor", "", "stroma"]);
    }

    #[test]
    fn validate_csr_layout_rejects_invalid_indptr_length() {
        let err = validate_csr_layout(2, 3, 2, &[0, 1], &[0, 2]).unwrap_err();
        assert!(err
            .to_string()
            .contains("X/indptr length (2) does not match n_obs + 1 (3)"));
    }

    #[test]
    fn validate_csr_layout_rejects_non_monotonic_indptr() {
        let err = validate_csr_layout(3, 3, 2, &[0, 1], &[0, 2, 1, 2]).unwrap_err();
        assert!(err
            .to_string()
            .contains("X/indptr must be monotonically non-decreasing"));
    }

    #[test]
    fn validate_csr_layout_rejects_terminal_count_mismatch() {
        let err = validate_csr_layout(2, 3, 2, &[0, 1], &[0, 1, 1]).unwrap_err();
        assert!(err
            .to_string()
            .contains("X/indptr terminal value (1) does not match X/data length (2)"));
    }

    #[test]
    fn validate_csr_layout_rejects_out_of_bounds_indices() {
        let err = validate_csr_layout(2, 2, 1, &[2], &[0, 1, 1]).unwrap_err();
        assert!(err
            .to_string()
            .contains("X/indices contains column 2 outside matrix width (2)"));
    }

    #[test]
    fn read_h5ad_rejects_obs_column_row_mismatch() {
        with_temp_path("obs-row-mismatch", |path| {
            let file = create_base_h5ad(path);
            let obs = file.group("obs").unwrap();
            write_string_dataset(&obs, "sample", &["s1"]);
            drop(file);

            let err = match read_h5ad(path, &["sample"], &[], false, false, None, None) {
                Ok(_) => panic!("expected obs row count mismatch"),
                Err(err) => err,
            };
            assert!(err
                .to_string()
                .contains("obs column 'sample' row count (1) does not match obs count (2)"));
        });
    }

    #[test]
    fn read_h5ad_rejects_embedding_row_mismatch() {
        with_temp_path("embedding-row-mismatch", |path| {
            let file = create_base_h5ad(path);
            let obsm = file.group("obsm").unwrap();
            obsm.new_dataset_builder()
                .with_data(&arr2(&[[1.0, 2.0]]))
                .create("X_pca")
                .unwrap();
            drop(file);

            let err = match read_h5ad(path, &[], &["X_pca"], false, false, None, None) {
                Ok(_) => panic!("expected embedding row count mismatch"),
                Err(err) => err,
            };
            assert!(err
                .to_string()
                .contains("obsm/X_pca row count (1) does not match obs count (2)"));
        });
    }

    #[test]
    fn read_h5ad_rejects_expression_row_mismatch() {
        with_temp_path("expression-row-mismatch", |path| {
            let file = create_base_h5ad(path);
            file.new_dataset_builder()
                .with_data(&arr2(&[[1.0f32, 2.0f32]]))
                .create("X")
                .unwrap();
            drop(file);

            let err = match read_h5ad(path, &[], &[], true, false, None, None) {
                Ok(_) => panic!("expected expression row count mismatch"),
                Err(err) => err,
            };
            assert!(err
                .to_string()
                .contains("expression matrix X row count (1) does not match obs count (2)"));
        });
    }

    #[test]
    fn read_h5ad_allows_missing_categorical_codes() {
        with_temp_path("categorical-missing", |path| {
            let file = create_base_h5ad(path);
            let obs = file.group("obs").unwrap();
            let sample = obs.create_group("sample").unwrap();
            sample
                .new_dataset_builder()
                .with_data(&[0i32, -1i32])
                .create("codes")
                .unwrap();
            write_string_dataset(&sample, "categories", &["tumor"]);
            drop(file);

            let adata = read_h5ad(path, &["sample"], &[], false, false, None, None).unwrap();
            assert_eq!(
                adata.obs["sample"],
                vec!["tumor".to_string(), String::new()]
            );
        });
    }

    fn create_base_h5ad(path: &Path) -> hdf5::File {
        let file = hdf5::File::create(path).unwrap();

        let obs = file.create_group("obs").unwrap();
        write_string_dataset(&obs, "_index", &["cell1", "cell2"]);

        let var = file.create_group("var").unwrap();
        write_string_dataset(&var, "_index", &["gene1", "gene2"]);

        let obsm = file.create_group("obsm").unwrap();
        obsm.new_dataset_builder()
            .with_data(&arr2(&[[0.0f64, 0.0f64], [1.0f64, 1.0f64]]))
            .create("spatial")
            .unwrap();

        file
    }

    fn write_string_dataset(parent: &hdf5::Group, name: &str, values: &[&str]) {
        let encoded: Vec<VarLenUnicode> = values
            .iter()
            .map(|value| VarLenUnicode::from_str(value).unwrap())
            .collect();
        parent
            .new_dataset_builder()
            .with_data(&encoded)
            .create(name)
            .unwrap();
    }

    fn with_temp_path(name: &str, f: impl FnOnce(&Path)) {
        let path = temp_path(name);
        f(&path);
        let _ = fs::remove_file(&path);
    }

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("spatialrs-{name}-{}-{nanos}.h5ad", process::id()))
    }
}
