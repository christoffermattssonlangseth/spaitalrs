# spatialrs

A fast command-line toolkit for spatial transcriptomics analysis, written in Rust.

Reads `.h5ad` files and provides graph construction, interaction counting, composition profiling, NMF factorization, spatial aggregation, GMM niche detection, Moran's I (global + permutation p-values), Local Moran's I (LISA), bivariate Moran's I, Geary's C, spatially variable genes (SVG), graph statistics (QC), Ripley's K/L, cross-Ripley K/L, niche transition matrices with permutation tests, niche abundance differential testing, spatial entropy, niche summaries, differential composition, and pooled niche marker detection — all parallelized with Rayon.

---

## Crates

| Crate | Role |
|---|---|
| `spatialrs-io` | HDF5 reader: obs/var names, spatial coords, obsm embeddings, expression matrix X |
| `spatialrs-core` | Algorithms: neighbors, interactions, composition, NMF, aggregation, GMM, Moran's I, LISA, bivariate Moran's I, Ripley's K/L, cross-Ripley, transitions, permutation tests, diff-composition, diff-niches, markers |
| `spatialrs-cli` | Binary: CLI wiring with Clap |

---

## Building

```bash
cargo build --release
# binary at target/release/spatialrs
```

Requires a system HDF5 library (the `hdf5-metno` crate links against it).

---

## Subcommands

### `radius` — radius-based neighbour graph

```bash
spatialrs radius data.h5ad --radius 50 --groupby sample --output edges.csv
```

Output columns: `cell_i, cell_j, distance, group`
Edges are bidirectional (both i→j and j→i).

---

### `knn` — k-nearest-neighbour graph

```bash
spatialrs knn data.h5ad --k 6 --groupby sample --output edges.csv
```

Output columns: `cell_i, cell_j, distance, group`

---

### `interactions` — cell-type pair interaction counts

```bash
spatialrs interactions data.h5ad \
    --cell-type cell_type \
    --radius 50 \
    --groupby sample \
    --output interactions.csv
```

Counts co-localised cell-type pairs within `radius`. Each pair is counted once (canonical alphabetical order). Output columns: `group, cell_type_a, cell_type_b, count`

Add `--output-stats stats.csv` to also run a permutation test (`n_permutations`, default 1000) and write enrichment statistics: `observed, expected_mean, expected_std, z_score, p_value`.

---

### `composition` — per-cell neighbourhood composition

```bash
spatialrs composition data.h5ad \
    --cell-type cell_type \
    --radius 50 \
    --groupby sample \
    --output composition.csv
```

For each cell, computes the fraction of each cell type among its neighbours within `radius`. Isolated cells are omitted. Output columns: `cell_i, cell_type, fraction, group`

Add `--output-entropy entropy.csv` to also write per-cell Shannon entropy (bits) of the neighbourhood composition:

```bash
spatialrs composition data.h5ad \
    --cell-type cell_type \
    --radius 50 \
    --groupby sample \
    --output composition.csv \
    --output-entropy entropy.csv
```

Output columns for entropy: `cell_i, entropy, group`

---

### `nmf` — non-negative matrix factorization

Factorizes the gene expression matrix X (cells × genes) into W (cell factors) and H (gene loadings) using multiplicative update rules (Lee & Seung 2001).

**All cells are factorized together** so that component indices are directly comparable across samples. `--groupby` is used only to label output records, not to split the factorization.

```bash
spatialrs nmf data.h5ad --n-components 20 --groupby sample \
    --output-w w_factors.csv \
    --output-h h_loadings.csv \
    --output-diagnostics diagnostics.csv
```

| Flag | Default | Description |
|---|---|---|
| `--n-components` | 10 | Number of NMF components |
| `--max-iter` | 200 | Maximum multiplicative update iterations |
| `--tol` | 1e-4 | Convergence tolerance on Frobenius error change |
| `--seed` | 42 | RNG seed for reproducible initialisation |
| `--groupby` | *(none)* | Optional obs column used to label output records |
| `--var-filter` | *(none)* | Obs column to filter genes (keeps `True` rows) |
| `--layer` | *(none)* | Load from `layers/<name>` instead of `X` |
| `--sparse` | false | Use sparse CSR NMF (recommended for >200k cells) |
| `--output-w` | stdout | W matrix CSV (long format: `cell_i, component, weight, group`) |
| `--output-h` | *(skip)* | H matrix CSV (long format: `gene, component, loading, group`) |
| `--output-diagnostics` | *(skip)* | Per-component explained variance + error trajectory CSV (`metric, index, value`) |

The diagnostics CSV contains two record types:
- `metric="component_variance"` — fraction of `‖WH‖_F` explained by each component
- `metric="iteration_error"` — Frobenius error (or relative H change for sparse) at each checkpoint

---

### `aggregate` — distance-weighted spatial aggregation

For each cell, computes a weighted average of its neighbours' embedding vectors (e.g. `X_scVI`).

```bash
spatialrs aggregate data.h5ad \
    --embedding X_scVI \
    --radius 30 \
    --weighting gaussian \
    --sigma 15 \
    --groupby sample \
    --output agg.csv
```

**Graph mode** (pick one):

| Flag | Description |
|---|---|
| `--radius <f64>` | All neighbours within this distance |
| `--k <usize>` | k nearest neighbours |

**Weighting modes:**

| `--weighting` | Additional flag | Formula |
|---|---|---|
| `uniform` | — | w = 1 |
| `gaussian` | `--sigma <f64>` | w = exp(−d² / 2σ²) |
| `exponential` | `--decay <f64>` | w = exp(−λd) |
| `inverse-distance` | `--epsilon <f64>` (default 1e-9) | w = 1 / (d + ε) |

Output columns: `cell_i, dim, value, group` (long format, one row per cell × embedding dimension).
Cells with no neighbours within the search radius emit zeros.

---

### `gmm` — Gaussian Mixture Model niche detection

Fits a GMM to an embedding to identify spatial compartments / niches.

**All cells are clustered together** so that niche IDs are consistent across samples. `--groupby` labels output records only.

#### Recommended pipeline: NMF → aggregate → GMM

```bash
# 1. Factorize gene expression across all cells (pooled)
spatialrs nmf data.h5ad --n-components 20 --groupby sample --output-w /tmp/w.csv --output-h /tmp/h.csv

# 2. Aggregate spatial neighbourhoods using NMF factors (per sample — spatial coords don't mix)
spatialrs aggregate data.h5ad --nmf-w /tmp/w.csv --radius 75 --weighting gaussian --sigma 37 --groupby sample --output /tmp/agg.csv

# 3. Cluster aggregated embeddings into niches (pooled)
spatialrs gmm data.h5ad --agg /tmp/agg.csv -k 10 --groupby sample --output /tmp/niches.csv --output-probs /tmp/niches_probs.csv
```

You can also feed a raw obsm embedding or skip straight to GMM on the NMF factors:

```bash
spatialrs gmm data.h5ad --embedding X_scVI -k 10 --groupby sample --output niches.csv
spatialrs gmm data.h5ad --nmf-w w_factors.csv -k 10 --groupby sample --output niches.csv
```

**Embedding source** (exactly one required):

| Flag | Description |
|---|---|
| `--embedding <obsm_key>` | Load directly from obsm |
| `--nmf-w <path>` | Load from NMF W factors CSV |
| `--agg <path>` | Load from aggregation CSV (output of `spatialrs aggregate`) |

**GMM parameters:**

| Flag | Default | Description |
|---|---|---|
| `-k / --k` | *(required)* | Number of mixture components (niches) |
| `--covariance` | `diagonal` | `diagonal` or `spherical` covariance |
| `--max-iter` | 200 | Maximum EM iterations |
| `--tol` | 1e-6 | Convergence tolerance on log-likelihood change |
| `--seed` | 42 | RNG seed for K-means++ initialisation |
| `--reg-covar` | 1e-6 | Regularisation added to variance to prevent singularity |
| `--groupby` | `sample` | Obs column used to label output records |

**Outputs:**
- `--output` — hard assignments: `cell_i, niche, group`
- `--output-probs` *(optional)* — soft responsibilities: `cell_i, component, probability, group`
- `--output-model-stats` *(optional)* — fit statistics: `k, log_likelihood, bic, aic, n_iter, covariance_type, group`

---

### `gmm-sweep` — GMM BIC/AIC curve over a range of k

Runs GMM for every k in `[k_min, k_max]` (step `k_step`) and outputs model-fit statistics so you can select the optimal number of niches from the BIC/AIC elbow.

```bash
spatialrs gmm-sweep data.h5ad \
    --agg agg.csv \
    --k-min 2 --k-max 20 \
    --groupby sample \
    --output sweep.csv
```

Accepts the same `--embedding / --nmf-w / --agg` and GMM parameter flags as `gmm` (except `-k`).

Output columns: `k, log_likelihood, bic, aic, n_iter, covariance_type, group`

---

### `morans` — Moran's I over embedding dimensions or NMF components

Computes global Moran's I within each sample/group for either an `obsm` embedding or NMF W factors.

```bash
spatialrs morans data.h5ad --nmf-w w_factors.csv --radius 75 --groupby sample --output morans.csv
```

Output columns: `feature, moran_i, expected_i, variance_i, z_score, group`

Add `--n-permutations <N>` and `--output-perm perm.csv` to also write permutation-based p-values:

```bash
spatialrs morans data.h5ad --nmf-w w_factors.csv --radius 75 --groupby sample \
    --output morans.csv \
    --n-permutations 999 --output-perm morans_perm.csv
```

Output columns for permutation file: `feature, p_value_perm, group`

---

### `geary` — Geary's C spatial autocorrelation

Geary's C is a complement to Moran's I, more sensitive to local spatial dissimilarity.

```bash
spatialrs geary data.h5ad --nmf-w w_factors.csv --radius 75 --groupby sample --output geary.csv
spatialrs geary data.h5ad --embedding X_scVI --radius 50 --groupby sample --output geary.csv
```

```
C = ((n−1) / S₀) × Σ_{edge(i,j)} (zᵢ − zⱼ)² / Σᵢ zᵢ²
```

Under CSR, `expected_c = 1`. C < 1 → positive spatial autocorrelation (similar values cluster). C > 1 → negative spatial autocorrelation. The z-score uses the normality-assumption variance (Cliff & Ord 1981).

Output columns: `feature, geary_c, expected_c, variance_c, z_score, group`

---

### `svg` — spatially variable genes

Runs per-gene Moran's I on the expression matrix to identify genes whose expression is spatially structured.

```bash
spatialrs svg data.h5ad --radius 50 --groupby sample --output svg.csv
spatialrs svg data.h5ad --radius 50 --groupby sample --var-filter highly_variable --output svg.csv
spatialrs svg data.h5ad --radius 50 --groupby sample --layer counts --output svg.csv
```

The same `--var-filter` and `--layer` flags as `nmf` are supported. Outputs the same `MoranRecord` format as `morans`, one row per gene per group.

Output columns: `feature, moran_i, expected_i, variance_i, z_score, group`

---

### `graph-stats` — per-cell neighbourhood count (QC)

For each cell, counts how many other cells fall within `radius`. Useful for QC and for choosing the radius parameter.

```bash
spatialrs graph-stats data.h5ad --radius 50 --groupby sample --output graph_stats.csv
```

Output columns: `cell_i, n_neighbors, group`

---

### `lisa` — Local Moran's I (LISA) per cell

Computes per-cell Local Moran's I for each embedding dimension or NMF component, identifying spatial hot-spots and cold-spots.

```bash
spatialrs lisa data.h5ad --nmf-w w_factors.csv --radius 75 --groupby sample --output lisa.csv
spatialrs lisa data.h5ad --embedding X_scVI --radius 50 --groupby sample --output lisa.csv
```

For each cell i and feature f:

```
I_i = (z_i / m₂) × Σ_{j∈N(i)} z_j
```

The z-score uses the Anselin (1995) conditional-randomisation variance. Positive `local_i` / high `z_score` indicate a spatial cluster (hot-spot); negative `local_i` indicates a spatial outlier.

Output columns: `cell_i, feature, local_i, z_score, group`

---

### `transitions` — niche spatial co-occurrence matrix

Given niche assignments and a radius graph, counts how often each pair of niches is spatially adjacent.

```bash
spatialrs transitions data.h5ad \
    --niche-csv niches.csv \
    --radius 50 \
    --groupby sample \
    --output transitions.csv
```

Returns an upper-triangle K×K matrix (diagonal = same-niche edges). `fraction` is the proportion of all spatial edges occupied by each niche pair.

Output columns: `niche_a, niche_b, count, fraction, group`

Add `--output-stats stats.csv` to also run a permutation test (`--n-permutations`, default 1000) and write enrichment statistics per niche pair:

```bash
spatialrs transitions data.h5ad \
    --niche-csv niches.csv \
    --radius 50 \
    --groupby sample \
    --output transitions.csv \
    --output-stats transition_stats.csv \
    --n-permutations 1000
```

Output columns for stats: `niche_a, niche_b, observed, expected_mean, expected_std, z_score, p_value, group`

---

### `cross-ripley` — bivariate Ripley's K / L function

Estimates the cross-K function K_AB(r) between two cell types over a range of radii. Under CSR, `L_AB(r) ≈ 0`; positive values indicate that type B is more clustered around type A than expected by chance at scale r.

```bash
spatialrs cross-ripley data.h5ad \
    --cell-type cell_type \
    --type-a "T cell" \
    --type-b "Tumor cell" \
    --r-min 10 --r-max 200 --n-radii 20 \
    --groupby sample \
    --output cross_ripley.csv
```

```
K_AB(r) = (A / (nᴬ × nᴮ)) × Σ_{i∈A} Σ_{j∈B} 1(d(i,j) ≤ r)
L_AB(r) = √(K_AB(r) / π) − r
```

Output columns: `type_a, type_b, r, k_cross, l_cross, group`

---

### `bivariate-morans` — bivariate Moran's I

Computes the bivariate spatial autocorrelation statistic I_AB between all pairs of embedding dimensions or NMF components. Positive values indicate co-clustering in space; negative values indicate spatial anticorrelation.

```bash
spatialrs bivariate-morans data.h5ad \
    --nmf-w w_factors.csv \
    --radius 75 \
    --groupby sample \
    --output bivariate_morans.csv
```

Output columns: `feature_a, feature_b, bivariate_i, z_score, group`

---

### `diff-niches` — differential niche abundance test

Compares per-niche abundance (fraction of cells per sample) between two conditions using sample-level Mann–Whitney U tests with BH correction.

```bash
spatialrs diff-niches data.h5ad \
    --niche-csv niches.csv \
    --condition condition \
    --groupby sample \
    --group-a tumor \
    --group-b normal \
    --output diff_niches.csv
```

`--niche-csv` is the output of `spatialrs gmm`. The `group` column in the niche CSV is used as the sample identifier. `--condition` is an obs column that labels each cell's condition; `--groupby` identifies the sample for each cell.

Output columns: `niche, group_a, group_b, n_samples_a, n_samples_b, mean_fraction_a, mean_fraction_b, log2fc, z_score, p_value, q_value_bh`

---

### `summarize` — per-sample niche abundance summary

Computes the fraction of cells in each niche per sample directly from a niche assignment CSV. Useful for quick QC and for feeding into external differential abundance tools.

```bash
spatialrs summarize niches.csv --output niche_summary.csv
```

Output columns: `group, niche, n_cells, fraction`

---

### `diff-composition` — differential neighbourhood composition test

Compares per-cell neighbourhood composition fractions between two conditions using a Mann–Whitney U test.

```bash
spatialrs diff-composition data.h5ad \
    --composition-csv composition.csv \
    --condition condition \
    --group-a tumor \
    --group-b normal \
    --output diff_comp.csv
```

`--composition-csv` is the output of `spatialrs composition`. `--condition` is an obs column in the h5ad that labels each cell's condition. For each cell type, fractions from the two groups are compared and Benjamini–Hochberg FDR correction is applied.

Output columns: `cell_type, group_a, group_b, n_a, n_b, mean_a, mean_b, log2fc, z_score, p_value, q_value_bh`

---

### `ripley` — Ripley's K / L spatial clustering function

Estimates Ripley's K(r) and L(r) for a target cell type over a range of radii, using the bounding-box area as the CSR reference.

```bash
spatialrs ripley data.h5ad \
    --cell-type cell_type \
    --target-type "Tumor cell" \
    --r-min 10 --r-max 200 --n-radii 20 \
    --groupby sample \
    --output ripley.csv
```

```
K(r) = (A / n²) × Σ_{i≠j} 1(d(i,j) ≤ r)
L(r) = √(K(r) / π) − r
```

Under complete spatial randomness (CSR), `L(r) ≈ 0`. Positive values indicate clustering at scale r; negative values indicate regularity.

Output columns: `cell_type, r, k_r, l_r, group`

---

### `markers` — pooled niche marker genes

Reads niche assignments from `spatialrs gmm`, aligns them to `obs_names`, and runs one-vs-rest Wilcoxon tests over the full expression matrix.

```bash
spatialrs markers data.h5ad --niche-csv niches.csv --output markers.csv
```

This command is intentionally **pooled across all cells**. It does not split by sample and its output is not group-labeled.

Output columns: `niche, gene, mean_niche, mean_rest, log2fc, z_score, p_value, q_value_bh`

---

## Input format

Standard `.h5ad` files written by AnnData (Python).

- **Spatial coordinates** read from `obsm/spatial` or `obsm/X_spatial` (first two columns used).
- **Obs columns** read from `obs/{col}` — supports both categorical (codes/categories) and plain string datasets.
- **obsm embeddings** read from `obsm/{key}` — f32 or f64, all columns.
- **Expression matrix X** — supports sparse CSR groups (`X/data`, `X/indices`, `X/indptr`) and dense datasets.

---

## Design notes

- **NMF and GMM run on all cells pooled** so that factor/niche indices are comparable across samples. `--groupby` only labels output records.
- **`markers` also runs pooled** over all cells after aligning niche assignments to `obs_names`, so its output intentionally has no `group` column.
- **`aggregate`, `morans`, `lisa`, `bivariate-morans`, `transitions`, `interactions`, `composition`, `ripley`, `cross-ripley`, and `diff-composition` run per sample** (or per group) because spatial coordinates are sample-local.
- **`diff-niches` operates at the sample level**: it reads per-cell niche assignments and aggregates them to per-sample niche fractions before comparing conditions.
- **`summarize` reads only the niche CSV** (no h5ad needed) and emits per-sample niche fractions directly.
- Groups are processed in parallel with Rayon; spatial indexing uses an R* tree (`rstar`).
- NMF element-wise updates use `ndarray::Zip::par_for_each` across rayon threads.
- All CSV outputs are long format to be compatible with any number of components or embedding dimensions.
- Cells without neighbours in `aggregate` are kept in the output (zeros) to preserve a 1-to-1 join with the input.
