#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
REPO="/Users/christoffer/work/karolinska/development/spatialrs"
BIN="${REPO}/target/release/spatialrs"

H5AD="/Users/christoffer/work/karolinska/development/baloMS/data/baloMS_indep_clust_balo_MANA_balo.h5ad"
PROJECT="balo-take2"
OUTDIR="${REPO}/projects/${PROJECT}"

GROUPBY="sample_id"
CELL_TYPE="leiden_1"
EMBEDDING="X_scVI"   # existing obsm key for aggregate/morans direct embedding path
LAYER="counts"       # leave empty to use X
VAR_FILTER=""        # e.g. "highly_variable", leave empty to disable

RADIUS=200
KNN_K=10
NMF_COMPONENTS=20
GMM_K=12
N_PERMUTATIONS=1000
SEED=42

mkdir -p "$OUTDIR"

# =========================
# Outputs
# =========================
RADIUS_EDGES="${OUTDIR}/radius_edges.csv"
KNN_EDGES="${OUTDIR}/knn_edges.csv"
INTERACTIONS="${OUTDIR}/interactions.csv"
INTERACTION_STATS="${OUTDIR}/interaction_stats.csv"
COMPOSITION="${OUTDIR}/composition.csv"

NMF_W="${OUTDIR}/nmf_w.csv"
NMF_H="${OUTDIR}/nmf_h.csv"
AGG="${OUTDIR}/agg.csv"
NICHES="${OUTDIR}/niches.csv"
NICHE_PROBS="${OUTDIR}/niche_probs.csv"
MODEL_STATS="${OUTDIR}/model_stats.csv"
MORANS_NMF="${OUTDIR}/morans_nmf.csv"
MORANS_EMBED="${OUTDIR}/morans_embedding.csv"
MARKERS="${OUTDIR}/markers.csv"

log() {
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_file() {
  local f="$1"
  if [ ! -f "$f" ]; then
    echo "ERROR: missing file: $f" >&2
    exit 1
  fi
}

require_file "$H5AD"

# =========================
# Build
# =========================
log "Building spatialrs"
cargo build --release --manifest-path "${REPO}/Cargo.toml"

# =========================
# Optional inspect
# =========================
log "Inspecting dataset"
python3 - "$H5AD" <<'PYEOF'
import sys
import anndata as ad

adata = ad.read_h5ad(sys.argv[1], backed="r")
print(f"shape   : {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
print(f"obs     : {list(adata.obs.columns)}")
print(f"obsm    : {list(adata.obsm.keys())}")
print(f"layers  : {list(adata.layers.keys()) or '(none)'}")
PYEOF

# =========================
# Graph utilities
# =========================
log "Running radius graph"
"$BIN" radius "$H5AD" \
--radius "$RADIUS" \
--groupby "$GROUPBY" \
--output "$RADIUS_EDGES"

log "Running kNN graph"
"$BIN" knn "$H5AD" \
--k "$KNN_K" \
--groupby "$GROUPBY" \
--output "$KNN_EDGES"

log "Running interactions"
"$BIN" interactions "$H5AD" \
--cell-type "$CELL_TYPE" \
--radius "$RADIUS" \
--groupby "$GROUPBY" \
--n-permutations "$N_PERMUTATIONS" \
--seed "$SEED" \
--output "$INTERACTIONS" \
--output-stats "$INTERACTION_STATS"

log "Running composition"
"$BIN" composition "$H5AD" \
--cell-type "$CELL_TYPE" \
--radius "$RADIUS" \
--groupby "$GROUPBY" \
--output "$COMPOSITION"

# =========================
# Main niche pipeline
# =========================
log "Running NMF"
"$BIN" nmf "$H5AD" \
--n-components "$NMF_COMPONENTS" \
--max-iter 200 \
--tol 1e-4 \
--seed "$SEED" \
--groupby "$GROUPBY" \
${LAYER:+--layer "$LAYER"} \
${VAR_FILTER:+--var-filter "$VAR_FILTER"} \
--output-w "$NMF_W" \
--output-h "$NMF_H"

log "Running aggregation from NMF W"
"$BIN" aggregate "$H5AD" \
--nmf-w "$NMF_W" \
--radius "$RADIUS" \
--weighting gaussian \
--sigma "$((RADIUS / 3))" \
--groupby "$GROUPBY" \
--output "$AGG"

log "Running GMM on aggregated embedding"
"$BIN" gmm "$H5AD" \
--agg "$AGG" \
-k "$GMM_K" \
--covariance diagonal \
--max-iter 200 \
--tol 1e-6 \
--seed "$SEED" \
--reg-covar 1e-6 \
--groupby "$GROUPBY" \
--output "$NICHES" \
--output-probs "$NICHE_PROBS" \
--output-model-stats "$MODEL_STATS"

log "Running Moran's I on NMF W"
"$BIN" morans "$H5AD" \
--nmf-w "$NMF_W" \
--radius "$RADIUS" \
--groupby "$GROUPBY" \
--output "$MORANS_NMF"

log "Running markers"
"$BIN" markers "$H5AD" \
--niche-csv "$NICHES" \
${LAYER:+--layer "$LAYER"} \
--output "$MARKERS"

# =========================
# Direct embedding branches
# =========================
log "Running aggregation from direct obsm embedding"
"$BIN" aggregate "$H5AD" \
--embedding "$EMBEDDING" \
--radius "$RADIUS" \
--weighting gaussian \
--sigma "$((RADIUS / 2))" \
--groupby "$GROUPBY" \
--output "${OUTDIR}/agg_from_${EMBEDDING}.csv"

log "Running Moran's I on direct obsm embedding"
"$BIN" morans "$H5AD" \
--embedding "$EMBEDDING" \
--radius "$RADIUS" \
--groupby "$GROUPBY" \
--output "$MORANS_EMBED"

log "All current spatialrs commands completed"
