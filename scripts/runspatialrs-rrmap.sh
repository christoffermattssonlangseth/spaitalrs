#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configuration
# =========================
PROJECT="RRmap"
H5AD="/Volumes/processing2/RRmap/data/RRmap_metadata_fixed_update.h5ad"
RADIUS=100

# Set to the layer name holding the values you want NMF/markers to use,
# e.g. LAYER="counts" -> reads adata.layers['counts']
# Leave empty to use adata.X
LAYER="counts"

OUTDIR="/Users/christoffer/work/karolinska/development/spatialrs/projects/${PROJECT}"
mkdir -p "$OUTDIR"

# =========================
# Step toggles
# Set true/false depending on what you want to run
# =========================
RUN_INSPECT=false
RUN_NMF=false
RUN_AGG=false
RUN_GMM=true
RUN_MARKERS=true

# =========================
# Skip steps automatically if outputs already exist
# Set to false if you want to force rerun enabled steps
# =========================
SKIP_IF_EXISTS=false

# =========================
# Output files
# =========================
NMF_W="${OUTDIR}/nmf_w.csv"
NMF_H="${OUTDIR}/nmf_h.csv"
AGG="${OUTDIR}/agg.csv"
NICHES="${OUTDIR}/niches.csv"
NICHE_PROBS="${OUTDIR}/niche_probs.csv"
MODEL_STATS="${OUTDIR}/model_stats.csv"
MARKERS="${OUTDIR}/markers.csv"

# =========================
# Helpers
# =========================
log() {
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

skip_or_run() {
  local step_name="$1"
  local output_file="$2"

  if [ "$SKIP_IF_EXISTS" = true ] && [ -f "$output_file" ]; then
    log "Skipping ${step_name} (already exists: ${output_file})"
    return 0
  fi

  return 1
}

require_file() {
  local f="$1"
  if [ ! -f "$f" ]; then
    echo "ERROR: Required file not found: $f" >&2
    exit 1
  fi
}

# =========================
# Basic checks
# =========================
require_file "$H5AD"

# =========================
# Inspect H5AD
# =========================
if [ "$RUN_INSPECT" = true ]; then
  log "Inspecting h5ad: $H5AD"
  python3 - "$H5AD" <<'PYEOF'
import sys
import anndata as ad

adata = ad.read_h5ad(sys.argv[1], backed='r')
print(f"  shape    : {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
print(f"  X dtype  : {adata.X.dtype}")

sample = adata.X[:200, :200]
if hasattr(sample, "toarray"):
    sample = sample.toarray()

print(f"  X range  : [{sample.min():.3g}, {sample.max():.3g}]  (first 200×200 block)")
print(f"  layers   : {list(adata.layers.keys()) or '(none)'}")
PYEOF
fi

# =========================
# NMF
# =========================
if [ "$RUN_NMF" = true ]; then
  if ! skip_or_run "NMF" "$NMF_W"; then
    log "Running NMF"
    spatialrs nmf "$H5AD" \
      --n-components 20 \
      --seed 42 \
      --groupby sample_id \
      --sparse \
      ${LAYER:+--layer "$LAYER"} \
      --output-w "$NMF_W" \
      --output-h "$NMF_H"
  fi
else
  log "NMF disabled"
fi

# =========================
# Aggregate
# =========================
if [ "$RUN_AGG" = true ]; then
  require_file "$NMF_W"
  if ! skip_or_run "Aggregate" "$AGG"; then
    log "Running aggregation"
    spatialrs aggregate "$H5AD" \
      --nmf-w "$NMF_W" \
      --radius "$RADIUS" \
      --weighting gaussian \
      --sigma $(( RADIUS / 2 )) \
      --groupby sample_id \
      --output "$AGG"
  fi
else
  log "Aggregate disabled"
fi

# =========================
# GMM
# =========================
if [ "$RUN_GMM" = true ]; then
  require_file "$AGG"
  if ! skip_or_run "GMM" "$NICHES"; then
    log "Running GMM"
    spatialrs gmm "$H5AD" \
      --agg "$AGG" \
      -k 20 \
      --groupby sample_id \
      --output "$NICHES" \
      --output-probs "$NICHE_PROBS" \
      --output-model-stats "$MODEL_STATS"
  fi
else
  log "GMM disabled"
fi

# =========================
# Markers
# =========================
if [ "$RUN_MARKERS" = true ]; then
  require_file "$NICHES"
  if ! skip_or_run "Markers" "$MARKERS"; then
    log "Running markers"
    spatialrs markers "$H5AD" \
      --niche-csv "$NICHES" \
      ${LAYER:+--layer "$LAYER"} \
      --output "$MARKERS"
  fi
else
  log "Markers disabled"
fi

log "Done"