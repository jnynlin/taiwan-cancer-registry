#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# C15 Esophageal Cancer — Full Analysis Pipeline
# Taiwan Cancer Registry 2006–2020  |  Single-center cohort  |  n=2,367
#
# Usage:
#   bash run_all.sh              # full pipeline (all 7 steps)
#   bash run_all.sh --from 03   # resume from step 03 onwards
#   bash run_all.sh --only 07   # run one step only
#
# Steps:
#   01  Extract C15 cases from registry xlsx (requires data/raw/cancer_registry_92-109.xlsx)
#   02  Descriptive analysis — Table 1, incidence trend, figures
#   03  Survival analysis — KM curves, Cox regression
#   04  Deep learning — autoencoder, UMAP, k-means, DeepSurv
#   05  Summary PDF (brief, 5 pages)
#   06  Chemo & surgery impact — regimen, dose, sequence, margins, LN
#   07  Paper PDF (near-submission, 12 pages, 300 DPI)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

FROM_STEP=${1:-"--from"}; FROM_VAL=${2:-"01"}
ONLY_STEP=""
if [[ "${1:-}" == "--only" ]]; then ONLY_STEP="$2"; fi
if [[ "${1:-}" == "--from" ]]; then FROM_VAL="$2"; fi
[[ "${1:-}" != "--from" && "${1:-}" != "--only" ]] && FROM_VAL="01"

run_step() {
    local num="$1"; local label="$2"; local script="$3"
    [[ -n "$ONLY_STEP" && "$ONLY_STEP" != "$num" ]] && return 0
    [[ -z "$ONLY_STEP" && "$num" < "$FROM_VAL" ]] && return 0
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${num}/07]  ${label}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 "$script" 2>/dev/null
    echo "  ✓  Done: ${script}"
}

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   C15 Esophageal Cancer — Analysis Pipeline  (Taiwan Registry)  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"

run_step "01" "Extract C15 cases from registry"           01_extract.py
run_step "02" "Descriptive analysis & Table 1"            02_descriptive.py
run_step "03" "Survival analysis (KM + Cox)"              03_survival.py
run_step "04" "Deep learning (autoencoder / UMAP / DL)"   04_deep_learning.py
run_step "05" "Brief summary PDF (5 pages)"               05_summary_pdf.py
run_step "06" "Chemo & surgery impact"                    06_chemo_surgery_impact.py
run_step "07" "Near-submission paper PDF (12 pages)"      07_paper_pdf.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Pipeline complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Key outputs:"
echo "    results/C15_Paper_Draft.pdf           (12-page near-submission)"
echo "    results/C15_Esophageal_Cancer_Summary.pdf  (5-page brief)"
echo "    results/cover_letter_Cancers.md"
echo "    results/data_quality_feedback.md"
echo "    data/c15_enriched.csv                 (2,367 × 300+ cols)"
echo "    results/04_deep_learning/c15_final_annotated.csv"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
