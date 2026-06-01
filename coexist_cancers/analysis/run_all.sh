#!/bin/bash
# Co-existing Cancer Pattern Discovery Pipeline
# Taiwan Cancer Registry 78,621 patients, all cancer types
set -euo pipefail
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Co-existing Cancer Patterns — Unsupervised MLL Pipeline        ║"
echo "║  Taiwan Cancer Registry 2006–2020  |  N=78,621 patients         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""; echo "━━ [1/4] Build patient × cancer matrix ━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 01_build_patient_matrix.py 2>/dev/null

echo ""; echo "━━ [2/4] Association rule mining ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 02_association_rules.py 2>/dev/null

echo ""; echo "━━ [3/4] NMF cancer program decomposition ━━━━━━━━━━━━━━━━━━━━━━━"
python3 03_nmf_patterns.py 2>/dev/null

echo ""; echo "━━ [4/5] Autoencoder + UMAP + clustering ━━━━━━━━━━━━━━━━━━━━━━━━"
python3 04_autoencoder_clustering.py 2>/dev/null

echo ""; echo "━━ [5/7] SIR + disease-trajectory mining ━━━━━━━━━━━━━━━━━━━━━━━"
python3 07_sir_trajectories.py 2>/dev/null

echo ""; echo "━━ [6/7] Assemble illustrated report ━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 05_report_pdf.py 2>/dev/null

echo ""; echo "━━ [7/7] Assemble near-submission manuscript ━━━━━━━━━━━━━━━━━━━"
python3 06_manuscript_pdf.py 2>/dev/null

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Pipeline complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Outputs:"
echo "    data/patient_cancer_matrix.csv          (78,578 × 46 sites)"
echo "    results/01_matrix/                      (prevalence + co-occurrence heatmap)"
echo "    results/02_associations/                (association rules, lift, OR, sex/age-stratified)"
echo "    results/03_nmf/                         (cancer programs, patient loadings)"
echo "    results/04_clustering/                  (autoencoder latent space, UMAP, clusters)"
echo "    results/Coexist_Cancers_Report.pdf      (6-page illustrated report)"
echo "    results/Coexist_Cancers_Manuscript.pdf  (8-page near-submission paper)"
echo "    results/cover_letter_CancerEpidemiology.md"
echo "    RESULTS.md                              (full writeup)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
