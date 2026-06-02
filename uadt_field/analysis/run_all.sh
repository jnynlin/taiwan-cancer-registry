#!/bin/bash
# UADT Field Cancerization — Master Pipeline
# Run all 6 scripts sequentially. Stop on any error.
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================"
echo " UADT Field Cancerization Pipeline"
echo " Started: $(date)"
echo "============================================"

echo ""
echo "[1/6] Building cohort…"
python3 01_cohort_build.py

echo ""
echo "[2/6] Co-occurrence pairs…"
python3 02_cooccurrence_pairs.py

echo ""
echo "[3/6] SIR analysis…"
python3 03_sir_field.py

echo ""
echo "[4/6] Trajectory analysis…"
python3 04_trajectories_field.py

echo ""
echo "[5/6] Survival (landmark)…"
python3 05_survival_landmark.py

echo ""
echo "[6/6] Assembling draft PDF…"
python3 06_draft_pdf.py

echo ""
echo "============================================"
echo " Done: $(date)"
echo " Draft PDF: ../results/UADT_Field_Draft.pdf"
echo "============================================"
