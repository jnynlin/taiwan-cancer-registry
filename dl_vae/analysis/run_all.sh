#!/usr/bin/env bash
# dl_vae/analysis/run_all.sh — run full VAE pipeline
# Usage: cd taiwan-cancer-registry/dl_vae/analysis && bash run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Taiwan Cancer Registry — VAE Axis Discovery Pipeline ==="
echo "Started: $(date)"
echo ""

cd "$SCRIPT_DIR"

echo "--- 01: Build cancer co-occurrence matrix ---"
python3 01_build_matrix.py

echo ""
echo "--- 02: Train VAE ---"
python3 02_vae_train.py

echo ""
echo "--- 03: Latent space exploration ---"
python3 03_latent_explore.py

echo ""
echo "--- 04: Draft report PDF ---"
python3 04_axis_report.py

echo ""
echo "=== Pipeline complete: $(date) ==="
echo "Output: ../results/VAE_Axes_Draft.pdf"
