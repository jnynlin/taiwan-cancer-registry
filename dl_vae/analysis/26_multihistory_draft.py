"""
Registry DL — Script 26: Multi-History Transformer Draft PDF

Principal finding: The Transformer's R@1=0.312 (full history, Script 08) is NOT
the result of using sequential cancer history causally. It reflects leave-one-out
training where all cancers (including future ones) are visible as context.

When evaluated in a clinically realistic left-to-right inference scenario
(predict cancer k+1 from cancers 1..k), performance is:
  k=1 (first cancer only):  R@1=0.067  (2.5× random)
  k=2 (first two cancers):  R@1=0.054  (2.0× random)
  k≥3:                       R@1≈0.045  (decreasing)

KEY FINDING — Training/inference mismatch:
  Leave-one-out (BERT-style) training exposes the model to future cancers as
  context at all masked positions. During causal inference, this context is
  unavailable. The model has NOT learned to predict from past alone.

  The model predicts C18 (colon, most common second primary) as top-1 for
  nearly all first-cancer inputs → collapsed to marginal distribution.

  To achieve true clinical sequential prediction, the model needs to be
  retrained with CAUSAL masking (each position only attends to prior positions),
  equivalent to a language model rather than a BERT-style MLM.

Output:
  results/MultiHistory_Draft.pdf
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

BASE = Path(__file__).parent.parent
R08  = BASE / "results/08_transformer_eval"
R12  = BASE / "results/12_surveillance"
R25  = BASE / "results/25_multihistory"
OUT  = BASE / "results/MultiHistory_Draft.pdf"

NAVY   = "#14304a"
ACCENT = "#2e7fbf"
RED    = "#d62728"
GREEN  = "#2ca02c"


def img(ax, path, title=None):
    from matplotlib.image import imread
    p = Path(path)
    if p.exists():
        ax.imshow(imread(str(p))); ax.axis("off")
        if title: ax.set_title(title, fontsize=8, pad=3)
    else:
        ax.text(0.5, 0.5, f"[missing: {p.name}]", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="red"); ax.axis("off")


def footer(ax, page, n=8):
    ax.text(0.99, 0.01, f"Taiwan Cancer Registry — Multi-History Eval  |  p. {page}/{n}",
            transform=ax.transAxes, fontsize=7, color="#888888", ha="right", va="bottom")


def main():
    print("=== Registry DL — 26: Multi-History Draft PDF ===")

    acc = pd.read_csv(R25 / "accuracy_by_context.csv")
    cal = pd.read_csv(R25 / "multihistory_calendar.csv")
    cal12 = pd.read_csv(R12 / "surveillance_calendar.csv")

    # Marginal (most common) baseline
    n_sites = 37
    random_r1 = 1 / n_sites

    with PdfPages(str(OUT)) as pdf:

        # ── Page 1: Title + key finding ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off"); fig.patch.set_facecolor("white")
        ax.add_patch(plt.Rectangle((0,0.88),1,0.12,
                     transform=ax.transAxes, color=NAVY, zorder=0))
        ax.text(0.5, 0.935, "Taiwan Cancer Registry", transform=ax.transAxes,
                ha="center", fontsize=16, color="white", fontweight="bold")
        ax.text(0.5, 0.895, "Multi-History Transformer Extension — Training/Inference Mismatch",
                transform=ax.transAxes, ha="center", fontsize=11, color="white")

        k1_row = acc[acc["context_len"]==1].iloc[0]
        summary = (
            "Motivation\n\n"
            "The cancer sequence Transformer (Script 07-08) achieved R@1=0.312\n"
            "using leave-one-out (BERT-style) masking over the full cancer history.\n"
            "The surveillance calendar (Script 12) used only the first cancer as\n"
            "context, reporting R@1=0.232 — the 'first-only' baseline.\n\n"
            "This analysis extends inference to k=1,2,3+ cancer context lengths\n"
            "to quantify the value of sequential cancer history.\n\n"
            "Principal Finding — Training/Inference Mismatch\n\n"
            f"  k=1 context (first cancer only):     R@1={k1_row['R_at_1']:.3f}  "
            f"({k1_row['R_at_1']/random_r1:.1f}× random)\n"
        )
        for _, row in acc[acc["context_len"]>1].iterrows():
            summary += (f"  k={int(row['context_len'])} context "
                        f"({'two' if row['context_len']==2 else 'three+'} cancers)"
                        f":  R@1={row['R_at_1']:.3f}  ({row['R_at_1']/random_r1:.1f}× random)  "
                        f"n={int(row['n'])}\n")
        summary += (
            "\n"
            "  Accuracy DECREASES with more context — the model does not use\n"
            "  additional cancer history in left-to-right (causal) inference.\n\n"
            "Root Cause\n\n"
            "  Leave-one-out BERT training exposes all cancers (including\n"
            "  future ones) as context at every masked position. During\n"
            "  clinical inference, only PAST cancers are available. The model\n"
            "  learned bidirectional patterns, not causal sequential patterns.\n\n"
            "  The model predicts C18 (colon) as top-1 for virtually all\n"
            "  first-cancer inputs — collapsed to the marginal distribution.\n\n"
            "  The Script 12 result (R@1=0.232) reflects a different val split\n"
            "  from Script 07's training split; Script 25 uses a stricter\n"
            "  reconstruction that reveals the true causal performance.\n\n"
            "Recommendation\n\n"
            "  Retrain with CAUSAL masking (autoregressive / GPT-style):\n"
            "  each position attends only to past positions. This would enable\n"
            "  true left-to-right sequential cancer prediction.\n"
        )
        ax.text(0.07, 0.82, summary, transform=ax.transAxes, fontsize=9,
                va="top", ha="left", fontfamily="monospace",
                bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", pad=8))
        footer(ax, 1)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 2: R@k by context length ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        img(ax, R25/"fig_accuracy_by_context.png",
            "R@k by context length (k=1 to k=6)")
        fig.suptitle("Transformer accuracy by context length — no improvement with more history",
                     fontsize=11, color=NAVY)
        footer(ax, 2)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 3: Context value curve ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        img(ax, R25/"fig_context_value.png",
            "Incremental context value — R@1 vs context length")
        fig.suptitle("Additional cancer history does not improve causal prediction",
                     fontsize=11, color=NAVY)
        footer(ax, 3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 4: Prediction collapse analysis ──────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: pred1 distribution from Script 25 (shows collapse)
        ax_l = axes[0]
        k1_preds = cal[cal["context_len"]==1]["pred1"].value_counts().head(10)
        ax_l.barh(k1_preds.index[::-1], k1_preds.values[::-1],
                  color="#d62728", alpha=0.75)
        ax_l.set_xlabel("N patients")
        ax_l.set_title("Script 25 (k=1 causal): pred1 distribution\n"
                        "(C18 dominant → model collapsed to marginal)", fontsize=9)

        # Right: pred1 distribution from Script 12 (shows variety)
        ax_r = axes[1]
        s12_preds = cal12["pred1"].value_counts().head(10)
        ax_r.barh(s12_preds.index[::-1], s12_preds.values[::-1],
                  color="#2e7fbf", alpha=0.75)
        ax_r.set_xlabel("N patients")
        ax_r.set_title("Script 12 (different val split): pred1 distribution\n"
                        "(C15 dominant, site-specific patterns — val split leakage)",
                        fontsize=9)
        fig.suptitle("Prediction distribution comparison: Script 25 (causal) vs Script 12 (potentially leaked val split)",
                     fontsize=10, color=NAVY)
        footer(ax_r, 4)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 5: Training objective diagram ────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis("off"); fig.patch.set_facecolor("white")

        # Training diagram (leave-one-out / BERT)
        ax.text(0.5, 0.95, "Training vs Inference: The Mismatch", transform=ax.transAxes,
                ha="center", fontsize=13, fontweight="bold", va="top")

        training_txt = (
            "TRAINING (Script 07) — Leave-one-out / BERT-style\n\n"
            "  Sequence:   C06  C15  C13  C06\n"
            "  Mask:       C06  [MASK]  C13  C06   ← random position\n"
            "  Model sees: C06  ???   C13  C06   ← BOTH past AND future\n"
            "  Model learns: bidirectional co-occurrence patterns\n\n"
            "  R@1 (val, full bidirectional context) = 0.312"
        )
        ax.text(0.05, 0.82, training_txt, transform=ax.transAxes,
                fontsize=10, va="top", fontfamily="monospace",
                bbox=dict(facecolor="#e8f4fd", edgecolor="#2e7fbf", pad=8))

        inference_txt = (
            "CAUSAL INFERENCE (what clinicians need) — left-to-right\n\n"
            "  At time of 2nd cancer: know C06\n"
            "  Query:      C06  [MASK]  ___  ___   ← only PAST available\n"
            "  Model sees: C06  ???   <unknown> <unknown>\n"
            "  Must predict WITHOUT future context\n\n"
            "  R@1 (val, first-only causal context) = 0.067  ← 4.7× worse"
        )
        ax.text(0.05, 0.50, inference_txt, transform=ax.transAxes,
                fontsize=10, va="top", fontfamily="monospace",
                bbox=dict(facecolor="#fff0f0", edgecolor="#d62728", pad=8))

        solution_txt = (
            "FIX: Causal / Autoregressive Masking\n\n"
            "  Use a causal attention mask (lower-triangular) during training:\n"
            "    Position k can only attend to positions 1..k\n"
            "    This is GPT-style training, not BERT-style\n\n"
            "  Alternative: Keep BERT training but add causal inference fine-tuning:\n"
            "    Fine-tune with the specific first-cancer-only inference scenario"
        )
        ax.text(0.05, 0.20, solution_txt, transform=ax.transAxes,
                fontsize=10, va="top", fontfamily="monospace",
                bbox=dict(facecolor="#f0fff0", edgecolor="#2ca02c", pad=8))

        footer(ax, 5)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 6: Calibration ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        img(ax, R25/"fig_prob_calibration.png",
            "Probability calibration by context length")
        fig.suptitle("Model probability calibration — causal inference setting",
                     fontsize=11, color=NAVY)
        footer(ax, 6)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 7: Site improvement plot ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        imp_path = R25 / "fig_site_improvement.png"
        img(ax, imp_path, "Per-site R@1: k=2 vs k=1 context")
        fig.suptitle("Per-site accuracy comparison: 2-cancer vs 1-cancer context",
                     fontsize=11, color=NAVY)
        footer(ax, 7)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 8: Recommendations ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off"); fig.patch.set_facecolor("white")
        ax.add_patch(plt.Rectangle((0,0.88),1,0.12,
                     transform=ax.transAxes, color=NAVY, zorder=0))
        ax.text(0.5, 0.935, "Conclusions and Next Steps",
                transform=ax.transAxes, ha="center", fontsize=14,
                color="white", fontweight="bold")
        ax.text(0.5, 0.895,
                "Multi-History Transformer Extension — Taiwan Cancer Registry",
                transform=ax.transAxes, ha="center", fontsize=10, color="white")

        conc = (
            "Conclusions\n\n"
            "1.  The Transformer's R@1=0.312 (Script 08) is NOT a causal\n"
            "    sequential prediction — it is a bidirectional co-occurrence\n"
            "    measure where all cancers serve as context simultaneously.\n\n"
            "2.  In clinically realistic causal inference (predict next from past):\n"
            "    - k=1 context R@1=0.067 (2.5× random, n=975)\n"
            "    - Adding more history does NOT improve performance\n"
            "    - Model predicts C18 (colon) as top-1 for ~40% of patients\n"
            "      regardless of first cancer type → marginal distribution collapse\n\n"
            "3.  The Script 12 surveillance calendar R@1=0.232 reflects a\n"
            "    different val split (810 patients from a differently-filtered\n"
            "    build_sequences reconstruction). The two val sets have only\n"
            "    18% overlap; the discrepancy is attributable to both different\n"
            "    patient selection and potential train-set leakage.\n\n"
            "4.  This is a fundamental training/inference mismatch. The BERT\n"
            "    leave-one-out objective is excellent for learning associations\n"
            "    between cancer sites but does not train the model to predict\n"
            "    forward from an incomplete sequence.\n\n"
            "Recommendations\n\n"
            "  SHORT TERM — Use Transformer as association model only:\n"
            "    • Report co-occurrence OR / SIR (Scripts 15-16) for clinical guidance\n"
            "    • Use site-specific SIR as the clinical 'risk of second cancer' estimate\n"
            "    • Abandon first-only-context prediction as a clinical tool\n\n"
            "  MEDIUM TERM — Retrain with causal masking:\n"
            "    • Replace attention mask with lower-triangular (causal) mask\n"
            "    • Train with objective: given first k cancers, predict k+1\n"
            "    • Expected improvement: R@1 should recover to 0.15-0.25 range\n"
            "    • Evaluate with both first-only and multi-history contexts\n\n"
            "  LONG TERM — Proper generative model:\n"
            "    • Train an autoregressive sequence model (GPT-style) on cancer\n"
            "      sequences sorted by date\n"
            "    • Conditioning on sex, age, birth cohort as prefix tokens\n"
            "    • Sample trajectories for probabilistic clinical planning\n"
            "    • External validation: SEER / UK Biobank multi-primary registry"
        )
        ax.text(0.07, 0.82, conc, transform=ax.transAxes, fontsize=9,
                va="top", ha="left", fontfamily="monospace",
                bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", pad=8))
        footer(ax, 8)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

    size_kb = OUT.stat().st_size / 1024
    print(f"  MultiHistory_Draft.pdf written — {size_kb:.0f} KB")
    print(f"  Path: {OUT}")


if __name__ == "__main__":
    main()
