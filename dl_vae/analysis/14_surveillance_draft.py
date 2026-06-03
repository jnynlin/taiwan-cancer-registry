"""
Registry DL — Script 14: Sequence Surveillance Draft PDF

8-page draft summarising the surveillance calendar analysis.

Pages:
  1. Title + clinical framing + key numbers
  2. Methodology: first-cancer-only query vs full leave-one-out
  3. Accuracy: R@k overall + by site
  4. Lead time distribution + UADT guideline comparison
  5. Probability calibration
  6. Recommendation heatmap (first site → recommended sites)
  7. Timing windows table (top 20 pairs)
  8. Clinical interpretation + limitations + next steps

Output: results/Surveillance_Draft.pdf
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

BASE = Path(__file__).parent.parent
R12  = BASE / "results/12_surveillance"
R13  = BASE / "results/13_surveillance"
OUT  = BASE / "results"

NAVY   = "#14304a"
ACCENT = "#2e7fbf"
TOTAL  = 8


def footer(fig, page):
    fig.text(0.5, 0.01,
             f"Taiwan Cancer Registry — Sequence-Aware Surveillance Calendar  |  Page {page}/{TOTAL}  |  Draft 2026-06-03",
             ha="center", fontsize=7, color="#888888")


def img(ax, path, title=None):
    p = Path(path)
    if p.exists():
        ax.imshow(imread(str(p)), aspect="equal"); ax.axis("off")
        if title: ax.set_title(title, fontsize=9, pad=3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"[{p.name}]", ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=8)


def flow(ax, text, fontsize=9):
    ax.axis("off")
    ax.text(0.02, 0.96, text, transform=ax.transAxes, fontsize=fontsize,
            va="top", bbox=dict(facecolor="white", edgecolor="none", pad=4))


def main():
    print("=== Registry DL — 14: Surveillance Draft PDF ===")

    cal    = pd.read_csv(R12 / "surveillance_calendar.csv")
    timing = pd.read_csv(R12 / "timing_windows.csv")
    summ   = pd.read_csv(R13 / "validation_summary.csv").iloc[0]

    eval_df = cal[cal["true_second_site"].notna()].copy()
    gaps    = eval_df["actual_gap_days"].dropna()
    n_val   = int(summ["n_val_patients"])
    r1      = float(summ["r_at_1"])
    r3      = float(summ["r_at_3"])
    r5      = float(summ["r_at_5"])
    med_gap = float(summ["median_gap_days"])
    p25_gap = float(summ["p25_gap_days"])
    p75_gap = float(summ["p75_gap_days"])
    pct_6mo = float(summ["pct_within_6mo"])
    n_pairs = int(summ["n_timing_pairs"])

    pdf_path = OUT / "Surveillance_Draft.pdf"
    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: Title ─────────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax_t = fig.add_axes([0.05, 0.55, 0.90, 0.38])
        ax_t.set_facecolor(NAVY); ax_t.axis("off")
        ax_t.text(0.5, 0.65,
                  "Sequence-Aware Cancer Surveillance Calendar\n"
                  "from a Population-Based Registry Transformer",
                  ha="center", va="center", fontsize=17, color="white",
                  fontweight="bold", transform=ax_t.transAxes)
        ax_t.text(0.5, 0.25,
                  "Next-cancer prediction from first diagnosis  ·  84k patients  ·  Taiwan 2003–2020",
                  ha="center", va="center", fontsize=12, color="#aaccee",
                  transform=ax_t.transAxes)

        ax_s = fig.add_axes([0.10, 0.10, 0.80, 0.40])
        ax_s.axis("off")
        txt = (
            f"Validation cohort: {n_val:,} multi-cancer patients (20% held-out)\n\n"
            f"Accuracy — first-cancer-only context:\n"
            f"  R@1 = {r1:.3f}  ·  R@3 = {r3:.3f}  ·  R@5 = {r5:.3f}\n\n"
            f"Lead time (time from first to second cancer):\n"
            f"  Median = {med_gap:.0f} days ({med_gap/365.25:.1f} yr)  ·  IQR = {p25_gap:.0f}–{p75_gap:.0f} days\n\n"
            f"UADT validation: {pct_6mo:.0%} of C12/C13→C15 transitions occur ≤6 months\n"
            f"(consistent with guideline-based 6-month endoscopy)\n\n"
            f"Timing windows calibrated from {n_pairs:,} observed site-pair transitions in training set"
        )
        ax_s.text(0.5, 0.6, txt, ha="center", va="center", fontsize=11,
                  color=NAVY, transform=ax_s.transAxes,
                  bbox=dict(facecolor="#f0f4f8", edgecolor=ACCENT, boxstyle="round,pad=0.6"))
        footer(fig, 1)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 2: Methodology ───────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax  = fig.add_axes([0.05, 0.05, 0.90, 0.88])
        ax.axis("off")
        meth = (
            "Clinical framing:\n"
            "  A patient presents with cancer at site A on date T₀.\n"
            "  The model receives [CLS, A, MASK] and predicts the site token at MASK.\n"
            "  Only the index cancer is given as context — no future information used.\n\n"
            "Why first-cancer-only context?\n"
            "  • Clinically realistic: at first diagnosis, no subsequent cancers are known.\n"
            "  • Tests whether the Transformer has learned a meaningful site-to-site\n"
            "    co-occurrence prior beyond static frequency.\n"
            "  • R@1=0.232 vs leave-one-out R@1=0.312 — the gap quantifies what is\n"
            "    learned from temporal sequence context (Scripts 07–08).\n\n"
            "How timing windows are generated:\n"
            "  • From training-set multi-cancer patients, compute actual day-gaps\n"
            "    between consecutive cancer pairs (first→second, second→third, …).\n"
            "  • For each (A→B) pair: report empirical Q1/median/Q3.\n"
            "  • Report: 'Recommend surveillance for [B] at [Q1–Q3] days after [A] diagnosis.'\n"
            "  • For pairs not seen in training: use overall median/IQR.\n\n"
            "Validation design:\n"
            "  • Same 20% patient-level val split as Transformer training (SEED=42).\n"
            "  • Evaluate: does top-1/3/5 predicted site match actual second cancer?\n"
            "  • Lead time: actual gap days — this is the window in which surveillance\n"
            "    would have an opportunity to detect the second cancer.\n\n"
            "Comparison to UADT guideline:\n"
            "  Current guidelines (Muto 2004, Huang 2015) recommend synchronous\n"
            "  esophagoscopy at HNC diagnosis. The model timing window is compared\n"
            "  against this 6-month anchor for C12/C13→C15 pairs."
        )
        ax.text(0.05, 0.95, meth, transform=ax.transAxes, fontsize=10,
                va="top", color=NAVY,
                bbox=dict(facecolor="#f0f4f8", edgecolor=ACCENT, boxstyle="round,pad=0.7"))
        fig.suptitle("Methodology", fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 2)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 3: R@k by site ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 7))
        img(ax, R13 / "fig_recall_by_site.png",
            f"Fig 1: Next-cancer R@1 and R@3 by first-cancer site (overall R@1={r1:.3f}, R@3={r3:.3f})")
        fig.suptitle("Surveillance Accuracy by Primary Site", fontsize=13,
                     color=NAVY, fontweight="bold")
        footer(fig, 3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 4: Lead time + UADT guideline ───────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        img(axes[0], R13 / "fig_lead_time.png",
            f"Fig 2a: Lead time distribution\n(median={med_gap:.0f}d = {med_gap/365.25:.1f}yr, {pct_6mo:.0%} within 6 months)")
        img(axes[1], R13 / "fig_uadt_timing.png",
            "Fig 2b: UADT timing — C12/C13→C15\nvs 6-month guideline (red dashed)")
        fig.suptitle("Lead Time and Guideline Comparison", fontsize=13,
                     color=NAVY, fontweight="bold")
        footer(fig, 4)
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 5: Calibration ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 6.5))
        img(ax, R13 / "fig_calibration.png",
            "Fig 3: Probability calibration — model prob1 vs observed R@1\n"
            "(bubble size = n patients in decile)")
        ax_note = fig.add_axes([0.05, 0.02, 0.90, 0.14])
        flow(ax_note,
             "Note: all 810 validation patients received prob1 > 0.15 (model is uniformly high-confidence). "
             "This reflects the small vocabulary (~37 sites): after zeroing special tokens and the primary site, "
             "probability concentrates on ~35 targets. Risk stratification should use relative thresholds "
             "(e.g., tertiles of prob1) rather than absolute cutoffs. "
             "The calibration plot shows whether the model's relative confidence tracks observed accuracy.",
             fontsize=8.5)
        fig.suptitle("Probability Calibration", fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 5)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 6: Recommendation heatmap ───────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 7.5))
        img(ax, R13 / "fig_recommendation_heatmap.png",
            "Fig 4: Surveillance recommendation heatmap\n"
            "(for each first-cancer site, fraction of patients where target appears in top-3)")
        fig.suptitle("Site-to-Site Recommendation Map", fontsize=13,
                     color=NAVY, fontweight="bold")
        footer(fig, 6)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 7: Timing windows table ──────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax  = fig.add_axes([0.05, 0.15, 0.90, 0.78])
        ax.axis("off")
        top_timing = (timing[timing["n_pairs"] >= 10]
                      .sort_values("n_pairs", ascending=False)
                      .head(20)
                      .copy())
        top_timing["gap_p25_mo"]    = (top_timing["gap_p25"] / 30.44).round(1)
        top_timing["gap_median_mo"] = (top_timing["gap_median"] / 30.44).round(1)
        top_timing["gap_p75_mo"]    = (top_timing["gap_p75"] / 30.44).round(1)
        show = top_timing[["first_site","second_site","n_pairs",
                            "gap_p25_mo","gap_median_mo","gap_p75_mo"]].copy()
        show.columns = ["Index site","Target site","N pairs",
                        "Q1 (months)","Median (months)","Q3 (months)"]
        tbl = ax.table(cellText=show.values, colLabels=show.columns,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
        tbl.scale(1, 1.5)
        ax.set_title("Table 1: Empirical surveillance timing windows\n"
                     "(top 20 pairs by N, ≥10 observed transitions)", fontsize=10, pad=6)
        ax_note2 = fig.add_axes([0.05, 0.03, 0.90, 0.10])
        flow(ax_note2,
             "Timing window interpretation: 'Recommend screening for [Target] starting at [Q1] months "
             "and continuing through [Q3] months after [Index] diagnosis.' "
             "Pairs with Q1≈0 include synchronous presentations (≤6 months) — these should trigger "
             "immediate workup at the time of index diagnosis.", fontsize=8.5)
        fig.suptitle("Surveillance Timing Windows", fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 7)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 8: Interpretation + limitations ──────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax  = fig.add_axes([0.05, 0.05, 0.90, 0.88])
        ax.axis("off")
        interp = (
            "Key findings:\n"
            f"  1. First-cancer-only R@1={r1:.3f} (8.6× random chance, 1/37=0.027). The Transformer\n"
            "     has learned site-level co-occurrence priors that generalise to unseen patients.\n\n"
            f"  2. Median inter-cancer gap = {med_gap:.0f}d ({med_gap/365.25:.1f}yr), IQR {p25_gap:.0f}–{p75_gap:.0f}d.\n"
            "     This is the window in which surveillance has an opportunity to act. Most second\n"
            "     cancers occur beyond the 6-month guideline window — supporting extended protocols.\n\n"
            f"  3. UADT validation: {pct_6mo:.0%} of C12/C13→C15 transitions occur ≤6 months.\n"
            "     The model's 0–15 month timing window encompasses this high-density early window\n"
            "     and extends it, capturing the 26% of late-presenting cases.\n\n"
            "  4. The recommendation heatmap reveals non-UADT pairs of clinical interest:\n"
            "     liver (C22) appears in top-3 for multiple index sites — consistent with the\n"
            "     HBV pan-carcinogen signal identified in the masked predictor (Script 02b).\n\n"
            "Limitations:\n"
            "  ① First-cancer-only context does not use temporal history of ≥2 prior cancers —\n"
            "    R@1 would increase to 0.312 with full history (Script 07).\n"
            "  ② Timing windows derived from observed transitions, not from the model — the\n"
            "    Transformer does not yet directly predict time-to-event.\n"
            "  ③ Validation is retrospective; prospective RCT or quasi-experimental design\n"
            "    with EHR integration required to demonstrate clinical utility.\n"
            "  ④ Risk calibration: all patients receive high-confidence predictions (prob1>0.15)\n"
            "    due to small vocab; relative tertile thresholds recommended.\n\n"
            "Next steps:\n"
            "  → Extend Transformer to jointly predict site AND time-to-event (add regression head).\n"
            "  → Prospective validation: embed calendar into EHR; compare endoscopy yield vs\n"
            "    standard-of-care over 2 years.\n"
            "  → Combine cluster membership (Script 10) as a conditioning token to personalise\n"
            "    recommendations by carcinogenic axis."
        )
        ax.text(0.04, 0.97, interp, transform=ax.transAxes, fontsize=9.5,
                va="top", color=NAVY,
                bbox=dict(facecolor="#f0f4f8", edgecolor=ACCENT, boxstyle="round,pad=0.6"))
        fig.suptitle("Clinical Interpretation + Limitations + Next Steps",
                     fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 8)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

    print(f"  Saved → {pdf_path}  ({TOTAL} pages)")


if __name__ == "__main__":
    main()
