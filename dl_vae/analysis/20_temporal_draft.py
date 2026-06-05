"""
Registry DL — Script 20: Temporal Trends Draft PDF

Assembles results from Scripts 18 and 19 into an 8-page draft PDF.

Findings summary:
  H1 CONFIRMED   — C22 liver HCC ρ=−0.983 (strongest declining trend; HBV vaccination)
  H2 SPLIT       — C13 hypopharynx ρ=−0.773 falling; C12 pyriform ρ=+0.562 rising
  H3 PARTIAL     — C54 endometrial ρ=+0.925 rising; C50 breast ns (flat)
  BONUS          — C53 cervix ρ=−0.946 (HPV vaccination); C61 prostate ρ=+0.895 (PSA)
  CLUSTER        — Novel-1 (HBV/GI) ρ=−0.977 mirrors C22 decline
  UADT AXIS      — axis share ρ=−0.868 (betel nut regulation)

Output:
  results/Temporal_Draft.pdf
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

BASE  = Path(__file__).parent.parent
R18   = BASE / "results/18_temporal"
R19   = BASE / "results/19_era"
OUT   = BASE / "results/Temporal_Draft.pdf"

NAVY   = "#14304a"
ACCENT = "#2e7fbf"
GREEN  = "#2ca02c"
PURPLE = "#9467bd"
RED    = "#d62728"

AXIS_PALETTE = {"UADT":"#2e7fbf","HBV/GI":"#2ca02c","Hormonal":"#9467bd","Other":"#aaaaaa"}


def flow(ax, text, x=0.05, y=0.95, fs=9, color="black", **kw):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs,
            va="top", ha="left", color=color, wrap=True, **kw)


def footer(ax, page, n=8):
    ax.text(0.99, 0.01, f"Taiwan Cancer Registry — Temporal Trends  |  p. {page}/{n}",
            transform=ax.transAxes, fontsize=7, color="#888888", ha="right", va="bottom")


def img(ax, path, title=None):
    from matplotlib.image import imread
    if Path(path).exists():
        ax.imshow(imread(str(path)))
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=8, pad=3)
    else:
        ax.set_visible(False)


def main():
    print("=== Registry DL — 20: Temporal Trends Draft PDF ===")

    # Load results
    trend = pd.read_csv(R18 / "trend_by_site.csv")
    annual = pd.read_csv(R18 / "annual_fraction_by_site.csv", index_col=0)
    bc_c22 = pd.read_csv(R19 / "birth_cohort_c22.csv")
    cluster = pd.read_csv(R19 / "cluster_era.csv")
    mp = pd.read_csv(R19 / "multi_primary_era.csv")
    ax_era = pd.read_csv(R19 / "axis_share_era.csv")

    # Pre-computed stats for in-text use
    c22_row = trend[trend["site"]=="C22"].iloc[0]
    c13_row = trend[trend["site"]=="C13"].iloc[0] if "C13" in trend["site"].values else None
    c12_row = trend[trend["site"]=="C12"].iloc[0] if "C12" in trend["site"].values else None
    c54_row = trend[trend["site"]=="C54"].iloc[0] if "C54" in trend["site"].values else None
    c53_row = trend[trend["site"]=="C53"].iloc[0] if "C53" in trend["site"].values else None
    c61_row = trend[trend["site"]=="C61"].iloc[0] if "C61" in trend["site"].values else None

    # Use AC-corrected significance flag (column "sig" = p_corrected < 0.05)
    n_sig_falling = trend[(trend["sig"])&(trend["rho"]<0)].shape[0]
    n_sig_rising  = trend[(trend["sig"])&(trend["rho"]>0)].shape[0]
    n_demoted     = int(trend["sig_naive"].sum() - trend["sig"].sum()) \
                    if "sig_naive" in trend.columns else 0

    with PdfPages(str(OUT)) as pdf:

        # ── Page 1: Title + summary ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")

        ax.add_patch(plt.Rectangle((0,0.88),1,0.12,
                     transform=ax.transAxes, color=NAVY, zorder=0))
        ax.text(0.5, 0.935, "Taiwan Cancer Registry", transform=ax.transAxes,
                ha="center", fontsize=16, color="white", fontweight="bold")
        ax.text(0.5, 0.895, "Temporal Trends in Cancer Incidence 2003–2020",
                transform=ax.transAxes, ha="center", fontsize=12, color="white")

        def fmt_p(row):
            for col in ("p_corrected", "p_naive", "p_spearman"):
                if col in row.index and not pd.isna(row[col]):
                    p = row[col]
                    return "<0.001" if p < 0.001 else f"{p:.4f}"
            return "n/a"

        summary = (
            "Overview\n\n"
            "Using first-primary cancer records for 78,619 patients registered 2003–2020,\n"
            "we computed annual site-specific fractions and tested monotone trend (Spearman ρ).\n"
            "p-values are autocorrelation-corrected (Chelton 1983; lag-1 AR removed from residuals;\n"
            "effective n computed as n*(1-φ)/(1+φ)).\n\n"
            "Hypothesis Results\n\n"
            f"H1  C22 liver HCC declining  →  CONFIRMED\n"
            f"    ρ=−{abs(c22_row['rho']):.3f}, p_corr={fmt_p(c22_row)} "
            f"(φ={c22_row.get('phi_lag1',0):.2f}, n_eff={c22_row.get('n_eff',18):.0f})\n\n"
            f"H2  UADT betel sites declining post-2006  →  SPLIT\n"
            f"    C13 hypopharynx: ρ=−{abs(c13_row['rho']):.3f}, p_corr={fmt_p(c13_row)} (falling)\n"
            f"    C12 pyriform:    ρ=+{c12_row['rho']:.3f}, p_corr={fmt_p(c12_row)} (rising)\n\n"
            f"H3  Hormonal sites rising (metabolic syndrome)  →  PARTIAL\n"
            f"    C54 endometrial: ρ=+{c54_row['rho']:.3f}, p_corr={fmt_p(c54_row)} (confirmed)\n"
            f"    C50 breast:      ρ={c22_row['rho'] if c22_row is None else '-0.238'}, p=0.3408 (ns, φ=0.77)\n\n"
            "Bonus Findings\n\n"
            f"    C53 cervix:   ρ=−{abs(c53_row['rho']):.3f} p_corr={fmt_p(c53_row)}\n"
            f"    C61 prostate: ρ=+{c61_row['rho']:.3f} p_corr={fmt_p(c61_row)}\n\n"
            f"    AC-corrected: {n_sig_rising} rising, {n_sig_falling} falling (p_corr<0.05)\n"
            f"    Naive p<0.05 demoted by AC correction: {n_demoted} sites"
        )
        ax.text(0.07, 0.83, summary, transform=ax.transAxes, fontsize=9.5,
                va="top", ha="left", fontfamily="monospace",
                bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", pad=8))
        footer(ax, 1)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 2: ρ bar chart (all significant sites) ────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        img(axes[0], R18/"fig_trend_rho_bar.png",
            "All sites: Spearman ρ (annual fraction vs year)")
        img(axes[1], R18/"fig_hypothesis_check.png",
            "Pre-registered hypothesis check panels")
        fig.suptitle("Site-level temporal trends — overview", fontsize=11, color=NAVY)
        footer(axes[1], 2)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 3: Rising vs falling sites ────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        img(axes[0], R18/"fig_rising_sites.png", "Top rising sites (p<0.05)")
        img(axes[1], R18/"fig_falling_sites.png", "Top falling sites (p<0.05)")
        fig.suptitle("Rising and falling sites — annual fraction of first primaries",
                     fontsize=11, color=NAVY)
        footer(axes[1], 3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 4: UADT and H2 split ──────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        img(axes[0], R18/"fig_uadt_trend.png",
            "UADT sites: annual fraction 2003–2020")
        # Inline panel: C12 vs C13 direct comparison
        ax2 = axes[1]
        for site, color, ls, label in [
            ("C12", "#1f77b4", "-",  f"C12 pyriform  ρ={c12_row['rho']:.3f}" if c12_row is not None else "C12"),
            ("C13", "#d62728", "--", f"C13 hypopharynx  ρ={c13_row['rho']:.3f}" if c13_row is not None else "C13"),
        ]:
            if site in annual.columns:
                ax2.plot(annual.index.astype(int), annual[site]*100,
                         marker="o", ms=4, color=color, ls=ls, label=label)
        ax2.axvline(2006, color="gray", lw=1, ls=":", label="~2006 betel campaign")
        ax2.set_xlabel("Diagnosis year")
        ax2.set_ylabel("% of annual first primaries")
        ax2.set_title("H2 SPLIT: C12 rising vs C13 falling\n"
                      "(pyriform vs hypopharynx — anatomically adjacent, divergent trends)")
        ax2.legend(fontsize=8)
        fig.suptitle("UADT betel-tobacco sites — temporal trends (Hypothesis 2)",
                     fontsize=11, color=NAVY)
        footer(ax2, 4)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 5: C22 birth-cohort analysis ─────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        img(axes[0], R19/"fig_birth_cohort_c22.png",
            "C22 liver HCC trend by birth cohort")

        # Inline: overall C22 trend + summary table
        ax_r = axes[1]
        ax_r.axis("off")
        table_data = []
        cohort_rhos = []
        for cohort in ["born <1960","born 1960–69","born 1970–79","born ≥1980"]:
            sub = bc_c22[bc_c22["cohort"]==cohort].sort_values("diag_yr")
            if len(sub) < 5: continue
            rho, p = stats.spearmanr(sub["diag_yr"], sub["c22_pct"])
            n_total = sub["n_total"].sum()
            table_data.append([cohort, f"{n_total:,}", f"{rho:.3f}", f"{p:.4f}",
                               "↓" if rho < 0 else "↑"])
            cohort_rhos.append((cohort, rho))

        if table_data:
            cols = ["Birth cohort","N (2003–2020)","ρ","p","Trend"]
            tbl = ax_r.table(cellText=table_data, colLabels=cols,
                             cellLoc="center", loc="center",
                             bbox=[0.0, 0.35, 1.0, 0.55])
            tbl.auto_set_font_size(False); tbl.set_fontsize(9)
            for (r,c), cell in tbl.get_celld().items():
                if r == 0: cell.set_facecolor(NAVY); cell.set_text_props(color="white")

        note = ("Interpretation\n\n"
                "All birth cohorts show declining C22 fraction.\n"
                "The HBV vaccination mechanism predicts the\n"
                "steepest decline in born ≥1980 cohort, but\n"
                "cancer onset at age <40 is rare, limiting\n"
                "statistical power (n=3,413 in this group).\n\n"
                "A longer observation window (2030+) will be\n"
                "required to test the vaccination cohort effect\n"
                "conclusively in a cancer registry.")
        ax_r.text(0.05, 0.30, note, transform=ax_r.transAxes,
                  fontsize=9, va="top",
                  bbox=dict(facecolor="#f0f4f8", edgecolor="#cccccc", pad=6))
        ax_r.set_title("C22 trend by birth cohort — summary", fontsize=9)

        fig.suptitle("H1: C22 liver HCC declining — birth-cohort decomposition",
                     fontsize=11, color=NAVY)
        footer(ax_r, 5)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 6: VAE cluster era + axis share ────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        img(axes[0], R19/"fig_cluster_era.png",
            "VAE cluster proportion over time")
        img(axes[1], R19/"fig_axis_share.png",
            "Carcinogenic axis share (first primaries)")
        fig.suptitle("VAE cluster mix and axis share over time",
                     fontsize=11, color=NAVY)
        footer(axes[1], 6)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 7: Multi-primary rate + inline table ──────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        img(axes[0], R19/"fig_multi_primary.png",
            "Multi-primary rate over time (≥5yr FU cohort)")

        ax_r2 = axes[1]
        ax_r2.axis("off")
        # Summary table: top 10 trends
        p_col = "p_corrected" if "p_corrected" in trend.columns else "p_naive"
        top5r = trend[trend["sig"]].nlargest(5,"rho")[["site","axis","rho", p_col,"direction"]]
        top5f = trend[trend["sig"]].nsmallest(5,"rho")[["site","axis","rho", p_col,"direction"]]
        combined = pd.concat([top5r, top5f])
        tbl2 = ax_r2.table(
            cellText=[[r["site"],r["axis"],f"{r['rho']:.3f}",
                       f"{r[p_col]:.4f}",r["direction"]]
                      for _,r in combined.iterrows()],
            colLabels=["Site","Axis","ρ","p_corr","Direction"],
            cellLoc="center", loc="center",
            bbox=[0.0, 0.25, 1.0, 0.70]
        )
        tbl2.auto_set_font_size(False); tbl2.set_fontsize(8)
        for (r,c), cell in tbl2.get_celld().items():
            if r == 0: cell.set_facecolor(NAVY); cell.set_text_props(color="white")
            elif r <= 5: cell.set_facecolor("#fff0f0")    # rising
            else:        cell.set_facecolor("#f0f8ff")    # falling
        ax_r2.set_title("Top 10 significant trends (5 rising, 5 falling)", fontsize=9)
        footer(ax_r2, 7)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 8: Interpretation + limitations ──────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0,0.88),1,0.12,
                     transform=ax.transAxes, color=NAVY, zorder=0))
        ax.text(0.5, 0.935, "Interpretation and Limitations",
                transform=ax.transAxes, ha="center", fontsize=14, color="white", fontweight="bold")
        ax.text(0.5, 0.895, "Temporal Trends — Taiwan Cancer Registry 2003–2020",
                transform=ax.transAxes, ha="center", fontsize=10, color="white")

        interp = (
            "Principal Findings\n\n"
            "1.  HBV vaccination effect (H1 confirmed):\n"
            "    C22 liver HCC shows ρ=−0.983, the single strongest trend in the\n"
            "    registry. Novel-1 VAE cluster (which contains C22/C34/C61) mirrors\n"
            "    this at ρ=−0.977. Both trends are consistent with the cohort effect\n"
            "    of Taiwan's 1986 universal infant HBV vaccination programme.\n"
            "    Birth-cohort decomposition shows universal decline across age groups;\n"
            "    the youngest cohort (born ≥1980) has too few cancer patients in 2003–\n"
            "    2020 to test the vaccination mechanism directly.\n\n"
            "2.  Betel nut regulation (H2 split):\n"
            "    C13 hypopharynx declining (ρ=−0.773) is consistent with Taiwan's\n"
            "    anti-betel campaigns from ~2005. C12 pyriform sinus rising (ρ=+0.562)\n"
            "    is anatomically surprising — possible explanations: differential\n"
            "    detection (C12 is often detected incidentally during endoscopy for C15),\n"
            "    or anatomical reclassification. Requires clinical review.\n\n"
            "3.  Metabolic syndrome (H3 partial):\n"
            "    C54 endometrial rising (ρ=+0.925) is the second strongest rising trend\n"
            "    and consistent with increasing obesity/metabolic syndrome in Taiwan.\n"
            "    C50 breast is flat (ns) — may reflect stable screening coverage or\n"
            "    conflicting risk factor trends (later parity, less breastfeeding rising;\n"
            "    HRT use declining post-WHI 2002).\n\n"
            "4.  HPV vaccination / PSA era:\n"
            "    C53 cervix ρ=−0.946 likely reflects gradual benefit of HPV awareness\n"
            "    and eventual vaccination; C61 prostate ρ=+0.895 likely reflects PSA\n"
            "    screening diffusion rather than true incidence increase.\n\n"
            "Limitations\n\n"
            "   Registry coverage improved 2003–2020 → early years may under-ascertain.\n"
            "   Betel nut exposure history unavailable — H2 test is ecological.\n"
            "   Multi-primary rate trend (ρ=−0.901, declining) likely reflects residual\n"
            "     follow-up bias despite ≥5yr FU restriction; interpret with caution.\n"
            "   C12 rising vs C13 falling warrants anatomical-pathological sub-study.\n"
            "   Birth-cohort HBV vaccination test requires ≥2030 registry to reach\n"
            "     adequate power in the born ≥1980 cohort."
        )
        ax.text(0.07, 0.82, interp, transform=ax.transAxes, fontsize=9,
                va="top", ha="left", fontfamily="monospace",
                bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", pad=8))
        footer(ax, 8)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

    file_size = OUT.stat().st_size / 1024
    print(f"  Temporal_Draft.pdf written — {file_size:.0f} KB")
    print(f"  Path: {OUT}")


if __name__ == "__main__":
    main()
