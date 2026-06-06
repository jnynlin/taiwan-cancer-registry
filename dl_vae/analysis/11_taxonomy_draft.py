"""
Registry DL — Script 11: Three-Axis Taxonomy Draft PDF

Assembles an 8-page publication-ready draft summarising:
  - Low-dimensional cancer co-occurrence structure (3 active axes)
  - Axis characterisation by demographic / temporal covariates
  - Cluster structure and silhouette justification
  - Survival impact of cluster membership

Depends on outputs from scripts 03, 09, 10.

Output: results/Taxonomy_Draft.pdf
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
R03  = BASE / "results/03_latent"
R09  = BASE / "results/09_axis_covariate"
R10  = BASE / "results/10_cluster_survival"
OUT  = BASE / "results"

NAVY   = "#14304a"
ACCENT = "#2e7fbf"
TOTAL  = 8


def footer(fig, page):
    fig.text(0.5, 0.01,
             f"CMUH Institutional Registry — Three-Axis Cancer Taxonomy  |  Page {page}/{TOTAL}  |  Draft 2026-06-06",
             ha="center", fontsize=7, color="#888888")


def flow(ax, text, fontsize=9):
    ax.axis("off")
    ax.text(0.02, 0.96, text, transform=ax.transAxes, fontsize=fontsize,
            va="top", wrap=True,
            bbox=dict(facecolor="white", edgecolor="none", pad=4))


def img(ax, path, title=None):
    p = Path(path)
    if p.exists():
        ax.imshow(imread(str(p)), aspect="equal")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=9, pad=3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"[{p.name} not found]", ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=8)


def load_stats():
    meta     = pd.read_csv(BASE / "data/patient_meta.csv", index_col="pid")
    interp   = pd.read_csv(R03 / "axis_interpretation.csv")
    clust    = pd.read_csv(R03 / "cluster_profiles.csv")
    cov      = pd.read_csv(R09 / "axis_covariate_stats.csv")
    sil      = pd.read_csv(R10 / "silhouette_scores.csv")
    surv_tbl = pd.read_csv(R10 / "cluster_survival_table.csv", index_col="pid")
    ph_df    = pd.read_csv(R10 / "ph_test_results.csv", index_col=0)
    split_df = pd.read_csv(R10 / "cox_timesplit_results.csv", index_col=0)
    return meta, interp, clust, cov, sil, surv_tbl, ph_df, split_df


def main():
    print("=== Registry DL — 11: Taxonomy Draft PDF ===")

    meta, interp, clust, cov, sil, surv_tbl, ph_df, split_df = load_stats()

    n_pts   = len(meta)
    n_multi = int(meta["multi_cancer"].sum())
    n_clust = len(clust)
    best_sil_k = int(sil.loc[sil["silhouette"].idxmax(), "k"])
    sil_k5     = float(sil[sil["k"] == 5]["silhouette"].iloc[0])
    n_active   = len(cov)

    pdf_path = OUT / "Taxonomy_Draft.pdf"
    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: Title ─────────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax_t = fig.add_axes([0.05, 0.55, 0.90, 0.38])
        ax_t.set_facecolor(NAVY); ax_t.axis("off")
        ax_t.text(0.5, 0.65,
                  "A Three-Axis Taxonomy of Cancer Co-occurrence Patterns\n"
                  "in a Hospital-Based Institutional Cancer Registry (CMUH)",
                  ha="center", va="center", fontsize=17, color="white",
                  fontweight="bold", transform=ax_t.transAxes)
        ax_t.text(0.5, 0.25,
                  "Unsupervised discovery via β-VAE  ·  84k patients  ·  2003–2020",
                  ha="center", va="center", fontsize=12, color="#aaccee",
                  transform=ax_t.transAxes)

        ax_s = fig.add_axes([0.10, 0.12, 0.80, 0.38])
        ax_s.axis("off")
        n_ph_violated = int((~ph_df["PH_OK"]).sum())
        txt = (
            f"Cohort: {n_pts:,} patients (CMUH hospital-based registry)  ·  Multi-cancer: {n_multi:,} ({n_multi/n_pts*100:.1f}%)\n\n"
            f"VAE latent space: 12 dimensions → {n_active} active axes (highest max |loading|)\n"
            f"KMeans clustering: k={n_clust} (pre-specified for consistency with upstream scripts), silhouette={sil_k5:.3f}\n\n"
            "Key finding: Cancer co-occurrence in this CMUH cohort is explained by three latent axes,\n"
            "corresponding to distinct carcinogenic exposures (UADT/field-cancerization,\n"
            "hormonal/gynecologic ×2). Five clusters identified: Hormonal, Multi-solid,\n"
            "Colorectal/Lymphatic, Gynaecologic-Oral, Hepatic/Rare.\n"
            f"NOTE: proportional hazards violated for {n_ph_violated}/6 model terms; time-split\n"
            "Cox (landmark=2yr) is the primary analysis — full-follow-up HRs are invalid and not reported."
        )
        ax_s.text(0.5, 0.6, txt, ha="center", va="center", fontsize=11,
                  color=NAVY, transform=ax_s.transAxes,
                  bbox=dict(facecolor="#f0f4f8", edgecolor=ACCENT, boxstyle="round,pad=0.6"))
        footer(fig, 1)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 2: Active axis identification ────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax_top = fig.add_axes([0.05, 0.52, 0.90, 0.40])
        img(ax_top, R09 / "fig_active_dims.png",
            "Fig 1: Max |Spearman ρ| per latent dimension — top 3 (blue) selected as active axes")

        ax_tbl = fig.add_axes([0.05, 0.08, 0.90, 0.40])
        ax_tbl.axis("off")
        if interp is not None:
            show = interp[["dim", "axis_name", "top3_pos", "top3_neg", "max_loading"]].copy()
            show.columns = ["Dim", "Axis name", "Top sites (+)", "Top sites (–)", "Max |ρ|"]
            show["Max |ρ|"] = show["Max |ρ|"].round(3)
            tbl = ax_tbl.table(cellText=show.values, colLabels=show.columns,
                               loc="center", cellLoc="left")
            tbl.auto_set_font_size(False); tbl.set_fontsize(8)
            tbl.scale(1, 1.5)
            ax_tbl.set_title("Axis interpretation (all 12 dims; active in bold)", fontsize=10, pad=4)
        fig.suptitle("Active Latent Axis Identification", fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 2)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 3: Axis × sex ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 6.5))
        img(ax, R09 / "fig_axis_sex.png",
            "Fig 2: Active axis values by sex (violin plots, Mann–Whitney U)")
        ax_note = fig.add_axes([0.05, 0.02, 0.90, 0.14])
        cov_txt = ""
        for _, r in cov.iterrows():
            cov_txt += (f"{r['dim']} [{r['axis_name']}]:  "
                        f"sex MWU p={r['sex_mw_p']:.2e} rbi={r['sex_rbi']:.3f}  |  "
                        f"age ρ={r['age_r']:.3f} p={r['age_p']:.2e}  |  "
                        f"era ρ={r['era_r']:.3f} p={r['era_p']:.2e}\n")
        flow(ax_note, cov_txt.strip(), fontsize=8.5)
        fig.suptitle("Axis Characterisation — Sex", fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 4: Axis × age and era ───────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        img(axes[0], R09 / "fig_axis_age.png", "Fig 3a: Active axes vs age at first diagnosis")
        img(axes[1], R09 / "fig_axis_era.png", "Fig 3b: Active axis temporal trend 2003–2020\n(vertical dashed: ~2010 betel nut regulation)")
        fig.suptitle("Axis Characterisation — Age and Temporal Trend", fontsize=13,
                     color=NAVY, fontweight="bold")
        footer(fig, 4)
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 5: Cluster structure ─────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        img(axes[0], R10 / "fig_silhouette.png",
            f"Fig 4a: Silhouette analysis k=2–8\n(k={best_sil_k} best; k=5 selected; sil={sil_k5:.3f})")
        img(axes[1], R03 / "fig_umap_clusters.png",
            "Fig 4b: UMAP of VAE latent μ — KMeans k=5 clusters")
        fig.suptitle("Cluster Structure", fontsize=13, color=NAVY, fontweight="bold")

        ax_note = fig.add_axes([0.05, 0.02, 0.90, 0.12])
        clust_txt = "  ".join(
            f"C{int(r['cluster'])}: {r['name']} (n={int(r['n']):,})"
            for _, r in clust.iterrows())
        flow(ax_note, clust_txt, fontsize=9)
        footer(fig, 5)
        fig.tight_layout(rect=[0, 0.15, 1, 0.94])
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 6: Cluster site enrichment ──────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 7))
        img(ax, R03 / "fig_cluster_heatmap.png",
            "Fig 5: Cluster × site fold-enrichment heatmap (top 20 sites, fold vs overall mean)")
        ax_tbl2 = fig.add_axes([0.05, 0.03, 0.90, 0.20])
        ax_tbl2.axis("off")
        show2 = clust[["cluster", "name", "n", "top5_enriched"]].copy()
        show2.columns = ["Cluster", "Name", "N", "Top-5 enriched sites"]
        tbl2 = ax_tbl2.table(cellText=show2.values, colLabels=show2.columns,
                             loc="center", cellLoc="left")
        tbl2.auto_set_font_size(False); tbl2.set_fontsize(8.5)
        tbl2.scale(1, 1.4)
        fig.suptitle("Cluster Site Enrichment", fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 6)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 7: Cluster survival (KM) ────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 7))
        img(ax, R10 / "fig_km_clusters.png",
            "Fig 6: Kaplan–Meier overall survival by VAE cluster (log-rank test, approximate survival times)\n"
            "⚠ Proportional hazards violated for 5/6 terms — KM curves are descriptive only.\n"
            "See Page 8 (Fig 7) for time-split Cox as the primary survival analysis.")
        ax_note2 = fig.add_axes([0.05, 0.02, 0.90, 0.13])
        surv_summary = ""
        for k in range(n_clust):
            m = surv_tbl[surv_tbl["cluster"] == k]
            if len(m):
                med = m["duration"].median() / 365.25
                surv_summary += f"C{k} [{clust.iloc[k]['name']}]: n={len(m):,}  median={med:.1f}yr  dead={int(m['dead'].sum())}  "
        flow(ax_note2, surv_summary.strip(), fontsize=8.5)
        fig.suptitle("Survival Impact of Cluster Membership", fontsize=13,
                     color=NAVY, fontweight="bold")
        footer(fig, 7)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 8: Time-split Cox + PH test + interpretation ────────────
        fig = plt.figure(figsize=(11, 8.5))

        # Left: time-split forest plot (primary result)
        ax_split = fig.add_axes([0.03, 0.42, 0.58, 0.50])
        img(ax_split, R10 / "fig_cox_timesplit.png",
            "Fig 7 (PRIMARY): Time-split Cox — landmark 2yr\n"
            "PH assumption violated for C1/C3/C4/age/sex — full-FU model invalid")

        # Right: interpretation
        ax_lim = fig.add_axes([0.63, 0.42, 0.35, 0.50])
        # Build time-split summary from data
        ts_lines = []
        for idx_name, row in split_df.iterrows():
            ts_lines.append(
                f"{idx_name.replace('_vs_C2','').replace('_',' ')}: "
                f"HR={row['HR_early']:.2f} early / {row['HR_late']:.2f} late")
        ts_txt = "\n".join(ts_lines)

        lim = (
            "Time-split Cox (landmark=2yr):\n"
            f"{ts_txt}\n\n"
            "Interpretation:\n"
            "• C3 (Mixed-GI/gynecologic) HR=44 in full model\n"
            "  is ARTIFACTUAL — PH violation confirmed\n"
            "  (χ²=177.6, p=10⁻⁴⁰). Time-split shows C3\n"
            "  is PROTECTIVE vs C2 in both periods.\n"
            "• C0 (Hormonal) satisfies PH — strongly protective\n"
            "  vs reference throughout follow-up.\n"
            "• C1/C4 show time-varying attenuation (early\n"
            "  excess hazard diminishes over time).\n\n"
            "Limitations:\n"
            "① PH violated for 5/6 terms — time-split Cox\n"
            "   used as primary; full-FU HRs not reported.\n"
            "② k=5 pre-specified; silhouette monotone → k=8.\n"
            "③ Survival time approximated from diag_yr Jan 1.\n"
            "④ Cluster labels require manual ICD-O annotation.\n"
            "⑤ No molecular/treatment data in registry."
        )
        ax_lim.axis("off")
        ax_lim.text(0.03, 0.98, lim, transform=ax_lim.transAxes,
                    fontsize=7.8, va="top", color=NAVY,
                    bbox=dict(facecolor="#f0f4f8", edgecolor=ACCENT,
                              boxstyle="round,pad=0.5"))

        # Bottom: full-FU Cox labelled as invalid + next steps
        ax_bot = fig.add_axes([0.03, 0.03, 0.94, 0.35])
        ax_bot.axis("off")
        # PH test summary table
        ph_show = ph_df[["test_statistic", "p", "PH_OK"]].copy()
        ph_show["test_statistic"] = ph_show["test_statistic"].map(lambda x: f"{x:.1f}")
        ph_show["p"]    = ph_show["p"].map(lambda x: f"{x:.1e}")
        ph_show["PH_OK"] = ph_show["PH_OK"].map({True: "✓ Yes", False: "✗ No"})
        ph_show.index   = [i.replace("_vs_", " vs ") for i in ph_show.index]
        tbl = ax_bot.table(
            cellText=ph_show.values, rowLabels=ph_show.index,
            colLabels=["χ²", "p", "PH OK"],
            loc="upper left", cellLoc="center", bbox=[0.0, 0.0, 0.38, 1.0])
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white")
            elif c == 2 and "No" in str(cell.get_text().get_text()):
                cell.set_facecolor("#FDECEA")
            elif c == 2 and "Yes" in str(cell.get_text().get_text()):
                cell.set_facecolor("#E8F5E9")
        ax_bot.set_title("Schoenfeld PH test  |  Next steps: ICD-O cluster annotation; β sensitivity; held-out validation",
                         fontsize=8, loc="left", pad=4)

        next_txt = (
            "Next steps: "
            "① Manual ICD-O annotation of Novel-1/Novel-2.  "
            "② HBV era-split (pre/post 1986).  "
            "③ β sensitivity (β=2,4).  "
            "④ Held-out 20% validation.  "
            "⑤ Cluster-conditioned Transformer (Script 07)."
        )
        ax_bot.text(0.42, 0.5, next_txt, transform=ax_bot.transAxes,
                    fontsize=8, va="center", color=NAVY,
                    bbox=dict(facecolor="#f0f4f8", edgecolor=ACCENT, boxstyle="round,pad=0.4"))

        fig.suptitle("Survival Analysis — Time-Split Cox (Primary) + PH Test",
                     fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 8)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

    print(f"  Saved → {pdf_path}  ({TOTAL} pages)")


if __name__ == "__main__":
    main()
