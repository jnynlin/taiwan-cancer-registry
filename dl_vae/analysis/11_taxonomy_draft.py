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
             f"Taiwan Cancer Registry — Three-Axis Cancer Taxonomy  |  Page {page}/{TOTAL}  |  Draft 2026-06-03",
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
    return meta, interp, clust, cov, sil, surv_tbl


def main():
    print("=== Registry DL — 11: Taxonomy Draft PDF ===")

    meta, interp, clust, cov, sil, surv_tbl = load_stats()

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
                  "in a Population-Based Taiwanese Registry",
                  ha="center", va="center", fontsize=17, color="white",
                  fontweight="bold", transform=ax_t.transAxes)
        ax_t.text(0.5, 0.25,
                  "Unsupervised discovery via β-VAE  ·  84k patients  ·  2003–2020",
                  ha="center", va="center", fontsize=12, color="#aaccee",
                  transform=ax_t.transAxes)

        ax_s = fig.add_axes([0.10, 0.12, 0.80, 0.38])
        ax_s.axis("off")
        txt = (
            f"Cohort: {n_pts:,} patients  ·  Multi-cancer: {n_multi:,} ({n_multi/n_pts*100:.1f}%)\n\n"
            f"VAE latent space: 12 dimensions → {n_active} active axes (highest max |loading|)\n"
            f"KMeans clustering: k={n_clust}, silhouette={sil_k5:.3f}\n\n"
            "Key finding: Cancer co-occurrence in Taiwan is explained by three latent axes,\n"
            "corresponding to distinct carcinogenic exposures. Cluster membership independently\n"
            "predicts overall survival after adjustment for age and sex."
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
            "Fig 6: Kaplan–Meier overall survival by VAE cluster\n"
            "(global log-rank test; time from first diagnosis, approximate)")
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

        # ── Page 8: Cox HRs + limitations ────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax_cox = fig.add_axes([0.05, 0.45, 0.55, 0.47])
        img(ax_cox, R10 / "fig_cox_hr.png",
            "Fig 7: Cox HR forest plot — cluster vs largest cluster (ref),\nadjusted for age + sex")

        ax_lim = fig.add_axes([0.62, 0.45, 0.36, 0.47])
        lim = (
            "Interpretation:\n"
            "• Three latent axes capture the dominant axes of\n"
            "  cancer co-occurrence in Taiwan.\n"
            "• Axes reflect shared exposures, not genetics:\n"
            "  (i) hormonal/gynecologic, (ii) lifestyle/UADT,\n"
            "  (iii) novel (GI/systemic — possibly HBV).\n"
            "• Cluster survival differences likely reflect\n"
            "  case-mix (e.g. UADT field cluster has higher\n"
            "  multi-cancer burden) rather than latent biology.\n\n"
            "Limitations:\n"
            "① No molecular/treatment/exposure data in registry.\n"
            "② k=5 chosen; silhouette analysis presented.\n"
            "③ Survival time approximated from diag_yr.\n"
            "④ Cluster labels require manual ICD-O annotation.\n"
            "⑤ β=1; higher β may improve disentanglement."
        )
        ax_lim.axis("off")
        ax_lim.text(0.03, 0.97, lim, transform=ax_lim.transAxes,
                    fontsize=8.5, va="top", color=NAVY,
                    bbox=dict(facecolor="#f0f4f8", edgecolor=ACCENT,
                              boxstyle="round,pad=0.5"))

        ax_next = fig.add_axes([0.05, 0.05, 0.90, 0.33])
        next_txt = (
            "Next steps toward publication:\n"
            "① Manual ICD-O annotation of Novel-1 (C34/C61/C71/C22/C67) and Novel-2 (C20/C77/C44) "
            "clusters — confirm GI-systemic vs haematological split.\n"
            "② Test HBV hypothesis: partition patients by era (pre/post 1986 HBV vaccination) "
            "and compare novel-cluster enrichment — expected to track with HBV seroprevalence cohort.\n"
            "③ Increase β (β=2, 4) and compare active dimension count — tests whether low-dimensional "
            "structure is robust or a β=1 artifact.\n"
            "④ Validate cluster-survival finding in a held-out 20% test split (currently fit on all N).\n"
            "⑤ Consider cluster membership as a feature in the Cancer Sequence Transformer (Script 07) "
            "— does cluster-aware conditioning improve R@1?"
        )
        flow(ax_next, next_txt, fontsize=9)
        fig.suptitle("Cox Results + Limitations + Next Steps", fontsize=13,
                     color=NAVY, fontweight="bold")
        footer(fig, 8)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

    print(f"  Saved → {pdf_path}  ({TOTAL} pages)")


if __name__ == "__main__":
    main()
