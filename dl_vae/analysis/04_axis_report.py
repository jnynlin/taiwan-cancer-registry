"""
Registry DL — Script 04: Axis Discovery Report PDF

Assembles a 7-page draft PDF summarising all VAE findings.

Pages:
  1. Title + cohort stats + training summary
  2. Fig 1a (site freq) + Fig 1b (n_sites dist)
  3. Fig 2a (training loss) + Fig 2b (KL per dim) + Fig 2c (recon per site)
  4. Fig 3a (UMAP clusters) + cluster profiles table
  5. Fig 3b (UMAP known axes) — alignment with UADT + hormonal
  6. Fig 3c (axis loadings heatmap) — per-dim interpretation table
  7. Fig 3d (cluster enrichment heatmap) + novel axis candidates + limitations

Output:
  results/VAE_Axes_Draft.pdf
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = Path(__file__).parent.parent
R01  = BASE / "results/01_matrix"
R02  = BASE / "results/02_vae"
R03  = BASE / "results/03_latent"
DOUT = BASE / "data"
OUT  = BASE / "results"
OUT.mkdir(parents=True, exist_ok=True)

NAVY   = "#14304a"
ACCENT = "#2e7fbf"
TOTAL_PAGES = 7


def flow(ax, text, fontsize=9, x=0.02, y=0.96, color="black"):
    ax.axis("off")
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment="top", color=color, wrap=True,
            bbox=dict(facecolor="white", edgecolor="none", pad=4))


def footer(fig, page, total=TOTAL_PAGES):
    fig.text(0.5, 0.01, f"Taiwan Cancer Registry — VAE Latent Axis Discovery  |  Page {page}/{total}  |  Draft 2026-06-02",
             ha="center", fontsize=7, color="#888888")


def img(ax, path, title=None):
    p = Path(path)
    if p.exists():
        from matplotlib.image import imread
        im = imread(str(p))
        ax.imshow(im, aspect="equal")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=9, pad=3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"[{p.name} not found]", ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=8)


def load_stats():
    try:
        X_df  = pd.read_csv(DOUT / "cancer_matrix.csv", index_col="pid")
        meta  = pd.read_csv(DOUT / "patient_meta.csv",  index_col="pid")
        hist  = pd.read_csv(R02  / "train_history.csv")
        intrp = pd.read_csv(R03  / "axis_interpretation.csv")
        clust = pd.read_csv(R03  / "cluster_profiles.csv")
        return X_df, meta, hist, intrp, clust
    except FileNotFoundError as e:
        print(f"  WARNING: {e} — some stats will be missing")
        return None, None, None, None, None


def main():
    print("=== Registry DL — 04: Axis Report PDF ===")

    X_df, meta, hist, intrp, clust = load_stats()

    # Derived stats
    n_pts   = len(X_df) if X_df is not None else "?"
    n_sites = X_df.shape[1] if X_df is not None else "?"
    n_multi = int(meta["multi_cancer"].sum()) if meta is not None else "?"
    best_ep = int(hist["epoch"].iloc[-1]) if hist is not None else "?"
    best_val= f"{hist['val_loss'].min():.4f}" if hist is not None else "?"
    n_dims  = len(intrp) if intrp is not None else 12
    n_novel = intrp[intrp["axis_name"]=="Novel"].shape[0] if intrp is not None else "?"

    pdf_path = OUT / "VAE_Axes_Draft.pdf"
    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: Title ─────────────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax_title = fig.add_axes([0.05, 0.55, 0.90, 0.38])
        ax_title.set_facecolor(NAVY)
        ax_title.axis("off")
        ax_title.text(0.5, 0.65, "Unsupervised Discovery of Cancer Co-occurrence Axes",
                      ha="center", va="center", fontsize=18, color="white", fontweight="bold",
                      transform=ax_title.transAxes)
        ax_title.text(0.5, 0.35,
                      "Variational Autoencoder on Taiwan Cancer Registry\n"
                      "84,161 patients · 46 sites · 2003–2020",
                      ha="center", va="center", fontsize=13, color="#aaccee",
                      transform=ax_title.transAxes)

        ax_stats = fig.add_axes([0.05, 0.12, 0.90, 0.38])
        ax_stats.axis("off")
        stats_text = (
            f"Cohort: {n_pts:,} patients · {n_sites} cancer sites (≥30 patients each)\n"
            f"Multi-cancer patients: {n_multi:,}\n\n"
            f"Model: VAE  |  Latent dim = {n_dims}  |  β = 1.0  |  Training stopped at epoch {best_ep}\n"
            f"Best validation ELBO: {best_val}\n\n"
            f"Active latent dimensions: see Fig 2b (KL > 0.1 threshold)\n"
            f"Novel axes identified (not matching UADT or hormonal): {n_novel}"
        )
        ax_stats.text(0.5, 0.7, stats_text, ha="center", va="center", fontsize=11,
                      color=NAVY, transform=ax_stats.transAxes,
                      bbox=dict(facecolor="#f0f4f8", edgecolor=ACCENT, boxstyle="round,pad=0.6"))
        footer(fig, 1)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 2: Cohort figures ─────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        img(axes[0], R01 / "fig_site_freq.png",   "Fig 1a: Cancer site frequency (all registry sites ≥30 patients)")
        img(axes[1], R01 / "fig_n_sites_dist.png","Fig 1b: Multi-cancer burden per patient")
        fig.suptitle("Cohort Description", fontsize=13, color=NAVY, fontweight="bold", y=0.98)
        footer(fig, 2)
        fig.tight_layout(rect=[0,0.03,1,0.96])
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 3: Training diagnostics ──────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        img(axes[0], R02 / "fig_loss.png",          "Fig 2a: Training curve")
        img(axes[1], R02 / "fig_kl_per_dim.png",    "Fig 2b: KL per latent dim (active dims in blue)")
        img(axes[2], R02 / "fig_recon_per_site.png","Fig 2c: Per-site reconstruction accuracy")
        fig.suptitle("VAE Training Diagnostics", fontsize=13, color=NAVY, fontweight="bold")

        if intrp is not None:
            ax_text = fig.add_axes([0.02, 0.01, 0.96, 0.22])
            rows = intrp[["dim","axis_name","top3_pos","max_loading"]].copy()
            rows.columns = ["Dim","Axis","Top sites (+)","Max |ρ|"]
            table = ax_text.table(cellText=rows.values, colLabels=rows.columns,
                                   loc="center", cellLoc="center")
            table.auto_set_font_size(False); table.set_fontsize(7)
            table.scale(1, 1.2)
            ax_text.axis("off")

        footer(fig, 3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 4: UMAP + cluster profiles ───────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax_umap = fig.add_axes([0.02, 0.35, 0.56, 0.58])
        img(ax_umap, R03 / "fig_umap_clusters.png", "Fig 3a: UMAP — KMeans clusters")

        ax_tbl = fig.add_axes([0.60, 0.35, 0.38, 0.58])
        ax_tbl.axis("off")
        if clust is not None:
            tbl_data = clust[["cluster","name","n","top5_enriched"]].values
            tbl = ax_tbl.table(cellText=tbl_data,
                               colLabels=["Cluster","Name","N","Top-5 enriched sites"],
                               loc="center", cellLoc="left")
            tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
            tbl.scale(1, 1.6)
            ax_tbl.set_title("Cluster profiles", fontsize=10, pad=6)

        ax_note = fig.add_axes([0.02, 0.02, 0.96, 0.28])
        note = (
            "KMeans clustering (k=5) applied to VAE latent μ vectors. "
            "Clusters are characterized by fold-enrichment in cancer site prevalence relative to the full registry.\n"
            "Known axes (UADT, hormonal) should map to distinct clusters if the VAE has learned meaningful structure. "
            "'Novel' clusters represent axes not predicted by prior hypotheses and warrant follow-up characterization."
        )
        flow(ax_note, note, fontsize=9)

        fig.suptitle("Latent Space Clustering", fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 4)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 5: Known-axis alignment ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 6.5))
        img(ax, R03 / "fig_umap_known_axes.png",
            "Fig 3b: Known cancer axes projected into VAE latent space\n"
            "(UADT field sites / Hormonal-gynecologic / Liver-GI)")
        fig.suptitle("Known Axis Alignment", fontsize=13, color=NAVY, fontweight="bold")

        ax_note = fig.add_axes([0.05, 0.02, 0.90, 0.15])
        flow(ax_note,
             "If VAE has captured biologically meaningful structure, patients with UADT field cancers should cluster "
             "separately from hormonal/GI patients. Overlap between known axes and novel clusters points to shared "
             "exposures worth investigating (e.g. radiation overlap between thyroid + breast + UADT).",
             fontsize=9)
        footer(fig, 5)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 6: Axis loadings heatmap ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 7))
        img(ax, R03 / "fig_axis_loadings.png",
            "Fig 3c: VAE axis loadings — site × latent dimension (Spearman ρ)\n"
            "Red = positive loading (site enriched at high z), Blue = negative")
        fig.suptitle("Per-Dimension Axis Loadings", fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 6)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 7: Enrichment heatmap + novel axes + limitations ─────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax_heat = fig.add_axes([0.02, 0.40, 0.56, 0.52])
        img(ax_heat, R03 / "fig_cluster_heatmap.png",
            "Fig 3d: Cluster × site fold-enrichment")

        ax_novel = fig.add_axes([0.60, 0.40, 0.38, 0.52])
        ax_novel.axis("off")
        if intrp is not None:
            novel = intrp[intrp["axis_name"] == "Novel"]
            if len(novel):
                novel_text = "Novel axes (not matching known biology):\n\n"
                for _, row in novel.iterrows():
                    novel_text += f"• {row['dim']}: top+ = {row['top3_pos']}\n"
                novel_text += "\nThese require manual ICD-O code lookup and literature review."
            else:
                novel_text = "No novel axes detected — all dimensions mapped to known biology."
            ax_novel.text(0.05, 0.95, novel_text, transform=ax_novel.transAxes,
                          fontsize=9, va="top", color=NAVY,
                          bbox=dict(facecolor="#fff8e8", edgecolor="#e0a020", boxstyle="round,pad=0.5"))

        ax_lim = fig.add_axes([0.02, 0.02, 0.96, 0.33])
        lim_text = (
            "Limitations:\n"
            "① Registry contains only cancer diagnoses — no molecular, treatment, or lifestyle data; axes reflect "
            "co-occurrence patterns not etiology.\n"
            "② Single-cancer patients (the majority) are represented as zero vectors; sparse input may cause "
            "VAE to place them all near the prior mean.\n"
            "③ KMeans k=5 is arbitrary; silhouette analysis and varied k recommended before publication.\n"
            "④ β=1 (standard ELBO); increasing β would encourage more axis-disentanglement but risks "
            "underfitting rare co-occurrences.\n"
            "⑤ UMAP projection is stochastic — cluster topology varies slightly across runs (seed=42 locked here).\n"
            "⑥ Site MIN_SITE_N=30 threshold removes ultra-rare cancers; results do not generalise to those."
        )
        flow(ax_lim, lim_text, fontsize=8.5)
        fig.suptitle("Novel Axis Candidates + Limitations", fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 7)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

    print(f"  Saved → {pdf_path}  ({TOTAL_PAGES} pages)")


if __name__ == "__main__":
    main()
