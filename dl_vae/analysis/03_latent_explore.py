"""
Registry DL — Script 03: Latent Space Exploration

Interprets the VAE latent space to discover cancer co-occurrence axes.

Analysis:
  1. UMAP 2D projection of latent μ
  2. Per-dimension axis interpretation: top cancer sites loading on each dim
  3. KMeans clustering (k=5) → characterize clusters by site enrichment
  4. Known-axis alignment check: do clusters reproduce UADT + hormonal axes?
  5. Novel axis flagging: clusters not matching known axes

Outputs:
  results/03_latent/axis_loadings.csv    — site × latent_dim correlation matrix
  results/03_latent/axis_interpretation.csv — per-dim top-3 sites + named axis
  results/03_latent/cluster_profiles.csv — per-cluster site enrichment
  results/03_latent/fig_umap_clusters.png
  results/03_latent/fig_umap_known_axes.png
  results/03_latent/fig_axis_loadings.png
  results/03_latent/fig_cluster_heatmap.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_OK = True
except ImportError:
    from sklearn.decomposition import PCA
    UMAP_OK = False
    print("  umap-learn not found; falling back to PCA for 2D projection.")

BASE  = Path(__file__).parent.parent
DOUT  = BASE / "data"
OUT   = BASE / "results/03_latent"
OUT.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 5
SEED       = 42

# Known axis membership (from EDA)
UADT_SITES      = {'C02','C03','C04','C05','C06','C09','C10','C12','C13','C15'}
HORMONAL_SITES  = {'C50','C53','C54','C56'}
LIVER_GI_SITES  = {'C18','C20','C22','C25'}
LUNG_SITES      = {'C34'}


def axis_name(top_sites):
    """Heuristic: name an axis by its dominant site cluster membership."""
    uadt  = sum(1 for s in top_sites if s in UADT_SITES)
    horm  = sum(1 for s in top_sites if s in HORMONAL_SITES)
    lgi   = sum(1 for s in top_sites if s in LIVER_GI_SITES)
    if uadt >= 2:   return "UADT/field-cancerization"
    if horm >= 2:   return "Hormonal/gynecologic"
    if lgi  >= 2:   return "Liver-GI"
    return "Novel"


def main():
    print("=== Registry DL — 03: Latent Space Exploration ===")

    mu       = np.load(DOUT / "latent_mu.npy")
    X_df     = pd.read_csv(DOUT / "cancer_matrix.csv", index_col="pid")
    meta     = pd.read_csv(DOUT / "patient_meta.csv", index_col="pid")
    sites    = X_df.columns.tolist()
    X        = X_df.values
    N, D_lat = mu.shape
    print(f"  Loaded: {N:,} patients · {D_lat} latent dims · {len(sites)} sites")

    # ── 1. Axis loadings (Spearman correlation: latent dim ↔ cancer site) ────

    print("  Computing axis loadings (Spearman)…")
    loadings = np.zeros((len(sites), D_lat))
    for j in range(D_lat):
        for i, site in enumerate(sites):
            r, _ = stats.spearmanr(mu[:, j], X[:, i])
            loadings[i, j] = r

    load_df = pd.DataFrame(loadings, index=sites,
                           columns=[f"z{j}" for j in range(D_lat)])
    load_df.to_csv(OUT / "axis_loadings.csv")

    # Per-dimension: top-3 positively and negatively loading sites
    interp_rows = []
    for j in range(D_lat):
        col = load_df[f"z{j}"]
        top3_pos = col.nlargest(3).index.tolist()
        top3_neg = col.nsmallest(3).index.tolist()
        ax_name  = axis_name(top3_pos + top3_neg)
        interp_rows.append({
            "dim": f"z{j}",
            "top3_pos": ", ".join(top3_pos),
            "top3_neg": ", ".join(top3_neg),
            "max_loading": col.abs().max(),
            "axis_name": ax_name
        })
        print(f"  z{j:2d}: {ax_name:30s}  top+ {top3_pos}  top- {top3_neg}")

    interp_df = pd.DataFrame(interp_rows)
    interp_df.to_csv(OUT / "axis_interpretation.csv", index=False)

    # ── 2. UMAP (or PCA) 2D embedding ─────────────────────────────────────────

    print("  Computing 2D projection…")
    scaler = StandardScaler()
    mu_sc  = scaler.fit_transform(mu)

    if UMAP_OK:
        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
        emb = reducer.fit_transform(mu_sc)
        proj_label = "UMAP"
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=SEED)
        emb = reducer.fit_transform(mu_sc)
        proj_label = "PCA"

    # ── 3. KMeans clustering ──────────────────────────────────────────────────

    print(f"  KMeans clustering (k={N_CLUSTERS})…")
    km     = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=20)
    labels = km.fit_predict(mu_sc)

    # Cluster profiles: mean site prevalence per cluster vs overall
    cluster_profiles = []
    overall_prev = X.mean(axis=0)
    for k in range(N_CLUSTERS):
        mask   = labels == k
        n_k    = mask.sum()
        prev_k = X[mask].mean(axis=0)
        enrich = prev_k / (overall_prev + 1e-9)  # fold-enrichment vs overall
        top5   = pd.Series(enrich, index=sites).nlargest(5).index.tolist()
        uadt_k = sum(1 for s in top5 if s in UADT_SITES)
        horm_k = sum(1 for s in top5 if s in HORMONAL_SITES)
        if uadt_k >= 2:   cname = "UADT field"
        elif horm_k >= 2: cname = "Hormonal"
        else:             cname = f"Novel-{k}"
        cluster_profiles.append({
            "cluster": k, "n": n_k, "name": cname, "top5_enriched": ", ".join(top5)
        })
        print(f"  Cluster {k} (n={n_k:,}) [{cname}]: top enriched = {top5}")

    pd.DataFrame(cluster_profiles).to_csv(OUT / "cluster_profiles.csv", index=False)

    # ── 4. Figures ────────────────────────────────────────────────────────────

    PALETTE = ["#2e7fbf","#e05c2e","#2ca02c","#9467bd","#8c564b"]

    # Fig A: UMAP colored by cluster
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(N_CLUSTERS):
        mask = labels == k
        ax.scatter(emb[mask, 0], emb[mask, 1], s=4, alpha=0.4,
                   color=PALETTE[k], label=f"C{k}: {cluster_profiles[k]['name']} (n={mask.sum():,})")
    ax.set_xlabel(f"{proj_label}-1"); ax.set_ylabel(f"{proj_label}-2")
    ax.set_title(f"VAE latent space — {proj_label} projection, KMeans k={N_CLUSTERS}")
    ax.legend(markerscale=4, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_umap_clusters.png", dpi=150)
    plt.close()

    # Fig B: UMAP colored by known-axis membership
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axis_defs = [
        ("UADT field sites", UADT_SITES, "#2e7fbf"),
        ("Hormonal/gynecologic", HORMONAL_SITES, "#e05c2e"),
        ("Liver-GI", LIVER_GI_SITES, "#2ca02c"),
    ]
    for ax, (title, site_set, color) in zip(axes, axis_defs):
        # patients with ≥1 site in the axis
        has_axis = X_df[list(site_set & set(sites))].any(axis=1).values if (site_set & set(sites)) else np.zeros(N, dtype=bool)
        ax.scatter(emb[~has_axis, 0], emb[~has_axis, 1], s=2, alpha=0.2, color="#cccccc")
        ax.scatter(emb[has_axis,  0], emb[has_axis,  1], s=6, alpha=0.6, color=color)
        ax.set_title(f"{title}\n(n={has_axis.sum():,})", fontsize=10)
        ax.set_xlabel(f"{proj_label}-1"); ax.set_ylabel(f"{proj_label}-2")
    fig.suptitle("Known cancer axes in VAE latent space", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_umap_known_axes.png", dpi=150)
    plt.close()

    # Fig C: axis loadings heatmap (sites × dims)
    n_show = min(30, len(sites))  # top 30 sites by max absolute loading
    top_sites_idx = load_df.abs().max(axis=1).nlargest(n_show).index
    show_df = load_df.loc[top_sites_idx]
    fig, ax = plt.subplots(figsize=(D_lat * 0.7 + 2, n_show * 0.35 + 1))
    im = ax.imshow(show_df.values, aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    ax.set_xticks(range(D_lat))
    ax.set_xticklabels([f"z{j}\n{interp_df.iloc[j]['axis_name'][:12]}" for j in range(D_lat)],
                       fontsize=7)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(top_sites_idx, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Spearman ρ")
    ax.set_title("VAE axis loadings: site × latent dimension (Spearman ρ)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_axis_loadings.png", dpi=150)
    plt.close()

    # Fig D: cluster enrichment heatmap (top 20 sites × clusters)
    enrich_mat = np.zeros((20, N_CLUSTERS))
    site_order = None
    for k in range(N_CLUSTERS):
        mask = labels == k
        prev_k = X[mask].mean(axis=0)
        enrich = prev_k / (overall_prev + 1e-9)
        enrich_mat[:, k] = pd.Series(enrich, index=sites).nlargest(20).values
        if site_order is None:
            site_order = pd.Series(enrich, index=sites).nlargest(20).index.tolist()

    fig, ax = plt.subplots(figsize=(N_CLUSTERS * 1.2 + 2, 7))
    im = ax.imshow(enrich_mat, aspect="auto", cmap="Reds", vmin=1, vmax=enrich_mat.max())
    ax.set_xticks(range(N_CLUSTERS))
    ax.set_xticklabels([f"C{k}\n{cluster_profiles[k]['name']}" for k in range(N_CLUSTERS)], fontsize=9)
    ax.set_yticks(range(20))
    ax.set_yticklabels(site_order, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Fold-enrichment vs overall")
    ax.set_title("Cluster site enrichment (top 20 sites, fold vs overall)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cluster_heatmap.png", dpi=150)
    plt.close()

    print(f"  Figures saved → results/03_latent/")
    print(f"  Novel axes (not UADT or Hormonal): "
          f"{sum(1 for r in interp_rows if r['axis_name'] == 'Novel')}")


if __name__ == "__main__":
    main()
