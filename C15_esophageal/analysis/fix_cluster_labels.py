"""Regenerate cluster UMAP and KM figures with 1-indexed labels (1/2/3 instead of 0/1/2)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from pathlib import Path

OUT = Path("/home/jnynlin/coding/taiwan-cancer-registry/C15_esophageal/results/04_deep_learning")

embedding = np.load(OUT / "umap_embedding.npy")
df = pd.read_csv(OUT / "c15_final_annotated.csv", low_memory=False)

# Map 0-indexed cluster labels → 1-indexed for manuscript
cluster_map = {0: 2, 1: 1, 2: 3}
# cluster 0 = adenocarcinoma-enriched → Cluster 2 in text
# cluster 1 = mainstream SCC          → Cluster 1 in text
# cluster 2 = cervical/upper          → Cluster 3 in text
df["cluster_display"] = df["cluster_kmeans"].map(cluster_map)

palette = ["tab:blue", "tab:orange", "tab:green"]
cluster_names = {1: "Mainstream SCC", 2: "Adeno-enriched", 3: "Cervical/upper"}

# ── UMAP with 1-indexed labels ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for idx, (c_disp, name) in enumerate(cluster_names.items()):
    mask = df["cluster_display"] == c_disp
    valid = mask & (df.index < len(embedding))
    ax.scatter(embedding[valid, 0], embedding[valid, 1],
               c=[palette[idx]], label=f"Cluster {c_disp} — {name} (n={valid.sum()})",
               alpha=0.6, s=15)
ax.set(title="UMAP — Autoencoder Patient Subtypes (k=3)", xlabel="UMAP-1", ylabel="UMAP-2")
ax.legend(markerscale=2, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "umap_kmeans_k3.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved umap_kmeans_k3.png")

# ── KM curves with 1-indexed labels ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
kmf = KaplanMeierFitter()
sub = df[["cluster_display", "os_months", "event"]].dropna(subset=["os_months", "event"])
sub = sub[sub["cluster_display"].notna()]
for idx, c_disp in enumerate([1, 2, 3]):
    g = sub[sub["cluster_display"] == c_disp]
    kmf.fit(g["os_months"], g["event"], label=f"Cluster {c_disp} — {cluster_names[c_disp]} (n={len(g)})")
    kmf.plot_survival_function(ax=ax, ci_show=False, color=palette[idx])
res = multivariate_logrank_test(sub["os_months"], sub["cluster_display"], sub["event"])
ax.text(0.62, 0.85, f"Log-rank p = {res.p_value:.4f}", transform=ax.transAxes,
        fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax.set(title="Overall Survival by Autoencoder Cluster", xlabel="Months",
       ylabel="Survival probability", ylim=(0, 1.05))
fig.tight_layout()
fig.savefig(OUT / "km_clusters_k3.png", dpi=150)
plt.close(fig)
print("Saved km_clusters_k3.png")
print(f"KM n per cluster:\n{sub['cluster_display'].value_counts().sort_index()}")
