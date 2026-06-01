"""
Deep learning: Autoencoder + UMAP + k-means on multi-hot cancer co-occurrence matrix.
Optimized: trains on multi-primary cohort only (n≈4k, fast); projects all patients
into latent space post-hoc; subsamples UMAP to ≤12k for tractable embedding.
Outputs: results/04_clustering/
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap as umap_lib

BASE = Path(__file__).parent.parent
MAT  = BASE / "data/patient_cancer_matrix.csv"
SITE = BASE / "data/cancer_site_labels.csv"
META = BASE / "data/patient_meta.csv"
OUT  = BASE / "results/04_clustering"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="white", font_scale=1.0)
PALETTE = sns.color_palette("tab10")
torch.manual_seed(42); np.random.seed(42)

# ── Model ─────────────────────────────────────────────────────────────────────
class CancerAE(nn.Module):
    def __init__(self, d, latent=12):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, 64),  nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, 64),     nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, d),      nn.Sigmoid(),
        )
    def forward(self, x):
        z = self.enc(x); return self.dec(z), z


def train_ae(X, latent=12, epochs=150, bs=128, lr=1e-3):
    Xt = torch.tensor(X, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt), batch_size=bs, shuffle=True)
    model = CancerAE(X.shape[1], latent)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit  = nn.BCELoss()
    losses = []
    for ep in range(epochs):
        model.train(); el = 0.0
        for (b,) in dl:
            r, _ = model(b); loss = crit(r, b)
            opt.zero_grad(); loss.backward(); opt.step(); el += loss.item()*len(b)
        sched.step(); losses.append(el/len(Xt))
        if (ep+1) % 30 == 0:
            print(f"    epoch {ep+1}/{epochs}  BCE={losses[-1]:.5f}")
    return model, losses


@torch.no_grad()
def encode(model, X, bs=512):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    parts = []
    for i in range(0, len(Xt), bs):
        _, z = model(Xt[i:i+bs]); parts.append(z)
    return torch.cat(parts).numpy()


def umap_embed(Z, n_neighbors=20, min_dist=0.1, max_samples=12_000):
    """Sub-sample for speed; embed subset, reuse reducer for full projection."""
    if len(Z) > max_samples:
        idx = np.random.choice(len(Z), max_samples, replace=False)
        print(f"  UMAP subsampling {len(Z):,} → {max_samples:,} for training")
    else:
        idx = np.arange(len(Z))
    reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                             n_components=2, metric="cosine",
                             random_state=42, low_memory=True)
    emb_sub = reducer.fit_transform(Z[idx])
    if len(Z) > max_samples:
        print("  Projecting remaining points…")
        emb_all = np.full((len(Z), 2), np.nan)
        emb_all[idx] = emb_sub
        # Transform remaining in batches
        rest = np.setdiff1d(np.arange(len(Z)), idx)
        for i in range(0, len(rest), 2000):
            chunk = rest[i:i+2000]
            emb_all[chunk] = reducer.transform(Z[chunk])
    else:
        emb_all = emb_sub
    return emb_all, reducer


def scatter_umap(emb, cats, title, fname, palette="tab10", s=6, alpha=0.45):
    uniq = [c for c in dict.fromkeys(cats) if pd.notna(c)]
    pal  = sns.color_palette(palette, len(uniq))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i, g in enumerate(uniq):
        m = cats == g
        ax.scatter(emb[m,0], emb[m,1], c=[pal[i]], s=s, alpha=alpha,
                   label=str(g), linewidths=0, rasterized=True)
    ax.set(title=title, xlabel="UMAP-1", ylabel="UMAP-2")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(markerscale=2.5, fontsize=7, bbox_to_anchor=(1.01,1), loc="upper left")
    fig.tight_layout(); fig.savefig(OUT/fname, dpi=150, bbox_inches="tight"); plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data…")
    matrix = pd.read_csv(MAT, index_col=0).astype(float)
    sites  = pd.read_csv(SITE)
    meta   = pd.read_csv(META)
    label  = dict(zip(sites["code"], sites["label"]))
    site_codes = matrix.columns.tolist()

    # Fix vital status (Taiwan: 0=Dead)
    all_raw = pd.read_csv(
        Path(__file__).parent.parent.parent / "data/processed/all_cancers.csv",
        low_memory=False, usecols=["病歷號(2)","生存狀態(27)"])
    all_raw["pid"]  = all_raw["病歷號(2)"].astype(str).str.strip()
    # Taiwan registry: 0=Dead, 1=Alive. Column is float (0.0/1.0) due to NaNs.
    all_raw["dead"] = (pd.to_numeric(all_raw["生存狀態(27)"], errors="coerce")==0).astype("Int64")
    death = all_raw.groupby("pid")["dead"].max().reset_index()
    meta["pid"] = meta["pid"].astype(str)
    meta = meta.merge(death, on="pid", how="left")

    # Multi-primary subset for AE training
    mp_mask  = matrix.sum(axis=1) >= 2
    mp_mat   = matrix[mp_mask]
    keep_col = mp_mat.sum(axis=0) >= 3
    X_mp     = mp_mat.loc[:, keep_col].values.astype(float)
    X_all    = matrix.loc[:, keep_col].values.astype(float)
    ks       = [c for c, k in zip(site_codes, keep_col) if k]
    n_mp, n_all = len(X_mp), len(X_all)
    print(f"  Multi-primary (training): {n_mp:,}  |  All patients (inference): {n_all:,}  |  Sites: {len(ks)}")

    # ── 1. Train AE on multi-primary ─────────────────────────────────────────
    print(f"\n[1/5] Training autoencoder on multi-primary cohort (latent=12, epochs=150)…")
    model, losses = train_ae(X_mp, latent=12, epochs=150)
    np.save(OUT/"ae_losses.npy", losses)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(losses, color="steelblue")
    ax.set(title="Autoencoder Training Loss (BCE)", xlabel="Epoch", ylabel="Loss")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT/"ae_loss.png", dpi=150); plt.close(fig)
    print(f"  Final BCE loss: {losses[-1]:.5f}")

    # Encode ALL patients (including single-primary)
    print("\n[2/5] Encoding all patients into latent space…")
    Z_mp  = encode(model, X_mp)
    Z_all = encode(model, X_all)
    np.save(OUT/"latent_z_multiprimary.npy", Z_mp)
    np.save(OUT/"latent_z_all.npy",          Z_all)
    print(f"  Latent shapes: mp={Z_mp.shape}  all={Z_all.shape}")

    # ── 2. UMAP (subsampled) ──────────────────────────────────────────────────
    print("\n[3/5] UMAP embedding (subsampled ≤12k for speed)…")
    emb_all, reducer = umap_embed(Z_all, max_samples=12_000)
    emb_mp = emb_all[mp_mask.values]
    np.save(OUT/"umap_all.npy", emb_all)
    np.save(OUT/"umap_mp.npy",  emb_mp)
    print("  Done.")

    # Color by n cancers
    n_canc = matrix.sum(axis=1).values
    n_cat  = pd.cut(n_canc, bins=[0,1,2,3,20],
                    labels=["1 cancer","2 cancers","3 cancers","4+ cancers"])
    scatter_umap(emb_all, np.array(n_cat),
                 "UMAP — All Patients by Number of Cancer Types",
                 "umap_n_cancers.png", palette="RdYlGn_r")

    # NMF program overlay (multi-primary only)
    W_path = BASE/"results/03_nmf/patient_nmf_loadings_multiprimary.csv"
    if W_path.exists():
        W_df = pd.read_csv(W_path, index_col=0)
        W_df.index = W_df.index.astype(str)
        prog_labels = W_df.idxmax(axis=1).reindex(mp_mat.index.astype(str)).values
        PROG_NAMES = {
            "P1":"P1-Colorectal","P2":"P2-Aerodigestive SCC",
            "P3":"P3-Lung","P4":"P4-Liver/GI",
            "P5":"P5-Oral cavity","P6":"P6-Female genital/breast",
            "P7":"P7-Urological"}
        prog_named = np.array([PROG_NAMES.get(p,p) if pd.notna(p) else "?" for p in prog_labels])
        scatter_umap(emb_mp, prog_named,
                     "UMAP (Multi-primary) — NMF Program Assignment",
                     "umap_nmf_programs.png", palette="tab10", s=10, alpha=0.65)

    # ── 3. k-means on latent space (multi-primary) ───────────────────────────
    print("\n[4/5] k-means on multi-primary latent space…")
    sil = {}
    for k in range(2, 8):
        km  = KMeans(n_clusters=k, random_state=42, n_init=20)
        lbl = km.fit_predict(Z_mp)
        sil[k] = silhouette_score(Z_mp, lbl, metric="cosine") if len(np.unique(lbl))>1 else 0
        print(f"  k={k}: sil={sil[k]:.4f}")
    best_k = max(sil, key=sil.get)
    print(f"  Best k={best_k}")

    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    cl_mp   = km_best.fit_predict(Z_mp)
    cl_all  = km_best.predict(Z_all)

    scatter_umap(emb_mp,
                 np.array([f"C{l+1}" for l in cl_mp]),
                 f"UMAP (Multi-primary) — k-Means Clusters (k={best_k})",
                 f"umap_kmeans_k{best_k}.png")
    scatter_umap(emb_all,
                 np.array([f"C{l+1}" for l in cl_all]),
                 f"UMAP (All patients) — k-Means Clusters (k={best_k})",
                 f"umap_kmeans_all_k{best_k}.png")

    # ── 4. Cluster profiles ───────────────────────────────────────────────────
    print(f"\n[5/5] Cluster profiles…")
    mp_df = mp_mat.copy()
    mp_df["cluster"] = [f"C{l+1}" for l in cl_mp]
    # Align join keys: matrix index is numeric pid, meta pid was cast to str.
    meta_idx = meta.copy()
    meta_idx["pid"] = pd.to_numeric(meta_idx["pid"], errors="coerce")
    mp_df = mp_df.join(meta_idx.set_index("pid"))

    PROG_SHORT = {"P1":"Colorectal","P2":"Aerodig SCC","P3":"Lung","P4":"Liver/GI",
                  "P5":"Oral cavity","P6":"Female genital","P7":"Urological"}

    rows = []
    print(f"\n  k={best_k} cluster profiles:")
    print(f"  {'Cluster':<10} {'n':>6} {'≥3 cancers':>11} {'age':>5} {'M%':>5} "
          f"{'death%':>7} {'top NMF prog':>14} {'top 4 sites'}")
    print("  " + "─"*90)
    for c in sorted(mp_df["cluster"].unique()):
        sub = mp_df[mp_df["cluster"]==c]
        top_sites = sub[ks].sum().nlargest(4)
        top_site_str = ", ".join(f"{label.get(s,s)[:12]}({int(v)})"
                                  for s,v in top_sites.items())
        # dominant NMF program in this cluster
        if W_path.exists():
            sub_w = W_df.reindex(sub.index.astype(str))
            top_prog = sub_w.idxmax(axis=1).value_counts().idxmax() if len(sub_w)>0 else "?"
            top_prog_name = PROG_SHORT.get(top_prog, top_prog)
        else:
            top_prog_name = "?"
        n3    = 100*(sub["n_cancers"]>=3).mean()
        death = 100*sub["dead"].mean() if "dead" in sub else float("nan")
        print(f"  {c:<10} {len(sub):>6} {n3:>10.1f}% {sub['age_first'].median():>5.0f}"
              f" {100*(sub['sex']=='M').mean():>4.0f}% {death:>6.1f}%"
              f" {top_prog_name:>14}  {top_site_str}")
        rows.append({"cluster":c,"n":len(sub),"pct_3plus":round(n3,1),
                      "median_age":sub["age_first"].median(),
                      "pct_male":round(100*(sub["sex"]=="M").mean(),1),
                      "pct_death":round(death,1),"dominant_nmf":top_prog_name,
                      "top_sites":top_site_str})

    prof = pd.DataFrame(rows)
    prof.to_csv(OUT/f"cluster_k{best_k}_profile.csv", index=False, encoding="utf-8-sig")

    # Cluster × cancer heatmap
    top25 = mp_mat[ks].sum().nlargest(25).index
    heat  = mp_df.groupby("cluster")[list(top25)].mean()
    heat.columns = [label.get(c,c)[:14] for c in heat.columns]
    fig, ax = plt.subplots(figsize=(14, max(4, best_k*0.9)))
    sns.heatmap(heat, cmap="YlOrRd", ax=ax, linewidths=0.3,
                annot=True, fmt=".2f", annot_kws={"size":6.5},
                cbar_kws={"label":"Mean prevalence"})
    ax.set(title=f"Cancer Prevalence by Cluster — Multi-primary (top 25 sites)")
    ax.tick_params(axis="x", rotation=45, labelsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT/f"cluster_cancer_heatmap_k{best_k}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Silhouette plot
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(list(sil.keys()), list(sil.values()), "o-", color="steelblue")
    ax.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    ax.set(title="Silhouette vs k", xlabel="k", ylabel="Silhouette (cosine)")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT/"silhouette.png", dpi=150); plt.close(fig)

    # Save annotated CSV
    mp_df.to_csv(OUT/f"patients_annotated_k{best_k}.csv", encoding="utf-8-sig")
    print(f"\n  Done. All outputs in {OUT}")
