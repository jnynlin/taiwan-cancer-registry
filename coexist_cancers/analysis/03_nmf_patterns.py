"""
Non-negative Matrix Factorization (NMF) to discover latent cancer co-occurrence programs.
Analogous to mutational signature decomposition but applied to multi-cancer co-occurrence.
Each NMF component = a "cancer program" (set of cancers that appear together in patients).
Outputs: results/03_nmf/
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

BASE = Path(__file__).parent.parent
MAT  = BASE / "data/patient_cancer_matrix.csv"
SITE = BASE / "data/cancer_site_labels.csv"
META = BASE / "data/patient_meta.csv"
OUT  = BASE / "results/03_nmf"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)
np.random.seed(42)

PALETTE = sns.color_palette("tab10")


def load():
    matrix = pd.read_csv(MAT, index_col=0).astype(float)
    sites  = pd.read_csv(SITE)
    meta   = pd.read_csv(META)
    label  = dict(zip(sites["code"], sites["label"]))
    return matrix, sites, meta, label


def select_k(X, k_range=range(2, 12)):
    """Reconstruction error and silhouette to select optimal k."""
    recon_errors, sil_scores = [], []
    for k in k_range:
        model = NMF(n_components=k, init="nndsvda", max_iter=500, random_state=42)
        W = model.fit_transform(X)
        recon_errors.append(model.reconstruction_err_)
        labels = np.argmax(W, axis=1)
        if len(np.unique(labels)) > 1:
            sil_scores.append(silhouette_score(X, labels, metric="cosine"))
        else:
            sil_scores.append(0)
        print(f"  k={k:2d}  recon_err={model.reconstruction_err_:.3f}  sil={sil_scores[-1]:.4f}")
    return list(k_range), recon_errors, sil_scores


def plot_k_selection(k_range, recon_errors, sil_scores):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(k_range, recon_errors, "o-", color="steelblue")
    axes[0].set(title="NMF Reconstruction Error vs k", xlabel="k (number of components)",
                ylabel="Frobenius reconstruction error")
    axes[0].spines[["top","right"]].set_visible(False)
    axes[1].plot(k_range, sil_scores, "o-", color="tomato")
    best_k = k_range[np.argmax(sil_scores)]
    axes[1].axvline(best_k, color="gray", linestyle="--", label=f"Best k={best_k}")
    axes[1].set(title="Silhouette Score vs k", xlabel="k", ylabel="Silhouette (cosine)")
    axes[1].legend(); axes[1].spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "nmf_k_selection.png", dpi=150)
    plt.close(fig)
    return best_k


def fit_nmf(X, k):
    model = NMF(n_components=k, init="nndsvda", max_iter=1000,
                l1_ratio=0.1, alpha_W=0.01, random_state=42)
    W = model.fit_transform(X)   # patients × k
    H = model.components_         # k × cancer_sites
    print(f"  NMF fitted: k={k}, recon_err={model.reconstruction_err_:.4f}")
    return model, W, H


def plot_components(H, sites, label, k, fname="nmf_components.png"):
    """Bar chart showing top cancer sites per component (cancer program)."""
    H_df = pd.DataFrame(H, columns=sites,
                         index=[f"Program {i+1}" for i in range(k)])
    H_norm = H_df.div(H_df.sum(axis=1), axis=0)  # normalize rows

    fig, axes = plt.subplots(1, k, figsize=(4*k, 5), sharey=False)
    if k == 1: axes = [axes]
    for i, ax in enumerate(axes):
        prog = H_norm.iloc[i].nlargest(12)
        prog_labels = [label.get(c,c)[:16] for c in prog.index]
        colors = [PALETTE[i % len(PALETTE)]] * len(prog)
        ax.barh(prog_labels[::-1], prog.values[::-1], color=colors, edgecolor="white")
        ax.set(title=f"Program {i+1}", xlabel="Normalized loading")
        ax.tick_params(labelsize=8)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"NMF Cancer Co-occurrence Programs (k={k})",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_patient_program_heatmap(W, meta, k):
    """Heatmap of patient program loadings for multi-primary subset."""
    W_df = pd.DataFrame(W, columns=[f"P{i+1}" for i in range(k)])
    W_df = W_df.join(meta.set_index("pid")[["n_cancers","age_first","sex","any_death"]])
    mp   = W_df[W_df["n_cancers"]>=2].copy()
    if len(mp) > 500:
        mp = mp.sample(500, random_state=42)
    mp_sort = mp.sort_values(f"P1")
    prog_cols = [f"P{i+1}" for i in range(k)]
    fig, ax = plt.subplots(figsize=(k*1.5+3, 8))
    sns.heatmap(mp_sort[prog_cols].T, cmap="YlOrRd", ax=ax,
                xticklabels=False, yticklabels=True,
                cbar_kws={"label":"Program loading"})
    ax.set(title=f"Patient Program Loadings — Multi-primary Cohort (n≤500 sample)",
           ylabel="NMF Program")
    fig.tight_layout()
    fig.savefig(OUT / "patient_program_heatmap.png", dpi=150)
    plt.close(fig)
    return mp


def program_clinical_profile(W, meta, k):
    """For each dominant program, describe clinical characteristics."""
    W_df = pd.DataFrame(W, columns=[f"P{i+1}" for i in range(k)])
    W_df["dominant_program"] = W_df[[f"P{i+1}" for i in range(k)]].idxmax(axis=1)
    W_df = W_df.join(meta.set_index("pid"))
    rows = []
    for prog in [f"P{i+1}" for i in range(k)]:
        sub = W_df[W_df["dominant_program"]==prog]
        rows.append({
            "Program": prog,
            "n_patients": len(sub),
            "pct_multi_primary": f"{100*(sub['n_cancers']>=2).mean():.1f}%",
            "median_age_first": round(sub["age_first"].median(), 1),
            "pct_male": f"{100*(sub['sex']=='M').mean():.1f}%",
            "pct_death": f"{100*sub['any_death'].mean():.1f}%",
        })
    prof = pd.DataFrame(rows)
    prof.to_csv(OUT/"program_clinical_profile.csv", index=False, encoding="utf-8-sig")
    print("\n  Clinical profile by dominant NMF program:")
    print(prof.to_string(index=False))
    return prof


if __name__ == "__main__":
    print("Loading data...")
    matrix, sites, meta, label = load()
    X = matrix.values.astype(float)
    site_codes = matrix.columns.tolist()
    print(f"  Matrix: {X.shape}")

    print("\nSelecting optimal k...")
    k_range, recon, sil = select_k(X, k_range=range(2, 10))
    best_k = plot_k_selection(k_range, recon, sil)
    print(f"  → Best k = {best_k}")

    print(f"\nFitting NMF with k={best_k}...")
    model, W, H = fit_nmf(X, best_k)

    # Save W (patient loadings) and H (component weights)
    W_df = pd.DataFrame(W, index=matrix.index,
                         columns=[f"NMF_P{i+1}" for i in range(best_k)])
    H_df = pd.DataFrame(H, columns=site_codes,
                         index=[f"NMF_P{i+1}" for i in range(best_k)])
    W_df.to_csv(OUT/"patient_nmf_loadings.csv", encoding="utf-8-sig")
    H_df.to_csv(OUT/"component_weights.csv",    encoding="utf-8-sig")

    # Plots
    print("\nGenerating component plots...")
    plot_components(H, site_codes, label, best_k)
    plot_patient_program_heatmap(W_df, meta, best_k)
    program_clinical_profile(W_df, meta, best_k)

    # Top cancer sites per program summary
    print("\nTop 5 cancer sites per NMF program:")
    for i in range(best_k):
        top5 = H_df.iloc[i].nlargest(5)
        sites_str = ", ".join(f"{label.get(c,c)} ({v:.3f})" for c,v in top5.items())
        print(f"  Program {i+1}: {sites_str}")

    print(f"\nDone. Results in {OUT}")
