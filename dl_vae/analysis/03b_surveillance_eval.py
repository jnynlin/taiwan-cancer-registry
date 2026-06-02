"""
Registry DL — Script 03b: Surveillance Evaluation + Heatmap

Evaluates the masked predictor's clinical utility:

  1. Surveillance heatmap: for each primary cancer, ranked P(secondary)
  2. Model vs baseline comparison: does the model beat co-occurrence frequency?
  3. Case studies: C12 pyriform, C13 hypopharynx, C15 esophagus — exact predictions
  4. Demographic modulation: how much does age/sex shift predictions?

Outputs:
  results/03b_surveillance/fig_surveillance_heatmap.png
  results/03b_surveillance/fig_model_vs_baseline.png
  results/03b_surveillance/fig_case_studies.png
  results/03b_surveillance/fig_age_sex_effect.png
  results/03b_surveillance/surveillance_table.csv   — top-5 per primary cancer
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    print("ERROR: PyTorch not found.")
    raise SystemExit(1)

BASE  = Path(__file__).parent.parent
DOUT  = BASE / "data"
MOUT  = BASE / "models"
OUT   = BASE / "results/03b_surveillance"
OUT.mkdir(parents=True, exist_ok=True)

# Case studies (pre-registered UADT sites)
CASE_SITES = ["C12", "C13", "C15", "C06", "C34", "C50"]

NAVY   = "#14304a"
ACCENT = "#2e7fbf"


# ── Rebuild model skeleton (must match 02b exactly) ───────────────────────────

class CancerMaskedPredictor(nn.Module):
    def __init__(self, n_sites, hidden_dim=256, dropout=0.25):
        super().__init__()
        in_dim = n_sites + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_sites),
        )

    def forward(self, x):
        return self.net(x)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def predict_next(model, sites, primary_sites, age, sex_male,
                 age_mean, age_std):
    """Given known cancer sites + demographics, return P(next site = X) via softmax.

    Softmax is used instead of sigmoid because:
    - Sigmoid gives independent P(present) per site → saturates to 1.0 when model
      sees an atypical (sparse) input vs the training distribution
    - Softmax gives a proper probability distribution: "of all possible next sites,
      how likely is each one?" — directly interpretable for surveillance ranking
    """
    n_s = len(sites)
    x   = np.zeros(n_s + 2, dtype=np.float32)
    for s in primary_sites:
        if s in sites:
            x[sites.index(s)] = 1.0
    x[n_s]     = (age - age_mean) / age_std
    x[n_s + 1] = 1.0 if sex_male else 0.0
    with torch.no_grad():
        logits = model(torch.tensor(x).unsqueeze(0)).squeeze(0).numpy()
    # Zero out sites already present before softmax (so they rank last)
    for s in primary_sites:
        if s in sites:
            logits[sites.index(s)] = -1e9
    probs = softmax(logits)
    return probs


def main():
    print("=== Registry DL — 03b: Surveillance Evaluation ===")

    X_df  = pd.read_csv(DOUT / "cancer_matrix.csv", index_col="pid")
    meta  = pd.read_csv(DOUT / "patient_meta.csv",  index_col="pid")
    sites = X_df.columns.tolist()
    N_S   = len(sites)

    # Demographics for inference
    age_mean = meta["age_first"].mean()
    age_std  = meta["age_first"].std() + 1e-6

    # Load model
    model = CancerMaskedPredictor(N_S)
    model.load_state_dict(torch.load(MOUT / "predictor_weights.pt",
                                     map_location="cpu"))
    model.eval()
    print(f"  Loaded model · {N_S} sites")

    # ── 1. Surveillance heatmap ───────────────────────────────────────────────
    # For each primary site: query at median demographic for that site's patients
    print("  Building surveillance heatmap…")
    heatmap = np.zeros((N_S, N_S))
    demo_profiles = {}

    for i, primary in enumerate(sites):
        mask = X_df[primary] == 1
        if mask.sum() < 5:
            demo_profiles[primary] = (age_mean, True)
            continue
        age_p = meta.loc[mask, "age_first"].median()
        male_p = (meta.loc[mask, "sex"] == "M").mean() >= 0.5
        demo_profiles[primary] = (age_p, male_p)
        probs = predict_next(model, sites, [primary], age_p, male_p,
                             age_mean, age_std)
        heatmap[i, :] = probs

    # Raw co-occurrence rate baseline
    multi = meta[meta["n_sites"] >= 2].index
    cooc_rate = np.zeros((N_S, N_S))
    for i, s1 in enumerate(sites):
        n_s1 = X_df.loc[multi, s1].sum()
        if n_s1 == 0:
            continue
        for j, s2 in enumerate(sites):
            if i == j: continue
            n_both = ((X_df.loc[multi, s1] == 1) & (X_df.loc[multi, s2] == 1)).sum()
            cooc_rate[i, j] = n_both / (n_s1 + 1e-9)

    # ── 2. Top-5 per primary site table ──────────────────────────────────────
    rows = []
    for i, primary in enumerate(sites):
        if heatmap[i].sum() == 0:
            continue
        order = np.argsort(-heatmap[i])
        top5  = [(sites[j], round(float(heatmap[i, j]), 4)) for j in order[:5]]
        rows.append({"primary": primary,
                     "top1": f"{top5[0][0]} ({top5[0][1]:.3f})",
                     "top2": f"{top5[1][0]} ({top5[1][1]:.3f})",
                     "top3": f"{top5[2][0]} ({top5[2][1]:.3f})",
                     "top4": f"{top5[3][0]} ({top5[3][1]:.3f})",
                     "top5": f"{top5[4][0]} ({top5[4][1]:.3f})"})
    surv_df = pd.DataFrame(rows)
    surv_df.to_csv(OUT / "surveillance_table.csv", index=False)

    # ── 3. Case studies ───────────────────────────────────────────────────────
    print("  Case studies:")
    case_results = {}
    for cs in CASE_SITES:
        if cs not in sites:
            continue
        age_p, male_p = demo_profiles[cs]
        probs = predict_next(model, sites, [cs], age_p, male_p, age_mean, age_std)
        order = np.argsort(-probs)
        top10 = [(sites[j], probs[j]) for j in order[:10]]
        case_results[cs] = top10
        print(f"  {cs}: top-3 = {[f'{s}({p:.3f})' for s,p in top10[:3]]}")

    # ── 4. Demographic modulation (C15 as example) ───────────────────────────
    cs_demo = "C15" if "C15" in sites else sites[0]
    ages    = [40, 50, 55, 60, 65, 70]
    demo_probs = {}
    for ag in ages:
        p = predict_next(model, sites, [cs_demo], ag, True, age_mean, age_std)
        demo_probs[ag] = p

    # ── Figures ───────────────────────────────────────────────────────────────

    # Fig A: surveillance heatmap
    np.fill_diagonal(heatmap, np.nan)
    fig, ax = plt.subplots(figsize=(14, 12))
    masked = np.ma.masked_invalid(heatmap)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("white")
    im = ax.imshow(masked, cmap=cmap, aspect="auto",
                   vmin=0, vmax=np.nanpercentile(heatmap, 95))
    ax.set_xticks(range(N_S)); ax.set_xticklabels(sites, rotation=90, fontsize=7)
    ax.set_yticks(range(N_S)); ax.set_yticklabels(sites, fontsize=7)
    ax.set_xlabel("Predicted secondary cancer"); ax.set_ylabel("Primary cancer")
    ax.set_title("Model-predicted surveillance priority: P(next cancer = j | primary = i, demographics)\n"
                 "(softmax; diagonal masked; higher = higher surveillance priority)", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Softmax priority score")
    fig.tight_layout()
    fig.savefig(OUT / "fig_surveillance_heatmap.png", dpi=150)
    plt.close()

    # Fig B: model vs raw co-occurrence baseline (scatter)
    fig, ax = plt.subplots(figsize=(7, 6))
    h_flat = heatmap.copy(); np.fill_diagonal(h_flat, np.nan)
    c_flat = cooc_rate.copy(); np.fill_diagonal(c_flat, np.nan)
    mask_v = ~np.isnan(h_flat) & ~np.isnan(c_flat) & (c_flat > 0)
    ax.scatter(c_flat[mask_v], h_flat[mask_v], alpha=0.3, s=8, color=ACCENT)
    # Highlight UADT pairs
    uadt = {s for s in ['C02','C03','C04','C05','C06','C09','C10','C12','C13','C15']
            if s in sites}
    for s1 in uadt:
        for s2 in uadt:
            if s1 == s2: continue
            i, j = sites.index(s1), sites.index(s2)
            if not np.isnan(h_flat[i,j]):
                ax.scatter(c_flat[i,j], h_flat[i,j], s=40, color="#e05c2e", zorder=5)
    ax.set_xlabel("Raw co-occurrence rate (multi-cancer patients)")
    ax.set_ylabel("Model P(secondary)")
    ax.set_title("Model vs baseline: all site pairs\n(orange = UADT pairs)")
    lims = [0, max(c_flat[mask_v].max(), h_flat[mask_v].max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="y=x (perfect agreement)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_model_vs_baseline.png", dpi=150)
    plt.close()

    # Fig C: case study bar charts
    n_cases = len(case_results)
    if n_cases:
        fig, axes = plt.subplots(1, n_cases, figsize=(n_cases * 4, 5), sharey=False)
        if n_cases == 1: axes = [axes]
        for ax, cs in zip(axes, case_results):
            top10  = case_results[cs]
            labels = [s for s, _ in top10]
            vals   = [p for _, p in top10]
            colors = ["#e05c2e" if l in uadt else ACCENT for l in labels]
            ax.barh(range(len(labels))[::-1], vals, color=colors[::-1])
            ax.set_yticks(range(len(labels))[::-1])
            ax.set_yticklabels(labels[::-1], fontsize=9)
            ax.set_xlabel("P(secondary)")
            age_p, male_p = demo_profiles.get(cs, (age_mean, True))
            sex_s = "M" if male_p else "F"
            ax.set_title(f"Primary: {cs}\n(age={age_p:.0f}, sex={sex_s})", fontsize=10)
        fig.suptitle("Top-10 predicted secondary cancers — case studies\n(orange = UADT field sites)",
                     fontsize=11, color=NAVY, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / "fig_case_studies.png", dpi=150)
        plt.close()

    # Fig D: age effect on C15 predictions
    if cs_demo in sites and len(demo_probs) > 1:
        # Show top-6 sites and how their probability changes with age
        avg_p = np.mean([demo_probs[ag] for ag in ages], axis=0)
        avg_p[sites.index(cs_demo)] = 0  # exclude primary
        top6_idx = np.argsort(-avg_p)[:6]
        top6_sites = [sites[j] for j in top6_idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        for j, s in zip(top6_idx, top6_sites):
            probs_by_age = [demo_probs[ag][j] for ag in ages]
            color = "#e05c2e" if s in uadt else ACCENT
            ax.plot(ages, probs_by_age, marker="o", label=s, color=color)
        ax.set_xlabel("Age at primary diagnosis")
        ax.set_ylabel("P(secondary cancer)")
        ax.set_title(f"Age modulation of secondary cancer risk\n(primary = {cs_demo}, sex = M)")
        ax.legend(fontsize=9, loc="upper right")
        fig.tight_layout()
        fig.savefig(OUT / "fig_age_sex_effect.png", dpi=150)
        plt.close()

    print(f"  Figures saved → results/03b_surveillance/")
    print(f"  Surveillance table: {len(surv_df)} primary sites × top-5")


if __name__ == "__main__":
    main()
