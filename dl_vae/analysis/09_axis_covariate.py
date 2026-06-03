"""
Registry DL — Script 09: Axis Covariate Characterisation

For each of the N_ACTIVE most informative VAE latent axes (highest max |loading|),
compute associations with demographic and temporal covariates:
  - Sex (M/F): Mann-Whitney U + rank-biserial effect size
  - Age at first diagnosis: Spearman rho
  - Diagnosis year (era): Spearman rho + mean-per-year trend plot
  - Vital status (dead): point-biserial r

Active dims identified from axis_loadings.csv (max |Spearman rho| across all sites)
rather than KL divergence, which was not saved as CSV during training.

Outputs:
  results/09_axis_covariate/axis_covariate_stats.csv
  results/09_axis_covariate/fig_active_dims.png
  results/09_axis_covariate/fig_axis_sex.png
  results/09_axis_covariate/fig_axis_age.png
  results/09_axis_covariate/fig_axis_era.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE    = Path(__file__).parent.parent
DOUT    = BASE / "data"
R03     = BASE / "results/03_latent"
OUT     = BASE / "results/09_axis_covariate"
OUT.mkdir(parents=True, exist_ok=True)

N_ACTIVE = 3
PALETTE  = ["#2e7fbf", "#e05c2e", "#2ca02c"]


def main():
    print("=== Registry DL — 09: Axis Covariate Characterisation ===")

    mu      = np.load(DOUT / "latent_mu.npy")           # (N, 12)
    meta    = pd.read_csv(DOUT / "patient_meta.csv", index_col="pid")
    load_df = pd.read_csv(R03 / "axis_loadings.csv", index_col=0)   # sites × dims
    interp  = pd.read_csv(R03 / "axis_interpretation.csv", index_col="dim")

    N, D = mu.shape
    print(f"  Loaded: {N:,} patients · {D} latent dims")

    # Active dims: highest max |Spearman rho| across all sites
    max_abs = load_df.abs().max(axis=0)          # Series indexed z0..z11
    top_cols = max_abs.nlargest(N_ACTIVE).index  # e.g. ['z0','z2','z3']
    active_dims = [int(c[1:]) for c in top_cols]
    axis_labels = [interp.loc[c, "axis_name"] if c in interp.index else c
                   for c in top_cols]
    print(f"  Active dims: {list(zip(top_cols.tolist(), axis_labels))}")

    # Covariates
    is_male = (meta["sex"] == "M").values
    age     = meta["age_first"].values.astype(float)
    era     = meta["diag_yr"].values.astype(int)
    dead    = meta["dead"].values.astype(float)

    # ── Statistics ──────────────────────────────────────────────────────────
    rows = []
    for col, dim, label in zip(top_cols, active_dims, axis_labels):
        z = mu[:, dim]
        g_m, g_f = z[is_male], z[~is_male]

        mw_stat, mw_p = stats.mannwhitneyu(g_m, g_f, alternative="two-sided")
        rbi = 1 - (2 * mw_stat) / (len(g_m) * len(g_f))

        r_age, p_age = stats.spearmanr(z, age)
        r_era, p_era = stats.spearmanr(z, era)
        r_dead, p_dead = stats.pointbiserialr(dead, z)

        rows.append(dict(dim=col, axis_name=label,
                         max_loading=max_abs[col],
                         sex_mw_p=mw_p, sex_rbi=rbi,
                         age_r=r_age, age_p=p_age,
                         era_r=r_era, era_p=p_era,
                         dead_r=r_dead, dead_p=p_dead))
        print(f"  {col} [{label}]: sex p={mw_p:.2e} rbi={rbi:.3f} | "
              f"age ρ={r_age:.3f} p={p_age:.2e} | "
              f"era ρ={r_era:.3f} p={p_era:.2e} | "
              f"dead r={r_dead:.3f} p={p_dead:.2e}")

    pd.DataFrame(rows).to_csv(OUT / "axis_covariate_stats.csv", index=False)

    # ── Fig A: Active dim identification ────────────────────────────────────
    bar_colors = ["#cccccc"] * D
    for d in active_dims:
        bar_colors[d] = "#2e7fbf"

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(range(D), max_abs.values, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(D))
    ax.set_xticklabels(
        [f"z{j}" + (f"\n{axis_labels[active_dims.index(j)][:10]}" if j in active_dims else "")
         for j in range(D)], fontsize=7)
    ax.axhline(max_abs.values[np.argsort(max_abs.values)[::-1][N_ACTIVE - 1]],
               color="#e05c2e", lw=1, ls="--", label="activity threshold")
    ax.set_ylabel("Max |Spearman ρ| across sites")
    ax.set_title(f"Active latent dimensions — top {N_ACTIVE} by max |loading| (blue)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_active_dims.png", dpi=150)
    plt.close()

    # ── Fig B: Violin by sex ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, N_ACTIVE, figsize=(5 * N_ACTIVE, 5))
    for ax, dim, label, color, row in zip(axes, active_dims, axis_labels, PALETTE, rows):
        z = mu[:, dim]
        vp = ax.violinplot([z[is_male], z[~is_male]], positions=[0, 1], showmedians=True)
        for pc in vp["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Male\n(n={is_male.sum():,})", f"Female\n(n={(~is_male).sum():,})"])
        ax.set_title(f"z{dim}: {label[:22]}\np={row['sex_mw_p']:.2e}  rbi={row['sex_rbi']:.3f}",
                     fontsize=9)
        ax.set_ylabel("Latent μ")
    fig.suptitle("Active axis values by sex", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_axis_sex.png", dpi=150)
    plt.close()

    # ── Fig C: Hexbin vs age ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, N_ACTIVE, figsize=(5 * N_ACTIVE, 4))
    for ax, dim, label, color, row in zip(axes, active_dims, axis_labels, PALETTE, rows):
        z = mu[:, dim]
        ax.hexbin(age, z, gridsize=40, cmap="Blues", mincnt=1)
        ax.set_xlabel("Age at first diagnosis")
        ax.set_ylabel("Latent μ")
        ax.set_title(f"z{dim}: {label[:22]}\nSpearman ρ={row['age_r']:.3f}  p={row['age_p']:.2e}",
                     fontsize=9)
    fig.suptitle("Active axis vs age at first diagnosis", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_axis_age.png", dpi=150)
    plt.close()

    # ── Fig D: Mean axis value by diagnosis year ─────────────────────────────
    fig, axes = plt.subplots(1, N_ACTIVE, figsize=(5 * N_ACTIVE, 4))
    for ax, dim, label, color, row in zip(axes, active_dims, axis_labels, PALETTE, rows):
        z = mu[:, dim]
        yr_mean = pd.Series(z).groupby(era).mean()
        yr_sem  = pd.Series(z).groupby(era).sem()
        ax.plot(yr_mean.index, yr_mean.values, color=color, lw=2, marker="o", ms=5)
        ax.fill_between(yr_mean.index,
                        yr_mean.values - yr_sem.values,
                        yr_mean.values + yr_sem.values,
                        alpha=0.2, color=color)
        ax.axvline(2010, color="gray", lw=1, ls="--", alpha=0.7, label="~betel regulation")
        ax.set_xlabel("Diagnosis year")
        ax.set_ylabel("Mean latent μ (±SEM)")
        ax.set_title(f"z{dim}: {label[:22]}\nSpearman ρ={row['era_r']:.3f}  p={row['era_p']:.2e}",
                     fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Active axis temporal trend 2003–2020", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_axis_era.png", dpi=150)
    plt.close()

    print(f"  Saved → {OUT}/")


if __name__ == "__main__":
    main()
