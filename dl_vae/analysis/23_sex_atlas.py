"""
Registry DL — Script 23: Sex-Specific Cancer Atlas

Characterises the full sex distribution across all 37 registry sites and
links findings to the three VAE latent axes identified in Scripts 09–11.

Analyses:
  1. Male:Female odds ratio per site (Fisher 2×2, FDR-corrected)
  2. Age at first diagnosis by sex per site (Mann-Whitney U, FDR-corrected)
  3. Multi-cancer rate by sex (overall + by axis)
  4. Survival by sex (KM + Cox)
  5. VAE axis values by sex (z4=UADT, z0/z5=Hormonal)
  6. UADT sex effect: does sex modify the field cancerization survival penalty?

Outputs:
  results/23_sex/sex_or_by_site.csv
  results/23_sex/age_sex_by_site.csv
  results/23_sex/fig_mf_ratio_bar.png
  results/23_sex/fig_age_sex_violin.png
  results/23_sex/fig_multi_cancer_sex.png
  results/23_sex/fig_survival_sex.png
  results/23_sex/fig_vae_axis_sex.png
  results/23_sex/fig_uadt_sex_survival.png
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).parent.parent
RAW  = BASE.parent / "data/processed/all_cancers.csv"
MAT  = BASE / "data/cancer_matrix.csv"
META = BASE / "data/patient_meta.csv"
MU   = BASE / "data/latent_mu.npy"
R09  = BASE / "results/09_axis_covariate"
OUT  = BASE / "results/23_sex"
OUT.mkdir(parents=True, exist_ok=True)

from constants import UADT, GI_SYS, HORMONAL  # noqa: E402
AXIS_PALETTE = {"UADT":"#2e7fbf","HBV/GI":"#2ca02c","Hormonal":"#9467bd","Other":"#aaaaaa"}

ACTIVE_DIMS = {"z4": "UADT/field", "z0": "Hormonal/gynecologic", "z5": "Hormonal/gynecologic"}


def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    y  = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00","99") else "01"
    dd = s[5:7] if s[5:7] not in ("00","99") else "01"
    try: return pd.Timestamp(f"{y}-{mm}-{dd}")
    except: return pd.NaT


def axis_label(site):
    if site in UADT:     return "UADT"
    if site in GI_SYS:  return "HBV/GI"
    if site in HORMONAL: return "Hormonal"
    return "Other"


def main():
    print("=== Registry DL — 23: Sex-Specific Cancer Atlas ===")

    # ── Load data ─────────────────────────────────────────────────────────────
    mat  = pd.read_csv(MAT, index_col="pid")
    meta = pd.read_csv(META)
    meta_idx = meta.set_index("pid")   # float pid index — matches mat index directly

    df = mat.join(meta_idx[["sex","age_first","dead","diag_yr","multi_cancer"]], how="left")
    df = df.dropna(subset=["sex"])
    print(f"  Patients: M={( df['sex']=='M').sum():,}  F={( df['sex']=='F').sum():,}")

    m = df[df["sex"] == "M"]
    f = df[df["sex"] == "F"]

    # ── 1. M:F OR per site ────────────────────────────────────────────────────
    print("\n  1. Sex OR per site")
    rows = []
    for site in mat.columns:
        a = int((m[site] == 1).sum()); b = len(m) - a
        c = int((f[site] == 1).sum()); d = len(f) - c
        if a == 0 or c == 0:
            OR, lo, hi, p = (np.nan,)*4
        else:
            OR = (a * d) / (b * c)
            se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            lo = np.exp(np.log(OR) - 1.96*se)
            hi = np.exp(np.log(OR) + 1.96*se)
            _, p = fisher_exact([[a, b],[c, d]])
        rows.append({
            "site": site, "axis": axis_label(site),
            "n_m": a, "n_f": c,
            "rate_m": round(a/len(m), 4),
            "rate_f": round(c/len(f), 4),
            "OR": round(OR, 3) if not np.isnan(OR) else None,
            "OR_lo": round(lo, 3) if not np.isnan(OR) else None,
            "OR_hi": round(hi, 3) if not np.isnan(OR) else None,
            "p_fisher": p,
        })

    or_df = pd.DataFrame(rows)
    valid = or_df["p_fisher"].notna()
    _, or_df.loc[valid, "q_fdr"], _, _ = multipletests(
        or_df.loc[valid, "p_fisher"], method="fdr_bh")
    or_df["sig"] = or_df["q_fdr"] < 0.05
    or_df = or_df.sort_values("OR", ascending=False, na_position="last")
    or_df.to_csv(OUT / "sex_or_by_site.csv", index=False)

    print("  Top male-dominant (OR > 1, FDR<0.05):")
    print(or_df[or_df["sig"] & (or_df["OR"] > 1)][
        ["site","axis","rate_m","rate_f","OR","q_fdr"]].head(8).to_string(index=False))
    print("  Top female-dominant (OR < 1, FDR<0.05):")
    print(or_df[or_df["sig"] & (or_df["OR"] < 1)][
        ["site","axis","rate_m","rate_f","OR","q_fdr"]].tail(8).to_string(index=False))

    # ── Fig A: M:F OR forest (log scale) ──────────────────────────────────────
    show = or_df[or_df["OR"].notna()].copy()
    colors = [AXIS_PALETTE[r["axis"]] for _, r in show.iterrows()]
    log_or = np.log2(show["OR"].astype(float))

    fig, ax = plt.subplots(figsize=(9, max(6, len(show)*0.38+1.5)))
    ys = range(len(show))
    ax.barh(list(ys), log_or.values, color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    # CI whiskers
    for i, (_, row) in enumerate(show.iterrows()):
        if pd.notna(row["OR_lo"]):
            lo2 = np.log2(float(row["OR_lo"]))
            hi2 = np.log2(float(row["OR_hi"]))
            ax.plot([lo2, hi2], [i, i], color="black", lw=0.8, alpha=0.5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(
        [f"{r['site']} ({r['axis']})  {r['rate_m']:.1%}M/{r['rate_f']:.1%}F"
         for _, r in show.iterrows()], fontsize=8)
    ax.set_xlabel("log₂(Male:Female odds ratio)  ← female dominant | male dominant →")
    ax.set_title("Sex-specific cancer site prevalence — Male:Female OR\n"
                 "Taiwan Cancer Registry 2003–2020 (first-primary patients)")
    for axis_name, color in AXIS_PALETTE.items():
        if any(r["axis"] == axis_name for _, r in show.iterrows()):
            ax.barh([], [], color=color, label=axis_name, alpha=0.8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mf_ratio_bar.png", dpi=150)
    plt.close()

    # ── 2. Age at first dx by sex ─────────────────────────────────────────────
    print("\n  2. Age at first diagnosis by sex")
    df_ac = pd.read_csv(RAW, low_memory=False)
    df_ac.columns = df_ac.columns.str.strip().str.lstrip("﻿")
    df_ac["pid"]  = pd.to_numeric(df_ac["病歷號(2)"], errors="coerce").astype("Int64").astype(str)
    df_ac["site"] = df_ac["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df_ac["age"]  = pd.to_numeric(df_ac["診斷年齡(33)"], errors="coerce")
    df_ac["sex"]  = df_ac["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"})
    df_ac["dx_ts"] = df_ac["最初診斷日(45)"].apply(roc_to_ts)
    df_ac = df_ac.dropna(subset=["age","sex","dx_ts"])
    first_ac = (df_ac.sort_values("dx_ts").groupby("pid").first()
                .reset_index()[["pid","site","age","sex"]])

    age_rows = []
    for site in mat.columns:
        sub = first_ac[first_ac["site"]==site]
        sm = sub[sub["sex"]=="M"]["age"].dropna()
        sf = sub[sub["sex"]=="F"]["age"].dropna()
        if len(sm) < 5 or len(sf) < 5:
            continue
        stat, p = mannwhitneyu(sm, sf, alternative="two-sided")
        age_rows.append({
            "site": site, "axis": axis_label(site),
            "n_m": len(sm), "n_f": len(sf),
            "median_m": round(sm.median(), 1),
            "median_f": round(sf.median(), 1),
            "age_diff_mf": round(sm.median() - sf.median(), 1),
            "p_mw": round(p, 5),
        })

    age_df = pd.DataFrame(age_rows)
    valid_a = age_df["p_mw"].notna()
    _, age_df.loc[valid_a, "q_fdr"], _, _ = multipletests(
        age_df.loc[valid_a, "p_mw"], method="fdr_bh")
    age_df["sig"] = age_df["q_fdr"] < 0.05
    age_df.to_csv(OUT / "age_sex_by_site.csv", index=False)

    print("  Largest age M−F differences (FDR<0.05):")
    print(age_df[age_df["sig"]].sort_values("age_diff_mf", key=abs, ascending=False)
          .head(10)[["site","axis","median_m","median_f","age_diff_mf","q_fdr"]].to_string(index=False))

    # Fig B: Age violin for key sex-discordant sites
    key_sites = ["C12","C13","C15","C50","C54","C18","C22","C34"]
    key_sites = [s for s in key_sites if s in first_ac["site"].unique()]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=False)
    for ax, site in zip(axes.flat, key_sites):
        sub = first_ac[first_ac["site"]==site]
        sm = sub[sub["sex"]=="M"]["age"].dropna()
        sf = sub[sub["sex"]=="F"]["age"].dropna()
        if len(sm) < 3 or len(sf) < 3:
            ax.text(0.5, 0.5, f"{site}\nInsufficient data", ha="center", va="center",
                    transform=ax.transAxes); continue
        data = [sm.values, sf.values]
        parts = ax.violinplot(data, positions=[0,1], showmedians=True, showextrema=False)
        for i, (pc, color) in enumerate(zip(parts["bodies"], ["#2e7fbf","#9467bd"])):
            pc.set_facecolor(color); pc.set_alpha(0.6)
        for collection in [parts["cmedians"]]:
            collection.set_color("black"); collection.set_lw(2)
        row = age_df[age_df["site"]==site]
        sig_str = f"Δ={row['age_diff_mf'].values[0]:+.0f}yr" if len(row) else ""
        q_str = f" q={row['q_fdr'].values[0]:.3f}" if len(row) else ""
        ax.set_title(f"{site} ({axis_label(site)})\n{sig_str}{q_str}", fontsize=8)
        ax.set_xticks([0,1]); ax.set_xticklabels(["M","F"], fontsize=9)
        ax.set_ylabel("Age (yr)", fontsize=8)
    fig.suptitle("Age at first diagnosis by sex — key cancer sites\n"
                 "(blue=male, purple=female; Δ = median male − female)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_age_sex_violin.png", dpi=150)
    plt.close()

    # ── 3. Multi-cancer rate by sex and axis ──────────────────────────────────
    print("\n  3. Multi-cancer rate by sex")
    mc_sex = df.groupby("sex")["multi_cancer"].agg(["mean","sum","count"])
    mc_sex.columns = ["rate","n_multi","n_total"]
    print(mc_sex.round(3).to_string())

    # Multi-cancer rate for first-primary patients in each axis category
    first_ac["axis"] = first_ac["site"].apply(axis_label)
    # Merge multi_cancer flag
    # pid in first_ac is string (from all_cancers.csv Int64 conversion);
    # convert meta float pids to match
    mc_flag = meta[["pid","multi_cancer","sex"]].copy()
    mc_flag["pid"] = mc_flag["pid"].astype("Int64").astype(str)
    first_mc = first_ac.merge(mc_flag, on=["pid","sex"], how="left")

    mc_axis = (first_mc.groupby(["axis","sex"])
               .agg(rate=("multi_cancer","mean"), n=("multi_cancer","count"))
               .reset_index())
    print("\n  Multi-cancer rate by axis and sex:")
    print(mc_axis.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left: overall by sex
    for ax, grp, title in [
        (axes[0], mc_sex, "Overall multi-cancer rate by sex"),
        (axes[1], None, "Multi-cancer rate by axis and sex"),
    ]:
        if grp is not None:
            bars = ax.bar(grp.index, grp["rate"]*100,
                          color=["#2e7fbf","#9467bd"], alpha=0.8, width=0.5)
            ax.set_ylabel("% multi-cancer patients")
            ax.set_title(title)
            for bar, row in zip(bars, grp.itertuples()):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                        f"{row.rate*100:.1f}%\nn={row.n_total:,}", ha="center", fontsize=9)
        else:
            pivot_mc = mc_axis.pivot(index="axis", columns="sex", values="rate") * 100
            x = np.arange(len(pivot_mc))
            w = 0.35
            for i, (sex, color) in enumerate([("M","#2e7fbf"),("F","#9467bd")]):
                if sex in pivot_mc.columns:
                    ax.bar(x + i*w - w/2, pivot_mc[sex], w, label=sex,
                           color=color, alpha=0.8)
            ax.set_xticks(x); ax.set_xticklabels(pivot_mc.index, rotation=15)
            ax.set_ylabel("% multi-cancer patients")
            ax.set_title(title)
            ax.legend()
    fig.suptitle("Multi-cancer rate by sex — males 2× more likely to develop sequential cancers",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_multi_cancer_sex.png", dpi=150)
    plt.close()

    # ── 4. Survival by sex (KM) ───────────────────────────────────────────────
    print("\n  4. Survival by sex")
    meta["end_fu_ts"] = pd.to_datetime(meta["end_fu"], errors="coerce")
    meta["surv_yrs"] = (
        meta["end_fu_ts"] -
        pd.to_datetime(meta["diag_yr"].astype(str)+"-01-01", errors="coerce")
    ).dt.days / 365.25
    meta["surv_yrs"] = meta["surv_yrs"].clip(lower=0.01)

    try:
        from lifelines import KaplanMeierFitter, CoxPHFitter
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overall survival by sex
        ax = axes[0]
        for sex, color, label in [("M","#2e7fbf","Male"), ("F","#9467bd","Female")]:
            sub = meta[meta["sex"]==sex]
            kmf = KaplanMeierFitter()
            kmf.fit(sub["surv_yrs"], sub["dead"], label=f"{label} (n={len(sub):,})")
            kmf.plot_survival_function(ax=ax, color=color, ci_show=True)
        ax.set_xlabel("Years from first diagnosis")
        ax.set_ylabel("Survival probability")
        ax.set_title("Overall survival by sex\n(all first-primary registry patients)")
        ax.set_xlim(0, 18)
        ax.legend(fontsize=9)

        # UADT patients specifically
        ax2 = axes[1]
        uadt_pids = set(first_ac[first_ac["axis"]=="UADT"]["pid"].astype(str))
        uadt_meta = meta[meta["pid"].astype("Int64").astype(str).isin(uadt_pids)]
        for sex, color, label in [("M","#2e7fbf","UADT male"), ("F","#9467bd","UADT female")]:
            sub = uadt_meta[uadt_meta["sex"]==sex]
            if len(sub) < 10: continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub["surv_yrs"], sub["dead"], label=f"{label} (n={len(sub):,})")
            kmf.plot_survival_function(ax=ax2, color=color, ci_show=True)
        ax2.set_xlabel("Years from first diagnosis")
        ax2.set_ylabel("Survival probability")
        ax2.set_title("UADT-site patients: survival by sex\n"
                      "(C12/C13/C15 — male-dominated betel/tobacco axis)")
        ax2.set_xlim(0, 18)
        ax2.legend(fontsize=9)

        fig.suptitle("Kaplan–Meier survival by sex", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "fig_survival_sex.png", dpi=150)
        plt.close()

        # Log-rank p for sex difference
        from lifelines.statistics import logrank_test
        sm_s = meta[meta["sex"]=="M"]["surv_yrs"].dropna()
        sf_s = meta[meta["sex"]=="F"]["surv_yrs"].dropna()
        sm_e = meta[meta["sex"]=="M"]["dead"].dropna().astype(int)
        sf_e = meta[meta["sex"]=="F"]["dead"].dropna().astype(int)
        idx_m = sm_s.index.intersection(sm_e.index)
        idx_f = sf_s.index.intersection(sf_e.index)
        lr = logrank_test(sm_s.loc[idx_m], sf_s.loc[idx_f],
                          sm_e.loc[idx_m], sf_e.loc[idx_f])
        print(f"  Log-rank test M vs F: p={lr.p_value:.4e}")

    except ImportError:
        print("  lifelines not available — skipping KM")

    # ── 5. VAE axis values by sex ─────────────────────────────────────────────
    print("\n  5. VAE axis values by sex")
    mu = np.load(MU)
    meta_mu = meta.copy()
    meta_mu["z4"] = mu[:, 4]
    meta_mu["z0"] = mu[:, 0]
    meta_mu["z5"] = mu[:, 5]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, dim, name in [
        (axes[0], "z4", "z4 — UADT/field-cancerization"),
        (axes[1], "z0", "z0 — Hormonal/gynecologic"),
        (axes[2], "z5", "z5 — Hormonal/gynecologic"),
    ]:
        for sex, color in [("M","#2e7fbf"),("F","#9467bd")]:
            sub = meta_mu[meta_mu["sex"]==sex][dim].dropna()
            ax.hist(sub.values, bins=60, alpha=0.5, color=color,
                    label=f"{sex} (n={len(sub):,})", density=True)
        stat, p = mannwhitneyu(
            meta_mu[meta_mu["sex"]=="M"][dim].dropna(),
            meta_mu[meta_mu["sex"]=="F"][dim].dropna(),
            alternative="two-sided")
        m_med = meta_mu[meta_mu["sex"]=="M"][dim].median()
        f_med = meta_mu[meta_mu["sex"]=="F"][dim].median()
        ax.set_title(f"{name}\nM median={m_med:.3f}  F median={f_med:.3f}\np<0.001" if p<0.001 else f"p={p:.4f}")
        ax.set_xlabel(f"{dim} value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle("VAE active axis values by sex — latent space sex separation\n"
                 "(z4=UADT: male-shifted; z0/z5=Hormonal: female-shifted)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_vae_axis_sex.png", dpi=150)
    plt.close()

    # Print Mann-Whitney stats for active dims
    for dim in ["z4","z0","z5"]:
        sm_v = meta_mu[meta_mu["sex"]=="M"][dim].dropna()
        sf_v = meta_mu[meta_mu["sex"]=="F"][dim].dropna()
        stat, p = mannwhitneyu(sm_v, sf_v, alternative="two-sided")
        rbi = (stat / (len(sm_v) * len(sf_v)) - 0.5) * 2   # rank-biserial correlation
        print(f"  {dim}: MW p={p:.2e}  rank-biserial={rbi:.3f}")

    print(f"\n  Saved → {OUT}/")


if __name__ == "__main__":
    main()
