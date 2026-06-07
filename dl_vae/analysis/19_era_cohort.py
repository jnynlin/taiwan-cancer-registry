"""
Registry DL — Script 19: Era / Birth-Cohort Decomposition

Tests the HBV vaccination mechanism by comparing C22 incidence trends
stratified by birth cohort. Taiwan began universal infant HBV vaccination in 1986;
individuals born 1970+ were partially covered; born 1980+ were fully covered.

Also characterises:
  - VAE cluster proportions over diagnosis year (Hormonal rising? Novel-1 falling?)
  - Multi-primary rate over time (improving survival → more second cancers detected)
  - UADT vs HBV/GI axis share over time

Outputs:
  results/19_era/birth_cohort_c22.csv
  results/19_era/cluster_era.csv
  results/19_era/multi_primary_era.csv
  results/19_era/axis_share_era.csv
  results/19_era/fig_birth_cohort_c22.png
  results/19_era/fig_cluster_era.png
  results/19_era/fig_multi_primary.png
  results/19_era/fig_axis_share.png
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans

# ── Locked constants (identical to Scripts 10 / 15 / 16) ─────────────────────
KMEANS_K     = 5
KMEANS_SEED  = 42
KMEANS_INIT  = 20
from constants import UADT, GI_SYS, HORMONAL  # noqa: E402

CLUSTER_NAMES = {0: "Hormonal", 1: "Novel-1", 2: "Novel-2", 3: "UADT-field", 4: "Novel-4"}
CLUSTER_COLORS = {
    "Hormonal":   "#9467bd",
    "Novel-1":    "#d62728",
    "Novel-2":    "#aaaaaa",
    "UADT-field": "#2e7fbf",
    "Novel-4":    "#ff7f0e",
}
# ─────────────────────────────────────────────────────────────────────────────

BASE  = Path(__file__).parent.parent
RAW   = BASE.parent / "data/processed/all_cancers.csv"
MU    = BASE / "data/latent_mu.npy"
META  = BASE / "data/patient_meta.csv"
OUT   = BASE / "results/19_era"
OUT.mkdir(parents=True, exist_ok=True)


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
    print("=== Registry DL — 19: Era / Birth-Cohort Decomposition ===")

    # ── Load registry (first cancer per patient) ──────────────────────────────
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip().str.lstrip("﻿")
    df["pid"]    = df["病歷號(2)"].astype(str).str.strip()
    df["site"]   = df["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df["age"]    = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]    = df["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"})
    df["dx_ts"]  = df["最初診斷日(45)"].apply(roc_to_ts)
    df["birth_ts"] = df["出生日期(7)"].apply(roc_to_ts)
    df = df.dropna(subset=["dx_ts","age","sex"])
    df["diag_yr"] = df["dx_ts"].dt.year

    # Birth year: prefer parsed birth_ts, fallback diag_yr - age
    df["birth_yr"] = np.where(
        df["birth_ts"].notna(),
        df["birth_ts"].dt.year,
        df["diag_yr"] - df["age"].round().astype("Int64"),
    )

    first = (df.sort_values("dx_ts")
               .groupby("pid").first()
               .reset_index()[["pid","site","diag_yr","age","sex","birth_yr"]])
    first = first[first["diag_yr"].between(2003, 2020)]

    # ── 1. Birth-cohort analysis for C22 ─────────────────────────────────────
    print("\n  1. C22 birth-cohort analysis")
    # Cohort boundaries: < 1960 / 1960-1969 / 1970-1979 / 1980+
    bins   = [-np.inf, 1959, 1969, 1979, np.inf]
    labels = ["born <1960", "born 1960–69", "born 1970–79", "born ≥1980"]
    first["cohort"] = pd.cut(first["birth_yr"], bins=bins, labels=labels)

    has_c22 = first["site"] == "C22"
    annual_total    = first.groupby(["diag_yr","cohort"]).size().rename("n_total")
    annual_c22      = (first[has_c22].groupby(["diag_yr","cohort"])
                       .size().rename("n_c22").reset_index())
    bc_df = annual_c22.merge(annual_total.reset_index(), on=["diag_yr","cohort"])
    bc_df["c22_pct"] = bc_df["n_c22"] / bc_df["n_total"] * 100
    bc_df.to_csv(OUT / "birth_cohort_c22.csv", index=False)

    # Spearman trend per cohort
    print("  C22 Spearman ρ by birth cohort:")
    for cohort in labels:
        sub = bc_df[bc_df["cohort"]==cohort].sort_values("diag_yr")
        if len(sub) < 5: continue
        rho, pval = stats.spearmanr(sub["diag_yr"], sub["c22_pct"])
        n_total = sub["n_total"].sum()
        print(f"    {cohort}: n={n_total:,}, ρ={rho:.3f}, p={pval:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    cohort_colors = {"born <1960":"#d62728","born 1960–69":"#ff7f0e",
                     "born 1970–79":"#2ca02c","born ≥1980":"#2e7fbf"}
    for cohort in labels:
        sub = bc_df[bc_df["cohort"]==cohort].sort_values("diag_yr")
        if len(sub) < 3: continue
        rho, pval = stats.spearmanr(sub["diag_yr"], sub["c22_pct"])
        ax.plot(sub["diag_yr"], sub["c22_pct"], marker="o", ms=4,
                color=cohort_colors.get(cohort,"#aaa"),
                label=f"{cohort}  ρ={rho:.2f} p={pval:.3f}")
    ax.axvline(2003, color="gray", lw=1, ls=":", alpha=0.5)
    ax.set_xlabel("Diagnosis year")
    ax.set_ylabel("C22 liver HCC as % of first primaries")
    ax.set_title("C22 incidence trend by birth cohort\n"
                 "HBV vaccination (1986): born ≥1980 cohort should show steepest decline")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_birth_cohort_c22.png", dpi=150)
    plt.close()

    # ── 2. VAE cluster proportions over time ──────────────────────────────────
    print("\n  2. VAE cluster era analysis")
    meta = pd.read_csv(META)
    meta["pid"] = meta["pid"].astype(str).str.split(".").str[0]

    latent_mu = np.load(MU)
    print(f"  Loaded latent_mu: {latent_mu.shape}")

    km = KMeans(n_clusters=KMEANS_K, random_state=KMEANS_SEED, n_init=KMEANS_INIT)
    meta["cluster"] = km.fit_predict(latent_mu)
    meta["cluster_name"] = meta["cluster"].map(CLUSTER_NAMES)

    meta_yr = meta[meta["diag_yr"].between(2003, 2020)]
    cluster_era = (meta_yr.groupby(["diag_yr","cluster_name"])
                   .size().rename("n").reset_index())
    annual_n    = meta_yr.groupby("diag_yr").size().rename("total")
    cluster_era = cluster_era.merge(annual_n, on="diag_yr")
    cluster_era["pct"] = cluster_era["n"] / cluster_era["total"] * 100
    cluster_era.to_csv(OUT / "cluster_era.csv", index=False)

    # Spearman trend per cluster
    print("  Cluster proportion Spearman ρ:")
    for cname in CLUSTER_NAMES.values():
        sub = cluster_era[cluster_era["cluster_name"]==cname].sort_values("diag_yr")
        if len(sub) < 5: continue
        rho, pval = stats.spearmanr(sub["diag_yr"], sub["pct"])
        print(f"    {cname}: ρ={rho:.3f}, p={pval:.4f}")

    pivot_cl = cluster_era.pivot(index="diag_yr", columns="cluster_name", values="pct").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    yrs = pivot_cl.index.values
    bottom = np.zeros(len(yrs))
    for cname in CLUSTER_NAMES.values():
        if cname not in pivot_cl.columns: continue
        vals = pivot_cl[cname].values
        ax.bar(yrs, vals, bottom=bottom, label=cname,
               color=CLUSTER_COLORS.get(cname, "#aaa"), alpha=0.8, width=0.85)
        bottom += vals
    ax.set_xlabel("Diagnosis year")
    ax.set_ylabel("% of patients")
    ax.set_title("VAE cluster proportions over time\n(stacked bar — shift indicates changing disease mix)")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cluster_era.png", dpi=150)
    plt.close()

    # ── 3. Multi-primary rate over time ───────────────────────────────────────
    print("\n  3. Multi-primary rate over time")
    # Use patient_meta multi_cancer flag; restrict to patients with ≥5yr follow-up
    # to avoid immortal-time bias (recent dx patients have less time for 2nd cancer)
    meta["end_fu_ts"] = pd.to_datetime(meta["end_fu"], errors="coerce")
    meta["fu_yrs"] = (meta["end_fu_ts"] -
                      pd.to_datetime(meta["diag_yr"].astype(str) + "-01-01",
                                     errors="coerce")).dt.days / 365.25
    meta_5yr = meta[(meta["fu_yrs"] >= 5) & meta["diag_yr"].between(2003, 2015)]

    mp_era = (meta_5yr.groupby("diag_yr")
              .agg(n_total=("multi_cancer","count"),
                   n_multi=("multi_cancer","sum"))
              .reset_index())
    mp_era["pct_multi"] = mp_era["n_multi"] / mp_era["n_total"] * 100
    mp_era.to_csv(OUT / "multi_primary_era.csv", index=False)

    rho_mp, p_mp = stats.spearmanr(mp_era["diag_yr"], mp_era["pct_multi"])
    print(f"  Multi-primary rate (≥5yr FU): ρ={rho_mp:.3f}, p={p_mp:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mp_era["diag_yr"], mp_era["pct_multi"], marker="o", ms=5,
            color="#ff7f0e", lw=2)
    slope, intercept, *_ = stats.linregress(mp_era["diag_yr"], mp_era["pct_multi"])
    ax.plot(mp_era["diag_yr"], intercept + slope*mp_era["diag_yr"].values,
            "--", color="#ff7f0e", alpha=0.6)
    ax.set_xlabel("Diagnosis year (patients with ≥5yr follow-up)")
    ax.set_ylabel("% multi-primary patients")
    ax.set_title(f"Multi-primary cancer rate over time\nρ={rho_mp:.3f}, p={p_mp:.4f}")
    ax.text(0.02, 0.95,
            "Restricted to patients with ≥5yr follow-up\n(2003–2015 cohort) to control censoring bias",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="white", edgecolor="gray", pad=3))
    fig.tight_layout()
    fig.savefig(OUT / "fig_multi_primary.png", dpi=150)
    plt.close()

    # ── 4. Axis share over time ───────────────────────────────────────────────
    print("\n  4. Axis share over time")
    first["axis"] = first["site"].apply(axis_label)
    axis_era = (first.groupby(["diag_yr","axis"]).size().rename("n").reset_index())
    axis_total = first.groupby("diag_yr").size().rename("total")
    axis_era = axis_era.merge(axis_total, on="diag_yr")
    axis_era["pct"] = axis_era["n"] / axis_era["total"] * 100
    axis_era.to_csv(OUT / "axis_share_era.csv", index=False)

    axis_colors = {"UADT":"#2e7fbf","HBV/GI":"#2ca02c","Hormonal":"#9467bd","Other":"#aaaaaa"}
    print("  Axis share Spearman ρ:")
    fig, ax = plt.subplots(figsize=(9, 5))
    for axis_name in ["UADT","HBV/GI","Hormonal"]:
        sub = axis_era[axis_era["axis"]==axis_name].sort_values("diag_yr")
        rho, pval = stats.spearmanr(sub["diag_yr"], sub["pct"])
        print(f"    {axis_name}: ρ={rho:.3f}, p={pval:.4f}")
        ax.plot(sub["diag_yr"], sub["pct"], marker="o", ms=4,
                color=axis_colors[axis_name],
                label=f"{axis_name}  ρ={rho:.2f} p={pval:.3f}")
    ax.set_xlabel("Diagnosis year")
    ax.set_ylabel("% of first-primary diagnoses")
    ax.set_title("Carcinogenic axis share over time\n"
                 "(first-primary patients; UADT = betel/tobacco; HBV/GI = viral/metabolic)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_axis_share.png", dpi=150)
    plt.close()

    print(f"\n  Saved → {OUT}/")


if __name__ == "__main__":
    main()
