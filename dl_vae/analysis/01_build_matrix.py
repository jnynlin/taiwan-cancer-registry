"""
Registry DL — Script 01: Build Cancer Co-occurrence Matrix

Loads the full Taiwan Cancer Registry and produces:
  - pid × N_sites multi-hot matrix (all sites, not just UADT)
  - Patient-level metadata: age_first, sex, dead, n_sites, diag_yr

Outputs:
  data/cancer_matrix.csv     — pid × N_sites binary (0/1)
  data/patient_meta.csv      — pid-level covariates
  data/site_index.csv        — site code ↔ column index mapping
  results/01_matrix/fig_site_freq.png
  results/01_matrix/fig_n_sites_dist.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY_END = pd.Timestamp("2020-12-31")
MIN_SITE_N = 30          # drop sites with fewer than this many patients

BASE = Path(__file__).parent.parent
RAW  = BASE.parent / "data/processed/all_cancers.csv"
DOUT = BASE / "data"
OUT  = BASE / "results/01_matrix"
OUT.mkdir(parents=True, exist_ok=True)
DOUT.mkdir(parents=True, exist_ok=True)


def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    y = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00", "99") else "01"
    dd = s[5:7] if s[5:7] not in ("00", "99") else "01"
    try: return pd.Timestamp(f"{y}-{mm}-{dd}")
    except: return pd.NaT


def load():
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip().str.lstrip('﻿')
    df["pid"]        = df["病歷號(2)"].astype(str).str.strip()
    df["site"]       = df["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df["dx"]         = df["最初診斷日(45)"].apply(roc_to_ts)
    df["age"]        = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]        = df["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"})
    df["dead"]       = (pd.to_numeric(df["生存狀態(27)"], errors="coerce")==0).astype(int)
    df["death_dt"]   = df["死亡日期(31)"].apply(roc_to_ts)
    df["contact_dt"] = df["最後聯絡日(30)"].apply(roc_to_ts)
    df = df.dropna(subset=["dx"])
    return df


def patient_endfu(df):
    def _endfu(g):
        if g["dead"].max() == 1 and g["death_dt"].notna().any():
            return g["death_dt"].dropna().min()
        if g["contact_dt"].notna().any():
            return g["contact_dt"].dropna().max()
        return g["dx"].max()
    return (df.groupby("pid")
              .apply(_endfu)
              .clip(upper=STUDY_END)
              .rename("end_fu")
              .reset_index())


def main():
    print("=== Registry DL — 01: Build Matrix ===")
    df = load()
    print(f"  Loaded: {df['pid'].nunique():,} patients · {len(df):,} records · {df['site'].nunique()} raw sites")

    # Per-patient, per-site: first diagnosis only
    first = (df.sort_values("dx")
               .groupby(["pid", "site"], as_index=False)
               .agg(dx=("dx","first"), age=("age","first"),
                    sex=("sex","first"), dead=("dead","max")))

    # Drop rare sites (hard to learn from)
    site_counts = first.groupby("site")["pid"].nunique()
    keep_sites = sorted(site_counts[site_counts >= MIN_SITE_N].index.tolist())
    first = first[first["site"].isin(keep_sites)]
    print(f"  Sites kept (≥{MIN_SITE_N} patients): {len(keep_sites)}")

    # Site index table
    site_idx = pd.DataFrame({"site": keep_sites, "col_idx": range(len(keep_sites))})
    site_idx.to_csv(DOUT / "site_index.csv", index=False)

    # Multi-hot matrix
    matrix = (first.assign(val=1)
                   .pivot_table(index="pid", columns="site", values="val", fill_value=0)
                   .reindex(columns=keep_sites, fill_value=0))
    matrix.index.name = "pid"

    # Patient-level metadata
    agg = (first.groupby("pid")
                .agg(age_first=("age","min"),
                     sex=("sex","first"),
                     dead=("dead","max"),
                     diag_yr=("dx", lambda x: x.min().year),
                     n_sites=("site","nunique")))
    endfu = patient_endfu(df)
    meta = agg.join(endfu.set_index("pid"), how="left")
    meta["multi_cancer"] = (meta["n_sites"] >= 2).astype(int)

    matrix.to_csv(DOUT / "cancer_matrix.csv")
    meta.to_csv(DOUT / "patient_meta.csv")
    print(f"  Matrix: {matrix.shape[0]:,} patients × {matrix.shape[1]} sites")
    print(f"  Multi-cancer patients: {meta['multi_cancer'].sum():,} ({meta['multi_cancer'].mean()*100:.1f}%)")
    print(f"  Saved → data/cancer_matrix.csv, patient_meta.csv, site_index.csv")

    # Figure: site frequencies
    freq = matrix.sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(freq)), freq.values, color="#2e7fbf")
    ax.set_xticks(range(len(freq)))
    ax.set_xticklabels(freq.index, rotation=90, fontsize=8)
    ax.set_ylabel("Patients")
    ax.set_title("Cancer site frequency (all 46 sites)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_site_freq.png", dpi=150)
    plt.close()

    # Figure: n_sites distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = meta["n_sites"].value_counts().sort_index()
    ax.bar(vc.index, vc.values, color="#14304a")
    ax.set_xlabel("Number of distinct cancer sites per patient")
    ax.set_ylabel("Patients")
    ax.set_title("Multi-cancer burden distribution")
    for x, y in zip(vc.index, vc.values):
        ax.text(x, y + 20, str(y), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_n_sites_dist.png", dpi=150)
    plt.close()

    print("  Figures saved → results/01_matrix/")
    return matrix, meta


if __name__ == "__main__":
    main()
