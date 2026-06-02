"""
UADT Field Cancerization — Script 01: Cohort Build

Extracts the upper aerodigestive tract (UADT) field cohort from the Taiwan
Cancer Registry, runs a batch-effect check, and emits three derived tables.

Outputs:
  data/field_patients.csv      — (pid, site, dx, age, sex, dead, end_fu)
  data/field_patient_matrix.csv — pid × 10 multi-hot
  data/field_meta.csv          — pid-level summary
  results/01_cohort/batch_check.csv
  results/01_cohort/fig1a_site_prevalence.png
  results/01_cohort/fig1b_temporal_trend.png
  results/01_cohort/fig1c_age_sex.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── LOCKED CONSTANTS (never modify after analysis begins) ─────────────────────
FIELD_SITES = ['C02','C03','C04','C05','C06','C09','C10','C12','C13','C15']
FIELD_LABELS = {
    'C02':'Tongue',      'C03':'Gum',        'C04':'Floor of mouth',
    'C05':'Palate',      'C06':'Oral NOS',   'C09':'Tonsil',
    'C10':'Oropharynx',  'C12':'Pyriform',   'C13':'Hypopharynx',
    'C15':'Esophagus'
}
SYNC_MO     = 6
LANDMARK_MO = 6
STUDY_END   = pd.Timestamp('2020-12-31')
MIN_OBS     = 5

assert len(FIELD_SITES) == 10, "FIELD_SITES tampered — abort"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
RAW  = BASE.parent / "data/processed/all_cancers.csv"
DOUT = BASE / "data"
OUT  = BASE / "results/01_cohort"
OUT.mkdir(parents=True, exist_ok=True)
DOUT.mkdir(parents=True, exist_ok=True)


# ── Helpers (copied verbatim from 07_sir_trajectories.py) ────────────────────

def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    y = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00","99") else "01"
    dd = s[5:7] if s[5:7] not in ("00","99") else "01"
    try: return pd.Timestamp(f"{y}-{mm}-{dd}")
    except: return pd.NaT


def load():
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip().str.lstrip('﻿')
    df["pid"]   = df["病歷號(2)"].astype(str).str.strip()
    df["site"]  = df["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df["dx"]    = df["最初診斷日(45)"].apply(roc_to_ts)
    df["age"]   = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]   = df["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"})
    df["dead"]  = (pd.to_numeric(df["生存狀態(27)"], errors="coerce")==0).astype(int)
    df["death_dt"]   = df["死亡日期(31)"].apply(roc_to_ts)
    df["contact_dt"] = df["最後聯絡日(30)"].apply(roc_to_ts)
    df["_sheet"] = df["_sheet"].astype(str).str.strip() if "_sheet" in df.columns else "unknown"
    df = df.dropna(subset=["dx"])
    return df


def patient_level(df):
    """One row per (patient, site): earliest dx. Plus patient end-of-followup."""
    first = (df.sort_values("dx").groupby(["pid","site"], as_index=False)
               .agg(dx=("dx","first"), age=("age","first"),
                    sex=("sex","first"), dead=("dead","max"),
                    _sheet=("_sheet","first")))
    endfu = df.groupby("pid").apply(
        lambda g: g["death_dt"].dropna().min()
        if g["dead"].max()==1 and g["death_dt"].notna().any()
        else (g["contact_dt"].dropna().max()
              if g["contact_dt"].notna().any() else g["dx"].max())
    ).rename("end_fu").reset_index()
    endfu["end_fu"] = endfu["end_fu"].clip(upper=STUDY_END)
    first = first.merge(endfu, on="pid", how="left")
    return first


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== UADT Field Cohort Build ===")
    print("Loading registry…")
    df = load()
    print(f"  Registry: {df['pid'].nunique():,} patients, {len(df):,} records")

    # Restrict to field sites
    field_df = df[df["site"].isin(FIELD_SITES)].copy()
    print(f"  UADT field records: {len(field_df):,}")

    first = patient_level(field_df)
    print(f"  Unique (pid, site) entries: {len(first):,}")

    n_pts       = first["pid"].nunique()
    n_multi     = (first.groupby("pid")["site"].nunique() >= 2).sum()
    n_triple    = (first.groupby("pid")["site"].nunique() >= 3).sum()
    print(f"  Unique patients: {n_pts:,}")
    print(f"  Multi-field (≥2 sites): {n_multi:,}")
    print(f"  Triple-field (≥3 sites): {n_triple:,}")

    # ── field_patients.csv ────────────────────────────────────────────────────
    first.to_csv(DOUT / "field_patients.csv", index=False, encoding="utf-8-sig")

    # ── field_patient_matrix.csv (pid × 10 multi-hot) ────────────────────────
    mat = (first.assign(val=1)
                .pivot_table(index="pid", columns="site", values="val",
                             aggfunc="max", fill_value=0))
    for s in FIELD_SITES:
        if s not in mat.columns: mat[s] = 0
    mat = mat[FIELD_SITES]
    mat.to_csv(DOUT / "field_patient_matrix.csv", encoding="utf-8-sig")

    # ── field_meta.csv ────────────────────────────────────────────────────────
    first_dx = first.sort_values("dx").groupby("pid").first().reset_index()
    meta = first.groupby("pid").agg(
        n_field_sites=("site","nunique"),
        diag_yr_first=("dx", lambda x: x.min().year),
        age_first=("age","min"),
        sex=("sex","first"),
        dead=("dead","max"),
        _sheet=("_sheet","first")
    ).reset_index()
    meta["multi_field"] = (meta["n_field_sites"] >= 2).astype(int)
    meta.to_csv(DOUT / "field_meta.csv", index=False, encoding="utf-8-sig")

    # ── Batch-effect check ────────────────────────────────────────────────────
    batch = meta.groupby("_sheet").agg(
        n_patients=("pid","count"),
        n_multi=("multi_field","sum")
    ).reset_index()
    batch["pct_multi"] = (batch["n_multi"] / batch["n_patients"] * 100).round(1)
    median_pct = batch["pct_multi"].median()
    batch["ratio_to_median"] = (batch["pct_multi"] / median_pct).round(2)
    batch.to_csv(OUT / "batch_check.csv", index=False, encoding="utf-8-sig")
    print("\n  Batch-effect check:")
    print(batch[["_sheet","n_patients","n_multi","pct_multi","ratio_to_median"]]
          .sort_values("pct_multi", ascending=False).to_string(index=False))
    outliers = batch[batch["ratio_to_median"] > 2]
    if len(outliers):
        print(f"\n  ⚠️  WARNING: {len(outliers)} batch(es) >2× median co-occurrence rate:")
        print(outliers[["_sheet","pct_multi"]].to_string(index=False))

    # ── Figure 1a: Site prevalence ────────────────────────────────────────────
    site_n = first.groupby("site")["pid"].nunique().reindex(FIELD_SITES)
    labels = [FIELD_LABELS[s] for s in FIELD_SITES]
    colors = (['#3b82f6']*5 + ['#f59e0b']*3 + ['#ef4444']*2)   # oral/pharyngeal/esophageal

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, site_n.values, color=colors, edgecolor='white', height=0.7)
    for bar, n in zip(bars, site_n.values):
        ax.text(n + 15, bar.get_y() + bar.get_height()/2,
                str(int(n)), va='center', fontsize=9)
    ax.set_xlabel("Unique patients", fontsize=11)
    ax.set_title("UADT Field Site — Patient Counts", fontsize=12, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    from matplotlib.patches import Patch
    legend_elements = [Patch(color='#3b82f6', label='Oral cavity (C02–C06)'),
                       Patch(color='#f59e0b', label='Pharyngeal (C09–C12)'),
                       Patch(color='#ef4444', label='Esophageal (C13, C15)')]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT / "fig1a_site_prevalence.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 1b: Temporal trend 2003-2020 ───────────────────────────────────
    first["diag_yr"] = first["dx"].dt.year
    yr_site = first.groupby(["diag_yr","site"])["pid"].nunique().unstack(fill_value=0)
    oral_sites    = [s for s in ['C02','C03','C04','C05','C06'] if s in yr_site.columns]
    pharynx_sites = [s for s in ['C09','C10','C12'] if s in yr_site.columns]
    eso_sites     = [s for s in ['C13','C15'] if s in yr_site.columns]
    yrs = sorted([y for y in yr_site.index if 2003 <= y <= 2020])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yrs, yr_site.loc[yrs, FIELD_SITES].sum(axis=1), 'k-o', lw=2,
            ms=5, label='Total field', zorder=5)
    ax.plot(yrs, yr_site.loc[yrs, oral_sites].sum(axis=1), 'b--s', lw=1.5,
            ms=4, label='Oral cavity', alpha=0.8)
    ax.plot(yrs, yr_site.loc[yrs, pharynx_sites].sum(axis=1), color='#f59e0b',
            ls='--', marker='^', lw=1.5, ms=4, label='Pharyngeal', alpha=0.8)
    ax.plot(yrs, yr_site.loc[yrs, eso_sites].sum(axis=1), 'r--D', lw=1.5,
            ms=4, label='Hypopharynx + Esophagus', alpha=0.8)
    ax.set_xlabel("Year of diagnosis", fontsize=11)
    ax.set_ylabel("New diagnoses", fontsize=11)
    ax.set_title("UADT Field Cancer Incidence Trend 2003–2020", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1b_temporal_trend.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 1c: Age distribution by site (violin) ─────────────────────────
    ages_by_site = {FIELD_LABELS[s]: first[first["site"]==s]["age"].dropna().values
                    for s in FIELD_SITES}
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), sharey=True)
    for ax, s in zip(axes.flat, FIELD_SITES):
        data = ages_by_site[FIELD_LABELS[s]]
        if len(data) > 1:
            parts = ax.violinplot(data, showmedians=True, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor('#93c5fd'); pc.set_alpha(0.7)
            parts['cmedians'].set_color('#1e40af'); parts['cmedians'].set_lw(2)
        ax.set_title(FIELD_LABELS[s], fontsize=8, fontweight='bold')
        ax.set_xticks([]); ax.spines[['top','right','bottom']].set_visible(False)
        med = float(np.median(data)) if len(data) else 0
        ax.text(1, med + 1, f'{med:.0f}', ha='center', va='bottom', fontsize=8,
                color='#1e40af')
    axes[0, 0].set_ylabel("Age at diagnosis (yr)", fontsize=10)
    axes[1, 0].set_ylabel("Age at diagnosis (yr)", fontsize=10)
    fig.suptitle("Age Distribution by UADT Field Site", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / "fig1c_age_sex.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n✓ Outputs written to {DOUT} and {OUT}")


if __name__ == "__main__":
    main()
