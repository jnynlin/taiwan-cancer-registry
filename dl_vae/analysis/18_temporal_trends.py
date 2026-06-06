"""
Registry DL — Script 18: Site-Level Temporal Trends (2003–2020)

For each cancer site, compute the annual fraction of first-primary diagnoses and
test for monotone trend (Spearman ρ). Sites are coloured by axis membership
(UADT / HBV-GI / Hormonal / Other).

Key hypotheses pre-registered at top of script:
  H1: C22 (HBV-driven HCC) fraction is declining 2003–2020
  H2: C12/C13 (UADT betel sites) fraction is declining post-2006
      (Taiwan government betel-nut control campaigns started ~2005)
  H3: Hormonal sites (C50 breast, C54 endometrial) fraction is rising

Outputs:
  results/18_temporal/trend_by_site.csv
  results/18_temporal/annual_fraction_by_site.csv
  results/18_temporal/fig_rising_sites.png
  results/18_temporal/fig_falling_sites.png
  results/18_temporal/fig_trend_rho_bar.png
  results/18_temporal/fig_uadt_trend.png
  results/18_temporal/fig_hypothesis_check.png
"""

# ── Pre-registered hypotheses ─────────────────────────────────────────────────
HYPOTHESIS_1 = "C22 annual fraction declines monotonically 2003–2020 (Spearman ρ < 0)"
HYPOTHESIS_2 = "C12/C13 annual fraction declines post-2006 (betel nut regulation)"
HYPOTHESIS_3 = "C50/C54 annual fraction rises 2003–2020 (metabolic syndrome / obesity)"
# ─────────────────────────────────────────────────────────────────────────────

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path(__file__).parent.parent
RAW  = BASE.parent / "data/processed/all_cancers.csv"
OUT  = BASE / "results/18_temporal"
OUT.mkdir(parents=True, exist_ok=True)

UADT     = {"C02","C03","C04","C05","C06","C09","C10","C12","C13","C15"}
GI_SYS   = {"C16","C17","C18","C19","C20","C22","C25","C34","C61","C67"}
HORMONAL = {"C50","C53","C54","C56"}

AXIS_PALETTE = {
    "UADT":       "#2e7fbf",
    "HBV/GI":     "#2ca02c",
    "Hormonal":   "#9467bd",
    "Other":      "#aaaaaa",
}

MIN_FIRST_PX = 30   # minimum first-primary patients ever to report a trend


def spearmanr_ac_corrected(years, y):
    """Spearman ρ with autocorrelation-corrected p-value.

    Annual time-series are autocorrelated → naive Spearman p-values
    underestimate the true p (effective df < n-2).

    Method (Chelton 1983 / Trenberth 1984):
      1. Remove linear trend to get detrended residuals
      2. Estimate lag-1 autocorrelation φ from residuals
      3. Effective n: n_eff = n * (1-φ) / (1+φ)
      4. t = ρ * sqrt(n_eff-2) / sqrt(1-ρ²); p from t-distribution

    Returns: rho, p_naive, phi, n_eff, p_corrected
    """
    n = len(y)
    rho, p_naive = stats.spearmanr(years, y)
    if n < 4:
        return rho, p_naive, np.nan, float(n), p_naive

    # Detrend before measuring autocorrelation
    slope, intercept = np.polyfit(years, y, 1)
    resid = y - (slope * np.asarray(years) + intercept)

    phi = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    phi = max(-0.999, min(0.999, phi))        # numerical clamp
    n_eff = max(3.0, n * (1.0 - phi) / (1.0 + phi))

    if abs(rho) >= 1.0:
        p_corrected = 0.0
    else:
        t_stat = rho * np.sqrt(n_eff - 2.0) / np.sqrt(1.0 - rho ** 2)
        p_corrected = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n_eff - 2.0)))

    return rho, p_naive, phi, n_eff, p_corrected


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
    print("=== Registry DL — 18: Site-Level Temporal Trends ===")
    print(f"H1: {HYPOTHESIS_1}")
    print(f"H2: {HYPOTHESIS_2}")
    print(f"H3: {HYPOTHESIS_3}\n")

    # ── Load registry ─────────────────────────────────────────────────────────
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip().str.lstrip("﻿")
    df["pid"]    = df["病歷號(2)"].astype(str).str.strip()
    df["site"]   = df["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df["seq"]    = pd.to_numeric(df["癌症發生順序(34)"], errors="coerce")
    df["dx_ts"]  = df["最初診斷日(45)"].apply(roc_to_ts)
    df["age"]    = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]    = df["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"})
    df           = df.dropna(subset=["dx_ts","age","sex"])
    df["diag_yr"] = df["dx_ts"].dt.year

    # First cancer per patient (sequence == 1 or min seq per patient)
    first = (df.sort_values("dx_ts")
               .groupby("pid").first()
               .reset_index()[["pid","site","diag_yr","age","sex"]])
    first = first[first["diag_yr"].between(2003, 2020)]

    print(f"  First-primary patients 2003–2020: {len(first):,}")
    print(f"  Sites with ≥{MIN_FIRST_PX} first-primary patients: "
          f"{(first.groupby('site').size() >= MIN_FIRST_PX).sum()}")

    # ── Annual fraction per site ──────────────────────────────────────────────
    annual_total = first.groupby("diag_yr").size().rename("total")
    annual_site  = first.groupby(["diag_yr","site"]).size().rename("n").reset_index()
    annual_site  = annual_site.merge(annual_total, on="diag_yr")
    annual_site["fraction"] = annual_site["n"] / annual_site["total"]

    # Pivot: site × year
    pivot = annual_site.pivot(index="diag_yr", columns="site", values="fraction").fillna(0)
    pivot.to_csv(OUT / "annual_fraction_by_site.csv")

    years = pivot.index.values
    site_totals = first.groupby("site").size()

    # ── Spearman ρ per site (with AC-corrected p) ─────────────────────────────
    rows = []
    for site in pivot.columns:
        if site_totals.get(site, 0) < MIN_FIRST_PX:
            continue
        y = pivot[site].values
        rho, p_naive, phi, n_eff, p_corr = spearmanr_ac_corrected(years, y)
        rows.append({
            "site":        site,
            "axis":        axis_label(site),
            "n_first":     int(site_totals.get(site, 0)),
            "rho":         round(rho, 3),
            "p_naive":     round(p_naive, 4),
            "phi_lag1":    round(phi, 3) if not np.isnan(phi) else np.nan,
            "n_eff":       round(n_eff, 1),
            "p_corrected": round(p_corr, 4),
            "direction":   "rising" if rho > 0 else "falling",
            "sig_naive":   p_naive < 0.05,
            "sig":         p_corr  < 0.05,   # primary significance flag
        })
    trend_df = pd.DataFrame(rows).sort_values("rho", ascending=False)
    trend_df.to_csv(OUT / "trend_by_site.csv", index=False)

    n_naive = trend_df["sig_naive"].sum()
    n_corr  = trend_df["sig"].sum()
    n_demoted = n_naive - n_corr
    print(f"\n  n={len(years)} annual points | "
          f"naive sig: {n_naive} sites | AC-corrected sig: {n_corr} sites | "
          f"demoted by correction: {n_demoted}")
    print("\n  Spearman trend — top 8 rising (AC-corrected p):")
    print(trend_df.head(8)[["site","axis","rho","p_naive","phi_lag1","n_eff","p_corrected","sig"]].to_string(index=False))
    print("\n  Spearman trend — top 8 falling (AC-corrected p):")
    print(trend_df.tail(8)[["site","axis","rho","p_naive","phi_lag1","n_eff","p_corrected","sig"]].to_string(index=False))

    # ── Hypothesis check ─────────────────────────────────────────────────────
    print("\n  Pre-registered hypothesis results (AC-corrected):")
    for site, label, h in [
        ("C22", "C22 liver HCC", "H1"),
        ("C12", "C12 pyriform", "H2"),
        ("C13", "C13 hypopharynx", "H2"),
        ("C50", "C50 breast", "H3"),
        ("C54", "C54 endometrial", "H3"),
    ]:
        r = trend_df[trend_df["site"]==site]
        if not r.empty:
            row = r.iloc[0]
            status = "CONFIRMED" if row["sig"] else ("NAIVE-ONLY" if row["sig_naive"] else "NOT SIG")
            print(f"  {h} [{label}]: ρ={row['rho']:.3f} "
                  f"p_naive={row['p_naive']:.4f} φ={row['phi_lag1']:.3f} "
                  f"n_eff={row['n_eff']:.1f} p_corr={row['p_corrected']:.4f} → {status}")
        else:
            print(f"  {h} [{label}]: insufficient data")

    # ── Fig A: Top-5 rising sites ─────────────────────────────────────────────
    top_rising  = trend_df[trend_df["sig"]].nlargest(5, "rho")
    top_falling = trend_df[trend_df["sig"]].nsmallest(5, "rho")

    for fname, subset, title in [
        ("fig_rising_sites.png",  top_rising,  "Top rising sites 2003–2020 (Spearman ρ > 0, p < 0.05)"),
        ("fig_falling_sites.png", top_falling, "Top falling sites 2003–2020 (Spearman ρ < 0, p < 0.05)"),
    ]:
        fig, ax = plt.subplots(figsize=(9,5))
        for _, row in subset.iterrows():
            if row["site"] not in pivot.columns: continue
            color = AXIS_PALETTE[row["axis"]]
            y = pivot[row["site"]].values * 100   # percent
            ax.plot(years, y, marker="o", ms=4, label=f"{row['site']} ({row['axis']}) ρ={row['rho']:.2f}",
                    color=color)
        ax.set_xlabel("Diagnosis year")
        ax.set_ylabel("% of annual first primaries")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=150)
        plt.close()

    # ── Fig B: ρ bar chart all significant sites ──────────────────────────────
    sig = trend_df[trend_df["sig"]].sort_values("rho")
    colors = [AXIS_PALETTE[r["axis"]] for _, r in sig.iterrows()]

    fig, ax = plt.subplots(figsize=(9, max(5, len(sig)*0.35 + 1.5)))
    bars = ax.barh(sig["site"], sig["rho"], color=colors, alpha=0.8)
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("Spearman ρ (annual fraction vs year, 2003–2020)")
    ax.set_title(f"Cancer site temporal trends\n"
                 f"(AC-corrected p<0.05; n={n_corr} sites; {n_demoted} demoted vs naive)")
    for axis_name, color in AXIS_PALETTE.items():
        if any(r["axis"] == axis_name for _, r in sig.iterrows()):
            ax.barh([], [], color=color, label=axis_name, alpha=0.8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_trend_rho_bar.png", dpi=150)
    plt.close()

    # ── Fig C: UADT sites trend ───────────────────────────────────────────────
    uadt_sites = [s for s in ["C12","C13","C15","C06","C10"] if s in pivot.columns]
    fig, ax = plt.subplots(figsize=(9,5))
    for site in uadt_sites:
        y = pivot[site].values * 100
        rho_row = trend_df[trend_df["site"]==site]
        rho_str = f"ρ={rho_row['rho'].values[0]:.2f}" if not rho_row.empty else ""
        ax.plot(years, y, marker="o", ms=4, label=f"{site} {rho_str}")
    ax.axvline(2006, color="gray", lw=1, ls="--", alpha=0.7, label="~2006 betel campaigns")
    ax.set_xlabel("Diagnosis year")
    ax.set_ylabel("% of annual first primaries")
    ax.set_title("UADT site temporal trends\n"
                 "H2: C12/C13 declining post-2006 (betel nut regulation)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_uadt_trend.png", dpi=150)
    plt.close()

    # ── Fig D: Hypothesis summary panel ──────────────────────────────────────
    hyp_sites = {
        "H1: C22 (HBV/GI)": (["C22"], "#2ca02c"),
        "H2: UADT betel (C12+C13)": (["C12","C13"], "#2e7fbf"),
        "H3: Hormonal (C50+C54)": (["C50","C54"], "#9467bd"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, (label, (sites, color)) in zip(axes, hyp_sites.items()):
        y_vals = sum(pivot[s].values for s in sites if s in pivot.columns) * 100
        ax.plot(years, y_vals, marker="o", ms=4, color=color, lw=2)
        # Trend line
        slope, intercept, *_ = stats.linregress(years, y_vals)
        ax.plot(years, intercept + slope*years, "--", color=color, alpha=0.6, lw=1.5)
        row = trend_df[trend_df["site"].isin(sites)]
        if not row.empty:
            rho_mean = row["rho"].mean()
            p_corr   = row["p_corrected"].min()
            p_naive  = row["p_naive"].min()
            sig_str  = "✓" if p_corr < 0.05 else "✗ (naive only)" if p_naive < 0.05 else "✗"
            ax.set_title(f"{label}\nρ≈{rho_mean:.2f} p_corr={p_corr:.4f} {sig_str}", fontsize=9)
        else:
            ax.set_title(label, fontsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("% first primaries")
    fig.suptitle("Pre-registered hypothesis checks — annual fraction of first-primary diagnoses",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hypothesis_check.png", dpi=150)
    plt.close()

    print(f"\n  Saved → {OUT}/")


if __name__ == "__main__":
    main()
