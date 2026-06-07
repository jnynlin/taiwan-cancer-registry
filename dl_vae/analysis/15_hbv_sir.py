"""
Registry DL — Script 15: HBV/GI-Systemic Axis — SIR Analysis

Core question: Is C22 (liver HCC) elevated as a second primary across ALL cancer
sites (pan-carcinogen), or selectively in GI/systemic sites (axis-specific)?

Finding from Script 12: C22 appears in top-3 predictions for GI/lung/prostate/
bladder sites (80–100%) but NOT for UADT or breast sites (0–3%).
This script confirms the pattern with SIR statistics.

Analyses:
  1. SIR of C22 as second primary for each index site (Poisson, sex×age_band stratified)
  2. Reverse SIR: C22 as index → second primary site distribution
  3. Diagnosis-year trend of C22 first-primary incidence (HBV vaccination cohort proxy)
  4. Age-specific C22 incidence trend 2003–2020 (young vs old strata)

Outputs:
  results/15_hbv/sir_c22_by_index.csv
  results/15_hbv/sir_reverse_from_c22.csv
  results/15_hbv/c22_trend.csv
  results/15_hbv/fig_sir_c22_forest.png
  results/15_hbv/fig_sir_reverse.png
  results/15_hbv/fig_c22_trend.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import poisson

BASE = Path(__file__).parent.parent
RAW  = BASE.parent / "data/processed/all_cancers.csv"
OUT  = BASE / "results/15_hbv"
OUT.mkdir(parents=True, exist_ok=True)

STUDY_END = pd.Timestamp("2020-12-31")
MIN_OBS   = 5     # minimum observed C22 second primaries to report SIR
TARGET    = "C22"

UADT_SITES = {"C02","C03","C04","C05","C06","C09","C10","C11","C12","C13","C15","C32","C34"}
GI_SITES   = {"C16","C17","C18","C19","C20","C22","C23","C24","C25"}


def spearmanr_ac_corrected(years, y):
    """Spearman ρ with autocorrelation-corrected p-value (Chelton 1983)."""
    n = len(y)
    rho, p_naive = stats.spearmanr(years, y)
    if n < 4:
        return rho, p_naive, np.nan, float(n), p_naive
    slope, intercept = np.polyfit(years, y, 1)
    resid = y - (slope * np.asarray(years, dtype=float) + intercept)
    phi = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    phi = max(-0.999, min(0.999, phi))
    n_eff = max(3.0, n * (1.0 - phi) / (1.0 + phi))
    if abs(rho) >= 1.0:
        p_corrected = 0.0
    else:
        t_stat = rho * np.sqrt(n_eff - 2.0) / np.sqrt(1.0 - rho ** 2)
        p_corrected = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n_eff - 2.0)))
    return rho, p_naive, phi, n_eff, p_corrected
AXIS_COLOR = {"UADT":"#2e7fbf", "GI":"#e05c2e", "Other":"#888888",
              "Hormonal":"#9467bd", "Lung/prostate":"#2ca02c"}


def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    y  = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00","99") else "01"
    dd = s[5:7] if s[5:7] not in ("00","99") else "01"
    try: return pd.Timestamp(f"{y}-{mm}-{dd}")
    except: return pd.NaT


def age_band(age):
    if age < 40:  return "<40"
    if age < 50:  return "40-49"
    if age < 60:  return "50-59"
    if age < 70:  return "60-69"
    return "70+"


def load_registry():
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip().str.lstrip("﻿")
    df["pid"]    = df["病歷號(2)"].astype(str).str.strip()
    df["site"]   = df["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df["dx"]     = df["最初診斷日(45)"].apply(roc_to_ts)
    df["fu_end"] = df["最後聯絡日(30)"].apply(roc_to_ts).fillna(STUDY_END).clip(upper=STUDY_END)
    df["age"]    = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]    = df["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"})
    df["dead"]   = (df["生存狀態(27)"].astype(str).str.strip() == "0").astype(int)
    df = df.dropna(subset=["dx","age","sex"])
    df["age_band"] = df["age"].apply(age_band)
    df["diag_yr"]  = df["dx"].dt.year
    return df


def compute_sir(df, index_site, target_site=TARGET):
    """
    SIR of target_site as second primary after index_site.
    Person-years at risk start at first index_site dx.
    """
    # Patients with first cancer = index_site (no prior target_site)
    first_dx = (df.sort_values("dx")
                  .groupby("pid")
                  .agg(first_site=("site","first"),
                       first_dx=("dx","first"),
                       sex=("sex","first"),
                       age=("age","first"),
                       fu_end=("fu_end","max"))
                  .reset_index())
    # Index patients: first cancer is index_site
    idx_pts = first_dx[first_dx["first_site"] == index_site].copy()
    if len(idx_pts) < 10:
        return None

    idx_pts["age_band"] = idx_pts["age"].apply(age_band)

    # Did they develop target_site after first_dx?
    target_after = (df[(df["site"] == target_site)]
                    .sort_values("dx")
                    .groupby("pid")["dx"]
                    .first()
                    .rename("target_dx"))
    idx_pts = idx_pts.join(target_after, on="pid", how="left")
    idx_pts["event"]   = idx_pts["target_dx"].notna()
    # Exclude patients where C22 was the first cancer or preceded index
    idx_pts = idx_pts[~(idx_pts["event"] & (idx_pts["target_dx"] <= idx_pts["first_dx"]))]
    # Person-years: first_dx → min(target_dx, fu_end)
    idx_pts["end_risk"] = idx_pts[["target_dx","fu_end"]].min(axis=1)
    idx_pts["py"]       = ((idx_pts["end_risk"] - idx_pts["first_dx"])
                           .dt.days.clip(lower=0) / 365.25)

    obs = int(idx_pts["event"].sum())
    if obs < MIN_OBS:
        return None

    # Expected: age×sex stratified target_site incidence from general registry
    # Rate = target_site first primaries per person-year in stratum
    # Use all patients NOT in idx_pts as reference
    ref_df = df[~df["pid"].isin(idx_pts["pid"])]
    ref_first = (ref_df.sort_values("dx")
                       .groupby("pid")
                       .agg(first_site=("site","first"),
                            sex=("sex","first"),
                            age=("age","first"),
                            first_dx=("dx","first"),
                            fu_end=("fu_end","max"))
                       .reset_index())
    ref_first["age_band"] = ref_first["age"].apply(age_band)
    ref_first["py"] = ((ref_first["fu_end"] - ref_first["first_dx"])
                       .dt.days.clip(lower=0) / 365.25)
    target_in_ref = set(ref_df[ref_df["site"] == target_site]["pid"])
    ref_first["is_target"] = ref_first["pid"].isin(target_in_ref).astype(int)

    # Rate per stratum
    rates = (ref_first.groupby(["sex","age_band"])
             .agg(target_n=("is_target","sum"), ref_py=("py","sum"))
             .reset_index())
    rates["rate"] = rates["target_n"] / (rates["ref_py"] + 1e-6)

    # Expected: sum over index patients
    expected = 0.0
    for _, r in rates.iterrows():
        mask = (idx_pts["sex"] == r["sex"]) & (idx_pts["age_band"] == r["age_band"])
        expected += idx_pts.loc[mask, "py"].sum() * r["rate"]

    if expected < 0.1:
        return None

    sir = obs / expected
    ci_lo = poisson.ppf(0.025, obs) / expected
    ci_hi = poisson.ppf(0.975, obs + 1) / expected
    p     = 2 * min(poisson.cdf(obs, expected), 1 - poisson.cdf(obs - 1, expected))

    return {"index_site": index_site, "n_index": len(idx_pts),
            "obs": obs, "exp": round(expected, 2),
            "SIR": round(sir, 3), "CI_lo": round(ci_lo, 3),
            "CI_hi": round(ci_hi, 3), "p": round(p, 4)}


def site_axis(site):
    if site in UADT_SITES:        return "UADT"
    if site in GI_SITES:          return "GI"
    if site in {"C50","C53","C54","C56"}: return "Hormonal"
    if site in {"C34","C61"}:     return "Lung/prostate"
    return "Other"


def main():
    print("=== Registry DL — 15: HBV/GI Axis SIR ===")

    df = load_registry()
    print(f"  Loaded: {df['pid'].nunique():,} patients · {len(df):,} records")
    print(f"  C22 patients: {(df['site']==TARGET).sum():,} records / "
          f"{df[df['site']==TARGET]['pid'].nunique():,} unique")

    # ── 1. SIR of C22 after each index site ───────────────────────────────
    print("\n  Computing SIR of C22 as second primary…")
    all_sites = df.groupby("site")["pid"].nunique()
    top_sites = all_sites[all_sites >= 200].index.tolist()
    top_sites = [s for s in top_sites if s != TARGET]

    sir_rows = []
    for site in sorted(top_sites):
        r = compute_sir(df, site)
        if r:
            r["axis"] = site_axis(site)
            sir_rows.append(r)
            print(f"  {site} ({r['axis']:12s}): SIR={r['SIR']:.2f} "
                  f"[{r['CI_lo']:.2f}–{r['CI_hi']:.2f}] "
                  f"obs={r['obs']} exp={r['exp']} p={r['p']:.3f}")

    sir_df = pd.DataFrame(sir_rows).sort_values("SIR", ascending=False)
    sir_df.to_csv(OUT / "sir_c22_by_index.csv", index=False)

    # ── Fig A: Forest plot — SIR by index site ────────────────────────────
    palette = {ax: AXIS_COLOR[ax] for ax in AXIS_COLOR}
    fig, ax = plt.subplots(figsize=(9, max(6, len(sir_df) * 0.38 + 1.5)))
    y = range(len(sir_df))
    colors = [palette.get(r["axis"], "#888888") for _, r in sir_df.iterrows()]

    ax.scatter(sir_df["SIR"], list(y), c=colors, zorder=5, s=55)
    ax.hlines(list(y), sir_df["CI_lo"], sir_df["CI_hi"],
              colors=colors, lw=2, alpha=0.7)
    ax.axvline(1.0, color="gray", lw=1, ls="--")

    ylabels = [f"{r['index_site']} ({r['axis']})  n={r['n_index']:,}"
               for _, r in sir_df.iterrows()]
    ax.set_yticks(list(y)); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("SIR (95% CI) — C22 liver HCC as second primary")
    ax.set_title(f"Standardised incidence ratio: C22 after index cancer\n"
                 f"(reference = registry C22 rates; min obs={MIN_OBS})")

    # Legend
    for axis_name, color in AXIS_COLOR.items():
        if any(r["axis"] == axis_name for _, r in sir_df.iterrows()):
            ax.scatter([], [], c=color, label=axis_name, s=40)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sir_c22_forest.png", dpi=150)
    plt.close()

    # ── 2. Reverse SIR: after C22, what comes next? ───────────────────────
    print("\n  Computing reverse SIR (second primaries after C22)…")
    rev_rows = []
    for target in sorted(top_sites):
        r = compute_sir(df, index_site=TARGET, target_site=target)
        if r:
            r["second_site"] = target
            r["axis"]        = site_axis(target)
            rev_rows.append(r)

    rev_df = pd.DataFrame(rev_rows).sort_values("SIR", ascending=False)
    rev_df.to_csv(OUT / "sir_reverse_from_c22.csv", index=False)
    print(f"  Top reverse SIRs after C22:")
    print(rev_df[["second_site","axis","SIR","CI_lo","CI_hi","obs"]].head(10).to_string())

    # Fig B: Reverse SIR forest (top 15)
    show_rev = rev_df.head(15)
    fig, ax = plt.subplots(figsize=(9, max(5, len(show_rev) * 0.42 + 1.5)))
    y2  = range(len(show_rev))
    c2  = [palette.get(r["axis"], "#888888") for _, r in show_rev.iterrows()]
    ax.scatter(show_rev["SIR"], list(y2), c=c2, zorder=5, s=55)
    ax.hlines(list(y2), show_rev["CI_lo"], show_rev["CI_hi"],
              colors=c2, lw=2, alpha=0.7)
    ax.axvline(1.0, color="gray", lw=1, ls="--")
    ylabels2 = [f"{r['second_site']} ({r['axis']})  obs={r['obs']}"
                for _, r in show_rev.iterrows()]
    ax.set_yticks(list(y2)); ax.set_yticklabels(ylabels2, fontsize=8)
    ax.set_xlabel("SIR (95% CI) — second primary after C22 liver HCC")
    ax.set_title("Second primaries after C22 — reverse SIR\n(top 15 by SIR)")
    for axis_name, color in AXIS_COLOR.items():
        if any(r["axis"] == axis_name for _, r in show_rev.iterrows()):
            ax.scatter([], [], c=color, label=axis_name, s=40)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sir_reverse.png", dpi=150)
    plt.close()

    # ── 3. C22 incidence trend 2003–2020 ──────────────────────────────────
    print("\n  C22 incidence trend by year…")
    first_dx_all = (df.sort_values("dx")
                      .groupby("pid")
                      .agg(first_site=("site","first"),
                           diag_yr=("diag_yr","first"),
                           age=("age","first"),
                           sex=("sex","first"))
                      .reset_index())

    c22_first = first_dx_all[first_dx_all["first_site"] == TARGET].copy()
    all_first = first_dx_all.copy()

    trend = (all_first.groupby("diag_yr")
             .agg(total_pts=("pid","count"),
                  c22_pts=("first_site", lambda x: (x==TARGET).sum()))
             .reset_index())
    trend["c22_rate_pct"] = trend["c22_pts"] / trend["total_pts"] * 100

    # Age-stratified trend
    for label, lo, hi in [("Age<50", 0, 49), ("Age 50-64", 50, 64), ("Age≥65", 65, 120)]:
        mask = (all_first["age"] >= lo) & (all_first["age"] <= hi)
        yr_grp = all_first[mask].groupby("diag_yr").agg(
            total=("pid","count"),
            c22=("first_site", lambda x: (x==TARGET).sum())).reset_index()
        yr_grp["rate"] = yr_grp["c22"] / yr_grp["total"] * 100
        trend[f"rate_{label}"] = trend["diag_yr"].map(
            yr_grp.set_index("diag_yr")["rate"])

    trend.to_csv(OUT / "c22_trend.csv", index=False)

    # Spearman trend tests — with autocorrelation correction
    valid = trend[trend["diag_yr"].between(2003, 2020)]
    r_all, p_naive_all, phi_all, neff_all, p_corr_all = spearmanr_ac_corrected(
        valid["diag_yr"].values, valid["c22_rate_pct"].values)
    print(f"  C22 overall trend ρ={r_all:.3f} "
          f"p_naive={p_naive_all:.4f} φ={phi_all:.3f} "
          f"n_eff={neff_all:.1f} p_corrected={p_corr_all:.4f}")

    # Save trend stats
    trend_stats = pd.DataFrame([{
        "stratum": "All ages",
        "rho": round(r_all, 3),
        "p_naive": round(p_naive_all, 4),
        "phi_lag1": round(phi_all, 3),
        "n_eff": round(neff_all, 1),
        "p_corrected": round(p_corr_all, 4),
        "sig_corrected": p_corr_all < 0.05,
    }])

    fig, ax = plt.subplots(figsize=(10, 5))
    sig_str = f"p_corr={p_corr_all:.4f} {'✓' if p_corr_all<0.05 else '✗'}"
    ax.plot(trend["diag_yr"], trend["c22_rate_pct"], "o-",
            color="#e05c2e", lw=2.5,
            label=f"All ages ρ={r_all:.2f}, {sig_str} (φ={phi_all:.2f}, n_eff={neff_all:.0f})")
    for label, color, ls in [
        ("Age<50",  "#2ca02c",  "--"),
        ("Age 50-64","#2e7fbf", "--"),
        ("Age≥65",  "#9467bd",  ":")
    ]:
        col = f"rate_{label}"
        if col in trend.columns:
            sub = trend[trend[col].notna()]
            rv, pnv, phiv, neffv, pcrv = spearmanr_ac_corrected(
                sub["diag_yr"].values, sub[col].values)
            trend_stats = pd.concat([trend_stats, pd.DataFrame([{
                "stratum": label,
                "rho": round(rv, 3), "p_naive": round(pnv, 4),
                "phi_lag1": round(phiv, 3), "n_eff": round(neffv, 1),
                "p_corrected": round(pcrv, 4), "sig_corrected": pcrv < 0.05,
            }])], ignore_index=True)
            sig_s = f"p_corr={pcrv:.3f} {'✓' if pcrv<0.05 else '✗'}"
            ax.plot(sub["diag_yr"], sub[col], ls, color=color, lw=1.5,
                    label=f"{label} ρ={rv:.2f}, {sig_s}")

    trend_stats.to_csv(OUT / "c22_trend_ac_stats.csv", index=False)
    print(f"  Saved: c22_trend_ac_stats.csv")
    ax.axvline(2010, color="gray", lw=1, ls=":", alpha=0.5,
               label="~HBV vaccination cohort enters adulthood")
    ax.set_xlabel("Diagnosis year")
    ax.set_ylabel("C22 as % of first-primary cancers")
    ax.set_title("C22 liver HCC incidence trend 2003–2020\n"
                 "(proportion of first primaries; by age stratum)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_c22_trend.png", dpi=150)
    plt.close()

    print(f"\n  Saved → {OUT}/")
    print(f"  SIR summary: {len(sir_df)} index sites with obs≥{MIN_OBS}")
    print(f"  Highest C22-after-index SIR: {sir_df.iloc[0]['index_site']} "
          f"SIR={sir_df.iloc[0]['SIR']:.2f}")
    print(f"  Axis breakdown:")
    print(sir_df.groupby("axis")["SIR"].agg(["mean","count"]).round(2).to_string())


if __name__ == "__main__":
    main()
