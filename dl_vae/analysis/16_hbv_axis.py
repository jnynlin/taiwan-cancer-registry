"""
Registry DL — Script 16: HBV/GI-Systemic Axis — Co-occurrence Network

The SIR analysis (Script 15) showed SIR << 1 for C22 as second primary across ALL
index sites — consistent with HBV patients developing HCC as their FIRST cancer,
not as a sequential second event. The co-occurrence structure is therefore
bidirectional: C22 clusters with GI/systemic sites but as index rather than target.

This script characterises the GI-systemic co-occurrence axis directly:
  1. Among multi-cancer patients: co-occurrence rates of C22 with each other site
  2. C22 vs non-C22 patients: enrichment of GI/systemic multi-cancer patterns
  3. Masked predictor evidence: C22 rank in top-K predictions by first_site
  4. Axis membership: which sites co-cluster with C22 (GI axis) vs UADT vs hormonal

Outputs:
  results/16_hbv/c22_cooccurrence.csv
  results/16_hbv/fig_c22_network.png
  results/16_hbv/fig_axis_comparison.png
  results/16_hbv/fig_predictor_c22_rank.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path(__file__).parent.parent
RAW  = BASE.parent / "data/processed/all_cancers.csv"
R12  = BASE / "results/12_surveillance"
OUT  = BASE / "results/16_hbv"
OUT.mkdir(parents=True, exist_ok=True)

TARGET   = "C22"
UADT     = {"C02","C03","C04","C05","C06","C09","C10","C12","C13","C15"}
GI_SYS   = {"C16","C17","C18","C19","C20","C25","C34","C61","C67"}
HORMONAL = {"C50","C53","C54","C56"}

AXIS_PALETTE = {
    "C22 hub": "#e05c2e",
    "GI/systemic": "#2ca02c",
    "UADT": "#2e7fbf",
    "Hormonal": "#9467bd",
    "Other": "#aaaaaa",
}


def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    y  = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00","99") else "01"
    dd = s[5:7] if s[5:7] not in ("00","99") else "01"
    try: return pd.Timestamp(f"{y}-{mm}-{dd}")
    except: return pd.NaT


def axis_label(site):
    if site == TARGET:   return "C22 hub"
    if site in GI_SYS:  return "GI/systemic"
    if site in UADT:     return "UADT"
    if site in HORMONAL: return "Hormonal"
    return "Other"


def main():
    print("=== Registry DL — 16: HBV/GI Axis Co-occurrence Network ===")

    # Load registry
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip().str.lstrip("﻿")
    df["pid"]  = df["病歷號(2)"].astype(str).str.strip()
    df["site"] = df["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df["age"]  = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]  = df["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"})
    df = df.dropna(subset=["age","sex"])

    # Site-level multi-hot: which sites does each patient have?
    site_counts = df.groupby("site")["pid"].nunique()
    top_sites   = sorted(site_counts[site_counts >= 100].index.tolist())

    pivot = (df[df["site"].isin(top_sites)]
             .groupby(["pid","site"])
             .size().unstack(fill_value=0)
             .clip(upper=1))          # multi-hot

    print(f"  Multi-hot matrix: {pivot.shape[0]:,} patients × {pivot.shape[1]} sites")

    has_c22 = pivot.index[pivot.get(TARGET, pd.Series(0, index=pivot.index)) == 1]
    no_c22  = pivot.index.difference(has_c22)
    print(f"  C22 patients: {len(has_c22):,}  Non-C22: {len(no_c22):,}")

    # ── 1. Co-occurrence rates among C22 patients ──────────────────────────
    c22_mat  = pivot.loc[has_c22]
    base_mat = pivot.loc[no_c22]

    rows = []
    for site in top_sites:
        if site == TARGET: continue
        if site not in pivot.columns: continue
        rate_c22  = c22_mat[site].mean()
        rate_base = base_mat[site].mean()
        # OR (Cornfield)
        a = int((c22_mat[site] == 1).sum());   b = len(has_c22) - a
        c = int((base_mat[site] == 1).sum());  d = len(no_c22) - c
        if b * c == 0:
            or_val, or_lo, or_hi = np.nan, np.nan, np.nan
        else:
            or_val = (a * d) / (b * c)
            se_log = np.sqrt(1/a + 1/b + 1/c + 1/d) if all(x > 0 for x in [a,b,c,d]) else np.nan
            or_lo  = np.exp(np.log(or_val) - 1.96 * se_log) if not np.isnan(se_log) else np.nan
            or_hi  = np.exp(np.log(or_val) + 1.96 * se_log) if not np.isnan(se_log) else np.nan
        rows.append({
            "site": site, "axis": axis_label(site),
            "n_c22_pts": len(has_c22),
            "rate_in_c22": round(rate_c22, 4),
            "rate_in_base": round(rate_base, 4),
            "OR": round(or_val, 3) if not np.isnan(or_val) else None,
            "OR_lo": round(or_lo, 3) if not np.isnan(or_lo) else None,
            "OR_hi": round(or_hi, 3) if not np.isnan(or_hi) else None,
        })

    cooc_df = pd.DataFrame(rows).sort_values("OR", ascending=False)
    cooc_df.to_csv(OUT / "c22_cooccurrence.csv", index=False)

    print("\n  Co-occurrence OR (C22 patients vs non-C22) — top 15:")
    print(cooc_df[["site","axis","rate_in_c22","rate_in_base","OR",
                   "OR_lo","OR_hi"]].head(15).to_string())

    # ── Fig A: OR forest by site, colored by axis ─────────────────────────
    show = cooc_df[cooc_df["OR"].notna()].head(25)
    colors = [AXIS_PALETTE.get(r["axis"], "#aaaaaa") for _, r in show.iterrows()]

    fig, ax = plt.subplots(figsize=(9, max(7, len(show)*0.38 + 1.5)))
    y = range(len(show))
    ax.scatter(show["OR"], list(y), c=colors, zorder=5, s=55)
    valid_ci = show[show["OR_lo"].notna()]
    y_valid  = [i for i, (_, r) in enumerate(show.iterrows()) if pd.notna(r["OR_lo"])]
    ax.hlines(y_valid, valid_ci["OR_lo"].values, valid_ci["OR_hi"].values,
              colors=[colors[i] for i in y_valid], lw=2, alpha=0.7)
    ax.axvline(1.0, color="gray", lw=1, ls="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [f"{r['site']} ({r['axis']})  {r['rate_in_c22']:.1%} vs {r['rate_in_base']:.1%}"
         for _, r in show.iterrows()], fontsize=8)
    ax.set_xlabel("Odds ratio for co-occurrence with C22\n(C22 patients vs non-C22 patients)")
    ax.set_title("Co-occurrence with C22 liver HCC\n"
                 "(within multi-hot patient matrix — bidirectional)")
    for axis_name, color in AXIS_PALETTE.items():
        if any(r["axis"] == axis_name for _, r in show.iterrows()):
            ax.scatter([], [], c=color, label=axis_name, s=40)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_c22_network.png", dpi=150)
    plt.close()

    # ── 2. Axis comparison: UADT vs GI/sys OR with C22 ────────────────────
    uadt_ors = cooc_df[cooc_df["axis"] == "UADT"]["OR"].dropna()
    gisy_ors = cooc_df[cooc_df["axis"] == "GI/systemic"]["OR"].dropna()
    horm_ors = cooc_df[cooc_df["axis"] == "Hormonal"]["OR"].dropna()

    mw_stat, mw_p = stats.mannwhitneyu(gisy_ors, uadt_ors, alternative="greater")
    print(f"\n  Axis OR comparison:")
    print(f"  GI/systemic median OR={gisy_ors.median():.2f}  UADT median OR={uadt_ors.median():.2f}")
    print(f"  Mann-Whitney GI>UADT: p={mw_p:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([gisy_ors.values, uadt_ors.values, horm_ors.values],
               labels=[f"GI/systemic\n(n={len(gisy_ors)})",
                       f"UADT\n(n={len(uadt_ors)})",
                       f"Hormonal\n(n={len(horm_ors)})"],
               patch_artist=True,
               boxprops=dict(facecolor="#2ca02c", alpha=0.5),
               medianprops=dict(color="black", lw=2))
    for patch, color in zip(ax.patches,
                            ["#2ca02c","#2e7fbf","#9467bd"]):
        patch.set_facecolor(color); patch.set_alpha(0.45)
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_ylabel("Odds ratio for co-occurrence with C22")
    ax.set_title(f"GI/systemic vs UADT axis: C22 co-occurrence OR\n"
                 f"Mann-Whitney GI>UADT: p={mw_p:.4f}")
    ax.text(0.98, 0.98,
            f"GI median: {gisy_ors.median():.2f}\nUADT median: {uadt_ors.median():.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray", pad=3))
    fig.tight_layout()
    fig.savefig(OUT / "fig_axis_comparison.png", dpi=150)
    plt.close()

    # ── 3. Transformer predictor evidence: C22 rank by first site ─────────
    cal = pd.read_csv(R12 / "surveillance_calendar.csv")
    for col in ["pred1","pred2","pred3","pred4","pred5"]:
        cal[f"is_c22_{col}"] = (cal[col] == TARGET)
    cal["c22_in_top3"] = cal[["is_c22_pred1","is_c22_pred2","is_c22_pred3"]].any(axis=1)
    cal["c22_in_top1"] = cal["is_c22_pred1"]

    site_pred = (cal.groupby("first_site")
                 .agg(n=("c22_in_top3","count"),
                      pct_top1=("c22_in_top1","mean"),
                      pct_top3=("c22_in_top3","mean"))
                 .reset_index()
                 .sort_values("pct_top3", ascending=False))
    site_pred["axis"] = site_pred["first_site"].apply(axis_label)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, col, label in [
        (axes[0], "pct_top3", "Fraction with C22 in top-3 predictions"),
        (axes[1], "pct_top1", "Fraction with C22 as top-1 prediction"),
    ]:
        show_p = site_pred[site_pred["n"] >= 5].copy()
        colors_p = [AXIS_PALETTE.get(axis_label(s), "#aaaaaa") for s in show_p["first_site"]]
        ys = range(len(show_p))
        axes_p = ax
        axes_p.barh(list(ys), show_p[col].values, color=colors_p, alpha=0.8)
        axes_p.set_yticks(list(ys))
        axes_p.set_yticklabels(
            [f"{r['first_site']} ({r['axis']})  n={r['n']}"
             for _, r in show_p.iterrows()], fontsize=8)
        axes_p.set_xlabel(label)
        axes_p.axvline(cal["c22_in_top3"].mean() if col=="pct_top3" else
                       cal["c22_in_top1"].mean(),
                       color="gray", lw=1, ls="--", label="overall mean")
        axes_p.legend(fontsize=8)
    for axis_name, color in AXIS_PALETTE.items():
        if any(axis_label(s) == axis_name for s in site_pred["first_site"]):
            axes[0].scatter([], [], c=color, label=axis_name, s=35)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Transformer surveillance predictions: C22 by first-cancer site\n"
                 "(GI/systemic sites predict C22 far more often than UADT/hormonal)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_predictor_c22_rank.png", dpi=150)
    plt.close()

    print(f"\n  Predictor: C22 in top-3 — GI/sys sites avg: "
          f"{site_pred[site_pred['axis']=='GI/systemic']['pct_top3'].mean():.2f}  "
          f"UADT: {site_pred[site_pred['axis']=='UADT']['pct_top3'].mean():.2f}")
    print(f"  Saved → {OUT}/")


if __name__ == "__main__":
    main()
