"""
Registry DL — Script 13: Surveillance Calendar Validation

Evaluates the surveillance calendar from Script 12 against actual outcomes.

Analyses:
  1. Accuracy: R@1/3/5 overall and by first-cancer site
  2. Lead time: distribution of actual gap days (time available for surveillance)
  3. Probability calibration: prob1 vs hit@1 across deciles
  4. Risk stratification: high (prob1>0.15) / medium / low — precision by tier
  5. UADT comparison: C12/C13→C15 model timing vs 6-month guideline
  6. Recommendation heatmap: for each first site, top-3 recommended sites

Outputs:
  results/13_surveillance/
    fig_recall_by_site.png
    fig_lead_time.png
    fig_calibration.png
    fig_risk_tiers.png
    fig_uadt_timing.png
    fig_recommendation_heatmap.png
    validation_summary.csv
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent.parent
R12  = BASE / "results/12_surveillance"
OUT  = BASE / "results/13_surveillance"
OUT.mkdir(parents=True, exist_ok=True)

from constants import UADT  # noqa: E402
HIGH_THRESH   = 0.15
MEDIUM_THRESH = 0.07
GUIDELINE_DAYS = 180   # 6-month endoscopy guideline


def main():
    print("=== Registry DL — 13: Surveillance Validation ===")

    cal     = pd.read_csv(R12 / "surveillance_calendar.csv")
    timing  = pd.read_csv(R12 / "timing_windows.csv")

    # Restrict to patients with a known second cancer
    eval_df = cal[cal["true_second_site"].notna()].copy()
    N       = len(eval_df)
    print(f"  Validation set: {N:,} patients with known second cancer")

    r1 = eval_df["hit@1"].mean()
    r3 = eval_df["hit@3"].mean()
    r5 = eval_df["hit@5"].mean()
    print(f"  R@1={r1:.3f}  R@3={r3:.3f}  R@5={r5:.3f}")

    # ── 1. R@k by first-cancer site ─────────────────────────────────────────
    site_stats = (eval_df.groupby("first_site")
                  .agg(n=("hit@1","count"),
                       r1=("hit@1","mean"),
                       r3=("hit@3","mean"))
                  .reset_index()
                  .sort_values("n", ascending=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    top_sites = site_stats[site_stats["n"] >= 10].head(20)
    for ax, col, label, color in [
        (axes[0], "r1", "R@1", "#2e7fbf"),
        (axes[1], "r3", "R@3", "#e05c2e")
    ]:
        ys = range(len(top_sites))
        ax.barh(list(ys), top_sites[col].values, color=color, alpha=0.8)
        ax.set_yticks(list(ys))
        ax.set_yticklabels(
            [f"{r['first_site']} (n={int(r['n'])})" for _, r in top_sites.iterrows()],
            fontsize=8)
        ax.set_xlabel(label)
        ax.set_title(f"Next-cancer {label} by first-cancer site (n≥10)")
        ax.axvline(eval_df["hit@1"].mean() if col=="r1" else eval_df["hit@3"].mean(),
                   color="gray", lw=1, ls="--", label="overall")
        ax.legend(fontsize=8)
    fig.suptitle("Surveillance accuracy by primary site", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_recall_by_site.png", dpi=150)
    plt.close()

    # ── 2. Lead time distribution ────────────────────────────────────────────
    gaps = eval_df["actual_gap_days"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(gaps / 365.25, bins=40, color="#2e7fbf", alpha=0.75, edgecolor="white")
    ax.axvline(GUIDELINE_DAYS / 365.25, color="#e05c2e", lw=2, ls="--",
               label=f"6-month guideline ({GUIDELINE_DAYS}d)")
    ax.axvline(gaps.median() / 365.25, color="#2ca02c", lw=2, ls="--",
               label=f"Data median ({gaps.median():.0f}d = {gaps.median()/365.25:.1f}yr)")
    within_6mo = (gaps <= GUIDELINE_DAYS).mean()
    ax.text(0.98, 0.95, f"{within_6mo:.1%} of second cancers\noccur ≤6 months",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray", pad=4))
    ax.set_xlabel("Time from first to second cancer (years)")
    ax.set_ylabel("Number of patients")
    ax.set_title("Lead time: available surveillance window\n"
                 "(time between first and second cancer diagnosis)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lead_time.png", dpi=150)
    plt.close()

    # ── 3. Probability calibration ───────────────────────────────────────────
    eval_df["prob1_decile"] = pd.qcut(eval_df["prob1"], q=10, labels=False, duplicates="drop")
    cal_stats = eval_df.groupby("prob1_decile").agg(
        mean_prob=("prob1","mean"),
        hit_rate=("hit@1","mean"),
        n=("hit@1","count")).reset_index()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(cal_stats["mean_prob"], cal_stats["hit_rate"],
               s=cal_stats["n"] / cal_stats["n"].max() * 300 + 20,
               color="#2e7fbf", alpha=0.8, zorder=5)
    for _, r in cal_stats.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_prob"], r["hit_rate"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    lim = max(cal_stats["mean_prob"].max(), cal_stats["hit_rate"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color="gray", lw=1, ls="--", label="perfect calibration")
    ax.set_xlabel("Model probability (pred1)")
    ax.set_ylabel("Observed hit@1 rate")
    ax.set_title("Probability calibration\n(deciles of pred1 probability)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration.png", dpi=150)
    plt.close()

    # ── 4. Risk tiers ────────────────────────────────────────────────────────
    def tier(p):
        if p >= HIGH_THRESH:   return "High"
        if p >= MEDIUM_THRESH: return "Medium"
        return "Low"
    eval_df["risk_tier"] = eval_df["prob1"].apply(tier)
    tier_stats = eval_df.groupby("risk_tier").agg(
        n=("hit@1","count"),
        r1=("hit@1","mean"),
        r3=("hit@3","mean"),
        mean_prob=("prob1","mean")).reset_index()
    tier_stats = tier_stats.set_index("risk_tier").loc[
        [t for t in ["High","Medium","Low"] if t in tier_stats["risk_tier"].values]]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(tier_stats))
    w = 0.35
    b1 = ax.bar([xi - w/2 for xi in x], tier_stats["r1"], width=w,
                color="#14304a", label="R@1")
    b3 = ax.bar([xi + w/2 for xi in x], tier_stats["r3"], width=w,
                color="#2e7fbf", label="R@3")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{t}\n(n={int(tier_stats.loc[t,'n'])}, p≥"
                        f"{'0.15' if t=='High' else '0.07' if t=='Medium' else '0'})"
                        for t in tier_stats.index], fontsize=9)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.set_title("Surveillance precision by risk tier\n(tier defined by top-1 predicted probability)")
    ax.legend(fontsize=9)
    print("\n  Risk tier breakdown:")
    print(tier_stats[["n","mean_prob","r1","r3"]].to_string())
    fig.tight_layout()
    fig.savefig(OUT / "fig_risk_tiers.png", dpi=150)
    plt.close()

    # ── 5. UADT analysis — model timing vs 6-month guideline ─────────────────
    uadt_df = eval_df[eval_df["first_site"].isin({"C12","C13"}) &
                      (eval_df["true_second_site"] == "C15")].copy()
    print(f"\n  UADT guideline comparison:")
    print(f"  C12/C13→C15 cases in val: {len(uadt_df)}")

    if len(uadt_df) >= 5:
        uadt_timing = timing[
            timing["first_site"].isin({"C12","C13"}) &
            (timing["second_site"] == "C15")]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(uadt_df["actual_gap_days"] / 30.44, bins=20,
                color="#2e7fbf", alpha=0.7, label="Actual gap (months)")
        ax.axvline(6, color="#e05c2e", lw=2, ls="--", label="Guideline: 6-month endoscopy")
        if len(uadt_timing):
            med = uadt_timing["gap_median"].mean()
            p25 = uadt_timing["gap_p25"].mean()
            p75 = uadt_timing["gap_p75"].mean()
            ax.axvspan(p25 / 30.44, p75 / 30.44, alpha=0.15, color="#2ca02c",
                       label=f"Model IQR window ({p25/30.44:.0f}–{p75/30.44:.0f} mo)")
            ax.axvline(med / 30.44, color="#2ca02c", lw=2, ls="--",
                       label=f"Model median ({med/30.44:.0f} mo)")
        within_6 = (uadt_df["actual_gap_days"] <= GUIDELINE_DAYS).mean()
        ax.set_xlabel("Months from first to C15 diagnosis")
        ax.set_ylabel("Patients")
        ax.set_title(f"UADT surveillance: C12/C13→C15 timing\n"
                     f"({within_6:.0%} of C15 cases occur ≤6 months after index)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "fig_uadt_timing.png", dpi=150)
        plt.close()
        print(f"  C15 within 6mo: {within_6:.0%}")
        if len(uadt_timing):
            print(f"  Model timing window: {p25/30.44:.0f}–{p75/30.44:.0f} months")

    # ── 6. Recommendation heatmap ─────────────────────────────────────────────
    top_sites = (eval_df.groupby("first_site")["hit@1"]
                 .count().nlargest(15).index.tolist())
    heatmap_rows = []
    for fs in top_sites:
        sub = eval_df[eval_df["first_site"] == fs]
        pred_counts = pd.concat([
            sub["pred1"], sub["pred2"], sub["pred3"]
        ]).value_counts(normalize=True)
        row = {s: pred_counts.get(s, 0.0) for s in top_sites}
        row["first_site"] = fs
        heatmap_rows.append(row)

    heat_df = pd.DataFrame(heatmap_rows).set_index("first_site")
    heat_df = heat_df[[c for c in top_sites if c in heat_df.columns]]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heat_df.values, cmap="Blues", aspect="auto",
                   vmin=0, vmax=heat_df.values.max())
    ax.set_xticks(range(len(heat_df.columns)))
    ax.set_xticklabels(heat_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index, fontsize=8)
    for i in range(len(heat_df.index)):
        for j in range(len(heat_df.columns)):
            v = heat_df.values[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.3 else "black")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Fraction in top-3 recommendations")
    ax.set_xlabel("Recommended site")
    ax.set_ylabel("First (index) cancer site")
    ax.set_title("Surveillance recommendation heatmap\n"
                 "(fraction of patients where site appears in top-3 predictions)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_recommendation_heatmap.png", dpi=150)
    plt.close()

    # ── Summary CSV ──────────────────────────────────────────────────────────
    summary = {
        "n_val_patients": N,
        "r_at_1": round(r1, 4),
        "r_at_3": round(r3, 4),
        "r_at_5": round(r5, 4),
        "median_gap_days":   round(gaps.median()),
        "p25_gap_days":      round(gaps.quantile(0.25)),
        "p75_gap_days":      round(gaps.quantile(0.75)),
        "pct_within_6mo":    round(within_6mo, 4),
        "n_timing_pairs":    len(timing),
    }
    pd.DataFrame([summary]).to_csv(OUT / "validation_summary.csv", index=False)
    print(f"\n  Saved → {OUT}/")


if __name__ == "__main__":
    main()
