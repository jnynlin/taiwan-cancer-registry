"""
Association rule mining for cancer co-occurrence patterns.
Uses Apriori (mlxtend) on the multi-hot patient × cancer matrix.
Focuses on multi-primary patients (≥2 cancer types).
Outputs: results/02_associations/
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

BASE = Path(__file__).parent.parent
MAT  = BASE / "data/patient_cancer_matrix.csv"
SITE = BASE / "data/cancer_site_labels.csv"
META = BASE / "data/patient_meta.csv"
OUT  = BASE / "results/02_associations"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False
    print("  mlxtend not available — using manual pair-count method")


def load():
    matrix = pd.read_csv(MAT, index_col=0)
    sites  = pd.read_csv(SITE)
    meta   = pd.read_csv(META)
    label  = dict(zip(sites["code"], sites["label"]))
    return matrix, sites, meta, label


def manual_pair_counts(mat, min_support=0.005):
    """Enumerate all cancer pairs; compute support, lift, odds ratio."""
    n = len(mat)
    cols = mat.columns.tolist()
    prev = mat.mean()  # marginal prevalence per cancer

    rows = []
    for c1, c2 in combinations(cols, 2):
        n_both  = int((mat[c1] & mat[c2]).sum())
        n_c1    = int(mat[c1].sum())
        n_c2    = int(mat[c2].sum())
        if n_both < 3:
            continue
        support    = n_both / n
        conf_c1c2  = n_both / n_c1 if n_c1 > 0 else 0
        conf_c2c1  = n_both / n_c2 if n_c2 > 0 else 0
        exp_both   = (n_c1 / n) * (n_c2 / n) * n
        lift       = (n_both / exp_both) if exp_both > 0 else np.nan
        # Odds ratio (Fisher)
        a = n_both; b = n_c1 - a; c = n_c2 - a; d = n - a - b - c
        or_val = (a * d) / (b * c) if b > 0 and c > 0 else np.nan
        rows.append({
            "antecedent": c1, "consequent": c2,
            "n_co": n_both, "n_ant": n_c1, "n_con": n_c2,
            "support": round(support, 5),
            "confidence_A→B": round(conf_c1c2, 4),
            "confidence_B→A": round(conf_c2c1, 4),
            "lift": round(lift, 3) if pd.notna(lift) else np.nan,
            "odds_ratio": round(or_val, 3) if pd.notna(or_val) else np.nan,
        })
    return pd.DataFrame(rows)


def fig_top_associations(rules, label, fname, top_n=30):
    top = rules.nlargest(top_n, "lift").copy()
    top["pair"] = top.apply(
        lambda r: f"{label.get(r['antecedent'],r['antecedent'])[:14]}\n↔ {label.get(r['consequent'],r['consequent'])[:14]}",
        axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Lift bar chart
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top)))
    axes[0].barh(top["pair"][::-1], top["lift"][::-1], color=colors[::-1], edgecolor="white")
    axes[0].axvline(1, color="black", linestyle="--", linewidth=0.8, label="Lift=1 (independence)")
    axes[0].set(title=f"Top {top_n} Cancer Pairs by Lift", xlabel="Lift", ylabel="Cancer pair")
    axes[0].tick_params(labelsize=7.5)
    axes[0].legend(fontsize=8)
    axes[0].spines[["top","right"]].set_visible(False)

    # Scatter: support vs lift, sized by n_co
    sc = axes[1].scatter(top["support"]*100, top["lift"],
                         s=top["n_co"]*0.8+20,
                         c=top["odds_ratio"], cmap="RdYlGn",
                         alpha=0.8, edgecolors="white", linewidths=0.5)
    for _, row in top.iterrows():
        axes[1].annotate(
            f"{row['antecedent']}&{row['consequent']}",
            (row["support"]*100, row["lift"]),
            fontsize=5.5, alpha=0.75)
    plt.colorbar(sc, ax=axes[1], label="Odds Ratio")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set(title="Support vs Lift (bubble = n co-occurring patients)",
                xlabel="Support (%)", ylabel="Lift")
    axes[1].spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def fig_network_heatmap(rules, label, n_sites=20):
    """Lift heatmap for top n_sites."""
    top_sites = (rules.groupby("antecedent")["n_co"].sum() +
                 rules.groupby("consequent")["n_co"].sum()
                ).sort_values(ascending=False).head(n_sites).index.tolist()
    pivot = pd.DataFrame(np.nan, index=top_sites, columns=top_sites)
    for _, r in rules[rules["antecedent"].isin(top_sites) &
                      rules["consequent"].isin(top_sites)].iterrows():
        pivot.loc[r["antecedent"], r["consequent"]] = r["lift"]
        pivot.loc[r["consequent"], r["antecedent"]] = r["lift"]
    pivot.index   = [label.get(c,c)[:15] for c in pivot.index]
    pivot.columns = pivot.index
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.eye(len(pivot), dtype=bool)
    sns.heatmap(pivot, mask=mask, cmap="coolwarm", center=1, ax=ax,
                linewidths=0.3, annot=True, fmt=".2f", annot_kws={"size":6},
                cbar_kws={"label":"Lift (>1 = co-occur more than expected)"})
    ax.set_title(f"Cancer Co-occurrence Lift Matrix (top {n_sites} sites)")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "lift_heatmap.png", dpi=150)
    plt.close(fig)


def sex_stratified(matrix, meta, rules, label):
    """Run association rules separately for male vs female."""
    merged = matrix.merge(meta[["pid","sex"]], left_index=True, right_on="pid")
    results = {}
    for sex in ["M","F"]:
        sub = merged[merged["sex"]==sex].drop(columns="pid").set_index(
            merged[merged["sex"]==sex]["pid"])
        sub = sub.drop(columns="sex")
        r = manual_pair_counts(sub, min_support=0.003)
        r = r[r["lift"]>1.5].sort_values("lift",ascending=False)
        results[sex] = r
        print(f"  {sex}: {len(r)} high-lift pairs (lift>1.5)")
    # Top 10 unique to male vs female
    male_top  = set(zip(results["M"].head(20)["antecedent"],results["M"].head(20)["consequent"]))
    female_top = set(zip(results["F"].head(20)["antecedent"],results["F"].head(20)["consequent"]))
    male_unique   = male_top   - female_top
    female_unique = female_top - male_top
    print(f"\n  Male-specific top pairs:   {male_unique}")
    print(f"  Female-specific top pairs: {female_unique}")
    results["M"].to_csv(OUT/"association_rules_male.csv",   index=False, encoding="utf-8-sig")
    results["F"].to_csv(OUT/"association_rules_female.csv", index=False, encoding="utf-8-sig")
    return results


if __name__ == "__main__":
    print("Loading matrix...")
    matrix, sites, meta, label = load()
    print(f"  Matrix: {matrix.shape}")

    # Work on multi-primary patients for richer signal
    mp_mat = matrix[matrix.sum(axis=1) >= 2]
    print(f"  Multi-primary patients: {len(mp_mat):,}")

    print("\nComputing pairwise association rules (all patients)...")
    rules_all = manual_pair_counts(matrix, min_support=0.001)
    rules_all["antecedent_label"] = rules_all["antecedent"].map(label)
    rules_all["consequent_label"] = rules_all["consequent"].map(label)
    rules_sig = rules_all[rules_all["lift"] > 1.5].sort_values("lift", ascending=False)
    print(f"  Total pairs: {len(rules_all):,}  |  High-lift (>1.5): {len(rules_sig):,}")
    rules_all.to_csv(OUT/"association_rules_all.csv",  index=False, encoding="utf-8-sig")
    rules_sig.to_csv(OUT/"association_rules_sig.csv",  index=False, encoding="utf-8-sig")

    print("\nTop 20 cancer co-occurrence pairs by lift:")
    for _, r in rules_sig.head(20).iterrows():
        print(f"  {label.get(r['antecedent'],r['antecedent']):<22} ↔  "
              f"{label.get(r['consequent'],r['consequent']):<22}  "
              f"n={r['n_co']:>4}  lift={r['lift']:.2f}  OR={r['odds_ratio']:.2f}")

    print("\nGenerating figures...")
    fig_top_associations(rules_sig, label, "top_associations_lift.png", top_n=25)
    fig_network_heatmap(rules_all, label, n_sites=22)

    print("\nSex-stratified analysis...")
    sex_rules = sex_stratified(matrix, meta, rules_sig, label)

    # Age-stratified: early-onset (age<50) vs late-onset (age≥60)
    merged = matrix.merge(meta[["pid","age_first"]], left_index=True, right_on="pid")
    for grp, cond in [("early_onset","age_first<50"), ("late_onset","age_first>=60")]:
        sub = merged.query(cond).drop(columns=["pid","age_first"])
        r = manual_pair_counts(sub, min_support=0.002)
        r_sig = r[r["lift"]>1.5].sort_values("lift",ascending=False)
        r_sig.to_csv(OUT/f"association_rules_{grp}.csv", index=False, encoding="utf-8-sig")
        print(f"  {grp}: {len(r_sig)} high-lift pairs")

    print(f"\nDone. Results in {OUT}")
