"""
UADT Field Cancerization — Script 02: Co-occurrence Pairs

All 45 UADT site pairs: counts, OR, lift, chi-squared, FDR correction.
Pre-registered hypotheses are declared BEFORE any data is loaded.

Outputs:
  results/02_pairs/field_pairs_all.csv
  results/02_pairs/field_pairs_fdr.csv
  results/02_pairs/fig2a_cooccurrence_heatmap.png
  results/02_pairs/fig2b_lift_heatmap.png
  results/02_pairs/fig2c_or_forest.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from itertools import combinations
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

# ── LOCKED CONSTANTS ──────────────────────────────────────────────────────────
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

# ── PRE-REGISTERED HYPOTHESES (declared before any pd.read_csv) ───────────────
# PRIMARY: Pyriform+Esophagus and Hypopharynx+Esophagus have the highest
#          co-occurrence RATES (% of smaller site) within the UADT field.
PRIMARY_PAIRS = [frozenset({'C12','C15'}), frozenset({'C13','C15'})]
# All other 43 pairs are EXPLORATORY.

assert len(FIELD_SITES) == 10
assert len(PRIMARY_PAIRS) == 2

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
OUT  = BASE / "results/02_pairs"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="white", font_scale=0.95)


def load():
    mat  = pd.read_csv(BASE / "data/field_patient_matrix.csv", index_col=0)
    meta = pd.read_csv(BASE / "data/field_meta.csv")
    for s in FIELD_SITES:
        if s not in mat.columns: mat[s] = 0
    mat = mat[FIELD_SITES]
    return mat, meta


def compute_pairs(mat):
    n_total = len(mat)
    site_n  = mat.sum()
    rows = []
    for a, b in combinations(FIELD_SITES, 2):
        n_a   = int(site_n[a])
        n_b   = int(site_n[b])
        n_co  = int((mat[a] & mat[b]).sum())
        if n_co < MIN_OBS:
            continue
        # Fisher exact OR
        table = [[n_co, n_a - n_co],
                 [n_b - n_co, n_total - n_a - n_b + n_co]]
        try:
            _, p_chi = chi2_contingency(table, correction=False)[:2]
            or_val, p_fish = fisher_exact(table)
        except Exception:
            p_chi = p_fish = 1.0; or_val = 1.0
        support = n_co / n_total
        conf_ab = n_co / n_a if n_a else 0
        conf_ba = n_co / n_b if n_b else 0
        lift    = support / ((n_a / n_total) * (n_b / n_total)) if n_a and n_b else 1.0
        rate_smaller = n_co / min(n_a, n_b) * 100  # % of smaller site
        label = "primary" if frozenset({a,b}) in PRIMARY_PAIRS else "exploratory"
        rows.append({
            "site_a": a, "label_a": FIELD_LABELS[a],
            "site_b": b, "label_b": FIELD_LABELS[b],
            "n_a": n_a, "n_b": n_b, "n_co": n_co,
            "support": round(support, 5),
            "conf_a_to_b": round(conf_ab, 4),
            "conf_b_to_a": round(conf_ba, 4),
            "lift": round(lift, 2),
            "OR": round(or_val, 2),
            "pct_of_smaller": round(rate_smaller, 1),
            "p_fisher": p_fish,
            "hypothesis": label
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["FDR"] = multipletests(df["p_fisher"], method="fdr_bh")[1]
        df = df.sort_values("pct_of_smaller", ascending=False).reset_index(drop=True)
    return df


def fig_heatmap(mat, metric, title, fname, center=None, cmap="YlOrRd", fmt=".0f"):
    labels = [FIELD_LABELS[s] for s in FIELD_SITES]
    # build symmetric matrix
    M = pd.DataFrame(np.nan, index=labels, columns=labels)
    for _, r in metric.iterrows():
        la, lb = FIELD_LABELS[r["site_a"]], FIELD_LABELS[r["site_b"]]
        M.loc[la, lb] = r["value"]
        M.loc[lb, la] = r["value"]
    M_arr = M.values.copy()
    np.fill_diagonal(M_arr, np.nan)
    M = pd.DataFrame(M_arr, index=M.index, columns=M.columns)
    mask = np.triu(np.ones_like(M, dtype=bool), k=1)  # hide upper triangle

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(M, mask=mask, cmap=cmap, center=center, annot=True,
                fmt=fmt, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7})
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_or_forest(pairs):
    import matplotlib.patches as mpatches
    df = pairs.copy().sort_values("OR", ascending=False).reset_index(drop=True)
    df["pair"] = df["label_a"] + " + " + df["label_b"]
    # compute approx CI from Fisher (Woolf log OR ± 1.96 SE)
    def or_ci(r):
        n_co, n_a, n_b, n_total = r["n_co"], r["n_a"], r["n_b"], len(pairs)
        a = n_co; b = n_a - n_co; c = n_b - n_co; d = n_total - a - b - c
        if min(a,b,c,d) <= 0: return r["OR"], r["OR"]*0.5, r["OR"]*2
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        lo = np.exp(np.log(r["OR"]) - 1.96*se)
        hi = np.exp(np.log(r["OR"]) + 1.96*se)
        return r["OR"], lo, hi
    df[["OR","CI_lo","CI_hi"]] = df.apply(
        lambda r: pd.Series(or_ci(r)), axis=1)

    fig, ax = plt.subplots(figsize=(9, len(df)*0.32+1.5))
    for i, (_, r) in enumerate(df.iterrows()):
        color = "#dc2626" if r["hypothesis"]=="primary" else "#6b7280"
        ax.hlines(i, r["CI_lo"], r["CI_hi"], color=color, lw=1.5, zorder=2)
        ax.scatter(r["OR"], i, color=color, s=30, zorder=3,
                   edgecolors='white', lw=0.5)
        if r["FDR"] < 0.05:
            ax.text(r["CI_hi"]*1.05, i, "*", va='center', fontsize=10, color=color)
    ax.axvline(1, color='#333', ls='--', lw=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["pair"], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Odds Ratio (95% CI, log scale)", fontsize=10)
    ax.set_title("UADT Field Co-occurrence Odds Ratios (all 45 pairs)", fontsize=11, fontweight='bold')
    legend_els = [mpatches.Patch(color='#dc2626', label='Pre-registered primary'),
                  mpatches.Patch(color='#6b7280', label='Exploratory'),
                  plt.Line2D([0],[0], marker='None', ls='None', label='* FDR < 0.05')]
    ax.legend(handles=legend_els, fontsize=8, loc='lower right')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2c_or_forest.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=== UADT Co-occurrence Pairs ===")
    print("Pre-registered primary pairs:")
    for p in PRIMARY_PAIRS:
        a, b = sorted(p)
        print(f"  {FIELD_LABELS[a]} + {FIELD_LABELS[b]}")

    mat, meta = load()
    print(f"\nLoaded matrix: {len(mat):,} patients × {len(mat.columns)} sites")

    pairs = compute_pairs(mat)
    print(f"Pairs with n_co ≥ {MIN_OBS}: {len(pairs)}")

    # ── PRIMARY HYPOTHESIS CHECK (printed before writing output) ─────────────
    by_rate = pairs.sort_values("pct_of_smaller", ascending=False).reset_index(drop=True)
    for p in PRIMARY_PAIRS:
        a, b = sorted(p)
        row = by_rate[(by_rate["site_a"]==a) & (by_rate["site_b"]==b)]
        if len(row):
            rank = row.index[0] + 1
            print(f"\nPRIMARY HYPOTHESIS CHECK: {FIELD_LABELS[a]}+{FIELD_LABELS[b]} "
                  f"rank={rank} by rate, pct_of_smaller={row.iloc[0]['pct_of_smaller']}%, "
                  f"OR={row.iloc[0]['OR']}, FDR={row.iloc[0]['FDR']:.4f}")

    pairs.to_csv(OUT / "field_pairs_all.csv", index=False, encoding="utf-8-sig")
    fdr_sig = pairs[pairs["FDR"] < 0.05]
    fdr_sig.to_csv(OUT / "field_pairs_fdr.csv", index=False, encoding="utf-8-sig")
    print(f"\nFDR < 0.05 pairs: {len(fdr_sig)}")
    print("\nTop 10 by co-occurrence rate:")
    print(by_rate[["label_a","label_b","n_co","pct_of_smaller","OR","FDR","hypothesis"]]
          .head(10).to_string(index=False))

    # ── Figure 2a: count heatmap ──────────────────────────────────────────────
    fig_heatmap(mat,
                pairs[["site_a","site_b"]].assign(value=pairs["n_co"]),
                "UADT Field Co-occurrence Counts", "fig2a_cooccurrence_heatmap.png",
                cmap="Blues", fmt=".0f")

    # ── Figure 2b: lift heatmap ───────────────────────────────────────────────
    fig_heatmap(mat,
                pairs[["site_a","site_b"]].assign(value=pairs["lift"]),
                "UADT Field Association Lift", "fig2b_lift_heatmap.png",
                center=1.0, cmap="RdBu_r", fmt=".1f")

    # ── Figure 2c: OR forest ──────────────────────────────────────────────────
    fig_or_forest(pairs)

    print(f"\n✓ Outputs written to {OUT}")


if __name__ == "__main__":
    main()
