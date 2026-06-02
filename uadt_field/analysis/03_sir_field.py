"""
UADT Field Cancerization — Script 03: Standardized Incidence Ratios

Internal-reference SIR for within-field second primaries (10×10 = 90 ordered pairs).
Adapted from coexist_cancers/analysis/07_sir_trajectories.py.

Outputs:
  results/03_sir/sir_field.csv
  results/03_sir/fig3a_sir_forest.png
  results/03_sir/fig3b_sir_heatmap.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2, poisson
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
WASHOUT_MO  = 2

assert len(FIELD_SITES) == 10

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
OUT  = BASE / "results/03_sir"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)


def age_band(a):
    """Copy verbatim from 07_sir_trajectories.py."""
    if pd.isna(a): return "unk"
    if a < 40: return "<40"
    if a < 50: return "40-49"
    if a < 60: return "50-59"
    if a < 70: return "60-69"
    if a < 80: return "70-79"
    return "80+"


def compute_sir(first):
    """
    Adapted from coexist 07_sir_trajectories.py compute_sir().
    Restricted: both index_site and target_site must be in FIELD_SITES.
    Female strata: if any cell has <5 events, collapse to <60 / ≥60.
    """
    first = first[first["site"].isin(FIELD_SITES)].copy()

    # index cancer = earliest field-site primary per patient
    idx = first.loc[first.groupby("pid")["dx"].idxmin()].copy()
    idx["band"]  = idx["age"].apply(age_band)
    idx["start"] = idx["dx"] + pd.to_timedelta(int(WASHOUT_MO*30.44), "D")
    idx["py"]    = (idx["end_fu"] - idx["start"]).dt.days / 365.25
    idx = idx[idx["py"] > 0]
    idx = idx.rename(columns={"site":"index_site"}).set_index("pid")

    # later field primaries (after washout)
    fp = first.merge(idx[["index_site","start","band","sex","py","end_fu"]],
                     left_on="pid", right_index=True, suffixes=("","_idx"))
    later = fp[(fp["dx"] > fp["start"]) &
               (fp["site"] != fp["index_site"]) &
               (fp["site"].isin(FIELD_SITES))].copy()

    strata = ["sex","band"]
    py_str = idx.groupby(strata)["py"].sum()
    total_py = idx["py"].sum()

    # check female sparsity — collapse if needed
    female_counts = later[later["sex"]=="F"].groupby(["site"]).size()
    collapse_female = female_counts.max() < 5 if len(female_counts) else True
    if collapse_female:
        def age_band_2(a):
            if pd.isna(a): return "unk"
            return "<60" if a < 60 else "60+"
        idx["band"]   = idx["age"].apply(age_band_2)
        later["band"] = later["age"].apply(age_band_2)
        py_str = idx.groupby(strata)["py"].sum()

    bg = (later.groupby(strata+["site"]).size()
               .rename("events").reset_index())
    bg = bg.merge(py_str.rename("py").reset_index(), on=strata, how="left")
    bg["rate"] = bg["events"] / bg["py"]
    rate_lookup = {(r.sex, r.band, r.site): r.rate for r in bg.itertuples()}

    idx_py_by_str = idx.groupby(["index_site","sex","band"])["py"].sum().reset_index()
    obs = later.groupby(["index_site","site"]).size().rename("O").reset_index()

    rows = []
    for isite in FIELD_SITES:
        sub_idx = idx[idx["index_site"]==isite]
        if len(sub_idx) == 0: continue
        sub_py = idx_py_by_str[idx_py_by_str["index_site"]==isite]
        n_idx  = len(sub_idx)
        for tsite in FIELD_SITES:
            if tsite == isite: continue
            E = sum(r.py * rate_lookup.get((r.sex, r.band, tsite), 0.0)
                    for r in sub_py.itertuples())
            o_row = obs[(obs["index_site"]==isite) & (obs["site"]==tsite)]
            O = int(o_row["O"].iloc[0]) if len(o_row) else 0
            if O < MIN_OBS or E <= 0: continue
            sir = O / E
            lo  = 0.5*chi2.ppf(0.025, 2*O)/E
            hi  = 0.5*chi2.ppf(0.975, 2*(O+1))/E
            p   = 2*min(poisson.cdf(O, E), 1-poisson.cdf(O-1, E))
            p   = min(p, 1.0)
            rows.append({
                "index": isite, "index_label": FIELD_LABELS[isite], "n_index": n_idx,
                "target": tsite, "target_label": FIELD_LABELS[tsite],
                "O": O, "E": round(E,2), "SIR": round(sir,2),
                "CI_low": round(lo,2), "CI_high": round(hi,2), "p": p
            })

    res = pd.DataFrame(rows)
    if len(res):
        res["FDR"] = multipletests(res["p"], method="fdr_bh")[1]
        res = res.sort_values("SIR", ascending=False).reset_index(drop=True)
    return res, total_py


def fig_sir_forest(sir, top=20):
    """Adapted from 07_sir_trajectories.py fig_sir_forest()."""
    s = sir[sir["FDR"] < 0.05].nlargest(top, "SIR").copy()
    if len(s) == 0: s = sir.nlargest(top, "SIR").copy()
    s["pair"] = s["index_label"].str[:13] + " → " + s["target_label"].str[:13]
    s = s.sort_values("SIR")
    y = range(len(s))
    fig, ax = plt.subplots(figsize=(9, len(s)*0.40+1.5))
    ax.hlines(list(y), s["CI_low"], s["CI_high"], color="#2980b9", lw=1.8, zorder=2)
    ax.scatter(s["SIR"], list(y), color="#c0392b", s=34, zorder=3,
               edgecolors="white", lw=0.5)
    ax.axvline(1, color="#333", ls="--", lw=0.9)
    ax.set_yticks(list(y)); ax.set_yticklabels(s["pair"], fontsize=8)
    ax.set_xscale("log")
    ax.set(title="UADT Within-Field SIR (index → later primary, FDR<0.05)",
           xlabel="Standardized Incidence Ratio (95% CI, log scale)")
    ax.spines[["top","right"]].set_visible(False)
    for i, (_, r) in enumerate(s.iterrows()):
        ax.text(ax.get_xlim()[1]*1.02, i,
                f"O={r.O}, SIR={r.SIR:.1f}", va="center", fontsize=6.5, color="#555")
    fig.tight_layout()
    fig.savefig(OUT / "fig3a_sir_forest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_sir_heatmap(sir):
    labels = [FIELD_LABELS[s] for s in FIELD_SITES]
    M = pd.DataFrame(np.nan, index=labels, columns=labels)
    for _, r in sir.iterrows():
        if r["FDR"] < 0.05 and r["O"] >= MIN_OBS:
            M.loc[FIELD_LABELS[r["index"]], FIELD_LABELS[r["target"]]] = r["SIR"]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(M, annot=True, fmt=".1f", cmap="Reds", linewidths=0.5, ax=ax,
                cbar_kws={"label": "SIR", "shrink": 0.8},
                annot_kws={"size": 7}, mask=M.isna())
    ax.set_title("Within-Field SIR Heatmap\n(rows=index, cols=target; grey=non-significant)",
                 fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xlabel("Target (2nd primary)"); ax.set_ylabel("Index (1st primary)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3b_sir_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=== UADT SIR Analysis ===")
    first = pd.read_csv(BASE / "data/field_patients.csv", parse_dates=["dx","end_fu"])

    sir, total_py = compute_sir(first)
    print(f"Total person-years at risk: {total_py:,.0f}")
    print(f"SIR pairs (O≥{MIN_OBS}): {len(sir)}  |  FDR<0.05: {(sir['FDR']<0.05).sum()}")

    # SECONDARY HYPOTHESIS CHECK
    for idx_s, tgt_s in [("C13","C15"), ("C15","C13"), ("C12","C15"), ("C15","C12")]:
        row = sir[(sir["index"]==idx_s) & (sir["target"]==tgt_s)]
        if len(row):
            r = row.iloc[0]
            print(f"  SIR {FIELD_LABELS[idx_s]}→{FIELD_LABELS[tgt_s]}: "
                  f"O={r.O}, SIR={r.SIR:.1f} ({r.CI_low:.1f}–{r.CI_high:.1f}), FDR={r.FDR:.4f}")
        else:
            print(f"  SIR {FIELD_LABELS[idx_s]}→{FIELD_LABELS[tgt_s]}: n<{MIN_OBS} or no events")

    print("\nTop 10 SIR pairs:")
    print(sir[["index_label","target_label","O","E","SIR","CI_low","CI_high","FDR"]]
          .head(10).to_string(index=False))

    sir.to_csv(OUT / "sir_field.csv", index=False, encoding="utf-8-sig")
    fig_sir_forest(sir)
    fig_sir_heatmap(sir)
    print(f"\n✓ Outputs written to {OUT}")


if __name__ == "__main__":
    main()
