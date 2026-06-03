"""
Registry DL — Script 21: Hereditary Syndrome Candidate Validation

Starting from the 128 candidates flagged by the autoencoder screen (Script 06),
this script applies phenotypic refinement criteria derived from published clinical
guidelines and assesses the clinical plausibility of each syndrome classification.

Key structural limitations discovered:
  LFS  — Effective registry sites {C50, C71} (C74 adrenal, C49 soft tissue absent).
          All 26 LFS candidates have C50 only; none have C71 brain tumour.
          match_score=0.5 therefore reflects breast cancer + unrelated cancer only.
  Cowden — Effective registry sites {C50, C54} (C73 thyroid absent).
            3 patients have match_score=1.0 (C50+C54) — the only high-confidence group.

Refinement criteria applied:
  LFS refined    : sites include C71 AND age_first < 50 (Chompret-like)
  Cowden refined : match_score == 1.0 (both C50 + C54 present)
  Other          : as reported by Script 06

Outputs:
  results/21_syndrome/refined_candidates.csv
  results/21_syndrome/timing_table.csv
  results/21_syndrome/fig_match_score_dist.png
  results/21_syndrome/fig_age_first_dist.png
  results/21_syndrome/fig_inter_cancer_gap.png
  results/21_syndrome/fig_survival_curves.png
  results/21_syndrome/fig_site_co_occurrence.png
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE  = Path(__file__).parent.parent
RAW   = BASE.parent / "data/processed/all_cancers.csv"
R06   = BASE / "results/06_syndrome"
OUT   = BASE / "results/21_syndrome"
OUT.mkdir(parents=True, exist_ok=True)

COWDEN_PUBLISHED_AGE_BREAST = (38, 46)   # published median range (Pilarski 2009)
LFS_PUBLISHED_AGE_FIRST     = 30          # median first cancer age in LFS families


def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    y  = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00","99") else "01"
    dd = s[5:7] if s[5:7] not in ("00","99") else "01"
    try: return pd.Timestamp(f"{y}-{mm}-{dd}")
    except: return pd.NaT


def main():
    print("=== Registry DL — 21: Hereditary Syndrome Candidate Validation ===")

    # ── Load candidates + registry ────────────────────────────────────────────
    cand = pd.read_csv(R06 / "syndrome_candidates.csv")
    cand["pid"] = cand["pid"].astype(str).str.split(".").str[0]
    actionable  = cand[cand["best_syndrome"].isin(["Li-Fraumeni","Cowden"])].copy()
    print(f"  Raw candidates: {len(cand)} total, {len(actionable)} actionable "
          f"(LFS={len(cand[cand['best_syndrome']=='Li-Fraumeni'])}, "
          f"Cowden={len(cand[cand['best_syndrome']=='Cowden'])})")

    # ── Phenotypic refinement ─────────────────────────────────────────────────
    def has_c71(sites_str):
        return "C71" in str(sites_str).split("+")

    # LFS: must have C71 (brain/CNS) AND age_first < 50
    lfs_mask = ((actionable["best_syndrome"] == "Li-Fraumeni") &
                actionable["sites"].apply(has_c71) &
                (actionable["age_first"] < 50))
    # Cowden: match_score == 1.0 (C50 + C54)
    cow_mask = ((actionable["best_syndrome"] == "Cowden") &
                (actionable["match_score"] == 1.0))

    actionable["refined"] = lfs_mask | cow_mask

    n_lfs_refined = lfs_mask.sum()
    n_cow_refined = cow_mask.sum()
    print(f"\n  After phenotypic refinement:")
    print(f"    LFS (C71 present + age<50): {n_lfs_refined}")
    print(f"    Cowden (match_score=1.0):   {n_cow_refined}")
    print(f"    Total high-confidence:       {n_lfs_refined + n_cow_refined}")

    # ── Load registry for timing data ─────────────────────────────────────────
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip().str.lstrip("﻿")
    df["pid"]    = pd.to_numeric(df["病歷號(2)"], errors="coerce").astype("Int64").astype(str)
    df["site"]   = df["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df["dx_ts"]  = df["最初診斷日(45)"].apply(roc_to_ts)
    df["fu_ts"]  = df["最後聯絡日(30)"].apply(roc_to_ts)
    df["dead"]   = pd.to_numeric(df["生存狀態(27)"], errors="coerce").fillna(0).astype(int)
    df           = df.dropna(subset=["dx_ts"])

    # Inter-cancer gap for multi-cancer candidates
    pids_of_interest = set(actionable["pid"].astype(str))
    reg_sub = df[df["pid"].isin(pids_of_interest)].copy()
    reg_sub = reg_sub.sort_values(["pid","dx_ts"])

    timing_rows = []
    for pid, grp in reg_sub.groupby("pid"):
        grp = grp.sort_values("dx_ts")
        sites_list = grp["site"].tolist()
        dates_list = grp["dx_ts"].tolist()
        for i in range(len(sites_list)-1):
            gap_days = (dates_list[i+1] - dates_list[i]).days
            timing_rows.append({
                "pid":       pid,
                "site_from": sites_list[i],
                "site_to":   sites_list[i+1],
                "gap_days":  gap_days,
                "gap_months": round(gap_days / 30.44, 1),
            })

    timing = pd.DataFrame(timing_rows)
    timing_with_meta = timing.merge(
        actionable[["pid","best_syndrome","match_score","refined","age_first","sex"]],
        on="pid", how="left"
    )
    timing_with_meta.to_csv(OUT / "timing_table.csv", index=False)
    print(f"\n  Timing table: {len(timing_with_meta)} inter-cancer transitions")
    if len(timing_with_meta):
        print(f"  Median inter-cancer gap: {timing_with_meta['gap_days'].median():.0f} days "
              f"({timing_with_meta['gap_months'].median():.1f} months)")

    # ── Survival table for candidates ─────────────────────────────────────────
    # Last event per patient
    last = (reg_sub.sort_values("dx_ts").groupby("pid")
            .agg(last_fu=("fu_ts","last"), dead=("dead","max"),
                 first_dx=("dx_ts","first")).reset_index())
    last["surv_days"] = (last["last_fu"] - last["first_dx"]).dt.days.clip(lower=1)
    last = last.merge(actionable[["pid","best_syndrome","refined","age_first"]], on="pid", how="left")

    # ── Fig A: match score distribution by syndrome ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, syndrome, color in [
        (axes[0], "Li-Fraumeni", "#d62728"),
        (axes[1], "Cowden",      "#9467bd"),
    ]:
        sub = actionable[actionable["best_syndrome"] == syndrome]["match_score"]
        ax.hist(sub, bins=np.arange(0.3, 1.15, 0.1), color=color, alpha=0.7, edgecolor="white")
        ax.axvline(sub.mean(), color="black", lw=1.5, ls="--",
                   label=f"mean={sub.mean():.2f}")
        ax.set_xlabel("Match score")
        ax.set_ylabel("N candidates")
        ax.set_title(f"{syndrome} (n={len(sub)})\n"
                     f"Effective sites: {'C50+C71' if syndrome=='Li-Fraumeni' else 'C50+C54'}")
        ax.legend(fontsize=9)
    fig.suptitle("Syndrome candidate match score distributions\n"
                 "(match_score = n_syndrome_sites_present / n_effective_sites_in_registry)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_match_score_dist.png", dpi=150)
    plt.close()

    # ── Fig B: age at first diagnosis ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for syndrome, color, published_age in [
        ("Li-Fraumeni", "#d62728", LFS_PUBLISHED_AGE_FIRST),
        ("Cowden",      "#9467bd", None),
    ]:
        sub = actionable[actionable["best_syndrome"]==syndrome]["age_first"].dropna()
        ax.hist(sub.values, bins=range(30, 90, 5), alpha=0.55, color=color,
                label=f"{syndrome} (n={len(sub)}, median={sub.median():.0f}yr)",
                edgecolor="white")
        if published_age:
            ax.axvline(published_age, color=color, lw=2, ls="--",
                       label=f"{syndrome} published median ~{published_age}yr")
    # Cowden published range
    ax.axvspan(*COWDEN_PUBLISHED_AGE_BREAST, alpha=0.08, color="#9467bd",
               label=f"Cowden published breast onset {COWDEN_PUBLISHED_AGE_BREAST[0]}–{COWDEN_PUBLISHED_AGE_BREAST[1]}yr")
    ax.axvline(50, color="gray", lw=1, ls=":", label="age 50 cutoff (LFS Chompret)")
    ax.set_xlabel("Age at first cancer diagnosis (years)")
    ax.set_ylabel("N candidates")
    ax.set_title("Age at first diagnosis — flagged syndrome candidates\nvs published syndrome age distributions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_age_first_dist.png", dpi=150)
    plt.close()

    # ── Fig C: inter-cancer gap for Cowden match_score=1.0 ────────────────────
    cow_hi = timing_with_meta[
        (timing_with_meta["best_syndrome"]=="Cowden") &
        (timing_with_meta["match_score"]==1.0)
    ]
    lfs_all = timing_with_meta[timing_with_meta["best_syndrome"]=="Li-Fraumeni"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, subset, label, color in [
        (axes[0], cow_hi,  "Cowden match=1.0 (C50+C54)\nInter-cancer gap", "#9467bd"),
        (axes[1], lfs_all, "Li-Fraumeni all candidates\nInter-cancer gap",   "#d62728"),
    ]:
        if len(subset) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_visible(True)
            continue
        gaps = subset["gap_months"].dropna()
        ax.hist(gaps.values, bins=20, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(gaps.median(), color="black", lw=1.5, ls="--",
                   label=f"median={gaps.median():.0f} months")
        ax.axvline(6, color="orange", lw=1, ls=":", label="6-month synchronous threshold")
        ax.set_xlabel("Inter-cancer gap (months)")
        ax.set_ylabel("N transitions")
        ax.set_title(label)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_inter_cancer_gap.png", dpi=150)
    plt.close()

    # ── Fig D: Survival curves (KM manual) for Cowden high-conf vs LFS vs registry ──
    # Load registry-wide comparison from patient_meta
    meta = pd.read_csv(BASE / "data/patient_meta.csv")
    meta["end_fu_ts"] = pd.to_datetime(meta["end_fu"], errors="coerce")
    meta["fu_yrs"] = (meta["end_fu_ts"] -
                      pd.to_datetime(meta["diag_yr"].astype(str)+"-01-01",
                                     errors="coerce")).dt.days / 365.25
    meta["surv_yrs"] = meta["fu_yrs"].clip(lower=0.01)

    def km_curve(times, events, label, ax, color):
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(times, events, label=label)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=(len(times)>30))

    try:
        from lifelines import KaplanMeierFitter
        fig, ax = plt.subplots(figsize=(9, 5))

        # Registry multi-cancer patients (comparator)
        multi_meta = meta[meta["multi_cancer"]==1]
        km_curve(multi_meta["surv_yrs"], multi_meta["dead"],
                 f"Registry multi-cancer (n={len(multi_meta):,})", ax, "#aaaaaa")

        # Cowden match=1.0
        if len(last[last["best_syndrome"]=="Cowden"]) > 0:
            cow_last = last[last["best_syndrome"]=="Cowden"]
            cow_hi_last = cow_last[cow_last["refined"]==True]
            if len(cow_hi_last) > 0:
                km_curve(cow_hi_last["surv_days"]/365.25, cow_hi_last["dead"],
                         f"Cowden match=1.0 (n={len(cow_hi_last)})", ax, "#9467bd")

        # LFS all
        lfs_last = last[last["best_syndrome"]=="Li-Fraumeni"]
        if len(lfs_last) > 0:
            km_curve(lfs_last["surv_days"]/365.25, lfs_last["dead"],
                     f"LFS candidates (n={len(lfs_last)})", ax, "#d62728")

        ax.set_xlabel("Years from first diagnosis")
        ax.set_ylabel("Survival probability")
        ax.set_title("Kaplan-Meier survival — syndrome candidates vs registry multi-cancer")
        ax.set_xlim(0, 18)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "fig_survival_curves.png", dpi=150)
        plt.close()
    except ImportError:
        print("  lifelines not available — skipping KM plot")

    # ── Fig E: site co-occurrence heatmap for Cowden candidates ──────────────
    cow_all = actionable[actionable["best_syndrome"]=="Cowden"].copy()
    all_sites_in_cow = set()
    for s in cow_all["sites"]:
        all_sites_in_cow.update(s.split("+"))
    site_list = sorted(all_sites_in_cow)

    co_mat = np.zeros((len(site_list), len(site_list)))
    for _, row in cow_all.iterrows():
        pts = row["sites"].split("+")
        for i, s1 in enumerate(site_list):
            for j, s2 in enumerate(site_list):
                if s1 in pts and s2 in pts:
                    co_mat[i,j] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(co_mat, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(site_list))); ax.set_xticklabels(site_list, rotation=90, fontsize=9)
    ax.set_yticks(range(len(site_list))); ax.set_yticklabels(site_list, fontsize=9)
    for i in range(len(site_list)):
        for j in range(len(site_list)):
            if co_mat[i,j] > 0:
                ax.text(j, i, int(co_mat[i,j]), ha="center", va="center", fontsize=8,
                        color="white" if co_mat[i,j] > co_mat.max()*0.6 else "black")
    plt.colorbar(im, ax=ax, label="N patients with both sites")
    ax.set_title(f"Site co-occurrence — Cowden candidates (n={len(cow_all)})\n"
                 "C50 breast + C54 endometrial = defining pair")
    fig.tight_layout()
    fig.savefig(OUT / "fig_site_co_occurrence.png", dpi=150)
    plt.close()

    # ── Save refined candidates ───────────────────────────────────────────────
    # Referral priority score
    def priority(row):
        score = row["match_score"] * 3
        score += 1.0 if row["age_first"] < 45 else (0.5 if row["age_first"] < 55 else 0)
        score += 0.5 if row["n_sites"] >= 3 else 0
        return round(score, 2)

    actionable["priority_score"] = actionable.apply(priority, axis=1)
    refined = actionable[actionable["refined"]].sort_values("priority_score", ascending=False)
    all_act = actionable.sort_values(["best_syndrome","priority_score"], ascending=[True,False])
    all_act.to_csv(OUT / "refined_candidates.csv", index=False)

    print(f"\n  High-confidence candidates (n={len(refined)}):")
    print(refined[["pid","best_syndrome","sites","match_score","age_first",
                   "sex","dead","priority_score"]].to_string(index=False))

    print(f"\n  LFS: {len(actionable[actionable['best_syndrome']=='Li-Fraumeni'])} raw → "
          f"{lfs_mask.sum()} refined (requires C71 + age<50)")
    print(f"  Cowden: {len(actionable[actionable['best_syndrome']=='Cowden'])} raw → "
          f"{cow_mask.sum()} refined (match_score=1.0, C50+C54)")
    print(f"\n  Saved → {OUT}/")


if __name__ == "__main__":
    main()
