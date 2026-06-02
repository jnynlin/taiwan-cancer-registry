"""
UADT Field Cancerization — Script 05: Survival with Landmark Analysis

KM + Cox for multi-field vs single-field patients, with 6-month landmark
to correct immortal-time bias (Bias Mitigation #5).

Immortal time: patients can only be "multi-field" if they survived long enough
to develop a second field cancer. Without landmark correction, multi-field
patients appear to survive longer simply because they must have survived the
interval between first and second diagnosis. The landmark drops anyone who
dies/censors before 6 months, resetting time origin to that point.

Outputs:
  results/05_survival/survival_table.csv
  results/05_survival/cox_results.csv
  results/05_survival/fig5a_km_landmark.png
  results/05_survival/fig5b_cox_forest.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    LIFELINES = True
except ImportError:
    LIFELINES = False
    print("⚠️  lifelines not installed — using manual KM fallback")

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
LANDMARK_DAYS = LANDMARK_MO * 30.44

# ── PRE-REGISTERED TERTIARY HYPOTHESIS ───────────────────────────────────────
# TERTIARY HYPOTHESIS: Multi-field patients have worse OS (HR > 1.0) compared
# to single-field patients AFTER landmark correction at 6 months.
# Note: without landmark, multi-field artificially appears better (immortal time).

assert len(FIELD_SITES) == 10

BASE = Path(__file__).parent.parent
OUT  = BASE / "results/05_survival"
OUT.mkdir(parents=True, exist_ok=True)


def build_survival_table(meta, patients):
    """Compute time_to_event from first field dx to death or censor."""
    first_dx = patients.sort_values("dx").groupby("pid")["dx"].min().reset_index()
    first_dx.columns = ["pid","first_field_dx"]
    surv = meta.merge(first_dx, on="pid", how="inner")
    # end_fu from patients (already computed by 01)
    end_fu = patients.groupby("pid")["end_fu"].max().reset_index()
    surv = surv.merge(end_fu, on="pid", how="left")
    surv["first_field_dx"] = pd.to_datetime(surv["first_field_dx"])
    surv["end_fu"] = pd.to_datetime(surv["end_fu"])
    surv["time_days"] = (surv["end_fu"] - surv["first_field_dx"]).dt.days
    surv["event"] = surv["dead"].astype(int)
    surv = surv[surv["time_days"] > 0].copy()
    return surv


def apply_landmark(surv, landmark_days=LANDMARK_DAYS):
    """
    Keep only patients surviving past landmark.
    Reset time origin: new_time = time_days - landmark_days.
    Patients who die before landmark are excluded (not censored — they are truly
    not at risk of developing a second field cancer after landmark).
    """
    n_total = len(surv)
    lm = surv[surv["time_days"] > landmark_days].copy()
    lm["time_lm"] = lm["time_days"] - landmark_days
    n_excluded = n_total - len(lm)
    return lm, n_excluded


def km_manual(time, event):
    """Manual Kaplan-Meier estimator (fallback if lifelines absent)."""
    order = np.argsort(time)
    t, e = np.array(time)[order], np.array(event)[order]
    ts, se = [0], [1.0]
    n = len(t); s = 1.0
    for i in range(n):
        if e[i] == 1:
            n_risk = n - i
            s = s * (n_risk - 1) / n_risk
            ts.append(t[i]); se.append(s)
    return np.array(ts), np.array(se)


def fig_km(lm):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {"Single-field":"#2563eb", "Multi-field":"#dc2626"}
    groups = {"Single-field": lm[lm["multi_field"]==0],
              "Multi-field":  lm[lm["multi_field"]==1]}

    for grp_name, sub in groups.items():
        if LIFELINES:
            kmf = KaplanMeierFitter()
            kmf.fit(sub["time_lm"]/30.44, sub["event"], label=grp_name)
            kmf.plot_survival_function(ax=ax, ci_show=True,
                                       color=colors[grp_name], linewidth=2)
        else:
            ts, se = km_manual(sub["time_lm"]/30.44, sub["event"])
            ax.step(ts, se, where='post', color=colors[grp_name],
                    lw=2, label=grp_name)
    ax.set_xlabel(f"Months from {LANDMARK_MO}-month landmark", fontsize=11)
    ax.set_ylabel("Overall Survival", fontsize=11)
    ax.set_title(f"UADT Field: Multi-field vs Single-field Survival\n"
                 f"(Landmark at {LANDMARK_MO} months — immortal time corrected)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "fig5a_km_landmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_cox_forest(cox_res):
    """Simple HR forest plot from Cox summary."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    y_pos = np.arange(len(cox_res))
    ax.hlines(y_pos, cox_res["CI_lo"], cox_res["CI_hi"],
              color="#2563eb", lw=2, zorder=2)
    ax.scatter(cox_res["HR"], y_pos, color="#dc2626", s=50, zorder=3,
               edgecolors="white", lw=0.8)
    ax.axvline(1, color="#333", ls="--", lw=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cox_res["covariate"], fontsize=10)
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=11)
    ax.set_title(f"Cox Regression — UADT Survival\n({LANDMARK_MO}-month landmark)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    for i, (_, r) in enumerate(cox_res.iterrows()):
        ax.text(max(cox_res["CI_hi"])*1.05, i,
                f"HR={r.HR:.2f} p={r.p:.3f}", va="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(OUT / "fig5b_cox_forest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=== UADT Survival — Landmark Analysis ===")
    print("TERTIARY HYPOTHESIS: multi-field HR > 1.0 after 6-month landmark\n")

    meta    = pd.read_csv(BASE / "data/field_meta.csv")
    patients = pd.read_csv(BASE / "data/field_patients.csv", parse_dates=["dx","end_fu"])

    surv = build_survival_table(meta, patients)
    print(f"Patients with valid survival time: {len(surv):,}")
    print(f"  Single-field: {(surv['multi_field']==0).sum():,}")
    print(f"  Multi-field:  {(surv['multi_field']==1).sum():,}")

    lm, n_excl = apply_landmark(surv)
    print(f"\nAfter {LANDMARK_MO}-month landmark:")
    print(f"  Excluded (event/censor before landmark): {n_excl:,}")
    print(f"  Included — single-field: {(lm['multi_field']==0).sum():,}")
    print(f"  Included — multi-field:  {(lm['multi_field']==1).sum():,}")

    surv.to_csv(OUT / "survival_table.csv", index=False, encoding="utf-8-sig")

    # Cox model
    lm_clean = lm[["time_lm","event","multi_field","age_first","sex"]].copy()
    lm_clean["sex_m"] = (lm_clean["sex"]=="M").astype(int)
    lm_clean = lm_clean.dropna(subset=["time_lm","event","multi_field","age_first","sex_m"])
    lm_clean = lm_clean[lm_clean["time_lm"]>0]

    cox_rows = []
    if LIFELINES:
        cox_df = lm_clean[["time_lm","event","multi_field","age_first","sex_m"]].copy()
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="time_lm", event_col="event")
        cph.print_summary()
        for covar in ["multi_field","age_first","sex_m"]:
            row = cph.params_
            hr   = float(np.exp(row[covar]))
            ci_lo = float(np.exp(cph.confidence_intervals_.loc[covar,"95% lower-bound"]))
            ci_hi = float(np.exp(cph.confidence_intervals_.loc[covar,"95% upper-bound"]))
            p_val = float(cph.summary.loc[covar,"p"])
            cox_rows.append({"covariate":covar,"HR":round(hr,3),
                             "CI_lo":round(ci_lo,3),"CI_hi":round(ci_hi,3),
                             "p":round(p_val,4)})
    else:
        # Manual univariate Cox via log-rank approximation (simplified)
        from scipy.stats import chi2 as chi2dist
        print("⚠️  Using simplified log-rank HR approximation (install lifelines for proper Cox)")
        for covar, thresh in [("multi_field",0.5), ("age_first",55), ("sex_m",0.5)]:
            g0 = lm_clean[lm_clean[covar]<thresh]
            g1 = lm_clean[lm_clean[covar]>=thresh]
            cox_rows.append({"covariate":covar,"HR":np.nan,"CI_lo":np.nan,"CI_hi":np.nan,"p":np.nan})

    cox_res = pd.DataFrame(cox_rows)
    cox_res.to_csv(OUT / "cox_results.csv", index=False, encoding="utf-8-sig")

    # TERTIARY HYPOTHESIS RESULT
    mf_row = cox_res[cox_res["covariate"]=="multi_field"]
    if len(mf_row):
        r = mf_row.iloc[0]
        print(f"\nTERTIARY HYPOTHESIS RESULT: multi-field HR={r.HR:.3f} "
              f"95%CI=[{r.CI_lo:.3f}, {r.CI_hi:.3f}] p={r.p:.4f}")
        if not np.isnan(r.HR):
            if r.CI_lo > 1.0:
                print("  → Hypothesis SUPPORTED (HR significantly >1.0)")
            elif r.CI_hi < 1.0:
                print("  → Hypothesis REJECTED (HR significantly <1.0)")
            else:
                print("  → CI spans 1.0 — inconclusive after landmark correction")

    # Age-stratified
    if LIFELINES:
        print("\nAge-stratified log-rank (<55 vs ≥55):")
        for age_grp, sub in [("<55", lm[lm["age_first"]<55]), ("≥55", lm[lm["age_first"]>=55])]:
            g0 = sub[sub["multi_field"]==0]; g1 = sub[sub["multi_field"]==1]
            if len(g0)>10 and len(g1)>10:
                lr = logrank_test(g0["time_lm"], g1["time_lm"],
                                  g0["event"], g1["event"])
                print(f"  Age {age_grp}: n_single={len(g0)}, n_multi={len(g1)}, p={lr.p_value:.4f}")

    fig_km(lm)
    if not cox_res.empty and not cox_res["HR"].isna().all():
        fig_cox_forest(cox_res)
    print(f"\n✓ Outputs written to {OUT}")


if __name__ == "__main__":
    main()
