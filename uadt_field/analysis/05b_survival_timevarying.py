"""
UADT Field Cancerization — Script 05b: Time-Varying Cox Model

Validates the TERTIARY HYPOTHESIS using a time-varying covariate approach,
which is the gold-standard fix for immortal-time bias.

PROBLEM WITH THE LANDMARK APPROACH (05):
  The 6-month landmark treats multi_field as a fixed baseline covariate.
  This is still biased: a patient diagnosed with a second field cancer at
  month 18 contributes "multi_field=1" from t=0 in the landmark cohort, but
  they were actually single-field for months 6–18.

SOLUTION — TIME-VARYING COX:
  Split each patient's follow-up at the date of the second field diagnosis:

    Single-field patients:
      start=0  stop=end_fu  event=dead  multi_field=0

    Multi-field patients:
      Row A: start=0       stop=t_2nd  event=0     multi_field=0
      Row B: start=t_2nd  stop=end_fu  event=dead  multi_field=1

  where t_2nd = days from first field dx to second field dx.

  This eliminates immortal time by construction: the patient is only
  "exposed" (multi_field=1) from the moment the second cancer is diagnosed.
  No observations are excluded; no arbitrary landmark date is chosen.

COMPARISON:
  Both models (landmark 05 and time-varying 05b) are reported side-by-side
  to show whether the HR<1 finding survives this stronger correction.

Outputs:
  results/05_survival/tv_table.csv            — start-stop dataset
  results/05_survival/cox_tv_results.csv      — time-varying Cox summary
  results/05_survival/fig5c_tv_comparison.png — HR comparison: landmark vs time-varying
  results/05_survival/fig5d_tv_schoenfeld.png — Schoenfeld residuals (PH check)
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from lifelines import CoxTimeVaryingFitter, CoxPHFitter
    LIFELINES = True
except ImportError:
    LIFELINES = False
    print("❌  lifelines not installed — cannot run time-varying Cox")
    raise SystemExit(1)

# ── LOCKED CONSTANTS ──────────────────────────────────────────────────────────
FIELD_SITES = ['C02','C03','C04','C05','C06','C09','C10','C12','C13','C15']
FIELD_LABELS = {
    'C02':'Tongue',      'C03':'Gum',        'C04':'Floor of mouth',
    'C05':'Palate',      'C06':'Oral NOS',   'C09':'Tonsil',
    'C10':'Oropharynx',  'C12':'Pyriform',   'C13':'Hypopharynx',
    'C15':'Esophagus'
}
STUDY_END   = pd.Timestamp('2020-12-31')
LANDMARK_MO = 6

assert len(FIELD_SITES) == 10

BASE = Path(__file__).parent.parent
OUT  = BASE / "results/05_survival"
OUT.mkdir(parents=True, exist_ok=True)


# ── Build start-stop dataset ──────────────────────────────────────────────────

def build_tv_table(patients, meta):
    """
    Build the start-stop (counting process) format for time-varying Cox.

    For each patient, time is measured in days from first field dx.
    Multi-field patients get two rows; single-field patients get one row.
    """
    patients = patients.copy()
    patients["dx"]     = pd.to_datetime(patients["dx"])
    patients["end_fu"] = pd.to_datetime(patients["end_fu"])

    # First field dx per patient
    first_dx = (patients.sort_values("dx")
                        .groupby("pid")["dx"].first()
                        .rename("first_dx").reset_index())

    # Second field dx per patient (if any)
    second_dx = (patients.sort_values("dx")
                         .groupby("pid")
                         .apply(lambda g: g["dx"].iloc[1] if len(g) >= 2 else pd.NaT)
                         .rename("second_dx").reset_index())

    # End of follow-up and event per patient
    end_info = meta[["pid","dead","sex","age_first"]].copy()
    end_fu   = patients.groupby("pid")["end_fu"].max().reset_index()

    df = (first_dx
          .merge(second_dx, on="pid", how="left")
          .merge(end_fu,    on="pid", how="left")
          .merge(end_info,  on="pid", how="left"))

    df["t_total"] = (df["end_fu"] - df["first_dx"]).dt.days
    df["t_2nd"]   = (df["second_dx"] - df["first_dx"]).dt.days
    df["sex_m"]   = (df["sex"] == "M").astype(int)
    df = df[df["t_total"] > 0].copy()

    rows = []
    for _, r in df.iterrows():
        pid      = r["pid"]
        t_total  = r["t_total"]
        dead     = int(r["dead"])
        age      = r["age_first"]
        sex_m    = r["sex_m"]
        t_2nd    = r["t_2nd"]
        multi    = (pd.notna(t_2nd) and t_2nd > 0 and t_2nd < t_total)

        if not multi:
            # Single-field: one interval
            rows.append({"pid": pid, "start": 0, "stop": t_total,
                         "event": dead, "multi_field": 0,
                         "age_first": age, "sex_m": sex_m})
        else:
            t2 = int(t_2nd)
            # Row A: before second dx (single-field period)
            rows.append({"pid": pid, "start": 0, "stop": t2,
                         "event": 0, "multi_field": 0,
                         "age_first": age, "sex_m": sex_m})
            # Row B: after second dx (multi-field period)
            rows.append({"pid": pid, "start": t2, "stop": t_total,
                         "event": dead, "multi_field": 1,
                         "age_first": age, "sex_m": sex_m})

    tv = pd.DataFrame(rows)
    tv = tv[tv["stop"] > tv["start"]].reset_index(drop=True)
    return tv


# ── Time-varying Cox ──────────────────────────────────────────────────────────

def fit_tv_cox(tv):
    ctv = CoxTimeVaryingFitter()
    ctv.fit(tv,
            id_col="pid",
            start_col="start",
            stop_col="stop",
            event_col="event",
            formula="multi_field + age_first + sex_m")
    return ctv


# ── Comparison figure: landmark HR vs time-varying HR ────────────────────────

def fig_comparison(tv_res, landmark_csv):
    """Side-by-side forest plot comparing the two Cox models."""
    try:
        lm = pd.read_csv(landmark_csv)
    except Exception:
        lm = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    covars    = ["multi_field", "age_first", "sex_m"]
    cov_labels = ["Multi-field status", "Age at first dx", "Sex (male)"]

    def plot_panel(ax, res_df, title):
        y = np.arange(len(covars))
        ax.axvline(1, color="#333", ls="--", lw=0.9, zorder=1)
        colors = ["#dc2626" if c == "multi_field" else "#6b7280" for c in covars]
        for i, (cov, label, col) in enumerate(zip(covars, cov_labels, colors)):
            row = res_df[res_df["covariate"] == cov]
            if len(row) == 0: continue
            r = row.iloc[0]
            ax.hlines(i, r["CI_lo"], r["CI_hi"], color=col, lw=2.5, zorder=2)
            ax.scatter(r["HR"], i, color=col, s=60, zorder=3,
                       edgecolors="white", lw=0.8)
            ax.text(max(res_df["CI_hi"]) * 1.05, i,
                    f"HR={r.HR:.3f}\n95%CI [{r.CI_lo:.3f}, {r.CI_hi:.3f}]\np={r.p:.4f}",
                    va="center", fontsize=7.5, color=col)
        ax.set_yticks(y); ax.set_yticklabels(cov_labels, fontsize=10)
        ax.set_xlabel("Hazard Ratio", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", color="#14304a")
        ax.spines[["top","right"]].set_visible(False)

    # Time-varying panel
    plot_panel(axes[1], tv_res, "Time-Varying Cox\n(gold standard, no landmark)")

    # Landmark panel
    if lm is not None:
        plot_panel(axes[0], lm, f"Landmark Cox ({LANDMARK_MO}-month)\n(Script 05 result)")
    else:
        axes[0].text(0.5, 0.5, "Run 05_survival_landmark.py first",
                     ha="center", va="center", transform=axes[0].transAxes,
                     fontsize=10, color="#aaa")
        axes[0].set_title(f"Landmark Cox ({LANDMARK_MO}-month)", fontsize=11, fontweight="bold")

    fig.suptitle("TERTIARY HYPOTHESIS VALIDATION\nLandmark vs Time-Varying Cox Comparison",
                 fontsize=12, fontweight="bold", color="#14304a", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig5c_tv_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_schoenfeld(ctv, tv):
    """Plot Schoenfeld residuals to check proportional hazards assumption."""
    try:
        resid = ctv.compute_residuals(tv, kind="schoenfeld")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        covars_plot = [c for c in ["multi_field","age_first","sex_m"]
                       if c in resid.columns]
        for ax, col in zip(axes, covars_plot):
            t = resid.index if resid.index.name == "stop" else range(len(resid))
            ax.scatter(t, resid[col], alpha=0.3, s=8, color="#6b7280")
            ax.axhline(0, color="#dc2626", ls="--", lw=1)
            ax.set_title(col, fontsize=9); ax.set_xlabel("Time (days)")
            ax.spines[["top","right"]].set_visible(False)
        fig.suptitle("Schoenfeld Residuals — Proportional Hazards Check",
                     fontsize=10, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / "fig5d_tv_schoenfeld.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  (Schoenfeld plot skipped: {e})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== UADT Survival — Time-Varying Cox ===")
    print("Gold-standard immortal-time correction: no landmark, exposure changes at t_2nd\n")

    meta     = pd.read_csv(BASE / "data/field_meta.csv")
    patients = pd.read_csv(BASE / "data/field_patients.csv",
                           parse_dates=["dx","end_fu"])

    print("Building start-stop table…")
    tv = build_tv_table(patients, meta)

    n_pts     = tv["pid"].nunique()
    n_rows    = len(tv)
    n_multi   = tv[tv["multi_field"]==1]["pid"].nunique()
    n_single  = n_pts - n_multi
    n_events  = tv.groupby("pid")["event"].max().sum()
    mf_time   = tv[tv["multi_field"]==1]["stop"].subtract(
                    tv[tv["multi_field"]==1]["start"]).sum()
    sf_time   = tv[tv["multi_field"]==0]["stop"].subtract(
                    tv[tv["multi_field"]==0]["start"]).sum()

    print(f"  Patients: {n_pts:,}  |  Rows: {n_rows:,}  |  Events: {int(n_events):,}")
    print(f"  Ever multi-field: {n_multi:,}  |  Always single-field: {n_single:,}")
    print(f"  Person-days exposed (multi_field=1): {mf_time:,.0f}")
    print(f"  Person-days exposed (multi_field=0): {sf_time:,.0f}")
    tv.to_csv(OUT / "tv_table.csv", index=False, encoding="utf-8-sig")

    print("\nFitting time-varying Cox model…")
    ctv = fit_tv_cox(tv)
    ctv.print_summary()

    # Extract results
    cox_rows = []
    for cov in ["multi_field","age_first","sex_m"]:
        try:
            hr    = float(np.exp(ctv.params_[cov]))
            ci_lo = float(np.exp(ctv.confidence_intervals_.loc[cov,"95% lower-bound"]))
            ci_hi = float(np.exp(ctv.confidence_intervals_.loc[cov,"95% upper-bound"]))
            p_val = float(ctv.summary.loc[cov,"p"])
            cox_rows.append({"covariate":cov, "HR":round(hr,3),
                             "CI_lo":round(ci_lo,3), "CI_hi":round(ci_hi,3),
                             "p":round(p_val,4)})
        except Exception:
            cox_rows.append({"covariate":cov,"HR":np.nan,"CI_lo":np.nan,
                             "CI_hi":np.nan,"p":np.nan})
    tv_res = pd.DataFrame(cox_rows)
    tv_res.to_csv(OUT / "cox_tv_results.csv", index=False, encoding="utf-8-sig")

    # TERTIARY HYPOTHESIS RESULT
    mf = tv_res[tv_res["covariate"]=="multi_field"].iloc[0]
    print(f"\nTERTIARY HYPOTHESIS — TIME-VARYING COX RESULT:")
    print(f"  multi_field HR = {mf.HR:.3f}  95%CI [{mf.CI_lo:.3f}, {mf.CI_hi:.3f}]  p={mf.p:.4f}")
    if not np.isnan(mf.HR):
        if mf.CI_lo > 1.0:
            verdict = "SUPPORTED — HR > 1.0 significantly (worse survival after second field dx)"
        elif mf.CI_hi < 1.0:
            verdict = "REJECTED — HR < 1.0 significantly (better survival after second field dx)"
        else:
            verdict = "INCONCLUSIVE — CI spans 1.0"
        print(f"  → {verdict}")

    # Comparison with landmark result
    lm_csv = OUT / "cox_results.csv"
    if lm_csv.exists():
        lm = pd.read_csv(lm_csv)
        lm_mf = lm[lm["covariate"]=="multi_field"].iloc[0]
        print(f"\n  COMPARISON:")
        print(f"  Landmark Cox (05):     HR={lm_mf.HR:.3f} 95%CI [{lm_mf.CI_lo:.3f}, {lm_mf.CI_hi:.3f}]")
        print(f"  Time-varying Cox (05b):HR={mf.HR:.3f} 95%CI [{mf.CI_lo:.3f}, {mf.CI_hi:.3f}]")
        delta = mf.HR - lm_mf.HR
        print(f"  ΔHR = {delta:+.3f}  "
              f"({'toward null' if (lm_mf.HR<1 and delta>0) or (lm_mf.HR>1 and delta<0) else 'away from null'})")

    print("\nGenerating figures…")
    fig_comparison(tv_res, lm_csv if lm_csv.exists() else None)
    fig_schoenfeld(ctv, tv)
    print(f"✓ Outputs written to {OUT}")


if __name__ == "__main__":
    main()
