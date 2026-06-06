"""
Generic site survival pipeline for CMUH cancer registry.

Generates KM + Cox survival tables for any ICD-O site code, then writes
CSV outputs compatible with the webapp's db.py and chart rendering.

Usage:
    python analysis/site_survival.py --site C34          # lung
    python analysis/site_survival.py --site C73          # thyroid
    python analysis/site_survival.py --site C18 C20      # colon + rectum

Outputs (written to results/site_survival/{SITE}/):
    km_stage_medians.csv      — median OS by AJCC stage group
    km_sex_medians.csv        — median OS by sex
    km_age_medians.csv        — median OS by age group (<55 / ≥55)
    cox_summary.csv           — multivariate Cox HR table
    site_summary.csv          — one-row cohort summary
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data/processed/all_cancers.csv"
OUTBASE = ROOT / "results/site_survival"

MISSING_STAGE = {"888", "999", "BBB", "NAN", "88", "99", "998", "000", "001"}
MISSING_DATE  = {"0000000", "NAN", "0", "00000000"}

# ── helpers ───────────────────────────────────────────────────────────────────

def roc_to_ad(val) -> pd.Timestamp:
    s = str(val).strip().split(".")[0]
    if not s.isdigit() or len(s) < 5:
        return pd.NaT
    s = s.zfill(7)
    yr = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00", "99") else "01"
    dd = s[5:7] if s[5:7] not in ("00", "99") else "01"
    try:
        return pd.Timestamp(f"{yr}-{mm}-{dd}")
    except Exception:
        return pd.NaT


def clean_stage(s) -> str | None:
    s = str(s).strip().upper()
    if s in MISSING_STAGE:
        return None
    return s


def stage_group(s) -> str | None:
    if s is None:
        return None
    m = str(s)[0]
    return {"1": "I", "2": "II", "3": "III", "4": "IV"}.get(m)


def km_medians(df: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    rows = []
    for grp in sorted(df[group_col].dropna().unique()):
        sub = df[df[group_col] == grp]
        n, ev = len(sub), int(sub["event"].sum())
        kmf = KaplanMeierFitter()
        kmf.fit(sub["os_months"], sub["event"], label=str(grp))
        med = kmf.median_survival_time_
        med_str = f"{med:.1f}" if pd.notna(med) and med < 1e6 else "NR"
        rows.append({"Group": str(grp), "n": n, "events": ev,
                     "Median OS (months)": med_str})
    return pd.DataFrame(rows)


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(site: str) -> None:
    site = site.upper().strip()
    print(f"\n{'='*55}")
    print(f"  Site survival pipeline  →  {site}")
    print(f"{'='*55}")

    if not DATA.exists():
        sys.exit(f"ERROR: {DATA} not found. Run from repo root.")

    # ── 1. Load & filter ──────────────────────────────────────────────────────
    df_raw = pd.read_csv(DATA, low_memory=False, encoding="utf-8-sig")
    df_raw["_site3"] = df_raw["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df = df_raw[df_raw["_site3"] == site].copy()
    print(f"  Raw rows for {site}: {len(df):,}")
    if len(df) < 30:
        print(f"  SKIP: Only {len(df)} rows — too few for survival analysis.")
        return

    # ── 2. Dates & OS ─────────────────────────────────────────────────────────
    df["diag_dt"]  = df["最初診斷日(45)"].apply(roc_to_ad)
    df["death_dt"] = df["死亡日期(31)"].apply(roc_to_ad)
    df["last_dt"]  = df["最後聯絡日(30)"].apply(roc_to_ad)

    df["event"] = (
        pd.to_numeric(df["生存狀態(27)"], errors="coerce").fillna(1) == 0
    ).astype(int)

    end_dt = df["death_dt"].where(df["event"] == 1, df["last_dt"])
    df["os_months"] = (end_dt - df["diag_dt"]).dt.days / 30.44
    df = df[df["os_months"] > 0].copy()
    print(f"  After OS filter: {len(df):,}  |  events: {int(df['event'].sum()):,}"
          f"  ({df['event'].mean()*100:.1f}%)")

    # ── 3. Covariates ────────────────────────────────────────────────────────
    df["sex"]     = pd.to_numeric(df["性別(5)"], errors="coerce")   # 1=M 2=F
    df["male"]    = (df["sex"] == 1).astype(int)
    df["age"]     = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["age55"]   = (df["age"] >= 55).astype(int)
    df["age_grp"] = np.where(df["age"].notna() & (df["age"] >= 55), "≥55", "<55")

    # ── 4. Stage ──────────────────────────────────────────────────────────────
    df["_path_stage"] = df["病理期別組合(101)"].apply(clean_stage)
    df["_clin_stage"] = df["臨床期別組合(95)"].apply(clean_stage)
    df["stage_raw"]   = df["_path_stage"].combine_first(df["_clin_stage"])
    df["stage_group"] = df["stage_raw"].apply(stage_group)

    staged = df[df["stage_group"].notna()].copy()
    pct_staged = len(staged) / len(df) * 100
    print(f"  Staged: {len(staged):,} ({pct_staged:.0f}%)  |  "
          + "  ".join(f"{g}:{(staged['stage_group']==g).sum()}"
                      for g in ["I","II","III","IV"]))

    # ── 5. KM tables ──────────────────────────────────────────────────────────
    OUT = OUTBASE / site
    OUT.mkdir(parents=True, exist_ok=True)

    # By stage
    if len(staged) >= 20:
        km_st = km_medians(staged, "stage_group", "stage")
        km_st.to_csv(OUT / "km_stage_medians.csv", index=False)
        print(f"  ✓ km_stage_medians.csv  ({len(km_st)} groups)")

    # By sex
    df_sex = df[df["sex"].isin([1, 2])].copy()
    df_sex["sex_label"] = df_sex["sex"].map({1: "Male", 2: "Female"})
    km_sx = km_medians(df_sex, "sex_label", "sex")
    km_sx.to_csv(OUT / "km_sex_medians.csv", index=False)
    print(f"  ✓ km_sex_medians.csv")

    # By age group
    df_age = df[df["age"].notna()].copy()
    km_ag = km_medians(df_age, "age_grp", "age")
    km_ag.to_csv(OUT / "km_age_medians.csv", index=False)
    print(f"  ✓ km_age_medians.csv")

    # ── 6. Cox ────────────────────────────────────────────────────────────────
    cox_df = staged[["os_months", "event", "age", "male", "stage_group"]].dropna()
    cox_df = cox_df[np.isfinite(cox_df["age"])].copy()
    if len(cox_df) >= 50 and cox_df["event"].sum() >= 10:
        for g in ["II", "III", "IV"]:
            cox_df[f"stage_{g}"] = (cox_df["stage_group"] == g).astype(int)
        cox_df = cox_df.drop(columns="stage_group")
        try:
            # Drop zero-variance covariates (sex-specific cancers: all-male prostate, all-female cervix)
            cov_cols = [c for c in cox_df.columns if c not in ("os_months", "event")]
            cov_cols = [c for c in cov_cols if cox_df[c].nunique() > 1]
            cox_df = cox_df[["os_months", "event"] + cov_cols]
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_df, "os_months", "event")
            s = cph.summary
            cox_out = pd.DataFrame({
                "covariate":  s.index.tolist(),
                "HR":         s["exp(coef)"].round(3).tolist(),
                "HR_lower95": s["exp(coef) lower 95%"].round(3).tolist(),
                "HR_upper95": s["exp(coef) upper 95%"].round(3).tolist(),
                "p":          s["p"].round(4).tolist(),
            })
            cox_out.to_csv(OUT / "cox_summary.csv", index=False)
            c_idx = cph.concordance_index_
            print(f"  ✓ cox_summary.csv  (C-index={c_idx:.3f})")
        except Exception as exc:
            print(f"  ⚠ Cox failed: {exc}")
    else:
        print(f"  ⚠ Cox skipped (n={len(cox_df)}, events={cox_df['event'].sum()})")

    # ── 7. Summary row ────────────────────────────────────────────────────────
    summary = pd.DataFrame([{
        "site": site,
        "n_total": len(df),
        "n_events": int(df["event"].sum()),
        "event_pct": round(df["event"].mean() * 100, 1),
        "n_staged": len(staged),
        "staged_pct": round(pct_staged, 1),
        "n_male": int((df["sex"] == 1).sum()),
        "n_female": int((df["sex"] == 2).sum()),
        "age_median": round(df["age"].median(), 1),
        "diag_yr_min": df["diag_dt"].dt.year.min(),
        "diag_yr_max": df["diag_dt"].dt.year.max(),
    }])
    summary.to_csv(OUT / "site_summary.csv", index=False)
    print(f"  ✓ site_summary.csv")
    print(f"  Output: {OUT}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic site survival pipeline")
    parser.add_argument("--site", nargs="+", required=True,
                        help="ICD-O site code(s), e.g. C34 C73 C18")
    args = parser.parse_args()
    for s in args.site:
        run(s)
    print("\nAll done.")
