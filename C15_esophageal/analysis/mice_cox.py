"""
MICE imputation for AJCC stage → Cox proportional hazards model.
Primary analysis for Round 3 revision: replaces zero-coding of stage-unknown patients.
Outputs: results/sensitivity/mice_cox_summary.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent.parent / "data/c15_enriched.csv"
OUT  = Path(__file__).parent.parent / "results/sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

# ── Code maps (same as 06_chemo_surgery_impact.py) ────────────────────────────
SURGERY_TYPE_MAP = {
    "00":"No surgery","10":"Incision only","12":"Endoscopic destruction",
    "20":"Local excision / polypectomy","27":"Endoscopic resection",
    "2E":"Endoscopic mucosal resection (EMR/ESD)","30":"Partial esophagectomy",
    "40":"Total esophagectomy","51":"Esophagectomy + reconstruction",
    "52":"Esophagogastrectomy","53":"Esophagectomy + LN dissection",
    "54":"Radical esophagectomy","55":"Radical esophagectomy + LN dissection","99":"Unknown",
}
SURGERY_GROUP_MAP = {
    "No surgery":"No surgery","Incision only":"Other/unknown",
    "Endoscopic destruction":"Other/unknown",
    "Local excision / polypectomy":"Endoscopic resection",
    "Endoscopic resection":"Endoscopic resection",
    "Endoscopic mucosal resection (EMR/ESD)":"Endoscopic resection",
    "Partial esophagectomy":"Partial esophagectomy",
    "Total esophagectomy":"Radical esophagectomy",
    "Esophagectomy + reconstruction":"Radical esophagectomy",
    "Esophagogastrectomy":"Radical esophagectomy",
    "Esophagectomy + LN dissection":"Radical esophagectomy",
    "Radical esophagectomy":"Radical esophagectomy",
    "Radical esophagectomy + LN dissection":"Radical esophagectomy",
    "Unknown":"Other/unknown",
}
MARGIN_MAP = {
    "0":"R0 (no residual)","2":"R1 (microscopic)","3":"R2 (macroscopic)",
    "7":"Not assessable","8":"No surgery","9":"Unknown",
    "B":"Unknown","C":"Unknown","D":"Unknown","E":"Unknown",
}
HIST_MAP = {
    "Squamous cell carcinoma": 0,
    "Carcinoma NOS": 1,
    "Adenocarcinoma": 2,
    "Other": 3,
}
STAGE_ORD = {"I": 1, "II": 2, "III": 3, "IV": 4}

# ── Load & decode ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, low_memory=False)
df["event"]     = (df["生存狀態(27)"].astype(str).str.strip() == "0").astype(int)
df["os_months"] = pd.to_numeric(df["os_days"], errors="coerce") / 30.44
df["age"]       = pd.to_numeric(df["age"], errors="coerce")
df["male"]      = (df["性別(5)"].astype(str).str.strip() == "1").astype(int)

# Surgery
df["surg_code"]  = df["原發部位手術方式(118)"].astype(str).str.strip().str.upper()
df["surg_label"] = df["surg_code"].map(SURGERY_TYPE_MAP).fillna("Unknown")
df["surg_group"] = df["surg_label"].map(SURGERY_GROUP_MAP).fillna("Other/unknown")
for g in ["Endoscopic resection", "Partial esophagectomy", "Radical esophagectomy"]:
    df[f"surg_{g.replace(' ','_')}"] = (df["surg_group"] == g).astype(int)

# Margin
df["margin"]      = df["原發部位手術邊緣(124)"].astype(str).str.strip()
df["margin_label"]= df["margin"].map(MARGIN_MAP).fillna("Unknown")
df["margin_R0"]   = (df["margin_label"] == "R0 (no residual)").astype(int)

# Chemotherapy
df["chemo_method"] = df["化學治療方式(160)_非"].astype(str).str.strip()
df["ccrt"]         = df["本院首療CCRT(159)_非"].astype(str).str.strip().map({"0":"No CCRT","1":"CCRT"})
df["ccrt_bin"]     = (df["ccrt"] == "CCRT").astype(int)
df["chemo_multi"]  = df["chemo_method"].isin(["2","4","A","C"]).astype(int)

# Stage (ordinal for MICE target, dummies for Cox)
df["stage_ord"] = df["stage_group"].map(STAGE_ORD)   # NaN for unknown

# Histology (for MICE predictor)
df["hist_num"] = df.get("histology_group", pd.Series(dtype=str)).map(HIST_MAP).fillna(3)

# Subsite (for MICE predictor) — abdominal/cervical as dummies
df["subsite_code"] = df["腫瘤部位(47)"].astype(str).str.strip()
df["subsite_abd"]  = df["subsite_code"].isin(["C154","C154.1"]).astype(int)
df["subsite_cerv"] = df["subsite_code"].isin(["C150","C150.9"]).astype(int)

print(f"Total patients: {len(df)}")
print(f"Stage known: {df['stage_ord'].notna().sum()}")
print(f"Stage unknown: {df['stage_ord'].isna().sum()} ({df['stage_ord'].isna().mean()*100:.1f}%)")

# ── MICE imputation for stage ─────────────────────────────────────────────────
# Imputation feature matrix: all predictors + stage_ord as target
mice_features = ["age","male","stage_ord",
                 "surg_Endoscopic_resection","surg_Partial_esophagectomy",
                 "surg_Radical_esophagectomy","margin_R0","ccrt_bin",
                 "chemo_multi","hist_num","subsite_abd","subsite_cerv"]

mice_df = df[mice_features].copy()

imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=50, random_state=42),
    max_iter=10,
    random_state=42,
    min_value=1,
    max_value=4,
)
mice_arr = imputer.fit_transform(mice_df)
mice_out = pd.DataFrame(mice_arr, columns=mice_features, index=df.index)

# Round imputed stage to nearest integer and clamp 1–4
mice_out["stage_imputed"] = mice_out["stage_ord"].round().clip(1, 4).astype(int)
print(f"\nImputed stage distribution (all patients):")
print(mice_out["stage_imputed"].value_counts().sort_index())
print(f"\nKnown stage distribution:")
print(df["stage_ord"].value_counts().sort_index())

# ── Cox model — MICE imputed (PRIMARY) ────────────────────────────────────────
feat_mice = pd.DataFrame({
    "os_months":           df["os_months"],
    "event":               df["event"],
    "age":                 df["age"],
    "male":                df["male"],
    "stage_II":            (mice_out["stage_imputed"] == 2).astype(int),
    "stage_III":           (mice_out["stage_imputed"] == 3).astype(int),
    "stage_IV":            (mice_out["stage_imputed"] == 4).astype(int),
    "surg_Endoscopic_resection":    df["surg_Endoscopic_resection"],
    "surg_Partial_esophagectomy":   df["surg_Partial_esophagectomy"],
    "surg_Radical_esophagectomy":   df["surg_Radical_esophagectomy"],
    "margin_R0":           df["margin_R0"],
    "ccrt":                df["ccrt_bin"],
    "chemo_multi":         df["chemo_multi"],
}).dropna()

print(f"\nMICE Cox cohort: n={len(feat_mice)}, events={int(feat_mice['event'].sum())}")

cph_mice = CoxPHFitter(penalizer=0.1)
cph_mice.fit(feat_mice, duration_col="os_months", event_col="event")
print(f"MICE Cox C-index: {cph_mice.concordance_index_:.4f}")

# ── Schoenfeld PH test (primary MICE model) ───────────────────────────────────
from lifelines.statistics import proportional_hazard_test

print("\n[PH test] Schoenfeld residuals — primary MICE Cox:")
ph_mice = proportional_hazard_test(cph_mice, feat_mice, time_transform="rank")
ph_out = ph_mice.summary[["test_statistic","p"]].copy()
ph_out["PH_OK"] = ph_out["p"] >= 0.05
ph_out.to_csv(OUT / "ph_test_mice.csv")
print(ph_out.to_string())

violated_mice = ph_out[~ph_out["PH_OK"]].index.tolist()
print(f"\nPH violations: {violated_mice if violated_mice else 'none'}")

# ── Time-split Cox at 12 months if violations detected ────────────────────────
SPLIT_MO = 12.0   # months — natural landmark for ESCC: early mortality vs survivors

if violated_mice:
    print(f"\n[Time-split Cox] landmark={SPLIT_MO}mo")

    fit_cols_mice = [c for c in feat_mice.columns if c not in ("os_months","event")]

    # Early period: 0 → SPLIT_MO
    early = feat_mice.copy()
    early["os_months_e"] = early["os_months"].clip(upper=SPLIT_MO)
    early["event_e"]     = ((early["event"] == 1) &
                            (early["os_months"] <= SPLIT_MO)).astype(int)
    early = early[early["os_months_e"] > 0]

    # Late period: SPLIT_MO → end (patients who survived past split)
    late = feat_mice[feat_mice["os_months"] > SPLIT_MO].copy()
    late["os_months_l"] = late["os_months"] - SPLIT_MO

    cph_early = CoxPHFitter(penalizer=0.1)
    cph_late  = CoxPHFitter(penalizer=0.1)

    cph_early.fit(early[["os_months_e","event_e"] + fit_cols_mice],
                  duration_col="os_months_e", event_col="event_e")
    cph_late.fit(late[["os_months_l","event"] + fit_cols_mice],
                 duration_col="os_months_l", event_col="event")

    print(f"\n  Early (≤{SPLIT_MO}mo): n={len(early)}, events={int(early['event_e'].sum())}, "
          f"C-index={cph_early.concordance_index_:.4f}")
    print(f"  Late  (>{SPLIT_MO}mo): n={len(late)},  events={int(late['event'].sum())},  "
          f"C-index={cph_late.concordance_index_:.4f}")

    key_ts = ["margin_R0","ccrt","surg_Endoscopic_resection",
              "surg_Radical_esophagectomy","stage_II","stage_III","stage_IV"]
    ts_rows = []
    print(f"\n  {'Variable':<35} {'HR_early':>9} {'HR_late':>9}  {'Direction'}")
    print("  " + "-"*70)
    for v in key_ts:
        hr_e = cph_early.summary["exp(coef)"].get(v, np.nan)
        hr_l = cph_late.summary["exp(coef)"].get(v, np.nan)
        p_e  = cph_early.summary["p"].get(v, np.nan)
        p_l  = cph_late.summary["p"].get(v, np.nan)
        if not (np.isnan(hr_e) or np.isnan(hr_l)):
            ratio = hr_l / hr_e
            direction = ("attenuates" if ratio < 0.85 else
                         "amplifies"  if ratio > 1.15 else "stable")
        else:
            ratio, direction = np.nan, "n/a"
        print(f"  {v:<35} {hr_e:9.3f} {hr_l:9.3f}  {direction}")
        ts_rows.append(dict(variable=v,
                            HR_early=round(hr_e,3), p_early=round(p_e,4),
                            HR_late=round(hr_l,3),  p_late=round(p_l,4),
                            late_vs_early=round(ratio,3) if not np.isnan(ratio) else np.nan,
                            direction=direction))
    pd.DataFrame(ts_rows).to_csv(OUT / "cox_timesplit_mice.csv", index=False)
    print(f"\n  Saved: cox_timesplit_mice.csv")

# ── Cox model — zero-coded (comparison) ───────────────────────────────────────
feat_zero = pd.DataFrame({
    "os_months":  df["os_months"],
    "event":      df["event"],
    "age":        df["age"],
    "male":       df["male"],
    "stage_II":   (df["stage_group"] == "II").astype(int),
    "stage_III":  (df["stage_group"] == "III").astype(int),
    "stage_IV":   (df["stage_group"] == "IV").astype(int),
    "surg_Endoscopic_resection":    df["surg_Endoscopic_resection"],
    "surg_Partial_esophagectomy":   df["surg_Partial_esophagectomy"],
    "surg_Radical_esophagectomy":   df["surg_Radical_esophagectomy"],
    "margin_R0":  df["margin_R0"],
    "ccrt":       df["ccrt_bin"],
    "chemo_multi":df["chemo_multi"],
}).dropna()

cph_zero = CoxPHFitter(penalizer=0.1)
cph_zero.fit(feat_zero, duration_col="os_months", event_col="event")
print(f"Zero-coded Cox C-index: {cph_zero.concordance_index_:.4f}")

# ── HR comparison table ───────────────────────────────────────────────────────
rows = []
key_vars = ["margin_R0","ccrt","surg_Endoscopic_resection",
            "surg_Radical_esophagectomy","stage_II","stage_III","stage_IV"]
print("\n{:35s}  {:>8s} {:>8s}  {:>8s} {:>8s}".format(
    "Variable","MICE HR","MICE p","Zero HR","Zero p"))
print("-"*75)
for v in key_vars:
    hr_m = cph_mice.summary["exp(coef)"].get(v, np.nan)
    p_m  = cph_mice.summary["p"].get(v, np.nan)
    hr_z = cph_zero.summary["exp(coef)"].get(v, np.nan)
    p_z  = cph_zero.summary["p"].get(v, np.nan)
    diff_pct = abs(hr_m - hr_z) / hr_z * 100 if not np.isnan(hr_z) else np.nan
    flag = " ⚠ >15%" if diff_pct > 15 else ""
    print(f"{v:35s}  {hr_m:8.3f} {p_m:8.4f}  {hr_z:8.3f} {p_z:8.4f}{flag}")
    rows.append({"variable": v,
                 "HR_MICE": round(hr_m,3), "p_MICE": round(p_m,4),
                 "HR_zero": round(hr_z,3), "p_zero": round(p_z,4),
                 "pct_diff": round(diff_pct,1)})

pd.DataFrame(rows).to_csv(OUT / "mice_cox_comparison.csv", index=False)
print(f"\nSaved: {OUT / 'mice_cox_comparison.csv'}")
print(f"\nMICE C-index: {cph_mice.concordance_index_:.4f}")
print(f"Zero C-index: {cph_zero.concordance_index_:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3  Recover stage_group from clin_stage for 2018–2020 (AJCC version NA)
# ══════════════════════════════════════════════════════════════════════════════
CLIN_TO_GROUP = {
    "0":  "I",
    "1": "I",  "1A": "I",  "1B": "I",
    "2": "II", "2A": "II", "2B": "II", "2E": "II",
    "3": "III","3A": "III","3B": "III","3C": "III",
    "4": "IV", "4A": "IV", "4B": "IV",
}

ajcc_col     = "AJCC癌症分期版本(104)"
df["ajcc_v"] = pd.to_numeric(df[ajcc_col], errors="coerce")

is_2018plus  = df["ajcc_v"].isna()
missing_sg   = df["stage_ord"].isna()

recovered = (
    df.loc[is_2018plus & missing_sg, "clin_stage"]
      .astype(str).str.strip().map(CLIN_TO_GROUP)
)
n_rec = recovered.notna().sum()
df.loc[is_2018plus & missing_sg & recovered.notna(), "stage_group"] = recovered[recovered.notna()]
df["stage_ord"] = df["stage_group"].map(STAGE_ORD)   # recompute

print(f"\n{'─'*60}")
print(f"STEP 3 — Stage recovery (2018+ from clin_stage): +{n_rec} cases")
print(f"  Stage known before: {missing_sg.sum()} missing → now {df['stage_ord'].isna().sum()} missing")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1  AJCC edition covariate dummies  (7th = reference)
# ══════════════════════════════════════════════════════════════════════════════
df["ajcc_6th"]    = (df["ajcc_v"] == 6).astype(int)
df["ajcc_unk"]    = (df["ajcc_v"] == 99).astype(int)
df["ajcc_2018p"]  = df["ajcc_v"].isna().astype(int)

# Re-run MICE with edition dummies added as predictors
mice_features_v2 = mice_features + ["ajcc_6th", "ajcc_unk", "ajcc_2018p"]
mice_df_v2 = df[mice_features_v2].copy()

imputer_v2 = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=50, random_state=42),
    max_iter=10, random_state=42, min_value=1, max_value=4,
)
mice_arr_v2 = imputer_v2.fit_transform(mice_df_v2)
mice_out_v2 = pd.DataFrame(mice_arr_v2, columns=mice_features_v2, index=df.index)
mice_out_v2["stage_imputed"] = mice_out_v2["stage_ord"].round().clip(1, 4).astype(int)

print(f"\nSTEP 1 — AJCC-adjusted MICE imputed stage distribution:")
print(mice_out_v2["stage_imputed"].value_counts().sort_index())

# Cox: AJCC-adjusted (7th = reference, adds 3 edition dummies)
feat_ajcc = pd.DataFrame({
    "os_months":                  df["os_months"],
    "event":                      df["event"],
    "age":                        df["age"],
    "male":                       df["male"],
    "stage_II":                   (mice_out_v2["stage_imputed"] == 2).astype(int),
    "stage_III":                  (mice_out_v2["stage_imputed"] == 3).astype(int),
    "stage_IV":                   (mice_out_v2["stage_imputed"] == 4).astype(int),
    "surg_Endoscopic_resection":  df["surg_Endoscopic_resection"],
    "surg_Partial_esophagectomy": df["surg_Partial_esophagectomy"],
    "surg_Radical_esophagectomy": df["surg_Radical_esophagectomy"],
    "margin_R0":                  df["margin_R0"],
    "ccrt":                       df["ccrt_bin"],
    "chemo_multi":                df["chemo_multi"],
    "ajcc_6th":                   df["ajcc_6th"],
    "ajcc_unk":                   df["ajcc_unk"],
    "ajcc_2018p":                 df["ajcc_2018p"],
}).dropna()

cph_ajcc = CoxPHFitter(penalizer=0.1)
cph_ajcc.fit(feat_ajcc, duration_col="os_months", event_col="event")
print(f"  AJCC-adjusted Cox n={len(feat_ajcc)}, C-index={cph_ajcc.concordance_index_:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2  7th-edition-only sensitivity (n=1,188, 2010–2017, clean staging)
# ══════════════════════════════════════════════════════════════════════════════
df7       = df[df["ajcc_v"] == 7].copy()
mo7       = mice_out_v2.loc[df7.index].copy()

feat_7th = pd.DataFrame({
    "os_months":                  df7["os_months"],
    "event":                      df7["event"],
    "age":                        df7["age"],
    "male":                       df7["male"],
    "stage_II":                   (mo7["stage_imputed"] == 2).astype(int),
    "stage_III":                  (mo7["stage_imputed"] == 3).astype(int),
    "stage_IV":                   (mo7["stage_imputed"] == 4).astype(int),
    "surg_Endoscopic_resection":  df7["surg_Endoscopic_resection"],
    "surg_Partial_esophagectomy": df7["surg_Partial_esophagectomy"],
    "surg_Radical_esophagectomy": df7["surg_Radical_esophagectomy"],
    "margin_R0":                  df7["margin_R0"],
    "ccrt":                       df7["ccrt_bin"],
    "chemo_multi":                df7["chemo_multi"],
}).dropna()

cph_7th = CoxPHFitter(penalizer=0.1)
cph_7th.fit(feat_7th, duration_col="os_months", event_col="event")
print(f"\nSTEP 2 — 7th-edition-only: n={len(feat_7th)}, C-index={cph_7th.concordance_index_:.4f}")

# ── Combined HR comparison: Primary vs AJCC-adjusted vs 7th-only ─────────────
print(f"\n{'─'*80}")
print(f"{'Variable':<32}  {'Primary':>8}  {'AJCC-adj':>8}  {'7th-only':>8}  {'Robust?':>8}")
print("─"*80)
rows2 = []
for v in key_vars:
    hr_p  = cph_mice.summary["exp(coef)"].get(v, np.nan)
    hr_a  = cph_ajcc.summary["exp(coef)"].get(v, np.nan)
    hr_7  = cph_7th.summary["exp(coef)"].get(v, np.nan)
    p_p   = cph_mice.summary["p"].get(v, np.nan)
    p_a   = cph_ajcc.summary["p"].get(v, np.nan)
    p_7   = cph_7th.summary["p"].get(v, np.nan)
    # Robust if max deviation across all 3 < 15%
    hrs   = [x for x in [hr_p, hr_a, hr_7] if not np.isnan(x)]
    max_dev = (max(hrs) - min(hrs)) / min(hrs) * 100 if len(hrs) > 1 else np.nan
    robust  = "✓" if max_dev < 15 else f"⚠ {max_dev:.0f}%"
    print(f"{v:<32}  {hr_p:8.3f}  {hr_a:8.3f}  {hr_7:8.3f}  {robust:>8}")
    rows2.append(dict(variable=v,
                      HR_primary=round(hr_p,3), p_primary=round(p_p,4),
                      HR_ajcc_adj=round(hr_a,3), p_ajcc_adj=round(p_a,4),
                      HR_7th_only=round(hr_7,3), p_7th_only=round(p_7,4),
                      max_pct_dev=round(max_dev,1)))

pd.DataFrame(rows2).to_csv(OUT / "mice_ajcc_sensitivity.csv", index=False)
print(f"\nSaved: {OUT / 'mice_ajcc_sensitivity.csv'}")
print(f"\nC-index summary:")
print(f"  Primary (n={len(feat_mice)}):       {cph_mice.concordance_index_:.4f}")
print(f"  AJCC-adjusted (n={len(feat_ajcc)}): {cph_ajcc.concordance_index_:.4f}")
print(f"  7th-only (n={len(feat_7th)}):      {cph_7th.concordance_index_:.4f}")
