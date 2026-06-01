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
