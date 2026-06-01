"""
Sensitivity analyses for manuscript revision:
1. Bootstrap OOB C-index (B=1000) — clarify training vs corrected
2. Stage-known-only Cox (n=1,758) vs primary (n=1,987)
3. Incremental cluster OS test (Cox: cluster + histology + subsite)
4. R1 vs R2 margin sensitivity
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import multivariate_logrank_test
import warnings
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent.parent / "data/c15_enriched.csv"
DEEP = Path(__file__).parent.parent / "results/04_deep_learning/c15_final_annotated.csv"

# ── Build feature matrix (mirrors cox_treatment in 06_chemo_surgery_impact.py) ─
def build_feat(df):
    from pathlib import Path
    # decode surgery
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
        "Endoscopic destruction":"Other/unknown","Local excision / polypectomy":"Endoscopic resection",
        "Endoscopic resection":"Endoscopic resection",
        "Endoscopic mucosal resection (EMR/ESD)":"Endoscopic resection",
        "Partial esophagectomy":"Partial esophagectomy",
        "Total esophagectomy":"Radical esophagectomy",
        "Esophagectomy + reconstruction":"Radical esophagectomy",
        "Esophagogastrectomy":"Radical esophagectomy",
        "Esophagectomy + LN dissection":"Radical esophagectomy",
        "Radical esophagectomy":"Radical esophagectomy",
        "Radical esophagectomy + LN dissection":"Radical esophagectomy","Unknown":"Other/unknown",
    }
    MARGIN_MAP = {
        "0":"R0 (no residual)","2":"R1 (microscopic)","3":"R2 (macroscopic)",
        "7":"Not assessable","8":"No surgery","9":"Unknown",
        "B":"Unknown","C":"Unknown","D":"Unknown","E":"Unknown",
    }
    df = df.copy()
    df["event"]     = (df["生存狀態(27)"].astype(str).str.strip() == "0").astype(int)
    df["os_months"] = pd.to_numeric(df["os_days"], errors="coerce") / 30.44
    df["surg_code"]  = df["原發部位手術方式(118)"].astype(str).str.strip().str.upper()
    df["surg_label"] = df["surg_code"].map(SURGERY_TYPE_MAP).fillna("Unknown")
    df["surg_group"] = df["surg_label"].map(SURGERY_GROUP_MAP).fillna("Other/unknown")
    df["margin"]     = df["原發部位手術邊緣(124)"].astype(str).str.strip()
    df["margin_label"]= df["margin"].map(MARGIN_MAP).fillna("Unknown")
    df["chemo_method"]= df["化學治療方式(160)_非"].astype(str).str.strip()
    df["ccrt"]       = df["本院首療CCRT(159)_非"].astype(str).str.strip().map({"0":"No CCRT","1":"CCRT"})

    feat = pd.DataFrame(index=df.index)
    feat["os_months"] = df["os_months"]
    feat["event"]     = df["event"]
    feat["age"]       = pd.to_numeric(df["age"], errors="coerce") if "age" in df else pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    feat["male"]      = (df["性別(5)"].astype(str).str.strip() == "1").astype(int)
    for s in ["II","III","IV"]:
        feat[f"stage_{s}"] = (df.get("stage_group") == s).astype(int)
    for g in ["Endoscopic resection","Partial esophagectomy","Radical esophagectomy"]:
        feat[f"surg_{g.replace(' ','_')}"] = (df["surg_group"] == g).astype(int)
    feat["ccrt"]        = (df["ccrt"] == "CCRT").astype(int)
    feat["chemo_multi"] = df["chemo_method"].isin(["2","4","A","C"]).astype(int)
    feat["margin_R0"]   = (df["margin_label"] == "R0 (no residual)").astype(int)
    return df, feat.dropna()

raw = pd.read_csv(DATA, low_memory=False)
df_enriched, feat_full = build_feat(raw)
print(f"Primary Cox cohort: n={len(feat_full)}, events={int(feat_full['event'].sum())}")

# ── 1. Training C-index & bootstrap OOB CI ─────────────────────────────────────
print("\n=== 1. Bootstrap OOB C-index (B=1000) ===")
cph = CoxPHFitter(penalizer=0.1)
cph.fit(feat_full, duration_col="os_months", event_col="event")
train_ci = cph.concordance_index_
print(f"Training C-index (apparent): {train_ci:.4f}")

np.random.seed(42)
n = len(feat_full)
feat_arr = feat_full.reset_index(drop=True)
boot_cis = []
for b in range(1000):
    idx = np.random.choice(n, n, replace=True)
    oob = np.setdiff1d(np.arange(n), np.unique(idx))
    if len(oob) < 15:
        continue
    try:
        m = CoxPHFitter(penalizer=0.1)
        m.fit(feat_arr.iloc[idx], duration_col="os_months", event_col="event")
        pred = m.predict_partial_hazard(feat_arr.iloc[oob])
        ci_b = concordance_index(feat_arr.iloc[oob]["os_months"], -pred, feat_arr.iloc[oob]["event"])
        boot_cis.append(ci_b)
    except:
        pass

ba = np.array(boot_cis)
print(f"Bootstrap OOB C-index: {ba.mean():.4f} (95% CI {np.percentile(ba,2.5):.4f}–{np.percentile(ba,97.5):.4f}), B={len(ba)}")
print(f"Optimism = training - OOB mean = {train_ci - ba.mean():.4f}")

# ── 2. Stage-known sensitivity analysis ────────────────────────────────────────
print("\n=== 2. Stage-known-only Cox (n=1,758 staged patients) ===")
# Stage-known: stage_II + stage_III + stage_IV + implicit stage_I > 0
# Actually: stage-unknown are those with all stage dummies = 0 AND stage_group is NaN
stage_known_mask = df_enriched["stage_group"].notna()
df_staged = df_enriched[stage_known_mask].copy()
_, feat_staged = build_feat(df_staged)
print(f"Stage-known Cox cohort: n={len(feat_staged)}, events={int(feat_staged['event'].sum())}")
cph2 = CoxPHFitter(penalizer=0.1)
cph2.fit(feat_staged, duration_col="os_months", event_col="event")
print(f"Stage-known C-index: {cph2.concordance_index_:.4f}")
print("\nHR comparison (primary vs stage-known):")
for col in ["margin_R0","ccrt","surg_Endoscopic_resection","surg_Radical_esophagectomy","stage_III","stage_IV"]:
    hr1 = cph.summary["exp(coef)"].get(col, float("nan"))
    hr2 = cph2.summary["exp(coef)"].get(col, float("nan"))
    p1  = cph.summary["p"].get(col, float("nan"))
    p2  = cph2.summary["p"].get(col, float("nan"))
    print(f"  {col:35s}: primary HR={hr1:.3f} (p={p1:.3f}) | staged-only HR={hr2:.3f} (p={p2:.3f})")

# ── 3. Incremental cluster OS test ─────────────────────────────────────────────
print("\n=== 3. Incremental cluster OS test ===")
# Load deep learning annotated file (has cluster_kmeans + histology features)
df_dl = pd.read_csv(DEEP, low_memory=False)
df_dl["event_dl"]    = (df_dl["生存狀態(27)"].astype(str).str.strip() == "0").astype(int) if "生存狀態(27)" in df_dl.columns else df_dl["event"]
df_dl["os_months_dl"]= pd.to_numeric(df_dl["os_months"], errors="coerce") if "os_months" in df_dl.columns else None

# Map cluster 0→2, 1→1, 2→3 (same as manuscript)
cluster_map = {0: 2, 1: 1, 2: 3}
df_dl["cluster_display"] = df_dl["cluster_kmeans"].map(cluster_map)

# Encode histology dummies
hist_dummies = pd.get_dummies(df_dl["histology"], prefix="hist", drop_first=True) if "histology" in df_dl.columns else pd.DataFrame()

# Encode subsite dummies
subsite_dummies = pd.get_dummies(df_dl["subsite"] if "subsite" in df_dl.columns else df_dl.get("腫瘤部位(47)", pd.Series(dtype=str)), prefix="subsite", drop_first=True)

# Build incremental test dataframe
inc = pd.DataFrame()
inc["os_months"] = df_dl["os_months_dl"] if "os_months_dl" in df_dl.columns else pd.to_numeric(df_dl["os_months"], errors="coerce")
inc["event"]     = df_dl["event_dl"] if "event_dl" in df_dl.columns else df_dl["event"]
inc = pd.concat([inc, hist_dummies, subsite_dummies], axis=1)
inc = inc.dropna(subset=["os_months","event"])

# Check if we have useful dummies
print(f"  Incremental test cohort (histology+subsite only): n={len(inc)}")

# Model A: histology + subsite only
try:
    cph_a = CoxPHFitter(penalizer=0.1)
    cph_a.fit(inc, duration_col="os_months", event_col="event")
    ci_a = cph_a.concordance_index_
    print(f"  Model A (hist + subsite): C-index = {ci_a:.4f}")
except Exception as e:
    print(f"  Model A failed: {e}")
    ci_a = None

# Model B: hist + subsite + cluster
inc_b = inc.copy()
cluster_aligned = df_dl["cluster_display"].reindex(inc.index)
inc_b["cluster_C2"] = (cluster_aligned == 2).astype(int)
inc_b["cluster_C3"] = (cluster_aligned == 3).astype(int)
inc_b = inc_b.dropna()
try:
    cph_b = CoxPHFitter(penalizer=0.1)
    cph_b.fit(inc_b, duration_col="os_months", event_col="event")
    ci_b = cph_b.concordance_index_
    print(f"  Model B (hist + subsite + cluster): C-index = {ci_b:.4f}")
    print(f"  Delta C-index from adding cluster: {ci_b - ci_a:.4f}")
    # Cluster HRs
    for col in ["cluster_C2","cluster_C3"]:
        hr = cph_b.summary["exp(coef)"].get(col, float("nan"))
        p  = cph_b.summary["p"].get(col, float("nan"))
        print(f"    {col}: HR={hr:.3f}, p={p:.4f}")
except Exception as e:
    print(f"  Model B failed: {e}")

# ── 4. R1 vs R2 sensitivity ────────────────────────────────────────────────────
print("\n=== 4. R1 vs R2 margin breakdown ===")
raw2 = pd.read_csv(DATA, low_memory=False)
df2, _ = build_feat(raw2)
print(df2["margin_label"].value_counts())
r0 = df2[df2["margin_label"] == "R0 (no residual)"]
r1 = df2[df2["margin_label"] == "R1 (microscopic)"]
r2 = df2[df2["margin_label"] == "R2 (macroscopic)"]
print(f"\nR0 n={len(r0)}, R1 n={len(r1)}, R2 n={len(r2)}")
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
for a_label, b_label, a_df, b_df in [("R0","R1",r0,r1),("R0","R2",r0,r2),("R1","R2",r1,r2)]:
    a = a_df.dropna(subset=["os_months","event"])
    b = b_df.dropna(subset=["os_months","event"])
    if len(a) < 5 or len(b) < 5:
        print(f"  {a_label} vs {b_label}: insufficient n")
        continue
    p = logrank_test(a["os_months"], b["os_months"], a["event"], b["event"]).p_value
    print(f"  {a_label} (n={len(a)}) vs {b_label} (n={len(b)}): log-rank p={p:.4f}")
    print(f"    {a_label} median OS: {a['os_months'].median():.1f} mo | {b_label} median OS: {b['os_months'].median():.1f} mo")
