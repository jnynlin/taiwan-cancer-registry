"""
Chemo regimen / cumulative dose  &  Surgery type impact on survival.
Outputs: results/06_chemo_surgery/
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import warnings
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent.parent / "data/c15_enriched.csv"
OUT  = Path(__file__).parent.parent / "results/06_chemo_surgery"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
COLORS = sns.color_palette("tab10")

# ── Code dictionaries ──────────────────────────────────────────────────────────
SURGERY_TYPE_MAP = {
    "00": "No surgery",
    "10": "Incision only",
    "12": "Endoscopic destruction",
    "20": "Local excision / polypectomy",
    "27": "Endoscopic resection",
    "2E": "Endoscopic mucosal resection (EMR/ESD)",
    "30": "Partial esophagectomy",
    "40": "Total esophagectomy",
    "51": "Esophagectomy + reconstruction",
    "52": "Esophagogastrectomy",
    "53": "Esophagectomy + LN dissection",
    "54": "Radical esophagectomy",
    "55": "Radical esophagectomy + LN dissection",
    "99": "Unknown",
}
SURGERY_GROUP_MAP = {
    "No surgery":                           "No surgery",
    "Incision only":                        "Other/unknown",
    "Endoscopic destruction":               "Other/unknown",
    "Local excision / polypectomy":         "Endoscopic resection",
    "Endoscopic resection":                 "Endoscopic resection",
    "Endoscopic mucosal resection (EMR/ESD)":"Endoscopic resection",
    "Partial esophagectomy":                "Partial esophagectomy",
    "Total esophagectomy":                  "Radical esophagectomy",
    "Esophagectomy + reconstruction":       "Radical esophagectomy",
    "Esophagogastrectomy":                  "Radical esophagectomy",
    "Esophagectomy + LN dissection":        "Radical esophagectomy",
    "Radical esophagectomy":                "Radical esophagectomy",
    "Radical esophagectomy + LN dissection":"Radical esophagectomy",
    "Unknown":                              "Other/unknown",
}
MARGIN_MAP = {
    "0": "R0 (no residual)", "2": "R1 (microscopic)",
    "3": "R2 (macroscopic)", "7": "Not assessable",
    "8": "No surgery",       "9": "Unknown",
    "B": "Unknown", "C": "Unknown", "D": "Unknown", "E": "Unknown",
}
LN_MAP = {
    0: "No LN dissection", 1: "Limited", 2: "Regional",
    3: "Extended", 4: "Systematic", 5: "Radical/3-field", 9: "Unknown",
}
MIS_MAP = {
    0: "Open", 1: "Thoracoscopic", 2: "Laparoscopic",
    3: "Robotic", 8: "No surgery", 9: "Unknown",
}
CHEMO_REGIMEN_MAP = {
    "0": "No chemo",
    "1": "Single agent",
    "2": "Multi-agent",
    "3": "Single agent + other",
    "4": "Multi-agent + other",
    "A": "Multi-agent + other",
    "C": "Combination (special)",
    "9": "Unknown",
}
# 申報醫院化學治療(163) grouping
def hosp_chemo_group(code):
    c = str(code).strip()
    if c in ("0",): return "No chemo"
    if c in ("1","86"): return "Single agent"
    if c in ("2",): return "Multi-agent (no RT)"
    if c in ("3",): return "CCRT (concurrent)"
    if c in ("82","83","84","85","87"): return "Multi-agent + targeted/immuno"
    if c in ("99",): return "Unknown"
    return "Unknown"

RT_SEQ_MAP = {
    0:  "Surgery only",
    -8: "Neoadjuvant RT → surgery",
    -7: "Neoadjuvant CCRT → surgery",
    -1: "Neoadjuvant chemo → surgery",
    4:  "Surgery → adjuvant RT",
    1:  "Surgery → adjuvant chemo",
    5:  "Surgery → adjuvant CCRT",
    99: "Unknown",
}


# ── Load & decode ──────────────────────────────────────────────────────────────
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False)
    df["event"]     = (df["生存狀態(27)"].astype(str).str.strip() == "0").astype(int)
    df["os_months"] = pd.to_numeric(df["os_days"], errors="coerce") / 30.44

    # Surgery type
    df["surg_code"]  = df["原發部位手術方式(118)"].astype(str).str.strip().str.upper()
    df["surg_label"] = df["surg_code"].map(SURGERY_TYPE_MAP).fillna("Unknown")
    df["surg_group"] = df["surg_label"].map(SURGERY_GROUP_MAP).fillna("Other/unknown")

    # Surgical margin
    df["margin"] = df["原發部位手術邊緣(124)"].astype(str).str.strip()
    df["margin_label"] = df["margin"].map(MARGIN_MAP).fillna("Unknown")

    # LN dissection extent
    df["ln_extent"] = pd.to_numeric(df["申報醫院區域淋巴結手術範圍(126)"], errors="coerce")
    df["ln_label"]  = df["ln_extent"].map(LN_MAP)

    # Minimally invasive surgery
    df["mis_code"]  = pd.to_numeric(df["*微創手術(B8)"], errors="coerce")
    df["mis_label"] = df["mis_code"].map(MIS_MAP)

    # Lymph node counts
    df["ln_examined"] = pd.to_numeric(df["区域淋巴結檢查數目(87)"] if "区域淋巴結檢查數目(87)" in df.columns
                                       else df["區域淋巴結檢查數目(87)"], errors="coerce").replace([97,98,99], np.nan)
    df["ln_positive"] = pd.to_numeric(df["區域淋巴結侵犯數目(88)"], errors="coerce").replace([97,98,99], np.nan)
    df["ln_ratio"]    = (df["ln_positive"] / df["ln_examined"]).replace([np.inf, -np.inf], np.nan)

    # Surgical margin distance (mm)
    df["margin_mm"] = pd.to_numeric(df["*原發部位手術切緣距(B9)"], errors="coerce").replace([988,999], np.nan)

    # Chemo regimen
    df["chemo_method"] = df["化學治療方式(160)_非"].astype(str).str.strip()
    df["chemo_label"]  = df["chemo_method"].map(CHEMO_REGIMEN_MAP).fillna("Unknown")
    df["chemo_hosp_group"] = df["申報醫院化學治療(163)"].apply(hosp_chemo_group)

    # CCRT flag
    df["ccrt"] = df["本院首療CCRT(159)_非"].astype(str).str.strip().map({"0":"No CCRT","1":"CCRT"})

    # Cycles (cumulative dose proxy)
    df["cycles"] = pd.to_numeric(df["化學治療次數-Cycle(161)_非"], errors="coerce").replace([98,99], np.nan)
    df["cycles_group"] = pd.cut(df["cycles"], bins=[-0.1, 0, 3, 6, 100],
                                labels=["0","1–3","4–6",">6"])

    # Chemo start lag (days from diagnosis)
    def to_date(x):
        from datetime import datetime
        s = str(x).split(".")[0].zfill(7)
        if len(s) != 7 or not s.isdigit() or s in ("0000000","9999999"):
            return pd.NaT
        yr = int(s[:3]) + 1911
        mm = s[3:5] if s[3:5] not in ("00","99") else "01"
        dd = s[5:7] if s[5:7] not in ("00","99") else "01"
        try:
            return pd.Timestamp(f"{yr}-{mm}-{dd}")
        except:
            return pd.NaT
    df["diag_ts"]       = pd.to_datetime(df["diag_date"], errors="coerce")
    df["chemo_start_ts"]= df["申報醫院化學治療開始日期(162)"].apply(to_date)
    df["chemo_lag_days"]= (df["chemo_start_ts"] - df["diag_ts"]).dt.days
    df.loc[df["chemo_lag_days"] < 0, "chemo_lag_days"] = np.nan
    df.loc[df["chemo_lag_days"] > 365, "chemo_lag_days"] = np.nan

    # RT-surgery sequence
    df["rt_surg_seq"] = pd.to_numeric(df["放射治療與手術順序(140)"], errors="coerce")
    df["rt_surg_label"] = df["rt_surg_seq"].map(RT_SEQ_MAP).fillna("Unknown")

    return df


# ── KM helper ─────────────────────────────────────────────────────────────────
def km_plot(df, group_col, order=None, title="", fname="km.png",
            min_n=20, time_col="os_months", event_col="event"):
    sub = df.dropna(subset=[time_col, event_col, group_col])
    sub = sub[sub[time_col] > 0]
    groups = order or sorted(sub[group_col].unique())
    groups = [g for g in groups if (sub[group_col] == g).sum() >= min_n]
    if len(groups) < 2:
        print(f"  Skipped {fname}: fewer than 2 groups with n≥{min_n}")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    kmf  = KaplanMeierFitter()
    meds = []
    for i, g in enumerate(groups):
        s = sub[sub[group_col] == g]
        kmf.fit(s[time_col], s[event_col], label=f"{g} (n={len(s)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=COLORS[i % 10])
        m = kmf.median_survival_time_
        meds.append({"Group": g, "n": len(s), "Events": int(s[event_col].sum()),
                     "Median OS (mo)": round(m, 1) if pd.notna(m) else "NR"})
    ax.set(title=title, xlabel="Months", ylabel="Survival probability", ylim=(0, 1.05))
    ax.legend(loc="upper right", fontsize=8)

    if len(groups) == 2:
        g1 = sub[sub[group_col] == groups[0]]; g2 = sub[sub[group_col] == groups[1]]
        p = logrank_test(g1[time_col], g2[time_col], g1[event_col], g2[event_col]).p_value
    else:
        p = multivariate_logrank_test(sub[time_col], sub[group_col], sub[event_col]).p_value
    ax.text(0.62, 0.85, f"Log-rank p = {p:.4f}", transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    med_df = pd.DataFrame(meds)
    axes[1].axis("off")
    tbl = axes[1].table(cellText=med_df.values, colLabels=med_df.columns,
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.1, 1.4)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150); plt.close(fig)
    med_df.to_csv(OUT / fname.replace(".png","_medians.csv"), index=False, encoding="utf-8-sig")
    return p


# ── Cox multi-treatment model ──────────────────────────────────────────────────
def cox_treatment(df: pd.DataFrame):
    """Cox model with chemo + surgery covariates."""
    feat = pd.DataFrame(index=df.index)
    feat["os_months"] = df["os_months"]
    feat["event"]     = df["event"]
    feat["age"]       = pd.to_numeric(df["age"], errors="coerce")
    feat["male"]      = (df["sex"] == "Male").astype(int) if "sex" in df else 0

    for s in ["II","III","IV"]:
        feat[f"stage_{s}"] = (df.get("stage_group") == s).astype(int)

    # Surgery groups
    for g in ["Endoscopic resection","Partial esophagectomy","Radical esophagectomy"]:
        feat[f"surg_{g.replace(' ','_')}"] = (df["surg_group"] == g).astype(int)

    # Chemo
    feat["ccrt"]        = (df["ccrt"] == "CCRT").astype(int)
    feat["chemo_multi"] = df["chemo_method"].isin(["2","4","A","C"]).astype(int)

    # Margin R0
    feat["margin_R0"] = (df["margin_label"] == "R0 (no residual)").astype(int)

    feat = feat.dropna()
    print(f"  Cox treatment model: n={len(feat)}, events={int(feat['event'].sum())}")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(feat, duration_col="os_months", event_col="event")
    cph.print_summary()

    # Forest plot
    s = cph.summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].copy()
    s.columns = ["HR","lo","hi","p"]
    s = s.reset_index()
    s.columns = ["Variable","HR","lo","hi","p"]
    s = s.sort_values("HR")
    fig, ax = plt.subplots(figsize=(8, len(s)*0.5 + 1))
    y = range(len(s))
    colors = ["#c0392b" if row.HR > 1 else "#27ae60" for _, row in s.iterrows()]
    ax.scatter(s["HR"], y, c=colors, zorder=3, s=60)
    ax.hlines(y, s["lo"], s["hi"], color=colors, linewidth=2)
    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(list(y)); ax.set_yticklabels(s["Variable"], fontsize=9)
    ax.set(title=f"Cox Model — Treatment Factors (C={cph.concordance_index_:.3f})",
           xlabel="Hazard Ratio (95% CI)")
    ax.set_xscale("log")
    fig.tight_layout(); fig.savefig(OUT / "cox_treatment_forest.png", dpi=150); plt.close(fig)

    s.to_csv(OUT / "cox_treatment_summary.csv", index=False, encoding="utf-8-sig")
    return cph


# ── Dose-response: cycles vs OS ───────────────────────────────────────────────
def cycles_dose_response(df: pd.DataFrame):
    sub = df[df["cycles"].notna() & df["chemo"] & df["os_months"].notna() & (df["os_months"] > 0)].copy()
    print(f"  Cycles analysis: n={len(sub)} chemo patients with cycle data")
    if len(sub) < 30:
        print("  Too few cases for cycles analysis.")
        return

    # Scatter + trend
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(sub["cycles"], sub["os_months"], alpha=0.35, color="steelblue", s=20)
    from scipy.stats import spearmanr
    r, p = spearmanr(sub["cycles"], sub["os_months"])
    z = np.polyfit(sub["cycles"].dropna(), sub.loc[sub["cycles"].notna(),"os_months"], 1)
    xr = np.linspace(sub["cycles"].min(), sub["cycles"].max(), 100)
    axes[0].plot(xr, np.poly1d(z)(xr), "r--", linewidth=2)
    axes[0].set(title=f"Cycles vs OS  (Spearman r={r:.2f}, p={p:.3f})",
                xlabel="Chemotherapy cycles (cumulative)", ylabel="OS (months)")

    # KM by cycle group
    cg = sub.dropna(subset=["cycles_group"])
    groups = ["1–3","4–6",">6"]
    groups = [g for g in groups if (cg["cycles_group"] == g).sum() >= 5]
    if len(groups) >= 2:
        kmf = KaplanMeierFitter()
        for i, g in enumerate(groups):
            s2 = cg[cg["cycles_group"] == g]
            kmf.fit(s2["os_months"], s2["event"], label=f"{g} cycles (n={len(s2)})")
            kmf.plot_survival_function(ax=axes[1], ci_show=True, color=COLORS[i % 10])
        axes[1].set(title="OS by Cycle Count Group", xlabel="Months",
                    ylabel="Survival probability", ylim=(0, 1.05))
        axes[1].legend(fontsize=9)
        if len(groups) == 2:
            g1 = cg[cg["cycles_group"] == groups[0]]
            g2 = cg[cg["cycles_group"] == groups[1]]
            pp = logrank_test(g1["os_months"], g2["os_months"], g1["event"], g2["event"]).p_value
        else:
            pp = multivariate_logrank_test(cg["os_months"], cg["cycles_group"], cg["event"]).p_value
        axes[1].text(0.62, 0.85, f"Log-rank p={pp:.4f}", transform=axes[1].transAxes,
                     fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout(); fig.savefig(OUT / "chemo_cycles_dose_response.png", dpi=150); plt.close(fig)


# ── Chemo timing: lag from diagnosis ──────────────────────────────────────────
def chemo_timing(df: pd.DataFrame):
    sub = df[df["chemo"] & df["chemo_lag_days"].notna() & df["os_months"].notna()].copy()
    print(f"  Chemo timing analysis: n={len(sub)}")
    if len(sub) < 30:
        return
    sub["lag_group"] = pd.cut(sub["chemo_lag_days"], bins=[-1, 30, 60, 120, 365],
                               labels=["≤30d","31–60d","61–120d",">120d"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(sub["chemo_lag_days"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set(title="Days from Diagnosis to Chemo Start",
                xlabel="Days", ylabel="Count")
    axes[0].axvline(sub["chemo_lag_days"].median(), color="red", linestyle="--",
                    label=f"Median {sub['chemo_lag_days'].median():.0f}d")
    axes[0].legend()

    kmf = KaplanMeierFitter()
    groups = [g for g in ["≤30d","31–60d","61–120d",">120d"]
              if (sub["lag_group"]==g).sum() >= 10]
    for i, g in enumerate(groups):
        s = sub[sub["lag_group"]==g]
        kmf.fit(s["os_months"], s["event"], label=f"{g} (n={len(s)})")
        kmf.plot_survival_function(ax=axes[1], ci_show=False, color=COLORS[i % 10])
    axes[1].set(title="OS by Time to Chemo Start", xlabel="Months",
                ylabel="Survival probability", ylim=(0,1.05))
    axes[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "chemo_timing.png", dpi=150); plt.close(fig)
    sub["chemo_lag_days"].describe().to_csv(OUT / "chemo_timing_stats.csv", encoding="utf-8-sig")


# ── Surgery subgroup: margins + LN ────────────────────────────────────────────
def surgery_subgroup(df: pd.DataFrame):
    op = df[df["surg_group"].isin(
        ["Partial esophagectomy","Radical esophagectomy","Endoscopic resection"]
    )].copy()
    print(f"  Surgical subgroup: n={len(op)}")

    # R0 vs R1/R2 (restricted to operated patients)
    op_margin = op[op["margin_label"].isin(
        ["R0 (no residual)","R1 (microscopic)","R2 (macroscopic)"]
    )]
    if len(op_margin) > 20:
        km_plot(op_margin, "margin_label",
                ["R0 (no residual)","R1 (microscopic)","R2 (macroscopic)"],
                "OS by Surgical Margin (Operated Cases)",
                "km_surgery_margin.png")

    # LN dissection extent (operated)
    op_ln = op[op["ln_label"].isin(["No LN dissection","Limited","Regional",
                                     "Extended","Systematic","Radical/3-field"])]
    if len(op_ln) > 20:
        km_plot(op_ln, "ln_label", None, "OS by LN Dissection Extent", "km_ln_extent.png")

    # LN ratio (continuous) scatter
    op_lnr = op[op["ln_ratio"].notna() & op["os_months"].notna() & (op["os_months"]>0)].copy()
    if len(op_lnr) > 20:
        from scipy.stats import spearmanr
        r, p = spearmanr(op_lnr["ln_ratio"], op_lnr["os_months"])
        op_lnr["ln_ratio_group"] = pd.cut(op_lnr["ln_ratio"],
                                           bins=[-0.01, 0, 0.2, 0.5, 1.01],
                                           labels=["LNR=0","LNR 0–0.2","LNR 0.2–0.5","LNR>0.5"])
        fig, axes = plt.subplots(1, 2, figsize=(13,5))
        axes[0].scatter(op_lnr["ln_ratio"], op_lnr["os_months"],
                        alpha=0.4, color="steelblue", s=20)
        axes[0].set(title=f"LN Ratio vs OS (Spearman r={r:.2f}, p={p:.3f})",
                    xlabel="LN Ratio (positive/examined)", ylabel="OS (months)")
        kmf = KaplanMeierFitter()
        groups = [g for g in ["LNR=0","LNR 0–0.2","LNR 0.2–0.5","LNR>0.5"]
                  if (op_lnr["ln_ratio_group"]==g).sum() >= 5]
        for i, g in enumerate(groups):
            s = op_lnr[op_lnr["ln_ratio_group"]==g]
            kmf.fit(s["os_months"], s["event"], label=f"{g} (n={len(s)})")
            kmf.plot_survival_function(ax=axes[1], ci_show=True, color=COLORS[i%10])
        axes[1].set(title="OS by LN Ratio", xlabel="Months",
                    ylabel="Survival probability", ylim=(0,1.05))
        axes[1].legend(fontsize=9)
        fig.tight_layout(); fig.savefig(OUT/"km_ln_ratio.png", dpi=150); plt.close(fig)

    # Minimally invasive vs open (where available)
    op_mis = op[op["mis_label"].isin(["Open","Thoracoscopic","Laparoscopic","Robotic"])].copy()
    print(f"  MIS data available: n={len(op_mis)}")
    if len(op_mis) > 20:
        op_mis["mis_group"] = op_mis["mis_label"].replace(
            {"Thoracoscopic":"Minimally invasive",
             "Laparoscopic":"Minimally invasive",
             "Robotic":"Minimally invasive"})
        km_plot(op_mis, "mis_group", ["Open","Minimally invasive"],
                "OS — Open vs Minimally Invasive Surgery", "km_mis_vs_open.png")

    # Neoadjuvant vs upfront surgery
    op["neoadj"] = op["rt_surg_label"].isin(
        ["Neoadjuvant RT → surgery","Neoadjuvant CCRT → surgery","Neoadjuvant chemo → surgery"])
    op["neoadj_label"] = op["neoadj"].map({True:"Neoadjuvant therapy", False:"Upfront surgery"})
    if op["neoadj_label"].value_counts().min() >= 10:
        km_plot(op, "neoadj_label", ["Neoadjuvant therapy","Upfront surgery"],
                "OS — Neoadjuvant vs Upfront Surgery", "km_neoadjuvant.png")

    return op


# ── Summary table ──────────────────────────────────────────────────────────────
def summary_table(df: pd.DataFrame):
    rows = []
    op = df[df["surg_group"].isin(["Partial esophagectomy","Radical esophagectomy",
                                    "Endoscopic resection"])]

    def s(col, grp=None):
        sub = grp if grp is not None else df
        n = sub[col].notna().sum()
        return n

    rows.append(("=== SURGERY ===", "", ""))
    for g, n in df["surg_group"].value_counts().items():
        med = df.loc[df["surg_group"]==g, "os_months"].median()
        rows.append((f"  {g}", f"n={n} ({100*n/len(df):.1f}%)",
                     f"Median OS={med:.1f} mo" if pd.notna(med) else ""))

    rows.append(("Surgical margin (operated)",  "", ""))
    for g, n in op["margin_label"].value_counts().items():
        med = op.loc[op["margin_label"]==g, "os_months"].median()
        rows.append((f"  {g}", f"n={n}", f"Median OS={med:.1f} mo" if pd.notna(med) else ""))

    rows.append(("LN dissection (operated)", "", ""))
    for g, n in op["ln_label"].value_counts().items():
        rows.append((f"  {g}", f"n={n}", ""))

    rows.append(("=== CHEMOTHERAPY ===", "", ""))
    rows.append(("CCRT",
                 f"n={int((df['ccrt']=='CCRT').sum())} ({100*(df['ccrt']=='CCRT').mean():.1f}%)",
                 f"Median OS={df.loc[df['ccrt']=='CCRT','os_months'].median():.1f} mo"))
    rows.append(("No CCRT",
                 f"n={int((df['ccrt']=='No CCRT').sum())} ({100*(df['ccrt']=='No CCRT').mean():.1f}%)",
                 f"Median OS={df.loc[df['ccrt']=='No CCRT','os_months'].median():.1f} mo"))
    for g, n in df["chemo_hosp_group"].value_counts().items():
        med = df.loc[df["chemo_hosp_group"]==g, "os_months"].median()
        rows.append((f"  Regimen: {g}", f"n={n}", f"Median OS={med:.1f} mo" if pd.notna(med) else ""))

    cyc = df["cycles"].dropna()
    rows.append(("Cycles (cumulative dose proxy)",
                 f"n={len(cyc)} with data",
                 f"Median={cyc.median():.1f}, Mean={cyc.mean():.1f}"))

    t = pd.DataFrame(rows, columns=["Variable","Value","OS"])
    t.to_csv(OUT / "chemo_surgery_summary.csv", index=False, encoding="utf-8-sig")
    return t


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load()
    print(f"  n={len(df)}, events={df['event'].sum()}, OS-evaluable={df['os_months'].notna().sum()}")

    print("\n=== SURGERY ANALYSIS ===")
    surg_order = ["No surgery","Endoscopic resection","Partial esophagectomy",
                  "Radical esophagectomy","Other/unknown"]
    km_plot(df, "surg_group", surg_order,
            "OS by Surgery Type", "km_surgery_type.png")

    km_plot(df, "rt_surg_label",
            ["Surgery only","Neoadjuvant RT → surgery","Neoadjuvant CCRT → surgery",
             "Surgery → adjuvant RT","Surgery → adjuvant CCRT"],
            "OS by RT-Surgery Sequence (Operated Only)",
            "km_rt_surgery_sequence.png",
            min_n=10)

    op_df = surgery_subgroup(df)

    print("\n=== CHEMO ANALYSIS ===")
    km_plot(df, "ccrt", ["No CCRT","CCRT"],
            "OS — CCRT vs No CCRT", "km_ccrt.png")

    km_plot(df, "chemo_hosp_group",
            ["No chemo","Single agent","Multi-agent (no RT)",
             "CCRT (concurrent)","Multi-agent + targeted/immuno"],
            "OS by Chemo Regimen", "km_chemo_regimen.png", min_n=15)

    km_plot(df[df["chemo"]], "chemo_label",
            ["Single agent","Multi-agent","Multi-agent + other"],
            "OS by Chemo Method (Chemo patients only)", "km_chemo_method.png", min_n=15)

    cycles_dose_response(df)
    chemo_timing(df)

    print("\n=== COMBINED TREATMENT COX MODEL ===")
    cox_treatment(df)

    print("\n=== SUMMARY TABLE ===")
    t = summary_table(df)
    print(t.to_string(index=False))

    print(f"\nDone. Results in {OUT}")
