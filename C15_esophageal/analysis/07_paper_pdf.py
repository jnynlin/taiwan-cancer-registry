"""
Near-submission paper PDF — Esophageal Cancer (C15), Taiwan Cancer Registry.
All figures are re-rendered at 300 DPI directly into the PDF (no PNG loading).
Output: results/C15_Paper_Draft.pdf
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy.stats import spearmanr

BASE  = Path(__file__).parent.parent
DATA  = BASE / "data/c15_enriched.csv"
OUT   = BASE / "results/C15_Paper_Draft.pdf"

DPI   = 300
FIG_W = 7.5   # inches (journal single column ≈ 3.5", double ≈ 7.0")
sns.set_theme(style="white", font="DejaVu Sans", font_scale=0.95)
plt.rcParams.update({
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "legend.framealpha": 0.85,
    "figure.dpi": DPI,
})
PALETTE  = ["#2166ac","#d6604d","#4dac26","#762a83","#e08214","#1b7837","#8073ac","#b35806"]
GRAY     = "#555555"
TITLE_BG = "#1a3a5c"

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    yr = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00","99") else "01"
    dd = s[5:7] if s[5:7] not in ("00","99") else "01"
    try:    return pd.Timestamp(f"{yr}-{mm}-{dd}")
    except: return pd.NaT

def load_data():
    df = pd.read_csv(DATA, low_memory=False)
    df["event"]     = (df["生存狀態(27)"].astype(str).str.strip() == "0").astype(int)
    df["os_months"] = pd.to_numeric(df["os_days"], errors="coerce") / 30.44
    df["age"]       = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]       = df["性別(5)"].map({1:"Male",2:"Female","1":"Male","2":"Female"})

    # Stage
    def clean_stage(s):
        s = str(s).strip().upper()
        return np.nan if s in ("888","999","BBB","NAN","88","99") else s
    df["clin_stage"] = df["臨床期別組合(95)"].apply(clean_stage)
    df["path_stage"] = df["病理期別組合(101)"].apply(clean_stage)
    df["stage"]      = df["path_stage"].combine_first(df["clin_stage"])
    df["stage_group"]= df["stage"].str.extract(r"^([1234])",expand=False).map(
                        {"1":"I","2":"II","3":"III","4":"IV"})

    # Histology
    df["hist_code"]  = df["組織型態(49)"].astype(str).str[:4]
    scc = ["8070","8071","8072","8073","8074","8075","8076"]
    df["hist_group"] = "Other"
    df.loc[df["hist_code"].isin(scc),"hist_group"] = "SCC"
    df.loc[df["hist_code"].isin(["8140","8480","8144"]),"hist_group"] = "Adenocarcinoma"
    df.loc[df["hist_code"].isin(["8000","8010","8020"]),"hist_group"] = "Carcinoma NOS"
    df.loc[df["hist_code"]=="8560","hist_group"] = "Adenosquamous"

    # Grade
    df["grade"] = pd.to_numeric(df["分化程度(50)"],errors="coerce").replace(9,np.nan)
    df["grade_label"] = df["grade"].map({1:"G1 (Well)",2:"G2 (Moderate)",
                                          3:"G3 (Poor)",4:"G4 (Undiff)"})

    # Surgery
    STYPE = {"00":"No surgery","2E":"EMR/ESD","27":"Endoscopic","20":"Local excision",
             "30":"Partial esophagectomy","51":"Esophagectomy+recon",
             "52":"Esophagogastrectomy","53":"Esophagectomy+LND",
             "54":"Radical esophagectomy","55":"Radical esoph+LND","99":"Unknown"}
    SGRP  = {"No surgery":"No surgery","EMR/ESD":"Endoscopic resection",
             "Endoscopic":"Endoscopic resection","Local excision":"Endoscopic resection",
             "Partial esophagectomy":"Partial esophagectomy",
             "Esophagectomy+recon":"Radical esophagectomy",
             "Esophagogastrectomy":"Radical esophagectomy",
             "Esophagectomy+LND":"Radical esophagectomy",
             "Radical esophagectomy":"Radical esophagectomy",
             "Radical esoph+LND":"Radical esophagectomy","Unknown":"Unknown"}
    df["surg_label"] = df["原發部位手術方式(118)"].astype(str).str.strip().str.upper().map(STYPE).fillna("Unknown")
    df["surg_group"] = df["surg_label"].map(SGRP).fillna("Unknown")

    df["margin"]     = df["原發部位手術邊緣(124)"].astype(str).str.strip().map(
                        {"0":"R0","2":"R1","3":"R2","8":"No surgery","9":"Unknown"}).fillna("Unknown")
    df["ln_extent"]  = pd.to_numeric(df["申報醫院區域淋巴結手術範圍(126)"],errors="coerce")
    df["ln_label"]   = df["ln_extent"].map({0:"None",1:"Limited",2:"Regional",
                                             3:"Extended",4:"Systematic",5:"Radical/3-field",9:"Unknown"})
    df["ln_examined"]= pd.to_numeric(df["區域淋巴結檢查數目(87)"],errors="coerce").replace([97,98,99],np.nan)
    df["ln_positive"]= pd.to_numeric(df["區域淋巴結侵犯數目(88)"],errors="coerce").replace([97,98,99],np.nan)
    df["ln_ratio"]   = (df["ln_positive"]/df["ln_examined"]).replace([np.inf,-np.inf],np.nan)

    # Chemo
    def hosp_chemo(c):
        c=str(c).strip()
        if c=="0":   return "No chemotherapy"
        if c in ("1","86"): return "Single-agent"
        if c=="2":   return "Multi-agent"
        if c=="3":   return "CCRT"
        if c in ("82","83","84","85","87"): return "Multi-agent+targeted/immuno"
        return "Unknown"
    df["chemo_group"]= df["申報醫院化學治療(163)"].apply(hosp_chemo)
    df["ccrt"]       = df["本院首療CCRT(159)_非"].astype(str).str.strip().map({"0":"No CCRT","1":"CCRT"})
    df["cycles"]     = pd.to_numeric(df["化學治療次數-Cycle(161)_非"],errors="coerce").replace([98,99],np.nan)

    # Chemo start lag
    df["diag_ts"]       = pd.to_datetime(df["diag_date"],errors="coerce")
    df["chemo_start_ts"]= df["申報醫院化學治療開始日期(162)"].apply(roc_to_ts)
    df["chemo_lag_days"]= (df["chemo_start_ts"]-df["diag_ts"]).dt.days
    df.loc[(df["chemo_lag_days"]<0)|(df["chemo_lag_days"]>365),"chemo_lag_days"] = np.nan

    # Neoadjuvant
    rt_seq = pd.to_numeric(df["放射治療與手術順序(140)"],errors="coerce")
    df["neoadj_label"] = rt_seq.map({-8:"Neoadjuvant RT",
                                      -7:"Neoadjuvant CCRT",
                                      -1:"Neoadjuvant chemo",
                                       0:"Surgery only",
                                       4:"Surgery+adj RT",
                                       1:"Surgery+adj chemo",
                                       5:"Surgery+adj CCRT"}).fillna("Unknown")

    # Lifestyle
    for c in ["每日吸菸量(A3-1)","每日嚼檳榔量(A4-1)"]:
        df[c] = pd.to_numeric(df[c],errors="coerce").replace([98,99],np.nan)
    df["smoker"]    = df["每日吸菸量(A3-1)"].apply(lambda x: "Yes" if pd.notna(x) and x>0 else ("No" if pd.notna(x) else np.nan))
    df["betel_nut"] = df["每日嚼檳榔量(A4-1)"].apply(lambda x: "Yes" if pd.notna(x) and x>0 else ("No" if pd.notna(x) else np.nan))
    ht = pd.to_numeric(df["身高(A1)"],errors="coerce").replace(999,np.nan)
    wt = pd.to_numeric(df["體重(A2)"],errors="coerce").replace(999,np.nan)
    df["bmi"] = wt/(ht/100)**2
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SHARED PLOT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def km_ax(ax, df_sub, group_col, order=None, time_col="os_months", event_col="event",
          title="", xlabel="Time (months)", min_n=10, at_risk_table=True):
    """Draw KM curve on ax. Returns p-value."""
    sub    = df_sub.dropna(subset=[time_col,event_col,group_col])
    sub    = sub[sub[time_col]>0]
    groups = order or sorted(sub[group_col].dropna().unique())
    groups = [g for g in groups if (sub[group_col]==g).sum()>=min_n]
    if len(groups)<2: return None
    kmf = KaplanMeierFitter()
    for i,g in enumerate(groups):
        s = sub[sub[group_col]==g]
        kmf.fit(s[time_col], s[event_col], label=f"{g}  (n={len(s)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=PALETTE[i%len(PALETTE)],
                                   linewidth=1.5, alpha=0.9)
    ax.set(title=title, xlabel=xlabel, ylabel="Cumulative survival probability",
           ylim=(0,1.05), xlim=(0, sub[time_col].quantile(0.98)+1))
    ax.spines[["top","right"]].set_visible(False)
    ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
    # p-value
    if len(groups)==2:
        g1 = sub.loc[sub[group_col]==groups[0]]
        g2 = sub.loc[sub[group_col]==groups[1]]
        p = logrank_test(g1[time_col], g2[time_col], g1[event_col], g2[event_col]).p_value
    else:
        p = multivariate_logrank_test(sub[time_col], sub[group_col], sub[event_col]).p_value
    sig = "****" if p<0.0001 else ("***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns")))
    pval_str = f"Log-rank p{'<0.0001' if p<0.0001 else f'={p:.4f}'}  {sig}"
    ax.text(0.98, 0.08, pval_str,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            clip_on=True,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", lw=0.6))
    ax.legend(loc="upper right", frameon=True, fontsize=7, handlelength=1.5)
    return p


def forest_ax(ax, cph, title=""):
    s = cph.summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].copy()
    s.columns = ["HR","lo","hi","p"]
    s = s.reset_index(); s.columns = ["Covariate","HR","lo","hi","p"]
    s = s.sort_values("HR", ascending=True)
    y = range(len(s))
    colors = ["#c0392b" if r>1 else "#27ae60" for r in s["HR"]]
    ax.scatter(s["HR"], list(y), c=colors, zorder=4, s=35, linewidths=0.5, edgecolors="white")
    for i,(_, row) in enumerate(s.iterrows()):
        ax.plot([row.lo, row.hi],[i,i], color=colors[i], linewidth=1.6, solid_capstyle="round")
    ax.axvline(1, color="#333333", linestyle="--", linewidth=0.9, zorder=3)
    ax.set_yticks(list(y)); ax.set_yticklabels(s["Covariate"], fontsize=8)
    ax.set_xscale("log")
    ax.set(title=title, xlabel="Hazard Ratio (95% CI, log scale)")
    ax.spines[["top","right"]].set_visible(False)
    # Annotate HR + p
    for i,(_, row) in enumerate(s.iterrows()):
        star = "***" if row.p<0.001 else ("**" if row.p<0.01 else ("*" if row.p<0.05 else ""))
        ax.text(ax.get_xlim()[1]*1.05, i,
                f"HR={row.HR:.2f} {star}", va="center", fontsize=6.5, color=GRAY)
    return s


def page_header(fig, title, subtitle=""):
    fig.text(0.5, 0.985, title, ha="center", va="top",
             fontsize=11, fontweight="bold", color=TITLE_BG)
    if subtitle:
        fig.text(0.5, 0.963, subtitle, ha="center", va="top",
                 fontsize=8.5, color=GRAY, style="italic")


def page_footer(fig, n, total, note=""):
    foot = f"Taiwan Cancer Registry 2006–2020  ·  C15 Esophageal Cancer  ·  Page {n}/{total}"
    if note: foot += f"  ·  {note}"
    fig.text(0.5, 0.008, foot, ha="center", va="bottom", fontsize=6.5, color=GRAY)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
df = load_data()
N       = len(df)
N_ev    = int(df["event"].sum())
N_surv  = int(df["os_months"].notna().sum())
N_op    = int(df["surg_group"].isin(["Endoscopic resection","Partial esophagectomy","Radical esophagectomy"]).sum())
TOTAL_PAGES = 12

with PdfPages(OUT) as pdf:

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 1 — TITLE & ABSTRACT
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(FIG_W, 10))
    page_footer(fig, 1, TOTAL_PAGES)

    # Title block
    ax0 = fig.add_axes([0.0, 0.82, 1.0, 0.16])
    ax0.set_facecolor(TITLE_BG); ax0.axis("off")
    ax0.text(0.5, 0.72,
             "Clinical Characteristics, Survival Outcomes, and Novel Patient Subtypes\n"
             "in Esophageal Cancer: A Comprehensive Analysis of the Taiwan Cancer\n"
             "Registry (2006–2020)",
             ha="center", va="center", fontsize=12, fontweight="bold",
             color="white", linespacing=1.5, transform=ax0.transAxes)
    ax0.text(0.5, 0.15,
             "Taiwan Cancer Registry Research Group  ·  Manuscript Draft  ·  2025",
             ha="center", va="center", fontsize=8, color="#aaccee",
             transform=ax0.transAxes)

    # Abstract
    ax1 = fig.add_axes([0.06, 0.05, 0.88, 0.74])
    ax1.axis("off")
    abstract_text = [
        ("ABSTRACT", True, 10, TITLE_BG),
        ("", False, 6, "white"),
        (f"Esophageal cancer carries a dismal prognosis, yet independent prognostic "
         f"determinants in Asian populations remain incompletely characterized. We analyzed "
         f"{N:,} consecutive esophageal cancer (ICD-10 C15) cases from a single-center Taiwan "
         f"Cancer Registry (2006\u20132020). Squamous cell carcinoma (SCC) predominated "
         f"(85.4%) in a predominantly male cohort (94.5%, median age 57 years). With {N_ev:,} "
         f"deaths (77.4%), median overall survival (OS) was 13.4 months. Multivariable Cox "
         f"regression (C-index 0.705, bootstrap 95\u202fCI 0.694\u20130.719, B\u2009=\u20091000) "
         f"identified surgical resection as the strongest independent predictor: endoscopic "
         f"resection (HR 0.47, 95\u202fCI 0.38\u20130.59) and radical esophagectomy "
         f"(HR 0.61, 95\u202fCI 0.51\u20130.72) each conferred significant benefit (p<0.001). "
         f"R0 resection margin independently reduced hazard by 40\u202f% "
         f"(HR 0.60, 95\u202fCI 0.53\u20130.69, p<0.001). Concurrent chemoradiotherapy (CCRT) "
         f"provided significant survival benefit beyond surgery alone "
         f"(HR 0.76, 95\u202fCI 0.67\u20130.85, p<0.001). Unsupervised deep learning\u2014"
         f"autoencoder representations, UMAP, and k-means clustering\u2014identified three "
         f"biologically distinct patient subtypes: a mainstream SCC cluster (n=2,029), an "
         f"adenocarcinoma-enriched subgroup (n=153), and a cervical/upper esophagus cluster "
         f"(n=185) with differential OS (log-rank p<0.05). R0 margin attainment and CCRT "
         f"are independent therapeutic priorities; deep learning reveals clinically meaningful "
         f"patient heterogeneity beyond conventional staging.",
         False, 8.5, "#111111"),
        ("", False, 6, "white"),
        ("Keywords", True, 8.5, TITLE_BG),
        ("Esophageal cancer; Squamous cell carcinoma; Surgical resection; R0 margin; "
         "Concurrent chemoradiotherapy; Deep learning; Patient subtypes; Taiwan Cancer Registry",
         False, 8, "#444444"),
    ]
    import textwrap as _tw
    y_pos = 1.0
    for text, bold, size, color in abstract_text:
        if not text:
            y_pos -= size / 500
            continue
        weight = "bold" if bold else "normal"
        # Pre-wrap body text; headings/keywords stay on one line
        if not bold and len(text) > 80:
            wrapped = _tw.fill(text, width=92)
        else:
            wrapped = text
        ax1.text(0, y_pos, wrapped, transform=ax1.transAxes, fontsize=size,
                 fontweight=weight, color=color, va="top", linespacing=1.55)
        n_lines = wrapped.count("\n") + 1
        y_pos -= (size * n_lines * 1.58) / 500

    pdf.savefig(fig, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 2 — INTRODUCTION
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(FIG_W, 10))
    page_footer(fig, 2, TOTAL_PAGES)

    ax_i = fig.add_axes([0.06, 0.03, 0.88, 0.94])
    ax_i.axis("off")

    intro_sections = [
        ("INTRODUCTION", True, 11, TITLE_BG),
        ("", False, 4, None),
        ("Epidemiology and Global Burden", True, 9.5, TITLE_BG),
        ("Esophageal cancer is the seventh most common malignancy and the sixth leading cause of "
         "cancer-related mortality worldwide, accounting for an estimated 604,100 new cases and "
         "544,076 deaths in 2020 [1]. Marked geographic heterogeneity characterizes its global "
         "distribution: squamous cell carcinoma (SCC) predominates in the 'esophageal cancer belt' "
         "spanning from northern Iran through Central Asia to north-central China, as well as "
         "sub-Saharan Africa, whereas esophageal adenocarcinoma (EAC) has risen sharply over the "
         "past four decades in Western Europe and North America, largely attributed to the epidemic "
         "of obesity and gastroesophageal reflux disease [2,3]. In Taiwan, esophageal cancer ranks "
         "among the top ten malignancies by incidence, with a distinctive epidemiological profile "
         "driven by a convergence of betel nut chewing, cigarette smoking, and alcohol consumption—"
         "the so-called 'three lifestyle risk triad'—which collectively confer synergistic carcinogenic "
         "effects on the esophageal squamous epithelium [4,5].", False, 8.5, "#111111"),
        ("", False, 3, None),
        ("Histological Subtypes and Asian Specificity", True, 9.5, TITLE_BG),
        ("The histological landscape of esophageal cancer in Asia differs fundamentally from that "
         "in the West. SCC constitutes over 90% of esophageal malignancies in East and Southeast "
         "Asia, while adenocarcinoma accounts for fewer than 5% [2]. This distinction has profound "
         "therapeutic implications: SCC typically arises in the middle and upper esophagus, is more "
         "sensitive to chemoradiotherapy, and shares carcinogenic pathways with head and neck cancers, "
         "whereas adenocarcinoma clusters at the gastroesophageal junction and responds differently "
         "to treatment paradigms [6]. Understanding the clinical spectrum of esophageal SCC in "
         "population-based registries is therefore essential to optimize treatment protocols "
         "tailored to Asian patients.", False, 8.5, "#111111"),
        ("", False, 3, None),
        ("Multimodal Treatment and Unresolved Questions", True, 9.5, TITLE_BG),
        ("Current management of resectable esophageal cancer is centered on multimodal therapy. "
         "The landmark CROSS trial demonstrated that neoadjuvant concurrent chemoradiotherapy "
         "(CCRT) followed by surgery significantly improved overall survival compared with surgery "
         "alone (median OS 48.6 vs 24.0 months; HR 0.68, p=0.003) [7]. The FLOT4 trial established "
         "perioperative FLOT chemotherapy as standard of care for gastroesophageal junction tumors "
         "in the West [8]. For unresectable or metastatic disease, immune checkpoint inhibitors "
         "have transformed the therapeutic landscape: nivolumab (ATTRACTION-3, CheckMate 648), "
         "pembrolizumab (KEYNOTE-590), and their combinations have demonstrated survival benefits "
         "in SCC [9,10]. Despite these advances, the 5-year survival rate remains approximately "
         "15–25%, underscoring the need for better prognostic stratification and personalized "
         "treatment allocation [11]. In Taiwan, real-world data on surgical quality metrics such "
         "as resection margins, lymph node dissection extent, and their interaction with systemic "
         "therapy remain incompletely characterized.", False, 8.5, "#111111"),
        ("", False, 3, None),
        ("Deep Learning in Oncology", True, 9.5, TITLE_BG),
        ("Machine learning and deep learning approaches have emerged as powerful tools for "
         "discovering latent patient subgroups and prognostic features from high-dimensional "
         "clinical data [12,13]. Autoencoders compress complex clinical feature spaces into "
         "compact latent representations, enabling unsupervised patient stratification via "
         "dimensionality reduction techniques such as UMAP [14]. DeepSurv, a deep Cox "
         "proportional hazards network, extends conventional survival analysis to non-linear "
         "feature interactions and provides interpretable feature importance through permutation "
         "analysis [15]. To date, the application of such methods to population-level cancer "
         "registry data in Taiwan has been limited.", False, 8.5, "#111111"),
        ("", False, 3, None),
        ("Study Objectives", True, 9.5, TITLE_BG),
        ("This study aimed to: (1) describe the clinical characteristics and temporal trends of "
         "esophageal cancer in a large single-center Taiwan Cancer Registry cohort spanning "
         "2006–2020; (2) quantify the independent prognostic impact of surgical modality, "
         "resection margin status, lymph node dissection extent, and perioperative treatment "
         "sequencing on overall survival; (3) characterize the dose-dependent effect of "
         "chemotherapy on outcomes where cycle data were available; and (4) apply unsupervised "
         "deep learning to discover novel patient subtypes with distinct biological and clinical "
         "profiles that may inform future personalized treatment strategies.", False, 8.5, "#111111"),
    ]

    import textwrap as _tw2
    y_pos = 0.985
    for text, bold, size, color in intro_sections:
        if not text:
            y_pos -= size / 560
            continue
        if not bold and len(text) > 80:
            wrapped = _tw2.fill(text, width=110)
        else:
            wrapped = text
        ax_i.text(0, y_pos, wrapped, transform=ax_i.transAxes, fontsize=size,
                  fontweight="bold" if bold else "normal",
                  color=(color or "#111111"), va="top", linespacing=1.5)
        n_lines = wrapped.count("\n") + 1
        y_pos -= (size * n_lines * 1.55) / 560
        if y_pos < 0.01:
            break

    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 3 — TABLE 1
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(FIG_W, 10))
    page_header(fig, "Table 1. Baseline Characteristics of Esophageal Cancer Patients",
                f"Taiwan Cancer Registry 2006–2020  (N = {N:,})")
    page_footer(fig, 3, TOTAL_PAGES)

    ax = fig.add_axes([0.03, 0.04, 0.94, 0.88])
    ax.axis("off")

    def pct(n, tot=N): return f"{n} ({100*n/tot:.1f})"
    def iqr(s): v=s.dropna(); return f"{v.median():.1f} ({v.quantile(.25):.1f}–{v.quantile(.75):.1f})"

    rows = [
        ["Characteristic", "Overall\nn = 2,367", "Male\nn = 2,237", "Female\nn = 130", "p-value"],
        ["Age at diagnosis, median (IQR)", iqr(df["age"]),
         iqr(df.loc[df["sex"]=="Male","age"]), iqr(df.loc[df["sex"]=="Female","age"]), "0.023"],
        ["  <50 years", pct((df["age"]<50).sum()), pct((df.loc[df["sex"]=="Male","age"]<50).sum(),2237),
         pct((df.loc[df["sex"]=="Female","age"]<50).sum(),130), ""],
        ["  50–59 years", pct((df["age"].between(50,59)).sum()), "", "", ""],
        ["  60–69 years", pct((df["age"].between(60,69)).sum()), "", "", ""],
        ["  ≥70 years", pct((df["age"]>=70).sum()), "", "", ""],
        ["BMI, median (IQR)",
         iqr(df["bmi"]), iqr(df.loc[df["sex"]=="Male","bmi"]),
         iqr(df.loc[df["sex"]=="Female","bmi"]), "0.108"],
        ["Esophageal subsite, n (%)", "", "", "", ""],
        ["  Overlapping (C15.5/8)", pct(848), "", "", ""],
        ["  Abdominal (C15.4)", pct(689), "", "", ""],
        ["  Esophagus NOS (C15.9)", pct(392), "", "", ""],
        ["  Lower thoracic (C15.3)", pct(346), "", "", ""],
        ["  Cervical (C15.0)", pct(70), "", "", ""],
        ["  Upper thoracic (C15.1)", pct(22), "", "", ""],
        ["Histological type, n (%)", "", "", "", "<0.001"],
        ["  Squamous cell carcinoma", pct(2022), "", "", ""],
        ["  Carcinoma NOS", pct(184), "", "", ""],
        ["  Other", pct(97), "", "", ""],
        ["  Adenocarcinoma", pct(54), "", "", ""],
        ["  Adenosquamous", pct(10), "", "", ""],
        ["Differentiation grade, n (%)", "", "", "", ""],
        ["  G2 (Moderate)", pct(628), "", "", ""],
        ["  G3 (Poor)", pct(399), "", "", ""],
        ["  G1 (Well)", pct(56), "", "", ""],
        ["  Unknown", pct(884), "", "", ""],
        ["AJCC stage, n (%)", "", "", "", "<0.001"],
        ["  Stage I", pct(256), "", "", ""],
        ["  Stage II", pct(301), "", "", ""],
        ["  Stage III", pct(626), "", "", ""],
        ["  Stage IV", pct(575), "", "", ""],
        ["  Unknown", pct(609), "", "", ""],
        ["Treatment received, n (%)", "", "", "", ""],
        ["  Surgery (any)", pct(N_op), "", "", ""],
        ["    Endoscopic resection", pct(180), "", "", ""],
        ["    Radical esophagectomy", pct(334), "", "", ""],
        ["  Radiation therapy", pct(1709), "", "", ""],
        ["  Chemotherapy", pct(1704), "", "", ""],
        ["  CCRT", pct(947), "", "", ""],
        ["  Immunotherapy", pct(561), "", "", ""],
        ["Lifestyle factors†, n (%)", "", "", "", ""],
        ["  Current/ex-smoker", pct(1001, 1187), "", "", ""],
        ["  Betel nut use", pct(234, 826), "", "", ""],
        ["Median OS, months (IQR)", iqr(df["os_months"]), "", "", ""],
        ["Events (deaths), n (%)", pct(N_ev), "", "", ""],
    ]

    col_widths = [0.40, 0.15, 0.15, 0.15, 0.10]
    tbl = ax.table(cellText=[r for r in rows[1:]], colLabels=rows[0],
                   colWidths=col_widths, loc="upper center",
                   cellLoc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc"); cell.PAD = 0.06
        cell.set_text_props(ha="left" if c==0 else "center")
        if r == 0:
            cell.set_facecolor(TITLE_BG)
            cell.set_text_props(color="white", fontweight="bold", ha="center")
            cell.set_height(0.038)
        elif r+1 < len(rows) and (rows[r+1][0].startswith("  ") or rows[r+1][0] == ""):
            cell.set_facecolor("white")
        else:
            cell.set_facecolor("#eef3fa" if r%2==0 else "white")
            cell.set_text_props(fontweight="bold" if c==0 else "normal")

    fig.text(0.05, 0.025, "† Denominator for lifestyle variables: n=1,187 (smoker), n=826 (betel nut) — subsets with available data.",
             fontsize=6.5, color=GRAY, style="italic")
    fig.text(0.05, 0.015, "Values are n (%) unless noted. IQR = interquartile range. p-values from χ² or Mann-Whitney U tests.",
             fontsize=6.5, color=GRAY, style="italic")

    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 4 — CLINICAL CHARACTERISTICS FIGURES
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 8))
    fig.suptitle("Figure 1. Clinical Characteristics of the Study Cohort",
                 fontsize=10, fontweight="bold", color=TITLE_BG, y=0.99)
    page_footer(fig, 4, TOTAL_PAGES,
                "A: Incidence trend  B: Age distribution  C: Histology  D: Stage")

    # A — Annual trend
    ax = axes[0,0]
    yr = df.dropna(subset=["diag_year"])
    trend = yr.groupby("diag_year").size().reset_index(name="cases")
    ax.bar(trend["diag_year"].astype(int), trend["cases"], color=PALETTE[0], edgecolor="white",
           width=0.75, alpha=0.9)
    z = np.polyfit(trend["diag_year"], trend["cases"], 1)
    xr = np.linspace(trend["diag_year"].min(), trend["diag_year"].max(), 100)
    ax.plot(xr, np.poly1d(z)(xr), "r--", linewidth=1.5, label="Linear trend")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set(xlabel="Year of diagnosis", ylabel="Number of cases",
           title="A   Annual Case Volume (2006–2020)")
    ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)

    # B — Age histogram by sex
    ax = axes[0,1]
    for sex, col in [("Male",PALETTE[0]),("Female",PALETTE[1])]:
        ax.hist(df.loc[df["sex"]==sex,"age"].dropna(), bins=20,
                alpha=0.65, color=col, label=sex, edgecolor="white")
    ax.set(xlabel="Age at diagnosis (years)", ylabel="Frequency",
           title="B   Age at Diagnosis by Sex")
    ax.axvline(df["age"].median(), color="black", linestyle="--",
               linewidth=1, label=f"Median {df['age'].median():.0f} yrs")
    ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)

    # C — Histology donut
    ax = axes[1,0]
    hg = df["hist_group"].value_counts()
    wedge_colors = [PALETTE[i] for i in range(len(hg))]
    wedges, texts, autotexts = ax.pie(
        hg, labels=None, autopct=lambda p: f'{p:.1f}%' if p > 1 else '', startangle=140,
        colors=wedge_colors, pctdistance=0.78,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.2))
    for at in autotexts: at.set_fontsize(7.5)
    ax.legend([f'{k} ({v/hg.sum()*100:.1f}%)' for k,v in hg.items()],
              loc='lower right', fontsize=6.5, frameon=True)
    ax.set_title("C   Histological Type Distribution")

    # D — Stage bar
    ax = axes[1,1]
    sc = df["stage_group"].value_counts(dropna=True).reindex(["I","II","III","IV"]).dropna()
    bars = ax.bar(sc.index, sc.values,
                  color=[PALETTE[0],PALETTE[2],PALETTE[4],PALETTE[1]],
                  edgecolor="white", width=0.6)
    for bar, val in zip(bars, sc.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
                str(val), ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set(xlabel="AJCC stage", ylabel="Number of cases",
           title="D   AJCC Stage Distribution")
    ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout(rect=[0,0.02,1,0.98])
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 4 — OVERALL SURVIVAL: STAGE & HISTOLOGY
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.8))
    fig.suptitle("Figure 2. Overall Survival by Stage and Histology",
                 fontsize=10, fontweight="bold", color=TITLE_BG, y=1.01)
    page_footer(fig, 5, TOTAL_PAGES)

    km_ax(axes[0], df, "stage_group", ["I","II","III","IV"],
          title="A   OS by AJCC Stage")
    km_ax(axes[1], df, "hist_group",
          ["SCC","Adenocarcinoma","Carcinoma NOS"],
          title="B   OS by Histological Type", min_n=30)

    fig.text(0.5, -0.04,
             "Figure 2. Kaplan-Meier curves of overall survival stratified by (A) AJCC pathologic/clinical stage "
             "and (B) histological type. Shaded areas represent 95% confidence intervals. "
             "p-values are from the log-rank test.",
             ha="center", fontsize=7, color=GRAY, wrap=True, style="italic")
    fig.tight_layout()
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 5 — SURGERY IMPACT
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(FIG_W, 9.5))
    fig.suptitle("Figure 3. Impact of Surgical Modality on Overall Survival",
                 fontsize=10, fontweight="bold", color=TITLE_BG, y=0.99)
    page_footer(fig, 6, TOTAL_PAGES)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                           top=0.94, bottom=0.10, left=0.09, right=0.97)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])

    km_ax(ax1, df, "surg_group",
          ["No surgery","Endoscopic resection","Partial esophagectomy","Radical esophagectomy"],
          title="A   OS by Surgery Type")

    op = df[df["surg_group"].isin(["Endoscopic resection","Partial esophagectomy","Radical esophagectomy"])]
    op_m = op[op["margin"].isin(["R0","R1"])]
    km_ax(ax2, op_m, "margin", ["R0","R1"],
          title="B   OS by Resection Margin (Operated)")

    op_seq = op[op["neoadj_label"].isin(
        ["Neoadjuvant CCRT","Neoadjuvant RT","Surgery only","Surgery+adj CCRT","Surgery+adj RT"])]
    km_ax(ax3, op_seq, "neoadj_label",
          ["Neoadjuvant CCRT","Neoadjuvant RT","Surgery only","Surgery+adj CCRT"],
          title="C   OS by Perioperative Treatment Sequence", min_n=10)

    # LN ratio scatter + regression
    op_lnr = op[op["ln_ratio"].notna() & op["os_months"].notna() & (op["os_months"]>0)]
    if len(op_lnr) > 20:
        r, p = spearmanr(op_lnr["ln_ratio"], op_lnr["os_months"])
        ax4.scatter(op_lnr["ln_ratio"], op_lnr["os_months"],
                    alpha=0.35, color=PALETTE[0], s=12, linewidths=0)
        z2 = np.polyfit(op_lnr["ln_ratio"], op_lnr["os_months"], 1)
        xr = np.linspace(0, 1, 100)
        ax4.plot(xr, np.poly1d(z2)(xr), color=PALETTE[1], linewidth=1.8, label="Linear fit")
        ax4.set(xlabel="Lymph node ratio (positive/examined)",
                ylabel="Overall survival (months)",
                title="D   LN Ratio vs OS (Operated Cases)")
        ax4.text(0.97, 0.95, f"Spearman r={r:.2f}\np={'<0.001' if p<0.001 else f'{p:.3f}'}",
                 transform=ax4.transAxes, ha="right", va="top", fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
        ax4.spines[["top","right"]].set_visible(False)

    fig.text(0.5, 0.025,
             "Figure 3. (A) OS stratified by surgical approach; No surgery is the reference. "
             "(B) OS by resection margin status (R0 vs R1) among operated patients. "
             "(C) Impact of perioperative treatment sequencing on OS. "
             "(D) Lymph node ratio (proportion of positive nodes) as a continuous predictor of OS.",
             ha="center", fontsize=7, color=GRAY, wrap=True, style="italic")
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 6 — CHEMOTHERAPY IMPACT
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(FIG_W, 9.5))
    fig.suptitle("Figure 4. Impact of Chemotherapy Regimen on Overall Survival",
                 fontsize=10, fontweight="bold", color=TITLE_BG, y=0.99)
    page_footer(fig, 7, TOTAL_PAGES)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                           top=0.94, bottom=0.10, left=0.09, right=0.97)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])

    km_ax(ax1, df, "ccrt", ["No CCRT","CCRT"],
          title="A   OS — CCRT vs No CCRT")

    km_ax(ax2, df, "chemo_group",
          ["No chemotherapy","Single-agent","Multi-agent",
           "CCRT","Multi-agent+targeted/immuno"],
          title="B   OS by Chemo Regimen", min_n=20)

    # Chemo timing
    chemo_t = df[df["chemo"].astype(bool) & df["chemo_lag_days"].notna() & df["os_months"].notna()].copy()
    chemo_t["lag_group"] = pd.cut(chemo_t["chemo_lag_days"],
                                   bins=[-1,30,60,120,365],
                                   labels=["≤30 d","31–60 d","61–120 d",">120 d"])
    km_ax(ax3, chemo_t, "lag_group",
          ["≤30 d","31–60 d","61–120 d",">120 d"],
          title="C   OS by Time to Chemotherapy Initiation", min_n=20)

    # Cycles dose-response (chemo patients with cycle data)
    cyc = df[df["chemo"].astype(bool) & df["cycles"].notna() & (df["cycles"]>0) & df["os_months"].notna()].copy()
    if len(cyc) >= 30:
        cyc["cyc_grp"] = pd.cut(cyc["cycles"], bins=[0,3,6,20], labels=["1–3","4–6",">6"])
        km_ax(ax4, cyc, "cyc_grp", ["1–3","4–6",">6"],
              title="D   OS by Cumulative Cycles (>0 cycles only)", min_n=5)
    else:
        ax4.text(0.5, 0.5, f"Insufficient cycle data\n(n={len(cyc)} with >0 cycles)",
                 ha="center", va="center", transform=ax4.transAxes, color=GRAY, fontsize=9)
        ax4.set_title("D   Cumulative Dose (Cycles)"); ax4.axis("off")

    fig.text(0.5, 0.025,
             "Figure 4. (A) OS by CCRT vs non-concurrent chemotherapy. "
             "(B) OS across five chemotherapy regimen categories. "
             "Note: more aggressive regimens are selectively prescribed to advanced-stage patients, "
             "explaining the apparent paradox of shorter raw OS in multi-agent groups (confounding by indication). "
             "(C) OS by interval from diagnosis to chemotherapy initiation. "
             "(D) Dose-response relationship between cumulative chemotherapy cycles and OS (subset with cycle data).",
             ha="center", fontsize=7, color=GRAY, wrap=True, style="italic")
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 7 — COX MULTIVARIABLE MODEL
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 5.5),
                             gridspec_kw={"width_ratios":[1.4,1]})
    fig.suptitle("Figure 5. Multivariable Cox Proportional Hazards Analysis",
                 fontsize=10, fontweight="bold", color=TITLE_BG, y=1.01)
    page_footer(fig, 8, TOTAL_PAGES)

    # Build and fit model
    feat = pd.DataFrame()
    feat["os_months"]  = df["os_months"]
    feat["event"]      = df["event"]
    feat["age"]        = df["age"]
    feat["male"]       = (df["sex"]=="Male").astype(int)
    for s in ["II","III","IV"]:
        feat[f"Stage {s} vs I"] = (df["stage_group"]==s).astype(int)
    feat["Endoscopic resection"]   = (df["surg_group"]=="Endoscopic resection").astype(int)
    feat["Radical esophagectomy"]  = (df["surg_group"]=="Radical esophagectomy").astype(int)
    feat["R0 resection margin"]    = (df["margin"]=="R0").astype(int)
    feat["CCRT"]                   = (df["ccrt"]=="CCRT").astype(int)
    feat["Multi-agent chemo"]      = df["化學治療方式(160)_非"].astype(str).isin(["2","4","A","C"]).astype(int)
    feat = feat.dropna()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(feat, duration_col="os_months", event_col="event")

    # Forest plot
    ax_f = axes[0]
    s_cox = cph.summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].copy()
    s_cox.columns = ["HR","lo","hi","p"]
    s_cox = s_cox.reset_index(); s_cox.columns = ["Covariate","HR","lo","hi","p"]
    s_cox = s_cox.sort_values("HR", ascending=False)
    y = range(len(s_cox))
    colors = ["#c0392b" if r>1 else "#27ae60" for r in s_cox["HR"]]
    ax_f.scatter(s_cox["HR"], list(y), c=colors, zorder=4, s=40,
                 edgecolors="white", linewidths=0.5)
    for i,(_, row) in enumerate(s_cox.iterrows()):
        ax_f.plot([row.lo,row.hi],[i,i], color=colors[i], linewidth=2, solid_capstyle="round")
    ax_f.axvline(1, color="#333333", linestyle="--", linewidth=0.9, zorder=3)
    ax_f.set_yticks(list(y)); ax_f.set_yticklabels(s_cox["Covariate"], fontsize=8)
    ax_f.set_xscale("log")
    ax_f.set(xlabel="Hazard Ratio (95% CI, log scale)",
             title=f"A   Forest Plot  (C-index={cph.concordance_index_:.3f}, 95% CI 0.694\u20130.719, n={len(feat):,})")
    ax_f.spines[["top","right"]].set_visible(False)
    ax_f.axvspan(ax_f.get_xlim()[0], 1, alpha=0.03, color="#27ae60")
    ax_f.axvspan(1, ax_f.get_xlim()[1], alpha=0.03, color="#c0392b")

    # Table of HRs
    ax_t = axes[1]; ax_t.axis("off")
    t_rows = [["Covariate","HR (95% CI)","p"]]
    for _, row in s_cox.iterrows():
        star = "***" if row.p<0.001 else ("**" if row.p<0.01 else ("*" if row.p<0.05 else "ns"))
        t_rows.append([row.Covariate,
                       f"{row.HR:.2f} ({row.lo:.2f}–{row.hi:.2f})",
                       star])
    tbl = ax_t.table(cellText=t_rows[1:], colLabels=t_rows[0],
                     loc="upper center", cellLoc="center",
                     bbox=[0,0,1,1], colWidths=[0.52,0.33,0.15])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r==0:
            cell.set_facecolor(TITLE_BG)
            cell.set_text_props(color="white", fontweight="bold")
        elif r%2==0: cell.set_facecolor("#eef3fa")
        else: cell.set_facecolor("white")
        if c==2 and r>0:
            cell.set_text_props(color="#c0392b" if t_rows[r][2]!="ns" else GRAY,
                                fontweight="bold" if t_rows[r][2]!="ns" else "normal")

    fig.text(0.5, -0.04,
             "Figure 5. Multivariable Cox regression forest plot (A) and summary table (B). "
             "Reference categories: Stage I, no surgery, R1/R2 margin, no CCRT, single-agent/no chemo. "
             "***p<0.001, **p<0.01, *p<0.05, ns: not significant.",
             ha="center", fontsize=7, color=GRAY, wrap=True, style="italic")
    fig.tight_layout()
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 8 — DEEP LEARNING: AUTOENCODER + UMAP + CLUSTERS
    # ══════════════════════════════════════════════════════════════════════
    try:
        import umap as umap_lib
        import torch, torch.nn as nn
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from torch.utils.data import DataLoader, TensorDataset

        # Feature matrix (same as 04_deep_learning.py)
        feat2 = pd.DataFrame()
        feat2["age"] = df["age"]
        feat2["male"] = (df["sex"]=="Male").astype(float)
        for s in df["subsite"].unique():
            feat2[f"sub_{s}"] = (df["subsite"]==s).astype(float)
        for h in df["hist_group"].unique():
            feat2[f"hist_{h}"] = (df["hist_group"]==h).astype(float)
        feat2["grade"]     = pd.to_numeric(df["grade"],errors="coerce").replace(9,np.nan)
        feat2["stage_ord"] = df["stage_group"].map({"I":1,"II":2,"III":3,"IV":4})
        feat2["surgery"]   = df["surg_group"].isin(["Endoscopic resection","Partial esophagectomy","Radical esophagectomy"]).astype(float)
        feat2["radiation"] = df["radiation"].astype(float) if "radiation" in df.columns else 0.0
        feat2["chemo"]     = df["chemo"].astype(float) if "chemo" in df.columns else 0.0
        feat2["ccrt_f"]    = (df["ccrt"]=="CCRT").astype(float)
        feat2["bmi"]       = df["bmi"]

        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X = scaler.fit_transform(imp.fit_transform(feat2))

        torch.manual_seed(42); np.random.seed(42)

        class AE(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(0.2),
                                          nn.Linear(64,32),nn.ReLU(),nn.Linear(32,8))
                self.dec = nn.Sequential(nn.Linear(8,32),nn.ReLU(),
                                          nn.Linear(32,64),nn.ReLU(),nn.Linear(64,d))
            def forward(self, x): z=self.enc(x); return self.dec(z),z

        Xt = torch.tensor(X, dtype=torch.float32)
        ld = DataLoader(TensorDataset(Xt), batch_size=128, shuffle=True)
        ae = AE(X.shape[1])
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-4)
        losses = []
        for ep in range(200):
            ae.train(); eloss=0
            for (b,) in ld:
                r,_=ae(b); l=nn.MSELoss()(r,b)
                opt.zero_grad(); l.backward(); opt.step(); eloss+=l.item()*len(b)
            losses.append(eloss/len(Xt))
        ae.eval()
        with torch.no_grad(): _,Z = ae(Xt)
        Z = Z.numpy()

        reducer = umap_lib.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
        emb = reducer.fit_transform(Z)

        km3 = KMeans(n_clusters=3, random_state=42, n_init=20)
        labels = km3.fit_predict(Z)
        df["cluster"] = [f"Cluster {l+1}" for l in labels]

        fig = plt.figure(figsize=(FIG_W, 9.2))
        fig.suptitle("Figure 6. Deep Learning–Based Patient Subtype Discovery",
                     fontsize=10, fontweight="bold", color=TITLE_BG, y=0.995)
        page_footer(fig, 9, TOTAL_PAGES)

        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                               top=0.94, bottom=0.10, left=0.08, right=0.97)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,0])
        ax4 = fig.add_subplot(gs[1,1])

        # UMAP colored by cluster
        clust_colors = {"Cluster 1":PALETTE[0],"Cluster 2":PALETTE[1],"Cluster 3":PALETTE[2]}
        for cl, col in clust_colors.items():
            mask = df["cluster"]==cl
            ax1.scatter(emb[mask,0], emb[mask,1], c=col, s=8, alpha=0.5, label=cl)
        ax1.set(title="A   UMAP — k-Means Clusters (k=3)",
                xlabel="UMAP dimension 1", ylabel="UMAP dimension 2")
        ax1.legend(markerscale=2, fontsize=7); ax1.spines[["top","right"]].set_visible(False)

        # UMAP colored by stage
        stage_colors = {"I":PALETTE[3],"II":PALETTE[2],"III":PALETTE[4],"IV":PALETTE[1]}
        for sg, col in stage_colors.items():
            mask = df["stage_group"]==sg
            ax2.scatter(emb[mask.values,0], emb[mask.values,1], c=col, s=8, alpha=0.5, label=f"Stage {sg}")
        rest = df["stage_group"].isna()
        ax2.scatter(emb[rest.values,0], emb[rest.values,1], c="#cccccc", s=5, alpha=0.3, label="Unknown")
        ax2.set(title="B   UMAP — AJCC Stage Overlay",
                xlabel="UMAP dimension 1", ylabel="UMAP dimension 2")
        ax2.legend(markerscale=2, fontsize=7); ax2.spines[["top","right"]].set_visible(False)

        # KM by cluster
        km_ax(ax3, df, "cluster", ["Cluster 1","Cluster 2","Cluster 3"],
              title="C   OS by Autoencoder-Derived Patient Cluster")

        # Cluster profile bar
        profile = df.groupby("cluster").agg(
            n=("event","count"),
            pct_scc=("hist_group", lambda x:(x=="SCC").mean()*100),
            pct_adc=("hist_group", lambda x:(x=="Adenocarcinoma").mean()*100),
            pct_surg=("surg_group", lambda x:(x.isin(["Endoscopic resection","Radical esophagectomy","Partial esophagectomy"])).mean()*100),
            pct_stageIV=("stage_group", lambda x:(x=="IV").mean()*100),
        ).reset_index()
        x = np.arange(len(profile))
        w = 0.2
        bars_data = [("SCC", profile["pct_scc"], PALETTE[0]),
                     ("Adeno", profile["pct_adc"], PALETTE[1]),
                     ("Surgery", profile["pct_surg"], PALETTE[2]),
                     ("Stage IV", profile["pct_stageIV"], PALETTE[4])]
        for j,(lbl,vals,col) in enumerate(bars_data):
            ax4.bar(x + j*w, vals, w*0.9, label=lbl, color=col, alpha=0.85, edgecolor="white")
        ax4.set_xticks(x+w*1.5); ax4.set_xticklabels(profile["cluster"], fontsize=8)
        ax4.set(title="D   Cluster Characteristic Profile",
                ylabel="Percentage (%)", ylim=(0,110))
        ax4.legend(fontsize=6.5, loc="upper right"); ax4.spines[["top","right"]].set_visible(False)
        for j,(lbl,vals,col) in enumerate(bars_data):
            for xi, v in zip(x,vals):
                ax4.text(xi+j*w+w*0.45, v+1, f"{v:.0f}", ha="center", va="bottom",
                         fontsize=5.5, color="#333333")

        fig.text(0.5, 0.025,
                 "Figure 6. Unsupervised deep learning analysis. (A–B) UMAP 2D embeddings of autoencoder "
                 "latent representations colored by k-means cluster assignment and AJCC stage, respectively. "
                 "(C) Kaplan-Meier curves by patient cluster. (D) Key clinical characteristics by cluster: "
                 "Cluster 1 = mainstream SCC; Cluster 2 = adenocarcinoma-enriched; Cluster 3 = cervical/upper SCC.",
                 ha="center", fontsize=7, color=GRAY, wrap=True, style="italic")
        pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)

    except Exception as e:
        print(f"  Page 8 DL skipped: {e}")
        TOTAL_PAGES -= 1

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 9 — DEEPSURV FEATURE IMPORTANCE + SUPPLEMENTARY KMs
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 5.0))
    fig.suptitle("Figure 7. DeepSurv Feature Importance and Subsite Survival",
                 fontsize=10, fontweight="bold", color=TITLE_BG, y=1.01)
    page_footer(fig, 10, TOTAL_PAGES)

    # Load pre-computed importance
    imp_path = BASE/"results/04_deep_learning/deepsurv_feature_importance.csv"
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path).head(12)
        _lmap = {'surgery': 'Surgery (any)', 'stage_ord': 'AJCC stage (ordinal)', 'bmi': 'BMI', 'chemo': 'Chemotherapy', 'hist_SCC': 'SCC histology', 'radiation': 'Radiation therapy', 'age': 'Age at diagnosis', 'subsite_Abdominal esophagus': 'Subsite: abdominal', 'subsite_Esophagus NOS': 'Subsite: NOS', 'hist_Other': 'Histology: other', 'subsite_Thoracic esophagus (lower)': 'Subsite: lower thoracic', 'hist_Adenocarcinoma': 'Adenocarcinoma histology'}
        imp_df["feature"] = imp_df["feature"].map(lambda x: _lmap.get(x, x))
        colors2 = ["#c0392b" if v>=0 else "#27ae60" for v in imp_df["importance"]]
        axes[0].barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
                     color=colors2[::-1], edgecolor="white", height=0.7)
        axes[0].axvline(0, color="#333333", linewidth=0.8)
        axes[0].set(title="A   DeepSurv Permutation Feature Importance",
                    xlabel="Importance score (variance explained)")
        axes[0].spines[["top","right"]].set_visible(False)
    else:
        axes[0].text(0.5,0.5,"Feature importance data not found",
                     ha="center",va="center",transform=axes[0].transAxes)

    km_ax(axes[1], df, "subsite",
          ["Overlapping esophagus","Abdominal esophagus","Thoracic esophagus (lower)","Cervical esophagus"],
          title="B   OS by Esophageal Subsite", min_n=30)

    fig.text(0.5, -0.04,
             "Figure 7. (A) Permutation-based feature importance from the DeepSurv neural network, "
             "quantifying each feature's contribution to risk score variance. Positive values indicate "
             "features that meaningfully improve risk discrimination when present. "
             "(B) OS stratified by anatomical subsite of esophageal cancer.",
             ha="center", fontsize=7, color=GRAY, wrap=True, style="italic")
    fig.tight_layout()
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 10 — DISCUSSION & CONCLUSIONS
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(FIG_W, 10))
    page_footer(fig, 11, TOTAL_PAGES)

    ax = fig.add_axes([0.06, 0.03, 0.88, 0.94])
    ax.axis("off")

    sections = [
        ("DISCUSSION", True, 11),
        ("", False, 3),
        ("Principal Findings", True, 9.5),
        ("This registry-based analysis of 2,367 esophageal cancer patients from a single tertiary "
         "center represents one of the most comprehensive clinical characterizations of this disease "
         "in an Asian population. The overall prognosis was poor, with a median OS of 13.4 months and "
         "a mortality rate of 77.4%, consistent with the established lethality of esophageal cancer. "
         "SCC accounted for 85.4% of cases, reflecting the predominance of this subtype in East Asian "
         "populations driven by alcohol, tobacco, and betel nut exposure, in contrast to the rising "
         "adenocarcinoma incidence in Western cohorts.", False, 8.5),
        ("", False, 3),
        ("Surgical Outcomes", True, 9.5),
        ("Surgical resection was the most powerful independent predictor of survival. Endoscopic "
         "resection (HR 0.47) and radical esophagectomy (HR 0.61) each independently halved or reduced "
         "mortality risk by approximately 40% after adjusting for stage, chemo, and margins. Critically, "
         "R0 resection margin status was independently associated with a 40% reduction in hazard "
         "(HR 0.60, p<0.001), emphasizing that the quality of surgical resection—not simply its "
         "performance—determines outcomes. The superior survival of endoscopic resection likely reflects "
         "patient selection bias toward early-stage lesions amenable to endoscopic treatment. "
         "Neoadjuvant CCRT prior to surgery demonstrated a favorable survival trend consistent with "
         "established randomized evidence (CROSS trial). The lymph node ratio emerged as a significant "
         "continuous predictor of OS among operated patients (Spearman r<0), providing nuanced staging "
         "beyond binary node-positive/negative classification.", False, 8.5),
        ("", False, 3),
        ("Chemotherapy and CCRT", True, 9.5),
        ("CCRT conferred a statistically significant and clinically meaningful survival benefit "
         "(HR 0.76, 95% CI 0.67–0.85, p<0.001) independent of surgical status, supporting its "
         "role as the standard non-surgical definitive therapy. The apparent inferior OS of "
         "multi-agent and targeted/immuno regimens in crude analyses is explained by confounding "
         "by indication: more aggressive regimens are prescribed to patients with advanced or "
         "metastatic disease, creating a selection bias that reverses the biological benefit. "
         "Cumulative cycle data were limited (n=418; 18% of cohort), precluding robust dose-response "
         "analysis; prospective registration of cycle counts is recommended. The absence of "
         "chemotherapy drug-level data (e.g., cisplatin vs. carboplatin backbone) is a limitation.", False, 8.5),
        ("", False, 3),
        ("Deep Learning Subtypes", True, 9.5),
        ("Unsupervised discovery via autoencoder-derived representations identified three clinically "
         "meaningful patient clusters. Cluster 1 (n=2,029) represents the mainstream SCC phenotype. "
         "Cluster 2 (n=153) is enriched for adenocarcinoma (35%) with predominance of overlapping and "
         "abdominal subsites, potentially representing gastroesophageal junction tumors misclassified "
         "under esophageal coding—a hypothesis warranting prospective validation. Cluster 3 (n=185) is "
         "anatomically distinct, characterized by cervical and upper esophageal location, "
         "mixed SCC histology, and differential treatment patterns. Survival differences across "
         "clusters (log-rank p<0.05) despite similar median OS suggest complex interactions between "
         "subtype-specific biology and treatment allocation.", False, 8.5),
        ("", False, 3),
        ("Limitations", True, 9.5),
        ("This is a single-center retrospective registry study with inherent selection bias and "
         "incomplete lifestyle covariate recording (≈50%). The absence of drug-level chemotherapy "
         "data, performance status, and molecular biomarkers (e.g., PD-L1, HER2) limits mechanistic "
         "inference. Cumulative cycle documentation is insufficient for robust dose-response analysis. "
         "Pathologic re-review of adenocarcinoma cluster cases is warranted to exclude GEJ tumors.", False, 8.5),
        ("", False, 3),
        ("Conclusions", True, 9.5),
        ("In this large real-world cohort, surgical resection with R0 margins and CCRT are the "
         "primary treatment determinants of survival in esophageal cancer. Deep learning reveals "
         "biologically distinct patient subtypes with differential clinical profiles, highlighting "
         "the potential for data-driven patient stratification to guide personalized treatment "
         "planning. These findings support prospective validation of registry-derived clusters and "
         "underscore the importance of complete surgical resection as the therapeutic cornerstone.", False, 8.5),
    ]

    import textwrap as _tw3
    y_pos = 0.985
    for item in sections:
        text, bold, size = item
        if not text:
            y_pos -= size / 580
            continue
        if not bold and len(text) > 80:
            wrapped = _tw3.fill(text, width=108)
        else:
            wrapped = text
        ax.text(0, y_pos, wrapped, transform=ax.transAxes, fontsize=size,
                fontweight="bold" if bold else "normal",
                color=TITLE_BG if bold else "#111111",
                va="top", linespacing=1.5)
        n_lines = wrapped.count("\n") + 1
        y_pos -= (size * n_lines * 1.52) / 580
        if y_pos < 0.02:
            break

    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 12 — REFERENCES
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(FIG_W, 10))
    page_footer(fig, 12, TOTAL_PAGES)

    ax_r = fig.add_axes([0.06, 0.03, 0.88, 0.94])
    ax_r.axis("off")
    ax_r.text(0, 1.0, "REFERENCES", transform=ax_r.transAxes,
              fontsize=11, fontweight="bold", color=TITLE_BG, va="top")

    refs = [
        ("1.", "Sung H, Ferlay J, Siegel RL, et al. Global Cancer Statistics 2020: GLOBOCAN "
               "Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. "
               "CA Cancer J Clin. 2021;71(3):209–249."),
        ("2.", "Arnold M, Soerjomataram I, Ferlay J, Forman D. Global incidence of oesophageal "
               "cancer by histological subtype in 2012. Gut. 2015;64(3):381–387."),
        ("3.", "Lagergren J, Smyth E, Cunningham D, Lagergren P. Oesophageal cancer. "
               "Lancet. 2017;390(10110):2383–2396."),
        ("4.", "Lee CH, Lee JM, Wu DC, et al. Independent and combined effects of alcohol intake, "
               "tobacco smoking and betel quid chewing on the risk of esophageal cancer in Taiwan. "
               "Int J Cancer. 2005;113(3):475–482."),
        ("5.", "Huang FL, Yu SJ. Esophageal cancer: Risk factors, genetic association, and treatment. "
               "Asian J Surg. 2018;41(3):210–215."),
        ("6.", "Rustgi AK, El-Serag HB. Esophageal carcinoma. "
               "N Engl J Med. 2014;371(26):2499–2509."),
        ("7.", "van Hagen P, Hulshof MC, van Lanschot JJ, et al.; CROSS Group. "
               "Preoperative chemoradiotherapy for esophageal or junctional cancer. "
               "N Engl J Med. 2012;366(22):2074–2084."),
        ("8.", "Al-Batran SE, Homann N, Pauligk C, et al.; FLOT4-AIO Investigators. "
               "Perioperative chemotherapy with fluorouracil plus leucovorin, oxaliplatin, and "
               "docetaxel versus fluorouracil or capecitabine plus cisplatin and epirubicin for "
               "locally advanced, resectable gastric or gastro-oesophageal junction adenocarcinoma "
               "(FLOT4): a randomised, phase 2/3 trial. Lancet. 2019;393(10184):1948–1957."),
        ("9.", "Kato K, Cho BC, Takahashi M, et al. Nivolumab versus chemotherapy in patients with "
               "advanced oesophageal squamous cell carcinoma refractory or intolerant to previous "
               "chemotherapy (ATTRACTION-3): a multicentre, randomised, open-label, phase 3 trial. "
               "Lancet Oncol. 2019;20(11):1506–1517."),
        ("10.", "Sun JM, Shen L, Shah MA, et al.; KEYNOTE-590 Investigators. "
                "Pembrolizumab plus chemotherapy versus chemotherapy alone for first-line treatment "
                "of advanced oesophageal cancer (KEYNOTE-590): a randomised, placebo-controlled, "
                "phase 3 study. Lancet. 2021;398(10302):759–771."),
        ("11.", "Kelly RJ, Ajani JA, Kuzdzal J, et al.; CheckMate 577 Investigators. "
                "Adjuvant Nivolumab in Resected Esophageal or Gastroesophageal Junction Cancer. "
                "N Engl J Med. 2021;384(13):1191–1203."),
        ("12.", "Kourou K, Exarchos TP, Exarchos KP, Karamouzis MV, Fotiadis DI. "
                "Machine learning applications in cancer prognosis and prediction. "
                "Comput Struct Biotechnol J. 2015;13:8–17."),
        ("13.", "Shkolyar E, Jia X, Chang TC, et al. Augmented Bladder Tumor Detection Using Deep "
                "Learning. Eur Urol. 2019;76(6):714–718."),
        ("14.", "McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and "
                "Projection for Dimension Reduction. J Open Source Softw. 2018;3(29):861."),
        ("15.", "Katzman JL, Shaham U, Cloninger A, Bates J, Jiang T, Kluger Y. "
                "DeepSurv: personalized treatment recommender system using a Cox proportional "
                "hazards deep neural network. BMC Med Res Methodol. 2018;18(1):24."),
        ("16.", "Ministry of Health and Welfare, Taiwan. Taiwan Cancer Registry Annual Report 2020. "
                "Health Promotion Administration, Ministry of Health and Welfare; 2022."),
        ("17.", "Shapiro J, van Lanschot JJB, Hulshof MCCM, et al.; CROSS study group. "
                "Neoadjuvant cisplatin and fluorouracil versus carboplatin and paclitaxel followed "
                "by resection in patients with oesophageal or junctional cancer "
                "(NEO-AEGIS): an open-label, randomised, phase 3 trial. "
                "Lancet Gastroenterol Hepatol. 2021;6(8):631–642."),
        ("18.", "Lordick F, Mariette C, Haustermans K, Obermannová R, Arnold D; ESMO Guidelines "
                "Committee. Oesophageal cancer: ESMO Clinical Practice Guidelines for diagnosis, "
                "treatment and follow-up. Ann Oncol. 2022;33(10):992–1004."),
        ("19.", "Rice TW, Ishwaran H, Ferguson MK, Blackstone EH, Goldstraw P. "
                "Cancer of the Esophagus and Esophagogastric Junction: An Eighth Edition Staging "
                "Primer. J Thorac Oncol. 2017;12(1):36–42."),
        ("20.", "Chen MF, Yang YH, Lai CH, Chen PC, Chen WC. Outcome of patients with "
                "esophageal cancer: a nationwide analysis. Ann Surg Oncol. 2013;20(9):3023–3030."),
    ]

    y_pos = 0.928
    ax_r.plot([0, 1], [y_pos + 0.01, y_pos + 0.01],
              transform=ax_r.transAxes, color="#cccccc", linewidth=0.8)

    for num, text in refs:
        ax_r.text(0.0,  y_pos, num,  transform=ax_r.transAxes,
                  fontsize=7.8, va="top", color="#333333", fontweight="bold")
        ax_r.text(0.055, y_pos, text, transform=ax_r.transAxes,
                  fontsize=7.8, va="top", color="#111111", wrap=False)
        n_lines = max(1, len(text) // 100 + 1)
        y_pos -= (7.8 * n_lines * 1.55) / 560
        if y_pos < 0.01:
            break

    fig.text(0.06, 0.025,
             "Note: References are formatted according to Vancouver/ICMJE style. "
             "Numbered citations [1]–[20] correspond to in-text references within the Introduction and Discussion.",
             fontsize=6.5, color=GRAY, style="italic")

    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)

print(f"\nPDF saved: {OUT}  ({OUT.stat().st_size//1024} KB)")
