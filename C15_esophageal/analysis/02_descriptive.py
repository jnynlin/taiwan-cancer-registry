"""
Descriptive analysis of esophageal cancer (C15) — clinical signatures.
Outputs: results/02_descriptive/  (CSV tables + PNG figures)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA = Path(__file__).parent.parent / "data/c15_all.csv"
OUT  = Path(__file__).parent.parent / "results/02_descriptive"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Code dictionaries ──────────────────────────────────────────────────────────
HISTOLOGY_MAP = {
    "8070": "Squamous cell carcinoma",
    "8071": "Keratinizing SCC",
    "8072": "Non-keratinizing SCC",
    "8073": "Small cell SCC",
    "8074": "Spindle cell SCC",
    "8075": "Basaloid SCC",
    "8076": "Microinvasive SCC",
    "8140": "Adenocarcinoma",
    "8480": "Mucinous adenocarcinoma",
    "8144": "Intestinal adenocarcinoma",
    "8000": "Malignant neoplasm NOS",
    "8010": "Carcinoma NOS",
    "8020": "Undifferentiated carcinoma",
    "8560": "Adenosquamous carcinoma",
    "8900": "Rhabdomyosarcoma",
    "8240": "Carcinoid tumor",
}
GRADE_MAP = {1: "Well differentiated (G1)", 2: "Moderately differentiated (G2)",
             3: "Poorly differentiated (G3)", 4: "Undifferentiated (G4)",
             9: "Unknown/NA"}
CONFIRM_MAP = {1: "Positive histology", 2: "Positive cytology", 3: "Positive histology (metastatic)",
               4: "Positive cytology (metastatic)", 5: "Positive lab/marker only",
               6: "Radiology/imaging only", 7: "Clinical diagnosis only",
               8: "Death certificate only", 9: "Unknown"}
ALCOHOL_MAP = {1: "Never", 2: "Past drinker", 3: "Current (social)", 4: "Current (regular)", 9: "Unknown", 999: "Unknown"}
STAGE_CLEAN = {"888": np.nan, "999": np.nan, "BBB": np.nan, "888": np.nan}


def clean_stage(s):
    s = str(s).strip().upper()
    if s in ("888", "999", "BBB", "NAN", "88", "99"):
        return np.nan
    return s


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False)

    # Histology: first 4 digits of 5-digit code
    df["histology_code"] = df["組織型態(49)"].astype(str).str[:4]
    df["histology"] = df["histology_code"].map(HISTOLOGY_MAP).fillna("Other")

    # Major histology group
    scc_codes = ["8070","8071","8072","8073","8074","8075","8076"]
    df["histology_group"] = "Other"
    df.loc[df["histology_code"].isin(scc_codes), "histology_group"] = "SCC"
    df.loc[df["histology_code"].isin(["8140","8480","8144"]), "histology_group"] = "Adenocarcinoma"
    df.loc[df["histology_code"].isin(["8000","8010","8020"]), "histology_group"] = "Carcinoma NOS"
    df.loc[df["histology_code"] == "8560", "histology_group"] = "Adenosquamous"

    # Grade
    df["grade"] = pd.to_numeric(df["分化程度(50)"], errors="coerce")
    df["grade_label"] = df["grade"].map(GRADE_MAP)

    # Confirmation method
    df["confirm"] = pd.to_numeric(df["癌症確診方式(51)"], errors="coerce")
    df["confirm_label"] = df["confirm"].map(CONFIRM_MAP)

    # Stage
    df["clin_stage"] = df["臨床期別組合(95)"].apply(clean_stage)
    df["path_stage"] = df["病理期別組合(101)"].apply(clean_stage)
    df["stage"] = df["path_stage"].combine_first(df["clin_stage"])
    df["stage_group"] = df["stage"].str.extract(r"^([1234])", expand=False).map(
        {"1": "I", "2": "II", "3": "III", "4": "IV"})

    # Treatment flags
    def has_surgery(x):
        x = str(x).strip()
        return x not in ("00", "88", "99", "nan", "NAN")
    df["surgery"] = df["原發部位手術方式(118)"].apply(has_surgery)
    df["radiation"] = df["放射治療開始日期(138)"].astype(str).apply(
        lambda x: x not in ("0", "nan", "NAN", "0.0"))
    df["chemo"] = pd.to_numeric(df["申報醫院化學治療(163)"], errors="coerce").apply(
        lambda x: x not in (0, np.nan) if pd.notna(x) else False)
    df["immunotherapy"] = pd.to_numeric(df["申報醫院免疫治療(169)"], errors="coerce").apply(
        lambda x: x not in (0, np.nan) if pd.notna(x) else False)
    df["targeted"] = pd.to_numeric(df["申報醫院標靶治療(A26)"], errors="coerce").apply(
        lambda x: x not in (0, np.nan) if pd.notna(x) else False)

    # Lifestyle (99/98 = unknown sentinel)
    for col in ["每日吸菸量(A3-1)", "吸菸年(A3-2)", "每日嚼檳榔量(A4-1)", "嚼檳榔年(A4-2)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([98, 99, 999], np.nan)
    df["alcohol"] = pd.to_numeric(df["喝酒行為(A5)"], errors="coerce").replace([9, 999], np.nan)
    df["alcohol_label"] = df["alcohol"].map({1: "Never", 2: "Past", 3: "Social", 4: "Regular"})
    df["smoker"] = df["每日吸菸量(A3-1)"].apply(lambda x: "Yes" if pd.notna(x) and x > 0 else ("No" if pd.notna(x) else np.nan))
    df["betel_nut"] = df["每日嚼檳榔量(A4-1)"].apply(lambda x: "Yes" if pd.notna(x) and x > 0 else ("No" if pd.notna(x) else np.nan))

    # BMI
    ht = pd.to_numeric(df["身高(A1)"], errors="coerce").replace([999], np.nan)
    wt = pd.to_numeric(df["體重(A2)"], errors="coerce").replace([999], np.nan)
    df["bmi"] = wt / (ht / 100) ** 2

    # Age groups
    age = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["age"] = age
    df["age_group"] = pd.cut(age, bins=[0,50,60,70,80,120],
                             labels=["<50","50–59","60–69","70–79","≥80"])

    # Sex
    df["sex"] = df["性別(5)"].map({1: "Male", 2: "Female"})

    return df


def table1(df: pd.DataFrame):
    rows = []
    total = len(df)

    def pct(n): return f"{n} ({100*n/total:.1f}%)"
    def med(s): v = s.dropna(); return f"{v.median():.1f} (IQR {v.quantile(.25):.1f}–{v.quantile(.75):.1f})"

    rows.append(("n", str(total), ""))
    rows.append(("--- Demographics ---", "", ""))
    rows.append(("Age at diagnosis (median, IQR)", med(df["age"]), ""))
    for g, n in df["age_group"].value_counts(sort=False).items():
        rows.append((f"  {g}", pct(n), ""))
    for g, n in df["sex"].value_counts().items():
        rows.append((f"Sex: {g}", pct(n), ""))

    rows.append(("--- Tumor characteristics ---", "", ""))
    for g, n in df["subsite"].value_counts().items():
        rows.append((f"Subsite: {g}", pct(n), ""))
    for g, n in df["histology_group"].value_counts().items():
        rows.append((f"Histology: {g}", pct(n), ""))
    for g, n in df["grade_label"].value_counts().items():
        rows.append((f"Grade: {g}", pct(n), ""))
    for g, n in df["stage_group"].value_counts(dropna=False).items():
        label = f"Stage: {g}" if pd.notna(g) else "Stage: Unknown/missing"
        rows.append((label, pct(n), ""))

    rows.append(("--- Treatment ---", "", ""))
    for col, label in [("surgery","Surgery"),("radiation","Radiation"),
                       ("chemo","Chemotherapy"),("immunotherapy","Immunotherapy"),
                       ("targeted","Targeted therapy")]:
        rows.append((label, pct(df[col].sum()), ""))

    rows.append(("--- Lifestyle (available subset) ---", "", ""))
    rows.append(("BMI (median, IQR)", med(df["bmi"]), ""))
    for g, n in df["smoker"].value_counts(dropna=False).items():
        rows.append((f"Smoker: {g}", pct(n), ""))
    for g, n in df["betel_nut"].value_counts(dropna=False).items():
        rows.append((f"Betel nut: {g}", pct(n), ""))
    for g, n in df["alcohol_label"].value_counts(dropna=False).items():
        rows.append((f"Alcohol: {g}", pct(n), ""))

    t1 = pd.DataFrame(rows, columns=["Variable", "Value", "Notes"])
    t1.to_csv(OUT / "table1_demographics.csv", index=False, encoding="utf-8-sig")
    return t1


def fig_incidence_trend(df: pd.DataFrame):
    yr = df.dropna(subset=["diag_year"])
    trend = yr.groupby("diag_year").size().reset_index(name="cases")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(trend["diag_year"], trend["cases"], color="steelblue", edgecolor="white")
    ax.set(title="Esophageal Cancer (C15) — Annual Cases", xlabel="Year", ylabel="Number of cases")
    fig.tight_layout()
    fig.savefig(OUT / "fig_incidence_trend.png", dpi=150)
    plt.close(fig)


def fig_age_sex(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Age distribution by sex
    for sex, grp in df.groupby("sex"):
        axes[0].hist(grp["age"].dropna(), bins=20, alpha=0.6, label=sex)
    axes[0].set(title="Age at Diagnosis by Sex", xlabel="Age (years)", ylabel="Count")
    axes[0].legend()

    # Age group bar
    ag = df.groupby(["age_group","sex"]).size().unstack(fill_value=0)
    ag.plot(kind="bar", ax=axes[1], colormap="Set1", edgecolor="white", width=0.7)
    axes[1].set(title="Age Group × Sex", xlabel="Age Group", ylabel="Count")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_age_sex.png", dpi=150)
    plt.close(fig)


def fig_histology_stage(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    hg = df["histology_group"].value_counts()
    axes[0].pie(hg, labels=hg.index, autopct="%1.1f%%", startangle=140,
                colors=sns.color_palette("Set2", len(hg)))
    axes[0].set_title("Histology Distribution")

    sg = df["stage_group"].value_counts(dropna=True).reindex(["I","II","III","IV"]).dropna()
    axes[1].bar(sg.index, sg.values, color=sns.color_palette("RdYlGn_r", 4), edgecolor="white")
    axes[1].set(title="AJCC Stage Distribution", xlabel="Stage", ylabel="Count")
    fig.tight_layout()
    fig.savefig(OUT / "fig_histology_stage.png", dpi=150)
    plt.close(fig)


def fig_treatment_matrix(df: pd.DataFrame):
    tx_cols = ["surgery","radiation","chemo","immunotherapy","targeted"]
    tx_labels = ["Surgery","Radiation","Chemo","Immunotherapy","Targeted"]
    tx = df[tx_cols].astype(int)

    # Co-occurrence heatmap
    comat_arr = tx.T.dot(tx).values.astype(float).copy()
    np.fill_diagonal(comat_arr, np.nan)
    comat = pd.DataFrame(comat_arr, index=tx.columns, columns=tx.columns)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(comat, annot=True, fmt=".0f", cmap="Blues", xticklabels=tx_labels,
                yticklabels=tx_labels, ax=ax, linewidths=0.5)
    ax.set_title("Treatment Co-occurrence Matrix")
    fig.tight_layout()
    fig.savefig(OUT / "fig_treatment_cooccurrence.png", dpi=150)
    plt.close(fig)

    # Treatment combination counts
    combos = df[tx_cols].apply(lambda r: "+".join(l for l, v in zip(tx_labels, r) if v) or "None", axis=1)
    top = combos.value_counts().head(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    top.sort_values().plot(kind="barh", ax=ax, color="teal", edgecolor="white")
    ax.set(title="Top 12 Treatment Combinations", xlabel="Count")
    fig.tight_layout()
    fig.savefig(OUT / "fig_treatment_combinations.png", dpi=150)
    plt.close(fig)

    # Save counts
    combos.value_counts().reset_index().rename(columns={"index":"combination","count":"n"}).to_csv(
        OUT / "treatment_combinations.csv", index=False, encoding="utf-8-sig")


def fig_lifestyle(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in [
        (axes[0], "smoker", "Smoking Status"),
        (axes[1], "betel_nut", "Betel Nut Use"),
        (axes[2], "alcohol_label", "Alcohol Behavior"),
    ]:
        vc = df[col].value_counts(dropna=True)
        ax.bar(vc.index, vc.values, color=sns.color_palette("muted", len(vc)), edgecolor="white")
        ax.set(title=title, ylabel="Count")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lifestyle.png", dpi=150)
    plt.close(fig)


def fig_subsite_stage_heatmap(df: pd.DataFrame):
    ct = pd.crosstab(df["subsite"], df["stage_group"])
    ct = ct.reindex(columns=["I","II","III","IV"]).fillna(0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, linewidths=0.5,
                cbar_kws={"label": "% within subsite"})
    ax.set(title="Stage Distribution by Esophageal Subsite (%)", xlabel="AJCC Stage")
    fig.tight_layout()
    fig.savefig(OUT / "fig_subsite_stage.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("Loading data...")
    df = load()
    print(f"  {len(df)} cases loaded")

    print("Table 1 — demographics/clinical summary...")
    t1 = table1(df)
    print(t1.to_string(index=False))

    print("\nGenerating figures...")
    fig_incidence_trend(df)
    fig_age_sex(df)
    fig_histology_stage(df)
    fig_treatment_matrix(df)
    fig_lifestyle(df)
    fig_subsite_stage_heatmap(df)

    # Save enriched dataset for downstream scripts
    df.to_csv(Path(__file__).parent.parent / "data/c15_enriched.csv", index=False, encoding="utf-8-sig")
    print(f"\nDone. Results in {OUT}")
    print(f"Enriched dataset saved for survival/DL scripts.")
