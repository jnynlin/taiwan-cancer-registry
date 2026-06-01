"""
Survival analysis for esophageal cancer (C15).
Outputs: results/03_survival/  (KM curves, Cox table, CSV)
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

DATA    = Path(__file__).parent.parent / "data/c15_enriched.csv"
OUT     = Path(__file__).parent.parent / "results/03_survival"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS  = sns.color_palette("tab10")


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False)
    # Taiwan registry: 生存狀態=0 → Dead (event=1), =1 → Alive (event=0)
    df["event"] = (df["生存狀態(27)"].astype(str).str.strip() == "0").astype(int)
    # Filter rows with valid OS and positive duration
    df = df[df["os_days"].notna() & (df["os_days"] > 0)].copy()
    df["os_months"] = df["os_days"] / 30.44
    print(f"  Survival-evaluable: {len(df)} cases  (events={df['event'].sum()})")
    return df


def km_plot(df, group_col, group_order=None, title="", fname="km.png",
            time_col="os_months", event_col="event", xlabel="Months"):
    groups = group_order or sorted(df[group_col].dropna().unique())
    groups = [g for g in groups if g in df[group_col].values]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    kmf = KaplanMeierFitter()
    stats, medians = [], []
    for i, grp in enumerate(groups):
        sub = df[df[group_col] == grp].dropna(subset=[time_col, event_col])
        if len(sub) < 5:
            continue
        kmf.fit(sub[time_col], sub[event_col], label=f"{grp} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=COLORS[i % len(COLORS)])
        med = kmf.median_survival_time_
        medians.append({"Group": grp, "n": len(sub), "events": int(sub[event_col].sum()),
                        "Median OS (months)": round(med, 1) if pd.notna(med) else "NR"})
    ax.set(title=title, xlabel=xlabel, ylabel="Survival probability", ylim=(0, 1.05))
    ax.legend(loc="upper right", fontsize=9)

    # Log-rank p-value (pairwise for 2 groups, multivariate otherwise)
    valid = df[df[group_col].isin(groups)].dropna(subset=[time_col, event_col])
    if len(groups) == 2:
        g1 = valid[valid[group_col] == groups[0]]
        g2 = valid[valid[group_col] == groups[1]]
        res = logrank_test(g1[time_col], g2[time_col], g1[event_col], g2[event_col])
        p = res.p_value
    else:
        res = multivariate_logrank_test(valid[time_col], valid[group_col], valid[event_col])
        p = res.p_value
    ax.text(0.65, 0.85, f"Log-rank p = {p:.4f}", transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Median table
    med_df = pd.DataFrame(medians)
    axes[1].axis("off")
    if not med_df.empty:
        tbl = axes[1].table(cellText=med_df.values, colLabels=med_df.columns,
                            loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.4)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    med_df.to_csv(OUT / (fname.replace(".png", "_medians.csv")), index=False, encoding="utf-8-sig")
    return p


def cox_model(df: pd.DataFrame):
    # Build feature matrix for Cox
    feat = pd.DataFrame()
    feat["os_months"] = df["os_months"]
    feat["event"] = df["event"]
    feat["age"] = df["age"]
    feat["male"] = (df["sex"] == "Male").astype(int)

    # Stage (I=ref)
    for s in ["II","III","IV"]:
        feat[f"stage_{s}"] = (df["stage_group"] == s).astype(int)

    # Histology (SCC=ref)
    for h in ["Adenocarcinoma","Carcinoma NOS","Adenosquamous","Other"]:
        feat[f"hist_{h.replace(' ','_')}"] = (df["histology_group"] == h).astype(int)

    # Treatment
    feat["surgery"] = df["surgery"].astype(int)
    feat["radiation"] = df["radiation"].astype(int)
    feat["chemo"] = df["chemo"].astype(int)
    feat["immunotherapy"] = df["immunotherapy"].astype(int)

    feat = feat.dropna()
    print(f"  Cox model: {len(feat)} cases with complete covariates")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(feat, duration_col="os_months", event_col="event")
    cph.print_summary()

    summary = cph.summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].round(4)
    summary.columns = ["HR","HR_lower95","HR_upper95","p"]
    summary.to_csv(OUT / "cox_summary.csv", encoding="utf-8-sig")

    # Forest plot
    s = summary.reset_index()
    s.columns = ["Variable"] + list(s.columns[1:])
    s = s[s["p"] < 1].sort_values("HR")
    fig, ax = plt.subplots(figsize=(8, len(s) * 0.5 + 1))
    y = range(len(s))
    ax.scatter(s["HR"], y, color="steelblue", zorder=3, s=60)
    ax.hlines(y, s["HR_lower95"], s["HR_upper95"], color="steelblue", linewidth=2)
    ax.axvline(1, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(s["Variable"], fontsize=9)
    ax.set(title="Cox Proportional Hazards — Forest Plot", xlabel="Hazard Ratio (95% CI)")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(OUT / "cox_forest_plot.png", dpi=150)
    plt.close(fig)
    return cph


if __name__ == "__main__":
    print("Loading enriched data...")
    df = load()

    print("\nKM — by Sex:")
    km_plot(df, "sex", ["Male","Female"], "OS by Sex", "km_sex.png")

    print("KM — by Stage:")
    km_plot(df, "stage_group", ["I","II","III","IV"], "OS by AJCC Stage", "km_stage.png")

    print("KM — by Histology group:")
    km_plot(df, "histology_group", None, "OS by Histology", "km_histology.png")

    print("KM — by Subsite:")
    km_plot(df, "subsite", None, "OS by Esophageal Subsite", "km_subsite.png")

    print("KM — by Surgery:")
    km_plot(df, "surgery", [True, False], "OS by Surgery", "km_surgery.png")

    print("KM — by Chemotherapy:")
    km_plot(df, "chemo", [True, False], "OS by Chemotherapy", "km_chemo.png")

    print("\nCox Proportional Hazards model:")
    cox_model(df)

    # Cancer-specific survival (died from C15 = event, others censored)
    df["css_event"] = df["event"].copy()
    df.loc[df["event"] == 1, "css_event"] = (
        df.loc[df["event"] == 1, "死亡原因碼(32)"].astype(str).str.startswith("C15")
    ).astype(int)
    print("\nKM — CSS by Stage (cancer-specific survival):")
    km_plot(df, "stage_group", ["I","II","III","IV"], "Cancer-Specific Survival by Stage",
            "km_css_stage.png", event_col="css_event")

    print(f"\nDone. Results in {OUT}")
