"""
Build patient × cancer-type multi-hot matrix from all-cancer registry data.
Outputs:
  data/patient_cancer_matrix.csv   — rows=patients, cols=cancer_site_codes, values=0/1
  data/patient_meta.csv            — demographics + sequence info per patient
  data/cancer_site_labels.csv      — ICD-O site code → readable label
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE  = Path(__file__).parent.parent
DATA  = Path(__file__).parent.parent.parent / "data/processed/all_cancers.csv"
OUT   = BASE / "data"
ROUT  = BASE / "results/01_matrix"
OUT.mkdir(exist_ok=True); ROUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)

# ICD-O major site groups (3-char C-code → label)
SITE_LABELS = {
    "C00":"Lip","C01":"Base of tongue","C02":"Tongue NOS","C03":"Gum",
    "C04":"Floor of mouth","C05":"Palate","C06":"Mouth NOS",
    "C07":"Parotid gland","C08":"Salivary gland","C09":"Tonsil",
    "C10":"Oropharynx","C11":"Nasopharynx","C12":"Pyriform sinus",
    "C13":"Hypopharynx","C14":"Lip/oral cavity/pharynx NOS",
    "C15":"Esophagus","C16":"Stomach","C17":"Small intestine",
    "C18":"Colon","C19":"Rectosigmoid junction","C20":"Rectum",
    "C21":"Anus","C22":"Liver","C23":"Gallbladder",
    "C24":"Biliary tract NOS","C25":"Pancreas",
    "C30":"Nasal cavity","C31":"Accessory sinuses","C32":"Larynx",
    "C33":"Trachea","C34":"Lung/bronchus","C37":"Thymus",
    "C38":"Heart/mediastinum","C40":"Bone (limbs)","C41":"Bone (other)",
    "C42":"Hematopoietic","C44":"Skin NOS","C47":"Peripheral nerves",
    "C48":"Peritoneum","C49":"Soft tissue",
    "C50":"Breast","C51":"Vulva","C52":"Vagina","C53":"Cervix uteri",
    "C54":"Corpus uteri","C55":"Uterus NOS","C56":"Ovary",
    "C57":"Female genital NOS","C58":"Placenta",
    "C60":"Penis","C61":"Prostate","C62":"Testis",
    "C63":"Male genital NOS","C64":"Kidney","C65":"Renal pelvis",
    "C66":"Ureter","C67":"Bladder","C68":"Urinary NOS",
    "C69":"Eye","C70":"Meninges","C71":"Brain",
    "C72":"CNS NOS","C73":"Thyroid","C74":"Adrenal gland",
    "C75":"Endocrine NOS","C76":"Ill-defined sites",
    "C77":"Lymph nodes","C80":"Primary unknown",
    "C81":"Hodgkin lymphoma","C82":"Follicular lymphoma",
    "C83":"Diffuse B-cell lymphoma","C84":"T/NK lymphoma",
    "C85":"NHL NOS","C88":"Malignant immunoproliferative",
    "C90":"Multiple myeloma","C91":"Lymphoid leukemia",
    "C92":"Myeloid leukemia","C95":"Leukemia NOS",
}

def roc_to_year(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s)!=7: return np.nan
    y = int(s[:3]) + 1911
    return y if 1990 < y < 2030 else np.nan


def load_and_clean():
    df = pd.read_csv(DATA, low_memory=False)
    df["site3"]    = df["腫瘤部位(47)"].astype(str).str.strip().str[:3].str.upper()
    df["site3"]    = df["site3"].where(df["site3"].str.match(r"C\d\d"), other=np.nan)
    df["seq"]      = pd.to_numeric(df["癌症發生順序(34)"], errors="coerce")
    df["diag_yr"]  = df["最初診斷日(45)"].apply(roc_to_year)
    df["age"]      = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]      = df["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"}).fillna("U")
    df["pid"]      = df["病歷號(2)"].astype(str).str.strip()
    # Taiwan registry: 0=Dead, 1=Alive. Column may be float (0.0/1.0) due to NaNs.
    df["event"]    = (pd.to_numeric(df["生存狀態(27)"], errors="coerce")==0).astype(int)
    df = df.dropna(subset=["pid","site3"])
    return df


def build_matrix(df):
    """Multi-hot encode: one row per patient, one col per cancer site."""
    # Use sites with ≥10 patients for tractable dimensions
    site_counts = df.groupby("site3")["pid"].nunique()
    valid_sites = site_counts[site_counts >= 10].index.tolist()
    valid_sites = sorted(valid_sites)
    print(f"  Cancer sites (≥10 patients): {len(valid_sites)}")

    # Pivot: patient × site → 1 if ever diagnosed
    piv = (df[df["site3"].isin(valid_sites)]
           .groupby(["pid","site3"])
           .size()
           .unstack(fill_value=0)
           .clip(upper=1)  # multi-hot: presence only
           .astype(np.uint8))
    piv.columns.name = None
    return piv, valid_sites


def build_patient_meta(df):
    """One row per patient: demographics, first/last diagnosis year, n cancers."""
    agg = df.groupby("pid").agg(
        n_cancers    = ("site3", "nunique"),
        age_first    = ("age", "min"),
        diag_yr_first= ("diag_yr", "min"),
        diag_yr_last = ("diag_yr", "max"),
        sex          = ("sex", "first"),
        any_death    = ("event", "max"),
    ).reset_index()
    agg["multi_primary"] = (agg["n_cancers"] >= 2).astype(int)
    agg["years_span"]    = agg["diag_yr_last"] - agg["diag_yr_first"]
    return agg


if __name__ == "__main__":
    print("Loading registry data...")
    df = load_and_clean()
    print(f"  Total diagnoses (clean): {len(df):,}")
    print(f"  Unique patients:         {df['pid'].nunique():,}")
    print(f"  Unique cancer sites:     {df['site3'].nunique()}")

    print("\nBuilding patient × cancer matrix...")
    matrix, sites = build_matrix(df)
    print(f"  Matrix shape: {matrix.shape}  ({matrix.shape[0]} patients × {matrix.shape[1]} sites)")

    # Basic stats
    n_multi = (matrix.sum(axis=1) >= 2).sum()
    print(f"  Patients with ≥2 cancer types: {n_multi:,} ({100*n_multi/len(matrix):.1f}%)")

    # Patient metadata
    print("\nBuilding patient metadata...")
    meta = build_patient_meta(df)

    # Site labels
    site_df = pd.DataFrame({
        "code":  sites,
        "label": [SITE_LABELS.get(s, s) for s in sites],
        "n_patients": [int(matrix[s].sum()) for s in sites],
    })

    # Save
    matrix.to_csv(OUT / "patient_cancer_matrix.csv", encoding="utf-8-sig")
    meta.to_csv(OUT / "patient_meta.csv", index=False, encoding="utf-8-sig")
    site_df.to_csv(OUT / "cancer_site_labels.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved matrix, meta, labels → {OUT}")

    # ── Figure 1: Cancer prevalence across all patients ──────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    top_sites = site_df.nlargest(30, "n_patients")
    top_sites["label_short"] = top_sites.apply(
        lambda r: f"{r['code']}\n{r['label'][:18]}", axis=1)
    colors = ["#c0392b" if v > 1000 else "#2980b9" for v in top_sites["n_patients"]]
    ax.bar(top_sites["label_short"], top_sites["n_patients"], color=colors, edgecolor="white")
    ax.set(title="Top 30 Cancer Sites — Patient Prevalence", ylabel="Number of patients")
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROUT / "cancer_prevalence.png", dpi=150)
    plt.close(fig)

    # ── Figure 2: Multi-primary distribution ────────────────────────────────
    n_canc_dist = matrix.sum(axis=1).value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(n_canc_dist.index, n_canc_dist.values,
                color=["#27ae60" if i==1 else "#e74c3c" for i in n_canc_dist.index])
    axes[0].set(title="Cancers per Patient (multi-hot count)",
                xlabel="Number of distinct cancer types", ylabel="Patients",
                yscale="log")
    axes[0].spines[["top","right"]].set_visible(False)

    # Top co-occurring pairs heatmap (multi-primary only)
    mp_mat = matrix[matrix.sum(axis=1) >= 2]
    top30 = site_df.nlargest(25,"n_patients")["code"].tolist()
    co_arr = mp_mat[top30].T.dot(mp_mat[top30]).values.astype(float).copy()
    np.fill_diagonal(co_arr, np.nan)
    co = pd.DataFrame(co_arr,
                      index=[SITE_LABELS.get(c, c)[:12] for c in top30],
                      columns=[SITE_LABELS.get(c, c)[:12] for c in top30])
    mask = np.zeros_like(co.values, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(co, mask=mask, cmap="YlOrRd", ax=axes[1], linewidths=0.3,
                cbar_kws={"label": "n co-occurring patients"},
                xticklabels=True, yticklabels=True)
    axes[1].tick_params(axis="both", labelsize=6)
    axes[1].set_title("Co-occurrence Heatmap (multi-primary patients, top 25 sites)")
    fig.tight_layout()
    fig.savefig(ROUT / "cooccurrence_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\nDone. Figures in {ROUT}")
    print(f"Matrix ready for association rules (02), NMF (03), and DL clustering (04).")
