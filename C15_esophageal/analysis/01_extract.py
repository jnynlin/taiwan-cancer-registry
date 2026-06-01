"""
Extract esophageal cancer (ICD-10 C15.x) from all registry sheets.
Output: C15_esophageal/data/c15_all.csv
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from load_registry import load_sheets

OUT_DIR = Path(__file__).parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def roc_to_ad(roc_str):
    """Convert ROC date (YYYMMDD as string or integer) to pd.Timestamp.
    Handles 6-digit integers (leading zero dropped by Excel) and
    imputes day/month=99 or 00 as 01 (first of period).
    """
    s = str(roc_str).strip().split(".")[0]  # drop decimal if float
    if not s.isdigit() or len(s) < 5:
        return pd.NaT
    s = s.zfill(7)  # restore leading zero: 920528 → 0920528
    yr = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00", "99") else "01"
    dd = s[5:7] if s[5:7] not in ("00", "99") else "01"
    try:
        return pd.Timestamp(f"{yr}-{mm}-{dd}")
    except Exception:
        return pd.NaT


def extract_c15(sheets: dict) -> pd.DataFrame:
    frames = []
    for sheet_name, df in sheets.items():
        c15 = df[df["腫瘤部位(47)"].astype(str).str.startswith("C15")].copy()
        c15["_sheet"] = sheet_name
        frames.append(c15)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert ROC dates to AD
    for col_roc, col_ad in [
        ("最初診斷日(45)", "diag_date"),
        ("死亡日期(31)", "death_date"),
        ("最後聯絡日(30)", "last_contact_date"),
    ]:
        if col_roc in df.columns:
            df[col_ad] = df[col_roc].apply(roc_to_ad)

    # Diagnosis year (AD)
    if "diag_date" in df.columns:
        df["diag_year"] = df["diag_date"].dt.year

    # Sex label
    if "性別(5)" in df.columns:
        df["sex"] = df["性別(5)"].map({"1": "Male", "2": "Female", 1: "Male", 2: "Female"})

    # Vital status: Taiwan registry codes 0=Dead, 1=Alive
    if "生存狀態(27)" in df.columns:
        df["vital_status"] = df["生存狀態(27)"].map({"0": "Dead", "1": "Alive", 0: "Dead", 1: "Alive"})

    # OS in days: dead → death_date, alive → last_contact_date
    if "diag_date" in df.columns and "death_date" in df.columns and "last_contact_date" in df.columns:
        end_date = df["death_date"].where(df["vital_status"] == "Dead", df["last_contact_date"])
        df["os_days"] = (end_date - df["diag_date"]).dt.days

    # Esophageal subsite label
    subsite_map = {
        "C150": "Cervical esophagus",
        "C151": "Thoracic esophagus (upper)",
        "C152": "Thoracic esophagus (middle)",
        "C153": "Thoracic esophagus (lower)",
        "C154": "Abdominal esophagus",
        "C155": "Overlapping esophagus",
        "C158": "Overlapping esophagus",
        "C159": "Esophagus NOS",
    }
    df["subsite"] = df["腫瘤部位(47)"].astype(str).map(subsite_map).fillna("Other/Unknown")

    return df


if __name__ == "__main__":
    print("Loading registry sheets...")
    sheets = load_sheets()

    print("Extracting C15 cases...")
    df = extract_c15(sheets)
    print(f"  Total C15 cases: {len(df)}")
    print(f"  Sheet breakdown:\n{df['_sheet'].value_counts().to_string()}")

    print("Adding derived columns...")
    df = add_derived_columns(df)

    out_path = OUT_DIR / "c15_all.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}  ({len(df)} rows x {df.shape[1]} cols)")

    # Quick summary
    print("\n--- Quick summary ---")
    print(f"Diagnosis years: {df['diag_year'].min():.0f}–{df['diag_year'].max():.0f}")
    print(f"Sex distribution:\n{df['sex'].value_counts().to_string()}")
    print(f"Subsite:\n{df['subsite'].value_counts().to_string()}")
    print(f"Vital status:\n{df['vital_status'].value_counts().to_string()}")
    print(f"OS days (median): {df['os_days'].median():.0f}")
