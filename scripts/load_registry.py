"""
Decrypt and load Taiwan Cancer Registry long-form Excel file.
Returns a dict of {sheet_name: DataFrame} or a merged DataFrame.
"""
import msoffcrypto
import io
import openpyxl
import pandas as pd
from pathlib import Path

RAW_FILE = Path(__file__).parent.parent / "data/raw/cancer_registry_92-109.xlsx"
PASSWORD = "3566"

SHEET_VERSIONS = [
    "92-95年長表Aver",
    "92-99年長表Bver",
    "100年長表Cver",
    "92-106年長表DEver",
    "92-108年長表FGver",
    "92-109年長表Hver",
]


def _decrypt(path: Path, password: str) -> io.BytesIO:
    with open(path, "rb") as f:
        office_file = msoffcrypto.OfficeFile(f)
        office_file.load_key(password=password)
        buf = io.BytesIO()
        office_file.decrypt(buf)
    buf.seek(0)
    return buf


def load_sheets(path: Path = RAW_FILE, password: str = PASSWORD) -> dict[str, pd.DataFrame]:
    buf = _decrypt(path, password)
    wb = openpyxl.load_workbook(buf, read_only=True)
    sheets = {}
    for name in SHEET_VERSIONS:
        ws = wb[name]
        rows = list(ws.iter_rows(values_only=True))
        cols = rows[0]
        sheets[name] = pd.DataFrame(rows[1:], columns=cols)
    wb.close()
    return sheets


def load_all(path: Path = RAW_FILE, password: str = PASSWORD) -> pd.DataFrame:
    """Combine all sheets, keeping only columns present in every sheet."""
    sheets = load_sheets(path, password)
    frames = []
    for name, df in sheets.items():
        df = df.copy()
        df["_sheet"] = name
        frames.append(df)
    common_cols = set(frames[0].columns)
    for df in frames[1:]:
        common_cols &= set(df.columns)
    common_cols = list(common_cols) + ["_sheet"]
    merged = pd.concat([df[common_cols] for df in frames], ignore_index=True)
    return merged
