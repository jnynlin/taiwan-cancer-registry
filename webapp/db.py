"""DuckDB loader for CMUH cancer registry aggregated result files."""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

REGISTRY_ROOT = Path("/home/jnynlin/coding/taiwan-cancer-registry")
ESCC_ROOT = Path("/home/jnynlin/coding/escc_cmuh")

MIN_GROUP_COUNT = 5

PID_COLUMNS = {"pid", "病歷號", "病歷號(2)", "patient_id", "subject_id", "reg_no"}

# ── simple CSV tables (load as-is) ───────────────────────────────────────────
_SIMPLE = {
    # C15 esophageal cancer
    "c15_km_stage":          REGISTRY_ROOT / "C15_esophageal/results/03_survival/km_stage_medians.csv",
    "c15_km_histology":      REGISTRY_ROOT / "C15_esophageal/results/03_survival/km_histology_medians.csv",
    "c15_km_sex":            REGISTRY_ROOT / "C15_esophageal/results/03_survival/km_sex_medians.csv",
    "c15_km_surgery":        REGISTRY_ROOT / "C15_esophageal/results/03_survival/km_surgery_medians.csv",
    "c15_km_chemo":          REGISTRY_ROOT / "C15_esophageal/results/03_survival/km_chemo_medians.csv",
    "c15_km_subsite":        REGISTRY_ROOT / "C15_esophageal/results/03_survival/km_subsite_medians.csv",
    "c15_km_css_stage":      REGISTRY_ROOT / "C15_esophageal/results/03_survival/km_css_stage_medians.csv",
    "c15_cox":               REGISTRY_ROOT / "C15_esophageal/results/03_survival/cox_summary.csv",
    "c15_km_ccrt":           REGISTRY_ROOT / "C15_esophageal/results/06_chemo_surgery/km_ccrt_medians.csv",
    "c15_km_surgery_type":   REGISTRY_ROOT / "C15_esophageal/results/06_chemo_surgery/km_surgery_type_medians.csv",
    "c15_km_chemo_regimen":  REGISTRY_ROOT / "C15_esophageal/results/06_chemo_surgery/km_chemo_regimen_medians.csv",
    "c15_km_chemo_method":   REGISTRY_ROOT / "C15_esophageal/results/06_chemo_surgery/km_chemo_method_medians.csv",
    "c15_km_surgery_margin": REGISTRY_ROOT / "C15_esophageal/results/06_chemo_surgery/km_surgery_margin_medians.csv",
    "c15_km_ln_extent":      REGISTRY_ROOT / "C15_esophageal/results/06_chemo_surgery/km_ln_extent_medians.csv",
    "c15_km_rt_sequence":    REGISTRY_ROOT / "C15_esophageal/results/06_chemo_surgery/km_rt_surgery_sequence_medians.csv",
    "c15_cox_treatment":     REGISTRY_ROOT / "C15_esophageal/results/06_chemo_surgery/cox_treatment_summary.csv",
    "c15_treatment_summary": REGISTRY_ROOT / "C15_esophageal/results/06_chemo_surgery/chemo_surgery_summary.csv",
    "c15_demographics":      REGISTRY_ROOT / "C15_esophageal/results/02_descriptive/table1_demographics.csv",
    "c15_treatment_combos":  REGISTRY_ROOT / "C15_esophageal/results/02_descriptive/treatment_combinations.csv",
    "c15_cluster_profiles":  REGISTRY_ROOT / "C15_esophageal/results/04_deep_learning/cluster_kmeans_profile.csv",
    "c15_feature_importance":REGISTRY_ROOT / "C15_esophageal/results/04_deep_learning/deepsurv_feature_importance.csv",
    "c15_sensitivity":       REGISTRY_ROOT / "C15_esophageal/results/sensitivity/cox_timesplit_mice.csv",
    # Registry-wide
    "site_trends":           REGISTRY_ROOT / "dl_vae/results/18_temporal/trend_by_site.csv",
    "c22_trend":             REGISTRY_ROOT / "dl_vae/results/15_hbv/c22_trend.csv",
    "sex_odds_ratios":       REGISTRY_ROOT / "dl_vae/results/23_sex/sex_or_by_site.csv",
    "sex_age_by_site":       REGISTRY_ROOT / "dl_vae/results/23_sex/age_sex_by_site.csv",
    "sir_second_primary":    REGISTRY_ROOT / "coexist_cancers/results/05_sir_trajectories/sir_second_primary.csv",
    "sir_uadt_field":        REGISTRY_ROOT / "uadt_field/results/03_sir/sir_field.csv",
    "sir_c22_by_index":      REGISTRY_ROOT / "dl_vae/results/15_hbv/sir_c22_by_index.csv",
    "assoc_rules":           REGISTRY_ROOT / "coexist_cancers/results/02_associations/association_rules_sig.csv",
    "assoc_rules_male":      REGISTRY_ROOT / "coexist_cancers/results/02_associations/association_rules_male.csv",
    "assoc_rules_female":    REGISTRY_ROOT / "coexist_cancers/results/02_associations/association_rules_female.csv",
    "assoc_rules_early":     REGISTRY_ROOT / "coexist_cancers/results/02_associations/association_rules_early_onset.csv",
    "assoc_rules_late":      REGISTRY_ROOT / "coexist_cancers/results/02_associations/association_rules_late_onset.csv",
    "vae_clusters":          REGISTRY_ROOT / "dl_vae/results/03_latent/cluster_profiles.csv",
    "vae_axis":              REGISTRY_ROOT / "dl_vae/results/03_latent/axis_interpretation.csv",
    "coexist_clusters":      REGISTRY_ROOT / "coexist_cancers/results/04_clustering/cluster_k3_profile.csv",
    "uadt_cox":              REGISTRY_ROOT / "uadt_field/results/05_survival/cox_results.csv",
    "uadt_cox_tv":           REGISTRY_ROOT / "uadt_field/results/05_survival/cox_tv_results.csv",
    "uadt_trajectories":     REGISTRY_ROOT / "uadt_field/results/04_trajectories/field_trajectories.csv",
    "uadt_pairs":            REGISTRY_ROOT / "uadt_field/results/02_pairs/field_pairs_fdr.csv",
    "model_performance":     REGISTRY_ROOT / "dl_vae/results/08_transformer_eval/model_comparison.csv",
    "surveillance_timing":   REGISTRY_ROOT / "dl_vae/results/12_surveillance/timing_windows.csv",
    "surveillance_summary":  REGISTRY_ROOT / "dl_vae/results/13_surveillance/validation_summary.csv",
    # ESCC CMUH
    "escc_survival":         ESCC_ROOT / "results/03_survival/survival_summary.csv",
    "escc_cox":              ESCC_ROOT / "results/03_survival/cox_primary_summary.csv",
    "escc_validation":       ESCC_ROOT / "results/04_validation/validation_vs_C15.csv",
    "escc_shap":             ESCC_ROOT / "results/05_dl/method2_shap_summary.csv",
    "escc_dl_cindex":        ESCC_ROOT / "results/05_dl/dl_summary_cindex.csv",
}


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM, lowercase, snake_case column names; drop PID columns; fix bool types."""
    df.columns = [
        re.sub(r"[^\w]", "_", c.strip().lstrip("﻿").lower()).strip("_")
        for c in df.columns
    ]
    drop = [c for c in df.columns if c in PID_COLUMNS or c.startswith("病歷")]
    df = df.drop(columns=drop, errors="ignore")

    # Convert "True"/"False" string columns to actual Python booleans so DuckDB
    # correctly handles WHERE col = true / col = false predicates.
    for col in df.columns:
        if df[col].dtype == object:
            uniq = set(df[col].dropna().unique())
            if uniq.issubset({"True", "False"}):
                df[col] = df[col].map({"True": True, "False": False})

    return df


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        return _normalise_cols(df)
    except Exception:
        return None


def _load_annual_proportions() -> pd.DataFrame | None:
    """Melt wide site×year fraction table to long format."""
    path = REGISTRY_ROOT / "dl_vae/results/18_temporal/annual_fraction_by_site.csv"
    df = _load_csv(path)
    if df is None:
        return None
    id_col = "diag_yr"
    site_cols = [c for c in df.columns if c != id_col]
    melted = df.melt(id_vars=id_col, value_vars=site_cols, var_name="site", value_name="fraction")
    melted["site"] = melted["site"].str.upper()
    return melted


def _load_nmf_weights() -> pd.DataFrame | None:
    """Melt wide NMF component×site weight table to long format."""
    path = REGISTRY_ROOT / "coexist_cancers/results/03_nmf/component_weights.csv"
    df = _load_csv(path)
    if df is None:
        return None
    # First column is component index
    id_col = df.columns[0]
    site_cols = [c for c in df.columns if c != id_col]
    melted = df.melt(id_vars=id_col, value_vars=site_cols, var_name="site", value_name="weight")
    melted.rename(columns={id_col: "component"}, inplace=True)
    melted["site"] = melted["site"].str.upper()
    return melted


class RegistryDB:
    """In-memory DuckDB with all safe aggregated result tables."""

    def __init__(self) -> None:
        self._con = duckdb.connect(database=":memory:")
        self._loaded: dict[str, int] = {}  # table_name → row count
        self._schema_cache: str | None = None
        self._load_all()

    def _register(self, name: str, df: pd.DataFrame | None) -> None:
        if df is None or df.empty:
            return
        self._con.register(name, df)
        self._loaded[name] = len(df)

    def _load_all(self) -> None:
        for name, path in _SIMPLE.items():
            self._register(name, _load_csv(path))

        self._register("annual_proportions", _load_annual_proportions())
        self._register("nmf_weights", _load_nmf_weights())

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def tables(self) -> list[str]:
        return sorted(self._loaded.keys())

    def table_row_count(self, name: str) -> int:
        return self._loaded.get(name, 0)

    def execute(self, sql: str) -> pd.DataFrame:
        """Execute a SELECT query; enforce privacy guardrail on result."""
        sql = sql.strip()
        if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
            raise ValueError("Only SELECT statements are permitted.")
        if re.search(r"\b(DROP|INSERT|UPDATE|DELETE|CREATE|ALTER|TRUNCATE)\b", sql, re.IGNORECASE):
            raise ValueError("DDL/DML statements are not permitted.")

        # Ensure LIMIT present
        if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
            sql = sql.rstrip(";") + " LIMIT 500"

        df = self._con.execute(sql).df()
        return self._privacy_filter(df)

    def _privacy_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Suppress rows where patient count < MIN_GROUP_COUNT."""
        count_cols = [c for c in df.columns if c in ("n", "n_index", "n_co", "obs", "n_m", "n_f", "count_")]
        for col in count_cols:
            numeric = pd.to_numeric(df[col], errors="coerce")
            mask = numeric < MIN_GROUP_COUNT
            if mask.any():
                df = df[~mask].copy()
                df.attrs["suppressed"] = int(mask.sum())
                break
        return df

    def sample(self, table: str, n: int = 5) -> pd.DataFrame:
        return self._con.execute(f"SELECT * FROM {table} LIMIT {n}").df()

    def schema_text(self) -> str:
        """Return a schema description string for the LLM system prompt."""
        if self._schema_cache:
            return self._schema_cache

        lines = ["Available DuckDB tables (all contain aggregate statistics, NO patient-level data):\n"]
        for name in sorted(self._loaded.keys()):
            try:
                df = self._con.execute(f"SELECT * FROM {name} LIMIT 3").df()
                cols = ", ".join(df.columns.tolist())
                sample_row = df.iloc[0].to_dict() if len(df) > 0 else {}
                sample_str = "; ".join(f"{k}={v}" for k, v in list(sample_row.items())[:5])
                lines.append(f"  {name}: columns=[{cols}] | example: {sample_str}")
            except Exception:
                lines.append(f"  {name}: (unavailable)")

        self._schema_cache = "\n".join(lines)
        return self._schema_cache


_singleton: RegistryDB | None = None


def get_db() -> RegistryDB:
    global _singleton
    if _singleton is None:
        _singleton = RegistryDB()
    return _singleton
