"""Natural-language query engine for CMUH cancer registry Q&A.

Two-tier approach:
  1. Rule-based: keyword → predefined SQL template (works without API key)
  2. Claude NL→SQL: fallback for complex/novel queries (requires ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import pandas as pd

from db import RegistryDB
from site_labels import extract_site, label as site_label

# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    question: str
    answer: str
    df: pd.DataFrame
    chart_type: str = "auto"          # km_bar | forest | sir_bar | line | bar | table
    chart_title: str = ""
    sql: str = ""
    source: str = ""
    suppressed: bool = False
    error: str = ""


# ── intent patterns ───────────────────────────────────────────────────────────

_INTENT_MAP = [
    # (regex pattern, intent_key)
    (r"\b(survival|median os|prognosis|die|death|os\b|overall survival|outcomes?|km\b)\b", "survival"),
    (r"\b(hazard ratios?|hr\b|cox|risk factor[s]?|covariate[s]?)\b", "cox"),
    (r"\b(sir\b|standardised|standardized|second cancer|second primary|second tumou?r)\b", "sir"),
    (r"\b(trends?|increasing|decreasing|over time|temporal|rising|falling|annual)\b", "trend"),
    (r"\b(male|female|sex|gender|sex.differ)\b", "sex"),
    (r"\b(co.?occur\w*|association rules?|association|together|lift|odds ratio)\b", "association"),
    (r"\b(cluster|subgroup|subtype|phenotype|profile|vae)\b", "cluster"),
    (r"\b(treatment|surgery|chemotherapy|chemo\b|radiation|radiotherapy|ccrt|concurrent)\b", "treatment"),
    (r"\b(how many|how much|count|number of cases|incidence|n=|patients)\b", "count"),
    (r"\b(model|transformer|deep learning|dl|r@1|r@3|surveillance|prediction|shap|feature importance)\b", "model"),
    (r"\b(uadt|field cancel|hypopharynx|larynx|pyriform|esophag)\b", "uadt"),
    (r"\b(liver|hbv|hepatitis|c22\b)\b", "hbv"),
]

_SURVIVAL_STRAT = [
    # More specific patterns first to avoid false matches
    (r"\bccrt\b|\bconcurrent\b", "ccrt"),
    (r"\bregimens?\b|\bplatinum\b|\bcisplatin\b|\bchemotherapy regimen\b", "chemo_regimen"),
    (r"\bsurgery type\b|\besophagect\b", "surgery_type"),
    (r"\bmargin\b|\bresect", "surgery_margin"),
    (r"\blymph node\b|\bln\b", "ln_extent"),
    (r"\bstage\b|\bajcc\b", "stage"),
    (r"\bhistolog\w*|\bscc\b|\badeno\w*", "histology"),
    (r"\bsex\b|\bmale\b|\bfemale\b", "sex"),
    (r"\bsurgery\b|\bsurgical\b|\boperat", "surgery"),
    (r"\bchemos?\b|\bchemotherapy\b", "chemo"),
    (r"\bsubsite\b|\bthoracic\b|\bcervical\b|\babdominal\b", "subsite"),
]


def _classify(question: str) -> tuple[str, str | None]:
    """Return (intent, site_code)."""
    q = question.lower()
    site = extract_site(q)

    for pattern, intent in _INTENT_MAP:
        if re.search(pattern, q, re.IGNORECASE):
            return intent, site

    return "general", site


def _survival_strat(question: str) -> str:
    q = question.lower()
    for pattern, strat in _SURVIVAL_STRAT:
        if re.search(pattern, q, re.IGNORECASE):
            return strat
    return "stage"


# ── SQL templates ─────────────────────────────────────────────────────────────

def _build_sql(intent: str, site: str | None, question: str) -> tuple[str, str, str, str]:
    """Return (sql, chart_type, chart_title, source_table)."""
    q = question.lower()

    if intent == "survival":
        if "escc" in q:
            return ("SELECT * FROM escc_survival", "km_bar",
                    "CMUH ESCC Detailed Dataset — Survival Summary", "escc_survival")
        strat = _survival_strat(question)
        if site and site != "C15":
            return (
                f"SELECT * FROM sex_age_by_site WHERE site = '{site}'",
                "table", f"Demographics for {site} {site_label(site)}", "sex_age_by_site"
            )
        table_map = {
            "stage":          ("c15_km_stage",          "km_bar",  "C15 Median OS by AJCC Stage"),
            "histology":      ("c15_km_histology",       "km_bar",  "C15 Median OS by Histology"),
            "sex":            ("c15_km_sex",             "km_bar",  "C15 Median OS by Sex"),
            "surgery":        ("c15_km_surgery",         "km_bar",  "C15 Median OS by Surgery"),
            "chemo":          ("c15_km_chemo",           "km_bar",  "C15 Median OS by Chemotherapy"),
            "subsite":        ("c15_km_subsite",         "km_bar",  "C15 Median OS by Tumour Subsite"),
            "ccrt":           ("c15_km_ccrt",            "km_bar",  "C15 Median OS by CCRT"),
            "chemo_regimen":  ("c15_km_chemo_regimen",   "km_bar",  "C15 Median OS by Chemo Regimen"),
            "surgery_type":   ("c15_km_surgery_type",    "km_bar",  "C15 Median OS by Surgery Type"),
            "surgery_margin": ("c15_km_surgery_margin",  "km_bar",  "C15 Median OS by Surgical Margin"),
            "ln_extent":      ("c15_km_ln_extent",       "km_bar",  "C15 Median OS by LN Extent"),
        }
        tbl, ct, ttl = table_map.get(strat, ("c15_km_stage", "km_bar", "C15 Median OS by Stage"))
        return f"SELECT * FROM {tbl}", ct, ttl, tbl

    if intent == "cox":
        if "escc" in q or "cmuh" in q:
            return ("SELECT * FROM escc_cox LIMIT 20", "forest",
                    "CMUH ESCC Detailed Cox Hazard Ratios", "escc_cox")
        if "uadt" in q or "field" in q:
            tv = "tv" in q or "time.vary" in q
            tbl = "uadt_cox_tv" if tv else "uadt_cox"
            return (f"SELECT * FROM {tbl}", "forest",
                    f"UADT Field Cox HR ({'time-varying' if tv else 'standard'})", tbl)
        if "treatment" in q or "surgery" in q or "chemo" in q:
            return ("SELECT * FROM c15_cox_treatment", "forest",
                    "C15 Treatment Cox Hazard Ratios", "c15_cox_treatment")
        return ("SELECT * FROM c15_cox", "forest",
                "C15 Esophageal Cox Hazard Ratios", "c15_cox")

    if intent == "sir":
        if site:
            return (
                f"SELECT * FROM sir_second_primary WHERE index = '{site}' "
                f"AND fdr < 0.05 ORDER BY sir DESC LIMIT 20",
                "sir_bar",
                f"SIR: Second primary cancers after {site} {site_label(site)}",
                "sir_second_primary",
            )
        if "c22" in q or "liver" in q or "hbv" in q:
            return (
                "SELECT * FROM sir_c22_by_index ORDER BY sir DESC LIMIT 20",
                "sir_bar", "SIR: Cancer risk by site after Liver (C22)", "sir_c22_by_index"
            )
        if "uadt" in q:
            return (
                "SELECT * FROM sir_uadt_field WHERE fdr < 0.05 ORDER BY sir DESC LIMIT 20",
                "sir_bar", "UADT Field Cancerization SIR", "sir_uadt_field"
            )
        return (
            "SELECT * FROM sir_second_primary WHERE fdr < 0.05 ORDER BY sir DESC LIMIT 30",
            "sir_bar", "Top Significant Second Primary Cancer SIRs", "sir_second_primary"
        )

    if intent == "trend":
        if site:
            return (
                f"SELECT * FROM site_trends WHERE site = '{site}'",
                "table", f"Temporal Trend for {site} {site_label(site)}", "site_trends"
            )
        if "rising" in q or "increas" in q:
            return (
                "SELECT site, direction, rho, n_first FROM site_trends "
                "WHERE direction = 'rising' AND sig = true ORDER BY rho DESC",
                "bar", "Cancer Sites with Significant Rising Incidence", "site_trends"
            )
        if "falling" in q or "decreas" in q:
            return (
                "SELECT site, direction, rho, n_first FROM site_trends "
                "WHERE direction = 'falling' AND sig = true ORDER BY rho",
                "bar", "Cancer Sites with Significant Falling Incidence", "site_trends"
            )
        return (
            "SELECT site, direction, rho, p_corrected, n_first FROM site_trends "
            "WHERE sig = true ORDER BY ABS(rho) DESC LIMIT 20",
            "bar", "Significant Temporal Trends (2003–2020)", "site_trends"
        )

    if intent == "sex":
        if site:
            return (
                f'SELECT site, "or", or_lo, or_hi, n_m, n_f, p_fisher FROM sex_odds_ratios WHERE site = \'{site}\'',
                "table", f"Sex Differences: {site} {site_label(site)}", "sex_odds_ratios"
            )
        return (
            'SELECT site, "or", or_lo, or_hi, n_m, n_f FROM sex_odds_ratios '
            "WHERE sig = true ORDER BY \"or\" DESC LIMIT 20",
            "forest", "Male-to-Female Odds Ratios by Cancer Site", "sex_odds_ratios"
        )

    if intent == "association":
        if site:
            return (
                f"SELECT * FROM assoc_rules WHERE antecedent = '{site}' OR consequent = '{site}' "
                f"ORDER BY lift DESC LIMIT 20",
                "bar", f"Cancer Co-occurrence Rules for {site} {site_label(site)}", "assoc_rules"
            )
        sex_filter = ""
        if "male" in q:
            tbl = "assoc_rules_male"
        elif "female" in q or "woman" in q:
            tbl = "assoc_rules_female"
        elif "early" in q:
            tbl = "assoc_rules_early"
        elif "late" in q:
            tbl = "assoc_rules_late"
        else:
            tbl = "assoc_rules"
        return (
            f"SELECT antecedent_label, consequent_label, lift, odds_ratio, n_co "
            f"FROM {tbl} ORDER BY lift DESC LIMIT 20",
            "bar", "Top Cancer Co-occurrence Association Rules", tbl
        )

    if intent == "cluster":
        if "c15" in q or "esophageal" in q:
            return ("SELECT * FROM c15_cluster_profiles", "table",
                    "C15 Esophageal Cancer Cluster Profiles", "c15_cluster_profiles")
        if "escc" in q or "cmuh" in q:
            return ("SELECT * FROM escc_dl_cindex", "table",
                    "CMUH ESCC Deep Learning C-index Summary", "escc_dl_cindex")
        return ("SELECT * FROM vae_clusters", "table",
                "VAE Latent-Space Cluster Profiles", "vae_clusters")

    if intent == "treatment":
        if "feature" in q or "importance" in q or "shap" in q:
            if "escc" in q:
                return ("SELECT * FROM escc_shap ORDER BY mean_abs_shap DESC", "bar",
                        "CMUH ESCC SHAP Feature Importance", "escc_shap")
            return ("SELECT * FROM c15_feature_importance ORDER BY importance DESC", "bar",
                    "C15 DeepSurv Feature Importance", "c15_feature_importance")
        if "combination" in q or "combo" in q:
            return ("SELECT * FROM c15_treatment_combos ORDER BY n DESC LIMIT 20", "bar",
                    "C15 Treatment Combinations", "c15_treatment_combos")
        return ("SELECT * FROM c15_treatment_summary", "table",
                "C15 Treatment Summary with Median OS", "c15_treatment_summary")

    if intent == "count":
        if "escc" in q:
            return ("SELECT * FROM escc_survival", "km_bar",
                    "CMUH ESCC Survival Summary", "escc_survival")
        if site:
            return (
                f"SELECT * FROM sex_odds_ratios WHERE site = '{site}'",
                "table", f"Case counts for {site} {site_label(site)}", "sex_odds_ratios"
            )
        return (
            "SELECT site, n_m + n_f AS n_total, n_m, n_f FROM sex_odds_ratios "
            "ORDER BY n_total DESC LIMIT 20",
            "bar", "Total Cases by Cancer Site", "sex_odds_ratios"
        )

    if intent == "model":
        if "shap" in q or "feature" in q or "importance" in q:
            if "escc" in q or "cmuh" in q:
                return ("SELECT * FROM escc_shap ORDER BY mean_abs_shap DESC", "bar",
                        "CMUH ESCC SHAP Feature Importance", "escc_shap")
            return ("SELECT * FROM c15_feature_importance ORDER BY importance DESC", "bar",
                    "C15 DeepSurv Feature Importance", "c15_feature_importance")
        if "surveillance" in q or "predict" in q or "timing" in q:
            return ("SELECT * FROM surveillance_summary", "table",
                    "Surveillance Calendar Validation Summary", "surveillance_summary")
        return ("SELECT * FROM model_performance", "table",
                "Deep Learning Model Performance (R@k)", "model_performance")

    if intent == "uadt":
        if "cox" in q or "survival" in q or "hr" in q:
            return ("SELECT * FROM uadt_cox_tv", "forest",
                    "UADT Multi-site Field Cox (Time-Varying)", "uadt_cox_tv")
        if "pair" in q or "co-occur" in q:
            return (
                "SELECT label_a, label_b, n_co, lift, or_ FROM uadt_pairs ORDER BY lift DESC LIMIT 20",
                "bar", "UADT Field Cancerization Pairs", "uadt_pairs"
            )
        return (
            "SELECT * FROM sir_uadt_field WHERE fdr < 0.05 ORDER BY sir DESC LIMIT 20",
            "sir_bar", "UADT Field SIR (FDR < 0.05)", "sir_uadt_field"
        )

    if intent == "hbv":
        return ("SELECT diag_yr, c22_rate_pct FROM c22_trend ORDER BY diag_yr",
                "line", "Liver Cancer (C22) Annual Rate 2003–2020", "c22_trend")

    # Fallback: demographics overview
    return ("SELECT * FROM c15_demographics LIMIT 30", "table",
            "C15 Esophageal Cancer — Cohort Overview", "c15_demographics")


# ── Claude NL→SQL fallback ────────────────────────────────────────────────────

def _claude_sql(question: str, schema: str, api_key: str) -> tuple[str, str, str]:
    """Use Claude tool-use to generate SQL. Returns (sql, chart_type, title)."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        tools = [{
            "name": "query_registry",
            "description": "Execute a DuckDB SELECT on CMUH cancer registry aggregated tables.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "Valid DuckDB SELECT statement."},
                    "chart_type": {
                        "type": "string",
                        "enum": ["km_bar", "forest", "sir_bar", "line", "bar", "table"],
                    },
                    "title": {"type": "string"},
                },
                "required": ["sql", "chart_type", "title"],
            },
        }]

        system = (
            "You are a DuckDB SQL expert for the CMUH cancer registry Q&A system. "
            "Generate ONLY SELECT statements. All tables contain aggregated statistics "
            "— no individual patient data. Use exact column names from the schema.\n\n"
            + schema
        )

        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system,
            tools=tools,
            messages=[{"role": "user", "content": question}],
        )

        for block in resp.content:
            if block.type == "tool_use":
                inp = block.input
                return inp.get("sql", ""), inp.get("chart_type", "table"), inp.get("title", "")

    except Exception:
        pass

    return "", "table", ""


# ── main engine ───────────────────────────────────────────────────────────────

class QueryEngine:
    def __init__(self, db: RegistryDB) -> None:
        self._db = db
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def set_api_key(self, key: str) -> None:
        self._api_key = key.strip()

    def answer(self, question: str) -> QueryResult:
        intent, site = _classify(question)
        sql, chart_type, chart_title, source = _build_sql(intent, site, question)

        try:
            df = self._db.execute(sql)
        except Exception as e:
            # Rule-based SQL failed — try Claude if key available
            if self._api_key:
                sql, chart_type, chart_title = _claude_sql(
                    question, self._db.schema_text(), self._api_key
                )
                if sql:
                    try:
                        df = self._db.execute(sql)
                        source = "claude_generated"
                    except Exception as e2:
                        return QueryResult(
                            question=question, answer="", df=pd.DataFrame(),
                            error=f"Query failed: {e2}"
                        )
                else:
                    return QueryResult(
                        question=question, answer="", df=pd.DataFrame(),
                        error=f"Could not generate query: {e}"
                    )
            else:
                return QueryResult(
                    question=question, answer="", df=pd.DataFrame(),
                    error=str(e)
                )

        suppressed = df.attrs.get("suppressed", 0) > 0
        answer_text = _narrate(df, chart_type, chart_title, site)

        return QueryResult(
            question=question,
            answer=answer_text,
            df=df,
            chart_type=chart_type,
            chart_title=chart_title,
            sql=sql,
            source=source,
            suppressed=suppressed,
        )


def _narrate(df: pd.DataFrame, chart_type: str, title: str, site: str | None) -> str:
    """Generate a one-paragraph natural language answer from the result."""
    if df.empty:
        return "No data found for this query."

    n = len(df)

    if chart_type == "km_bar":
        # Find best row (best survival)
        os_col = next((c for c in df.columns if "median_os" in c or "median_css" in c), None)
        grp_col = next((c for c in df.columns if c in ("group", "Group")), None)
        n_col   = next((c for c in df.columns if c in ("n", "N")), None)
        if os_col and grp_col:
            best = df.loc[pd.to_numeric(df[os_col], errors="coerce").idxmax()]
            worst = df.loc[pd.to_numeric(df[os_col], errors="coerce").idxmin()]
            return (
                f"**{title}** — {n} groups shown. "
                f"Best prognosis: **{best[grp_col]}** (median OS {best[os_col]} mo"
                + (f", n={best[n_col]}" if n_col else "")
                + f"). Worst prognosis: **{worst[grp_col]}** ({worst[os_col]} mo)."
            )

    if chart_type in ("forest", "sir_bar"):
        hr_col = next((c for c in df.columns if c in ("hr", "sir", "or_", "exp_coef", "exp_coef_")), None)
        lbl_col = next((c for c in df.columns if "label" in c or c == "covariate"), None)
        if hr_col and lbl_col:
            top = df.loc[pd.to_numeric(df[hr_col], errors="coerce").idxmax()]
            return (
                f"**{title}** — {n} covariates/sites. "
                f"Largest effect: **{top[lbl_col]}** ({hr_col.upper()}={top[hr_col]:.2f})."
            )

    if chart_type == "bar":
        val_col = next((c for c in df.columns if c in ("rho", "lift", "odds_ratio", "or_", "sir")), None)
        lbl_col = next((c for c in df.columns if "label" in c or "site" in c or c == "antecedent_label"), None)
        if val_col and lbl_col and not df.empty:
            top = df.iloc[0]
            return (
                f"**{title}** — top result: **{top[lbl_col]}** ({val_col}={top[val_col]:.2f}). "
                f"{n} results shown."
            )

    return f"**{title}** — {n} rows returned. See chart and table below."
