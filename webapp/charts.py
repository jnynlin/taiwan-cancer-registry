"""Plotly chart generators for CMUH registry Q&A results."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def km_bar(df: pd.DataFrame, title: str = "Median Overall Survival") -> go.Figure:
    """Horizontal bar chart of median OS by group with n annotation."""
    y_col = _find_col(df, ["group", "Group"])
    x_col = _find_col(df, ["median_os_months", "median_os__months_", "median_css_months",
                            "median_os__mo_", "median_os"])
    n_col = _find_col(df, ["n", "N"])

    if y_col is None or x_col is None:
        return _fallback_table(df, title)

    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df = df.dropna(subset=[x_col])

    label = x_col.replace("_", " ").replace("months", "(months)").title()
    fig = px.bar(
        df, x=x_col, y=y_col, orientation="h",
        text=n_col if n_col else None,
        color=y_col,
        labels={x_col: label, y_col: "Group"},
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_layout(showlegend=False, height=max(300, len(df) * 60 + 80),
                      margin=dict(l=160, r=40, t=50, b=40))
    return fig


def forest_plot(df: pd.DataFrame, title: str = "Hazard Ratios") -> go.Figure:
    """Forest plot for Cox HR or OR results."""
    label_col = _find_col(df, ["covariate", "variable", "feature", "term", "index_"])
    hr_col    = _find_col(df, ["hr", "exp_coef", "exp(coef)", "or", "or_", "odds_ratio", "sir"])
    lo_col    = _find_col(df, ["hr_lower95", "lo", "ci_lo", "or_lo", "or_lo_", "ci_low",
                                "coef_lower_95_", "exp_coef__lower_95_"])
    hi_col    = _find_col(df, ["hr_upper95", "hi", "ci_hi", "or_hi", "ci_high",
                                "coef_upper_95_", "exp_coef__upper_95_"])

    if any(c is None for c in [label_col, hr_col]):
        return _fallback_table(df, title)

    df = df.copy()
    for col in [hr_col, lo_col, hi_col]:
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[hr_col])

    err_plus  = (df[hi_col] - df[hr_col]).clip(lower=0) if hi_col else None
    err_minus = (df[hr_col] - df[lo_col]).clip(lower=0) if lo_col else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[hr_col],
        y=df[label_col].astype(str),
        mode="markers",
        marker=dict(size=10, color="steelblue"),
        error_x=dict(
            type="data", symmetric=False,
            array=err_plus.tolist() if err_plus is not None else None,
            arrayminus=err_minus.tolist() if err_minus is not None else None,
        ) if lo_col and hi_col else None,
        name="Estimate",
    ))
    fig.add_vline(x=1, line_dash="dash", line_color="grey", opacity=0.6)
    fig.update_xaxes(type="log", title=hr_col.upper().replace("_", " "))
    fig.update_layout(
        title=title,
        height=max(300, len(df) * 45 + 80),
        margin=dict(l=180, r=40, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def sir_bar(df: pd.DataFrame, title: str = "Standardised Incidence Ratios") -> go.Figure:
    """Horizontal bar chart for SIR results sorted by magnitude."""
    label_col = _find_col(df, ["target_label", "index_site", "index_label", "label"])
    sir_col   = _find_col(df, ["sir", "SIR"])
    lo_col    = _find_col(df, ["ci_low", "ci_lo", "lo"])
    hi_col    = _find_col(df, ["ci_high", "ci_hi", "hi"])

    if label_col is None or sir_col is None:
        return _fallback_table(df, title)

    df = df.copy()
    df[sir_col] = pd.to_numeric(df[sir_col], errors="coerce")
    df = df.dropna(subset=[sir_col]).sort_values(sir_col)

    err_plus  = (df[hi_col] - df[sir_col]).clip(lower=0).tolist() if hi_col else None
    err_minus = (df[sir_col] - df[lo_col]).clip(lower=0).tolist() if lo_col else None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df[label_col].astype(str),
        x=df[sir_col],
        orientation="h",
        error_x=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus)
            if err_plus else None,
        marker_color=[
            "crimson" if v > 1 else "steelblue" for v in df[sir_col]
        ],
    ))
    fig.add_vline(x=1, line_dash="dash", line_color="grey", opacity=0.6)
    fig.update_layout(
        title=title,
        xaxis_title="SIR (reference = 1)",
        height=max(300, len(df) * 30 + 80),
        margin=dict(l=180, r=40, t=50, b=40),
    )
    return fig


def line_chart(df: pd.DataFrame, x_col: str, y_col: str,
               color_col: str | None = None, title: str = "") -> go.Figure:
    """Line chart for temporal trends."""
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    fig = px.line(df, x=x_col, y=y_col, color=color_col,
                  title=title, markers=True,
                  color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(height=400, margin=dict(l=60, r=40, t=50, b=40))
    return fig


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                 label_col: str | None = None, title: str = "") -> go.Figure:
    df = df.copy()
    fig = px.scatter(df, x=x_col, y=y_col, text=label_col,
                     title=title,
                     color_discrete_sequence=["steelblue"])
    if label_col:
        fig.update_traces(textposition="top center")
    fig.update_layout(height=400)
    return fig


def ranked_bar(df: pd.DataFrame, x_col: str, y_col: str,
               title: str = "", top_n: int = 20) -> go.Figure:
    """Simple bar chart sorted by x_col, top N rows."""
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df = df.dropna(subset=[x_col]).sort_values(x_col, ascending=False).head(top_n)
    fig = px.bar(df, x=y_col, y=x_col, title=title,
                 color_discrete_sequence=["steelblue"])
    fig.update_layout(height=400, margin=dict(l=60, r=40, t=50, b=40))
    return fig


def _fallback_table(df: pd.DataFrame, title: str) -> go.Figure:
    """Show data as a plain table when no suitable chart type matches."""
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color="steelblue", font_color="white",
                    align="left"),
        cells=dict(values=[df[c].tolist() for c in df.columns],
                   fill_color="lavender", align="left"),
    )])
    fig.update_layout(title=title, height=max(300, len(df) * 30 + 120))
    return fig


def auto_chart(df: pd.DataFrame, chart_type: str, title: str = "") -> go.Figure:
    """Dispatch to the right chart based on chart_type hint."""
    ct = chart_type.lower()
    if ct in ("km_bar", "km"):
        return km_bar(df, title)
    if ct in ("forest_plot", "forest", "hr", "cox"):
        return forest_plot(df, title)
    if ct in ("sir_bar", "sir"):
        return sir_bar(df, title)
    if ct == "line":
        x = _find_col(df, ["diag_yr", "year", "epoch", "x"])
        y_candidates = [c for c in df.columns if c not in (x, "site", "axis", "direction")]
        y = y_candidates[0] if y_candidates else None
        color = _find_col(df, ["site", "axis", "group"])
        if x and y:
            return line_chart(df, x, y, color, title)
    if ct == "scatter":
        x = _find_col(df, ["k", "rho", "x"])
        y = _find_col(df, ["silhouette", "p_corrected", "y"])
        label = _find_col(df, ["site", "label", "name"])
        if x and y:
            return scatter_plot(df, x, y, label, title)
    if ct in ("bar", "ranked_bar"):
        x = _find_col(df, ["lift", "odds_ratio", "or_", "sir", "rho", "n"])
        y = _find_col(df, ["antecedent_label", "consequent_label", "target_label",
                            "site", "covariate", "feature", "label"])
        if x and y:
            return ranked_bar(df, x, y, title)

    return _fallback_table(df, title)
