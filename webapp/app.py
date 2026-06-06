"""CMUH Cancer Registry — Interactive Q&A Web App."""
import streamlit as st

st.set_page_config(
    page_title="CMUH Cancer Registry Q&A",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd

from db import get_db
from query_engine import QueryEngine
from charts import auto_chart
from site_labels import QUICK_SITES, label as site_label

# ── cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading registry database...")
def load_db():
    return get_db()


@st.cache_resource(show_spinner=False)
def load_engine(_db):
    return QueryEngine(_db)


db = load_db()
engine = load_engine(db)

# ── session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of QueryResult
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 CMUH Registry Q&A")
    st.caption("China Medical University Hospital  \nn=84,161 patients · 46 sites · 2003–2020")
    st.divider()

    # API key
    api_key = st.text_input(
        "Anthropic API key (optional)",
        type="password",
        placeholder="sk-ant-...",
        help="Required only for complex queries not matched by built-in patterns.",
    )
    if api_key:
        engine.set_api_key(api_key)

    st.divider()
    st.markdown("**Quick access — cancer sites**")
    cols = st.columns(2)
    for i, (code, name) in enumerate(QUICK_SITES):
        if cols[i % 2].button(f"{code} {name}", use_container_width=True):
            st.session_state.pending_question = f"Show survival outcomes for {code} {name}"

    st.divider()
    st.markdown("**Example questions**")
    examples = [
        "Median OS by stage for esophageal cancer",
        "Which cancer sites are rising in incidence?",
        "Sex differences across all cancer sites",
        "SIR for second primaries after liver cancer",
        "CCRT vs no CCRT survival in C15",
        "Top cancer co-occurrence association rules",
        "Hazard ratios for esophageal cancer treatment",
        "Transformer vs MLP model performance",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state.pending_question = ex

    st.divider()
    st.markdown("**Loaded tables**")
    st.caption(f"{len(db.tables)} aggregated result tables")
    with st.expander("View table list"):
        for t in db.tables:
            st.text(f"• {t}  ({db.table_row_count(t)} rows)")

    st.divider()
    st.caption(
        "Privacy: All data shown is pre-aggregated (group-level statistics only). "
        "Groups with n<5 are suppressed. No individual patient records are accessible."
    )

# ── main area ─────────────────────────────────────────────────────────────────
tab_query, tab_explorer = st.tabs(["💬 Q&A", "🗂 Data Explorer"])

# ── Q&A tab ───────────────────────────────────────────────────────────────────
with tab_query:
    st.markdown("### Ask anything about CMUH cancer registry data")
    st.caption(
        "Survival, incidence, hazard ratios, SIR, sex differences, co-occurrence, temporal trends…"
    )

    # Chat input
    user_input = st.chat_input("e.g. 'What is the median OS for esophageal cancer by stage?'")

    # Honour sidebar quick-buttons
    if st.session_state.pending_question and not user_input:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = ""

    if user_input:
        with st.spinner("Querying registry…"):
            result = engine.answer(user_input)
        st.session_state.history.insert(0, result)

    # Display history
    for result in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(result.question)

        with st.chat_message("assistant"):
            if result.error:
                st.error(f"Could not answer: {result.error}")
                continue

            if result.suppressed:
                st.warning(
                    "Some groups were suppressed because n < 5 (privacy threshold). "
                    "Suppressed rows are excluded from the chart and table."
                )

            # Answer text
            st.markdown(result.answer)

            # Chart + table side-by-side
            if not result.df.empty:
                col_chart, col_table = st.columns([3, 2])
                with col_chart:
                    try:
                        fig = auto_chart(result.df, result.chart_type, result.chart_title)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Chart rendering failed: {e}")

                with col_table:
                    st.dataframe(
                        result.df.head(30),
                        use_container_width=True,
                        hide_index=True,
                    )

            # SQL expander
            with st.expander("View SQL & source"):
                st.code(result.sql, language="sql")
                st.caption(f"Source table: `{result.source}`")

    if not st.session_state.history:
        st.info(
            "No questions yet. Try clicking a quick-access button in the sidebar, "
            "or type a question above."
        )

    # Clear history button
    if st.session_state.history:
        if st.button("Clear history", key="clear"):
            st.session_state.history = []
            st.rerun()

# ── Data Explorer tab ─────────────────────────────────────────────────────────
with tab_explorer:
    st.markdown("### Browse aggregated result tables directly")

    col1, col2 = st.columns([1, 3])
    with col1:
        selected = st.selectbox("Select table", db.tables)
        n_rows = st.slider("Rows to show", 5, 200, 30)
        show_schema = st.checkbox("Show column types")

    with col2:
        if selected:
            try:
                df_preview = db.execute(f"SELECT * FROM {selected} LIMIT {n_rows}")
                st.caption(f"**{selected}** — {db.table_row_count(selected)} total rows")
                st.dataframe(df_preview, use_container_width=True, hide_index=True)

                if show_schema:
                    types = df_preview.dtypes.reset_index()
                    types.columns = ["column", "dtype"]
                    st.dataframe(types, use_container_width=True, hide_index=True)

                # Auto-chart preview
                with st.expander("Quick chart preview"):
                    os_col = next(
                        (c for c in df_preview.columns if "median_os" in c or "sir" in c
                         or "hr" in c or "rho" in c),
                        None
                    )
                    if os_col and len(df_preview) > 1:
                        chart_type = (
                            "km_bar" if "os" in os_col else
                            "sir_bar" if "sir" in os_col else
                            "forest" if "hr" in os_col else
                            "bar"
                        )
                        try:
                            fig = auto_chart(df_preview, chart_type, selected)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            st.caption("Chart not available for this table.")

            except Exception as e:
                st.error(f"Could not load table: {e}")

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "CMUH Cancer Registry Q&A · Data: China Medical University Hospital institutional registry · "
    "Single-centre hospital-based, 2003–2020 · For research use only"
)
