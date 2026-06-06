"""CMUH Cancer Registry — Interactive Q&A (v2)."""
import streamlit as st

st.set_page_config(
    page_title="CMUH Cancer Registry Q&A",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="auto",   # collapsed on narrow viewports automatically
)

# ── custom CSS (desktop + RWD) ────────────────────────────────────────────────
st.markdown("""
<style>
/* ── base ───────────────────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* metric cards */
div[data-testid="metric-container"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 10px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}

/* answer highlight */
.ans-box {
    background: #f0f9ff;
    border-left: 4px solid #0ea5e9;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 6px 0 10px 0;
    line-height: 1.6;
}

/* suppression notice */
.suppress-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 4px 0 8px 0;
    font-size: 0.85rem;
}

/* sidebar site buttons — compact */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    font-size: 0.82rem;
    padding: 4px 10px;
    margin: 1px 0;
    border-radius: 6px;
}

/* ── tablet  (640 – 1023 px) ────────────────────────────────────────────── */
@media screen and (max-width: 1023px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* wrap columns → 2-per-row */
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }
    div[data-testid="column"] {
        min-width: calc(50% - 0.5rem) !important;
        flex: 0 1 calc(50% - 0.5rem) !important;
    }

    /* plotly chart – full width when stacked */
    div[data-testid="stPlotlyChart"] {
        width: 100% !important;
    }
}

/* ── mobile  (< 640 px) ──────────────────────────────────────────────────── */
@media screen and (max-width: 639px) {
    .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }

    /* single-column everything */
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    div[data-testid="column"] {
        min-width: 100% !important;
        flex: 0 1 100% !important;
    }

    /* smaller heading on phone */
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 1rem !important;   }

    /* metric label/value tighter */
    div[data-testid="metric-container"] {
        padding: 8px 12px !important;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.2rem !important;
    }

    /* ans-box narrower padding */
    .ans-box {
        padding: 10px 12px !important;
        font-size: 0.9rem !important;
    }

    /* chat messages full width */
    div[data-testid="stChatMessage"] {
        padding: 4px 0 !important;
    }

    /* topic card description text smaller */
    div[data-testid="stCaptionContainer"] p {
        font-size: 0.78rem !important;
    }

    /* buttons: larger touch targets */
    div[data-testid="stButton"] > button {
        min-height: 44px !important;
        font-size: 0.9rem !important;
    }

    /* hide "All loaded tables" expander in sidebar on phone */
    section[data-testid="stSidebar"] div[data-testid="stExpander"] {
        display: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

import pandas as pd

from db import get_db
from query_engine import QueryEngine
from charts import auto_chart
from site_labels import QUICK_SITES
from rag_engine import RAGEngine

# ── cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading registry database…")
def load_db():
    return get_db()

@st.cache_resource(show_spinner=False)
def load_engine(_db):
    return QueryEngine(_db)

@st.cache_resource(show_spinner=False)
def load_rag() -> RAGEngine:
    return RAGEngine()

db     = load_db()
engine = load_engine(db)
rag    = load_rag()

# ── session state ─────────────────────────────────────────────────────────────
if "history"  not in st.session_state: st.session_state.history  = []
if "pending"  not in st.session_state: st.session_state.pending  = ""

# ── topic catalogue ───────────────────────────────────────────────────────────
TOPICS = [
    ("📈", "Survival & OS",
     "Median OS by stage, treatment, sex, histology",
     "Median OS by stage for esophageal cancer"),
    ("📉", "Temporal Trends",
     "Rising / falling incidence trends 2003–2020",
     "Which cancer sites are rising in incidence?"),
    ("⚥",  "Sex Differences",
     "Male : female odds ratios across all 46 sites",
     "Sex differences male female odds ratios across all sites"),
    ("🔗", "Second Primary SIR",
     "Standardised incidence ratios after index cancer",
     "SIR for second primaries after esophageal cancer C15"),
    ("🔀", "Co-occurrence",
     "Cancer co-occurrence association rules and liver axis",
     "Top cancer co-occurrence association rules"),
    ("💊", "Treatment & Cox",
     "Hazard ratios, CCRT vs surgery, chemo regimens",
     "Hazard ratios for esophageal cancer treatment"),
    ("🤖", "Deep Learning",
     "Transformer R@k, SHAP importance, VAE clusters",
     "Deep learning transformer model performance R@1"),
    ("🏔", "UADT Field",
     "Field cancerization SIR, trajectories, Cox TV",
     "UADT field cancerization SIR pairs"),
    ("🔬", "ESCC Detailed",
     "CMUH chart-review cohort n=344, 2007–2010",
     "CMUH ESCC survival summary by group"),
]

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 CMUH Registry Q&A")
    st.caption(
        "China Medical University Hospital  \n"
        "n = 84,161 · 46 sites · 2003–2020  \n"
        "Single-centre, hospital-based"
    )

    st.divider()

    api_key = st.text_input(
        "🔑 Anthropic API key",
        type="password",
        placeholder="sk-ant-…  (optional, for complex queries)",
    )
    if api_key:
        engine.set_api_key(api_key)

    st.divider()
    st.markdown("**🎯 Quick cancer sites**")

    pairs = list(QUICK_SITES)
    for i in range(0, len(pairs), 2):
        c1, c2 = st.columns(2)
        code1, name1 = pairs[i]
        if c1.button(f"{code1} {name1}", key=f"qs_{code1}", use_container_width=True):
            st.session_state.pending = f"Show survival outcomes for {code1} {name1}"
        if i + 1 < len(pairs):
            code2, name2 = pairs[i + 1]
            if c2.button(f"{code2} {name2}", key=f"qs_{code2}", use_container_width=True):
                st.session_state.pending = f"Show survival outcomes for {code2} {name2}"

    st.divider()
    with st.expander("📋 All loaded tables"):
        for t in db.tables:
            st.caption(f"• {t}  ({db.table_row_count(t)} rows)")

    st.divider()
    st.caption(
        "🔒 **Privacy**: all data is pre-aggregated. "
        "Groups with n < 5 are suppressed. "
        "No individual patient records are accessible."
    )

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏥 CMUH Cancer Registry — Interactive Q&A")
st.caption("Hospital-based institutional registry · China Medical University Hospital, Taichung, Taiwan")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Patients", "84,161")
m2.metric("Cancer sites", "46")
m3.metric("Study period", "2003 – 2020")
m4.metric("Result tables", str(len(db.tables)))

st.divider()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_qa, tab_explore, tab_rag = st.tabs([
    "💬  Ask a Question",
    "🗂  Data Explorer",
    "📚  Search Manuscripts",
])

# ══════════════════════════════════════════════════════════════════════════════
with tab_qa:

    # ── topic grid (empty-state landing page) ─────────────────────────────────
    if not st.session_state.history:
        st.markdown("### Where would you like to start?")
        st.caption("Click a topic or type your own question below.")

        for row in range(0, len(TOPICS), 3):
            cols = st.columns(3, gap="medium")
            for col, (icon, title, desc, question) in zip(cols, TOPICS[row:row + 3]):
                with col:
                    with st.container(border=True):
                        label = f"{icon}  {title}"
                        if st.button(label, key=f"topic_{title}", use_container_width=True):
                            st.session_state.pending = question
                        st.caption(desc)

        st.markdown("")  # spacing before chat input

    # ── chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask anything… e.g. 'Median OS by stage for esophageal cancer'"
    )

    # honour sidebar / topic card clicks
    if st.session_state.pending and not user_input:
        user_input = st.session_state.pending
        st.session_state.pending = ""

    if user_input:
        with st.spinner("Querying registry…"):
            result = engine.answer(user_input)
        st.session_state.history.insert(0, result)
        st.rerun()

    # ── conversation history ──────────────────────────────────────────────────
    for result in st.session_state.history:

        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(f"**{result.question}**")

        with st.chat_message("assistant", avatar="🏥"):

            if result.error:
                st.error(f"Could not answer: {result.error}")
                continue

            if result.suppressed:
                st.markdown(
                    '<div class="suppress-box">'
                    "⚠️ Some groups suppressed — n&lt;5 privacy threshold"
                    "</div>",
                    unsafe_allow_html=True,
                )

            if result.answer:
                st.markdown(
                    f'<div class="ans-box">{result.answer}</div>',
                    unsafe_allow_html=True,
                )

            if not result.df.empty:
                chart_col, table_col = st.columns([3, 2], gap="medium")

                with chart_col:
                    try:
                        fig = auto_chart(result.df, result.chart_type, result.chart_title)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Chart could not render: {e}")

                with table_col:
                    st.dataframe(
                        result.df.head(25),
                        use_container_width=True,
                        hide_index=True,
                    )

            with st.expander("🔍 SQL & source"):
                st.code(result.sql, language="sql")
                st.caption(f"Source table: `{result.source}`")

    if st.session_state.history:
        st.markdown("")
        if st.button("🗑  Clear conversation"):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("### Browse pre-aggregated result tables directly")
    st.caption("All tables contain group-level statistics only — no individual patient records.")

    c1, c2 = st.columns([1, 3], gap="medium")

    with c1:
        selected = st.selectbox("Select table", db.tables)
        n_rows   = st.slider("Rows to preview", 5, 200, 30)
        show_dtypes = st.checkbox("Show column types")

        if selected:
            st.metric("Total rows", db.table_row_count(selected))

    with c2:
        if selected:
            try:
                df_preview = db.execute(f"SELECT * FROM {selected} LIMIT {n_rows}")
                st.dataframe(df_preview, use_container_width=True, hide_index=True)

                if show_dtypes:
                    types = df_preview.dtypes.reset_index()
                    types.columns = ["column", "dtype"]
                    st.dataframe(types, use_container_width=True, hide_index=True)

                with st.expander("📊 Quick chart preview"):
                    os_col = next(
                        (c for c in df_preview.columns
                         if any(k in c for k in ("median_os", "sir", "rho", "or", "lift"))),
                        None,
                    )
                    if os_col and len(df_preview) > 1:
                        ct = ("km_bar"    if "os"   in os_col else
                              "sir_bar"   if "sir"  in os_col else
                              "forest"    if os_col in ("or", "hr") else
                              "bar")
                        try:
                            fig = auto_chart(df_preview, ct, selected)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            st.caption("No automatic chart for this table.")
                    else:
                        st.caption("No automatic chart for this table.")

            except Exception as e:
                st.error(f"Could not load table: {e}")

# ══════════════════════════════════════════════════════════════════════════════
with tab_rag:
    st.markdown("### 📚 Search Manuscripts")
    st.caption(
        "Hybrid RAG (dense + BM25 + Claude synthesis) over ingested study drafts. "
        "The RAG server starts automatically on first use — allow ~60 s for model warm-up."
    )

    # ── server status ──────────────────────────────────────────────────────────
    with st.expander("🔧 Server & index status", expanded=False):
        if st.button("Check server", key="rag_check"):
            with st.spinner("Contacting RAG server…"):
                info = rag.status()
            if "error" in info:
                st.error(f"Server not reachable: {info['error']}")
            else:
                st.success(
                    f"RAG server online — "
                    f"{info.get('vectors_count', '?')} vectors indexed, "
                    f"status: {info.get('status', '?')}"
                )
            docs = rag.ingested_docs()
            if docs:
                st.markdown("**Ingested documents:**")
                for d in docs:
                    st.caption(f"• {d}")
            else:
                st.caption("No documents listed yet.")

    st.divider()

    # ── query interface ────────────────────────────────────────────────────────
    rag_question = st.text_area(
        "Research question",
        placeholder=(
            "e.g. What are the 5-year survival outcomes for stage III esophageal cancer?\n"
            "e.g. How does BRAF V600E affect thyroid cancer prognosis?\n"
            "e.g. What is the SIR for liver cancer after esophageal cancer?"
        ),
        height=100,
        key="rag_question_input",
    )

    col_topk, col_btn, col_clear = st.columns([1, 2, 1], gap="small")
    top_k = col_topk.slider("Chunks", min_value=2, max_value=12, value=6, key="rag_topk")

    run_query = col_btn.button(
        "🔍  Search & Synthesise",
        type="primary",
        disabled=not rag_question.strip(),
        use_container_width=True,
        key="rag_run",
    )
    if col_clear.button("🗑  Clear", use_container_width=True, key="rag_clear"):
        st.session_state.rag_history = []
        st.rerun()

    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    if run_query and rag_question.strip():
        with st.spinner("Starting RAG server and retrieving…  (first query may take ~60 s)"):
            result = rag.query(rag_question.strip(), top_k=top_k)
        st.session_state.rag_history.insert(0, (rag_question.strip(), result))
        st.rerun()

    # ── history ────────────────────────────────────────────────────────────────
    for q_text, res in st.session_state.rag_history:
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(f"**{q_text}**")

        with st.chat_message("assistant", avatar="📚"):
            if res.error:
                st.error(f"Error: {res.error}")
                continue

            st.markdown(
                f'<div class="ans-box">{res.answer}</div>',
                unsafe_allow_html=True,
            )

            if res.sources:
                with st.expander(f"📎 {len(res.sources)} source chunk(s)"):
                    import pandas as pd
                    src_df = pd.DataFrame(res.sources)
                    display_cols = [c for c in ("index", "source", "section", "page") if c in src_df.columns]
                    st.dataframe(
                        src_df[display_cols] if display_cols else src_df,
                        use_container_width=True,
                        hide_index=True,
                    )

    if not st.session_state.rag_history:
        st.markdown("")
        st.info(
            "💡 **Example questions**\n\n"
            "- What are the co-occurrence patterns between UADT cancers?\n"
            "- How does BRAF V600E status affect survival in papillary thyroid carcinoma?\n"
            "- What is the effect of CCRT vs surgery on esophageal cancer survival?\n"
            "- What clusters emerge from VAE latent space analysis of the cancer registry?\n"
            "- What is the HBV-related incidence trend for liver cancer in Taiwan?"
        )

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "CMUH Cancer Registry Q&A · "
    "Data: China Medical University Hospital institutional cancer registry · "
    "Single-centre hospital-based, 2003–2020 · "
    "For research use only · "
    "n < 5 groups suppressed"
)
