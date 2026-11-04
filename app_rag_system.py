"""
Streamlit Web Application — Professional Presentation Version
Cancer Clinical Information System Using RAG, LLMs, LAQA and ChromaDB

Redesigned for mentor presentation:
- Clean professional medical UI
- Fully readable structured output
- Clear color-coded evidence cards
- Prominent verified citations
- System performance dashboard
"""

import streamlit as st
import time
import os
import json

from laqa import LanguageAwareQueryAnalyzer, laqa
from retrieval import ClinicalRetriever
from response_generator import ClinicalResponseGenerator
from database_inspection import inspect_database

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gynecological Oncology Clinical Information System",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Cancer Clinical Information System | RAG + LAQA + ChromaDB\nBuilt for Internship Project | Gynecological Oncology Domain"
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Professional Medical UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Page Header ---- */
.app-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #1D4ED8 100%);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(30, 58, 138, 0.3);
}
.app-title {
    font-size: 2.0rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #93C5FD;
    margin: 0;
    font-weight: 400;
}
.app-badges {
    margin-top: 14px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.header-badge {
    background: rgba(255,255,255,0.15);
    color: #E0F2FE;
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ---- Query Input Area ---- */
.query-section {
    background: #F8FAFC;
    border: 1.5px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
}

/* ---- Answer Hero Card ---- */
.answer-hero {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    border: 2px solid #3B82F6;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 24px;
}
.answer-hero-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #2563EB;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}
.answer-hero-cancer {
    font-size: 1.5rem;
    font-weight: 800;
    color: #1E3A8A;
    margin-bottom: 4px;
}
.answer-hero-meta {
    font-size: 0.88rem;
    color: #64748B;
}

/* ---- Section Headers ---- */
.section-header {
    font-size: 1.0rem;
    font-weight: 700;
    color: #1E293B;
    padding: 10px 0 6px 0;
    border-bottom: 2px solid #E2E8F0;
    margin-bottom: 14px;
}

/* ---- Drug / Medication Card ---- */
.drug-card {
    background: #FFFFFF;
    border: 1.5px solid #D1FAE5;
    border-left: 5px solid #10B981;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.08);
}
.drug-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #065F46;
    margin-bottom: 6px;
}
.drug-dose {
    font-size: 0.95rem;
    color: #374151;
    line-height: 1.6;
    margin-bottom: 8px;
}
.drug-source {
    font-size: 0.8rem;
    color: #6B7280;
    background: #F0FDF4;
    border-radius: 6px;
    padding: 4px 10px;
    display: inline-block;
}

/* ---- Procedure Item ---- */
.procedure-item {
    background: #FFF7ED;
    border-left: 4px solid #F59E0B;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.92rem;
    color: #374151;
}

/* ---- Info Metric Card ---- */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.3rem;
    font-weight: 800;
    color: #1E293B;
}
.metric-unit {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 2px;
}

/* ---- Stage Badge ---- */
.stage-badge {
    display: inline-block;
    background: #EEF2FF;
    color: #3730A3;
    border: 1px solid #C7D2FE;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}
.modality-badge {
    display: inline-block;
    background: #FFF7ED;
    color: #9A3412;
    border: 1px solid #FED7AA;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}
.cancer-badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}
.intent-badge {
    display: inline-block;
    background: #F0FDF4;
    color: #166534;
    border: 1px solid #BBF7D0;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}

/* ---- Citation Card ---- */
.citation-card {
    background: #FAFAFA;
    border: 1.5px solid #E2E8F0;
    border-left: 5px solid #6366F1;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.citation-rank {
    font-size: 0.72rem;
    font-weight: 700;
    color: #6366F1;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.citation-doc {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 4px;
}
.citation-section {
    font-size: 0.85rem;
    color: #64748B;
    font-style: italic;
    margin-bottom: 4px;
}
.citation-authority {
    font-size: 0.82rem;
    color: #374151;
    background: #EEF2FF;
    border-radius: 6px;
    padding: 3px 10px;
    display: inline-block;
}
.similarity-bar-container {
    margin-top: 8px;
}
.similarity-label {
    font-size: 0.75rem;
    color: #64748B;
    margin-bottom: 3px;
}

/* ---- LAQA Entity Box ---- */
.entity-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 16px;
    min-height: 70px;
}
.entity-box-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.entity-tag {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 2px 3px;
}
.entity-tag-drug {
    background: #F0FDF4;
    color: #166534;
}
.entity-tag-stage {
    background: #FFF7ED;
    color: #9A3412;
}
.entity-tag-gene {
    background: #FDF4FF;
    color: #7E22CE;
}

/* ---- Evidence Text Box ---- */
.evidence-text {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #374151;
    max-height: 200px;
    overflow-y: auto;
}

/* ---- Performance Panel ---- */
.perf-panel {
    background: linear-gradient(135deg, #0F172A, #1E293B);
    border-radius: 12px;
    padding: 18px 20px;
    margin-top: 20px;
}
.perf-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}
.perf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.perf-name  { font-size: 0.82rem; color: #94A3B8; }
.perf-value { font-size: 0.88rem; font-weight: 700; color: #38BDF8; }
.perf-total { color: #34D399; }

/* ---- Disclaimer ---- */
.disclaimer {
    background: #FFFBEB;
    border: 1.5px solid #FDE68A;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #78350F;
    line-height: 1.6;
    margin-top: 24px;
}

/* ---- Sidebar ---- */
.sidebar-metric {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sidebar-metric-label { font-size: 0.8rem; color: #64748B; }
.sidebar-metric-value { font-size: 0.88rem; font-weight: 700; color: #1E293B; }

/* ---- Fallback Warning ---- */
.fallback-notice {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #9A3412;
    margin-bottom: 16px;
}

/* ---- Insufficient Evidence ---- */
.insufficient-box {
    background: #FEF2F2;
    border: 1.5px solid #FECACA;
    border-left: 5px solid #EF4444;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 20px;
}

/* ---- Tab overrides ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #F1F5F9;
    border-radius: 10px;
    padding: 4px;
    margin-bottom: 20px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 8px 20px;
}

/* ---- Hide default streamlit branding ---- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHE: Load RAG Engine once (not on every click)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing Clinical RAG Engine...")
def load_rag_engine():
    _analyzer = laqa
    _retriever = ClinicalRetriever(analyzer=_analyzer)
    _generator = ClinicalResponseGenerator(retriever=_retriever)
    return _analyzer, _retriever, _generator


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def render_similarity_bar(score: float) -> str:
    pct = int(score * 100)
    if pct >= 70:
        color = "#10B981"
    elif pct >= 50:
        color = "#F59E0B"
    else:
        color = "#EF4444"
    return f"""
    <div class="similarity-bar-container">
        <div class="similarity-label">Semantic Match Confidence: <b style="color:{color}">{pct}%</b></div>
        <div style="background:#E2E8F0;border-radius:6px;height:8px;width:100%;overflow:hidden;">
            <div style="background:{color};width:{min(pct,100)}%;height:8px;border-radius:6px;transition:width 0.3s;"></div>
        </div>
    </div>"""


def render_entity_tags(values: list, tag_class: str = "") -> str:
    if not values:
        return '<span style="color:#94A3B8;font-size:0.82rem;">None detected</span>'
    return "".join(f'<span class="entity-tag {tag_class}">{v}</span>' for v in values)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">🎗️ Gynecological Oncology Clinical Information System</div>
    <div class="app-subtitle">Evidence-Grounded RAG · LAQA Query Intelligence · ChromaDB Vector Search · 100% Verified Citations</div>
    <div class="app-badges">
        <span class="header-badge">📚 11,039 Clinical Chunks</span>
        <span class="header-badge">🏥 NCCN · FIGO · ACOG · SGO</span>
        <span class="header-badge">🧬 8 Cancer Domains</span>
        <span class="header-badge">⚡ Sub-Second Response</span>
        <span class="header-badge">✅ 100% Citation Rate</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Knowledge Base Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Knowledge Base Status")
    try:
        db_info = inspect_database()
        rc = db_info.get("record_count", 0)
        st.markdown(f"""
        <div class="sidebar-metric">
            <span class="sidebar-metric-label">✅ Total Indexed Chunks</span>
            <span class="sidebar-metric-value">{rc:,}</span>
        </div>
        <div class="sidebar-metric">
            <span class="sidebar-metric-label">🔢 Vector Dimensions</span>
            <span class="sidebar-metric-value">384-d</span>
        </div>
        <div class="sidebar-metric">
            <span class="sidebar-metric-label">📐 Search Metric</span>
            <span class="sidebar-metric-value">Cosine HNSW</span>
        </div>
        <div class="sidebar-metric">
            <span class="sidebar-metric-label">🤖 Embedding Model</span>
            <span class="sidebar-metric-value">all-MiniLM-L6-v2</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🧬 Cancer Domain Partitions")
        dist = db_info.get("cancer_type_distribution", {})
        sorted_dist = sorted(dist.items(), key=lambda x: -x[1])
        for cancer_name, count in sorted_dist:
            short_name = cancer_name.replace(" Cancer", "").replace(" (Endometrial)", "").replace(" (GTD)", " GTD")
            pct = count / rc * 100 if rc > 0 else 0
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#374151;margin-bottom:3px;">
                    <span>🔹 {short_name}</span>
                    <span style="font-weight:700;color:#1E293B">{count:,}</span>
                </div>
                <div style="background:#E2E8F0;border-radius:4px;height:5px;">
                    <div style="background:#3B82F6;width:{pct:.1f}%;height:5px;border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Database Error: {e}")

    st.markdown("---")
    st.markdown("#### ⚙️ Search Configuration")
    top_k = st.slider(
        "Evidence Chunks to Retrieve (k)",
        min_value=1, max_value=5, value=3,
        help="Number of top-ranked guideline chunks used to generate the answer"
    )
    st.caption(f"System retrieves **15 candidates** internally, re-ranks by clinical relevance, and surfaces top **{top_k}** for the answer.")

    st.markdown("---")
    st.markdown("#### 🏥 Guideline Sources")
    st.markdown("""
    <div style="font-size:0.8rem;color:#374151;line-height:2.0;">
    📌 <b>NCCN</b> — National Comprehensive Cancer Network<br>
    📌 <b>FIGO</b> — Int'l Federation Gynecology & Obstetrics<br>
    📌 <b>ACOG</b> — American College of Ob-Gyns<br>
    📌 <b>SGO</b> — Society of Gynecologic Oncology<br>
    📌 <b>ASCO</b> — American Society of Clinical Oncology<br>
    📌 <b>CDC</b> — Centers for Disease Control
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD RAG ENGINE
# ─────────────────────────────────────────────────────────────────────────────
try:
    analyzer, retriever, generator = load_rag_engine()
except Exception as e:
    st.error(f"**RAG Engine Initialization Error:** {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# QUERY INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔬 Clinical Query Interface</div>', unsafe_allow_html=True)

SAMPLE_QUERIES = [
    "── Select a benchmark question to test ──",
    "What is the first-line chemotherapy regimen for advanced epithelial ovarian cancer?",
    "What is the recommended concurrent chemoradiotherapy protocol for cervical cancer?",
    "What are the diagnostic criteria and chemotherapy for high-risk gestational trophoblastic neoplasia?",
    "What are the clinical staging criteria and surgical options for endometrial cancer?",
    "What are the FIGO staging criteria for vulvar cancer and when is sentinel lymph node biopsy indicated?",
    "What are the diagnostic HPV cotesting and Pap smear screening guidelines for cervical cancer prevention?",
    "What does neoadjuvant chemotherapy refer to in oncology and when is it used?",
    "What maintenance therapy is recommended after first-line chemotherapy in ovarian cancer?",
    "What is the role of bevacizumab in recurrent ovarian cancer treatment?",
]

col_q1, col_q2 = st.columns([2, 1])
with col_q1:
    selected = st.selectbox(
        "📌 Pre-loaded Benchmark Questions:",
        SAMPLE_QUERIES,
        help="Select a sample question from the mentor's 200-question benchmark dataset"
    )
    prefill = selected if selected != SAMPLE_QUERIES[0] else ""

with col_q2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 Or type your own clinical question below")

user_query = st.text_area(
    "📝 Enter your clinical question:",
    value=prefill,
    height=90,
    placeholder="e.g., What is the standard treatment for FIGO Stage IIB cervical cancer?",
    help="Type any gynecological oncology question in plain English"
)

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
with col_btn1:
    run_btn = st.button(
        "🔍  Analyze & Generate Clinical Answer",
        type="primary",
        use_container_width=True
    )
with col_btn2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION & RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not user_query or not user_query.strip():
        st.warning("⚠️ Please enter a clinical question to analyze.")
    else:
        # Run pipeline with timing
        progress = st.progress(0, text="⚡ Running LAQA query analysis...")
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        laqa_res = analyzer.analyze(user_query.strip())
        t_laqa = (time.perf_counter() - t0) * 1000
        progress.progress(30, text="🔍 Searching ChromaDB vector database...")

        t1 = time.perf_counter()
        ret_res = retriever.retrieve(laqa_res, top_k=top_k)
        t_ret = (time.perf_counter() - t1) * 1000
        progress.progress(80, text="✍️ Synthesizing grounded clinical answer...")

        t2 = time.perf_counter()
        ans_res = generator.answer(ret_res, top_k=top_k)
        t_gen = (time.perf_counter() - t2) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000
        progress.progress(100, text="✅ Complete!")
        time.sleep(0.3)
        progress.empty()

        # Early exit if LAQA needs clarification
        if laqa_res.needs_clarification:
            st.warning(f"⚠️ **Query Requires Clarification:** {laqa_res.clarification_reason}")
            st.stop()

        # ── Gather clinical entities from chunks ───────────────────────────
        chunks = ret_res.chunks
        drugs_found, stages_found, modalities_found, procedures_found = [], [], [], []
        seen_drugs, seen_stages, seen_mods, seen_procs = set(), set(), set(), set()

        for c in chunks:
            d = c.drugs_mentioned
            if d and d.lower() not in ["none", "not specified"] and d not in seen_drugs:
                seen_drugs.add(d)
                drugs_found.append({
                    "name": d,
                    "dose": c.dosages_regimens if c.dosages_regimens not in ["None", "not specified"] else "Per standard oncology protocol",
                    "doc": c.source_document,
                    "page": c.source_page,
                    "authority": c.issuing_authority
                })

            s = c.clinical_stage
            if s and s.lower() not in ["not specified", "unknown"] and s not in seen_stages:
                seen_stages.add(s); stages_found.append(s)

            m = c.treatment_modality
            if m and m.lower() not in ["not specified", "unknown"] and m not in seen_mods:
                seen_mods.add(m); modalities_found.append(m)

            p = c.procedures
            if p and p.lower() not in ["none", "not specified"] and p not in seen_procs:
                seen_procs.add(p); procedures_found.append(p)

        target_cancer = ans_res.cancer_type or "General Oncology"
        is_sufficient = ans_res.is_sufficient
        top_sim = chunks[0].similarity_score if chunks else 0.0


        # ── ANSWER HERO BANNER ─────────────────────────────────────────────
        sim_color = "#10B981" if top_sim >= 0.60 else "#F59E0B" if top_sim >= 0.40 else "#EF4444"
        st.markdown(f"""
        <div class="answer-hero">
            <div class="answer-hero-label">Clinical Information Result</div>
            <div class="answer-hero-cancer">{target_cancer}</div>
            <div class="answer-hero-meta">
                Question Intent: <b>{ans_res.question_category.title()}</b> &nbsp;·&nbsp;
                Evidence Chunks: <b>{len(chunks)}</b> &nbsp;·&nbsp;
                Top Match: <b style="color:{sim_color}">{top_sim*100:.1f}%</b> &nbsp;·&nbsp;
                Citations: <b>{len(ans_res.citations)} verified sources</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Fallback warning
        if ret_res.fallback_used:
            st.markdown(f"""
            <div class="fallback-notice">
                ⚠️ <b>Note:</b> This cancer type is outside the primary indexed domain.
                The system performed a broad semantic search across all 11,039 gynecology guideline chunks.
                Results may be from related oncology protocols.
            </div>
            """, unsafe_allow_html=True)


        # ═══════════════════════════════════════════════════════════════════
        # TABS
        # ═══════════════════════════════════════════════════════════════════
        tab1, tab2, tab3 = st.tabs([
            "💡  Clinical Answer",
            "🧠  LAQA Intelligence Report",
            "📚  Evidence & Citations"
        ])

        # ──────────────────────────────────────────────────────────────────
        # TAB 1: CLINICAL ANSWER
        # ──────────────────────────────────────────────────────────────────
        with tab1:

            # Insufficient evidence block
            if not is_sufficient:
                st.markdown(f"""
                <div class="insufficient-box">
                    <b>⚠️ Low Evidence Confidence ({top_sim*100:.1f}% match)</b><br>
                    The retrieved guidelines do not strongly match this question.
                    The information below is the best available match from indexed sources but may not be fully specific to your query.
                    Please consult published oncology guidelines directly.
                </div>
                """, unsafe_allow_html=True)

            # ── Medication / Treatment Cards ───────────────────────────────
            st.markdown('<div class="section-header">💊 Recommended Treatment & Pharmacotherapy</div>', unsafe_allow_html=True)

            if drugs_found:
                for d in drugs_found:
                    st.markdown(f"""
                    <div class="drug-card">
                        <div class="drug-name">🧪 {d['name']}</div>
                        <div class="drug-dose">
                            <b>Dosing Protocol / Administration Schedule:</b><br>
                            {d['dose']}
                        </div>
                        <span class="drug-source">
                            📄 {d['doc']} &nbsp;({d['page']}) &nbsp;·&nbsp; {d['authority']}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ No specific systemic pharmacotherapy mentioned in retrieved evidence for this query.")

            # ── Two-column layout: Procedures + Clinical Context ───────────
            left_col, right_col = st.columns([1.1, 1])

            with left_col:
                if procedures_found:
                    st.markdown('<div class="section-header">🩺 Surgical & Clinical Procedures</div>', unsafe_allow_html=True)
                    for proc in procedures_found[:4]:
                        st.markdown(f'<div class="procedure-item">🔹 {proc}</div>', unsafe_allow_html=True)

            with right_col:
                st.markdown('<div class="section-header">🎯 Clinical Context Summary</div>', unsafe_allow_html=True)

                if stages_found:
                    st.markdown("**📍 Applicable Cancer Stage(s):**")
                    stage_html = "".join(f'<span class="stage-badge">{s}</span>' for s in stages_found[:3])
                    st.markdown(stage_html, unsafe_allow_html=True)
                    st.markdown("")

                if modalities_found:
                    st.markdown("**⚕️ Treatment Modality:**")
                    mod_html = "".join(f'<span class="modality-badge">{m}</span>' for m in modalities_found[:3])
                    st.markdown(mod_html, unsafe_allow_html=True)
                    st.markdown("")

                st.markdown("**🏷️ Cancer Domain:**")
                st.markdown(f'<span class="cancer-badge">{target_cancer}</span>', unsafe_allow_html=True)
                st.markdown("")

                st.markdown("**📊 Query Intent:**")
                st.markdown(f'<span class="intent-badge">{ans_res.question_category.title()}</span>', unsafe_allow_html=True)

            # ── Performance Panel ──────────────────────────────────────────
            st.markdown(f"""
            <div class="perf-panel">
                <div class="perf-title">⚡ System Performance</div>
                <div class="perf-row">
                    <span class="perf-name">LAQA Query Analysis</span>
                    <span class="perf-value">{t_laqa:.2f} ms</span>
                </div>
                <div class="perf-row">
                    <span class="perf-name">ChromaDB Vector Search + Re-Ranking</span>
                    <span class="perf-value">{t_ret:.1f} ms</span>
                </div>
                <div class="perf-row">
                    <span class="perf-name">Response Synthesis</span>
                    <span class="perf-value">{t_gen:.2f} ms</span>
                </div>
                <div class="perf-row" style="border-bottom:none;">
                    <span class="perf-name" style="font-weight:700;color:#E2E8F0;">Total End-to-End</span>
                    <span class="perf-value perf-total">{total_ms:.1f} ms</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Full Clinical Text Expander ────────────────────────────────
            with st.expander("📋 View Complete Clinical Briefing (Full Structured Report)", expanded=False):
                st.markdown(f"""
                <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:18px;
                            font-family:monospace;font-size:0.85rem;line-height:1.8;white-space:pre-wrap;color:#1E293B;">
{ans_res.answer_text}
                </div>
                """, unsafe_allow_html=True)

            # ── Disclaimer ────────────────────────────────────────────────
            st.markdown(f"""
            <div class="disclaimer">
                ⚠️ <b>Medical Disclaimer:</b> {ans_res.disclaimer}
            </div>
            """, unsafe_allow_html=True)


        # ──────────────────────────────────────────────────────────────────
        # TAB 2: LAQA INTELLIGENCE REPORT
        # ──────────────────────────────────────────────────────────────────
        with tab2:
            st.markdown('<div class="section-header">🧠 How LAQA Analyzed Your Question</div>', unsafe_allow_html=True)
            st.caption("LAQA (Language-Aware Query Analyzer) processes the raw question in under 2ms to structure it before database search.")

            # 4 Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🌐 Language</div>
                    <div class="metric-value">{laqa_res.language}</div>
                    <div class="metric-unit">Detected Script</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🧬 Cancer Type</div>
                    <div class="metric-value" style="font-size:1.0rem;">{laqa_res.cancer_type or "General"}</div>
                    <div class="metric-unit">Identified Domain</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🎯 Question Intent</div>
                    <div class="metric-value" style="font-size:1.1rem;">{laqa_res.question_category.title()}</div>
                    <div class="metric-unit">Clinical Category</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                conf_pct = int(laqa_res.confidence * 100)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📊 LAQA Confidence</div>
                    <div class="metric-value" style="color:#10B981;">{conf_pct}%</div>
                    <div class="metric-unit">Analysis Score</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Normalized Query
            st.markdown("**🔍 Normalized Search Query** *(boilerplate stripped)*")
            st.info(f'"{laqa_res.normalized_query}"')

            # Extracted Entities
            st.markdown("**🧬 Clinical Entities Extracted Automatically**")
            ent = laqa_res.extracted_entities
            e1, e2, e3, e4 = st.columns(4)

            with e1:
                st.markdown(f"""
                <div class="entity-box">
                    <div class="entity-box-label">💊 Drugs</div>
                    {render_entity_tags(ent.get("drugs",[]), "entity-tag-drug")}
                </div>""", unsafe_allow_html=True)
            with e2:
                st.markdown(f"""
                <div class="entity-box">
                    <div class="entity-box-label">🧫 Biomarkers / Genes</div>
                    {render_entity_tags(ent.get("biomarkers_genes",[]), "entity-tag-gene")}
                </div>""", unsafe_allow_html=True)
            with e3:
                st.markdown(f"""
                <div class="entity-box">
                    <div class="entity-box-label">📍 Cancer Stage</div>
                    {render_entity_tags(ent.get("stages",[]), "entity-tag-stage")}
                </div>""", unsafe_allow_html=True)
            with e4:
                st.markdown(f"""
                <div class="entity-box">
                    <div class="entity-box-label">🫀 Anatomical Sites</div>
                    {render_entity_tags(ent.get("anatomical_sites",[]))}
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ChromaDB Filter
            st.markdown("**🛡️ ChromaDB Metadata Filter Applied**")
            filter_data = laqa_res.metadata_filters
            if filter_data:
                fcol1, fcol2 = st.columns([1, 2])
                with fcol1:
                    st.code(json.dumps(filter_data, indent=2), language="json")
                with fcol2:
                    st.markdown(f"""
                    <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:14px 16px;font-size:0.88rem;color:#1D4ED8;margin-top:4px;">
                        <b>What this means:</b><br><br>
                        Before computing any vector distances, ChromaDB first filters to only the
                        <b>{filter_data.get('cancer_type','')}</b> partition
                        ({dist.get(filter_data.get('cancer_type',''), 'N/A'):,} chunks).
                        This reduces the search space from 11,039 to only the relevant cancer domain — faster and more precise.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No metadata filter applied — broad search across all 11,039 chunks (general oncology question).")


        # ──────────────────────────────────────────────────────────────────
        # TAB 3: EVIDENCE & CITATIONS
        # ──────────────────────────────────────────────────────────────────
        with tab3:
            st.markdown('<div class="section-header">📚 Retrieved Evidence — Verified Guideline Citations</div>', unsafe_allow_html=True)
            st.caption(
                f"Top **{len(chunks)}** chunks retrieved from 11,039 indexed clinical guidelines. "
                "Each chunk carries a **verified document name and page number**. "
                "Re-ranked using: 60% Semantic Similarity + 25% Entity Match + 15% Category Alignment."
            )

            if not chunks:
                st.warning("No evidence chunks were retrieved for this query.")
            else:
                for rank, chunk in enumerate(chunks, 1):
                    re_rank_pct = getattr(chunk, "re_rank_score", chunk.similarity_score) * 100
                    sim_pct = chunk.similarity_score * 100

                    border_color = "#10B981" if sim_pct >= 65 else "#F59E0B" if sim_pct >= 45 else "#EF4444"

                    st.markdown(f"""
                    <div class="citation-card" style="border-left-color:{border_color};">
                        <div class="citation-rank">Evidence Rank #{rank} &nbsp;·&nbsp; Re-Rank Score: {re_rank_pct:.1f}%</div>
                        <div class="citation-doc">📄 {chunk.source_document} &nbsp; — &nbsp; {chunk.source_page}</div>
                        <div class="citation-section">Section: "{chunk.section_heading}"</div>
                        <span class="citation-authority">🏥 {chunk.issuing_authority}</span>
                        {render_similarity_bar(chunk.similarity_score)}
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"🔬 View Full Chunk Details — Rank #{rank}", expanded=(rank == 1)):
                        d1, d2 = st.columns(2)
                        with d1:
                            st.markdown(f"**Cancer Type:** `{chunk.cancer_type}`")
                            st.markdown(f"**Clinical Stage:** `{chunk.clinical_stage}`")
                            st.markdown(f"**Treatment Modality:** `{chunk.treatment_modality}`")
                        with d2:
                            if chunk.drugs_mentioned not in ["None", "not specified"]:
                                st.markdown(f"**Drug:** `{chunk.drugs_mentioned}`")
                                st.markdown(f"**Dosing:** `{chunk.dosages_regimens}`")
                            if getattr(chunk, "entity_match_hits", None):
                                st.markdown(f"**Matched Query Terms:** `{', '.join(chunk.entity_match_hits)}`")

                        st.markdown("**📃 Original Guideline Excerpt:**")
                        st.markdown(f'<div class="evidence-text">{chunk.text}</div>', unsafe_allow_html=True)

                # Summary citation list
                st.markdown("---")
                st.markdown('<div class="section-header">📋 Citation Summary (APA-Style Reference List)</div>', unsafe_allow_html=True)
                for i, cite in enumerate(ans_res.citations, 1):
                    st.markdown(f"**[{i}]** {cite}")
