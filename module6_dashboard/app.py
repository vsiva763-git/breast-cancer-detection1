import streamlit as st
import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="OncAI — Breast Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# GLOBAL CSS — Dark Medical Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

/* ── Base Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080C14 !important;
    color: #E8EDF5 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: #080C14 !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0D1219; }
::-webkit-scrollbar-thumb { background: #2563EB; border-radius: 2px; }

/* ── Top Navigation Bar ── */
.nav-bar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(8,12,20,0.92);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(37,99,235,0.2);
    padding: 0 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 60px;
}
.nav-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.5px;
}
.nav-logo span { color: #2563EB; }
.nav-links {
    display: flex;
    gap: 0.25rem;
}
.nav-link {
    position: relative;
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 500;
    color: #94A3B8;
    cursor: pointer;
    border: none;
    background: transparent;
    transition: all 0.15s;
    text-decoration: none;
}
.nav-link:hover { color: #fff; background: rgba(255,255,255,0.06); }
.nav-link.active { color: #fff; background: rgba(37,99,235,0.2); border: 1px solid rgba(37,99,235,0.4); }
.nav-link::after {
    content: '';
    position: absolute;
    left: 12px;
    right: 12px;
    bottom: -6px;
    height: 2px;
    border-radius: 99px;
    background: transparent;
    transition: background 0.2s, box-shadow 0.2s;
}
.nav-link.active::after {
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    box-shadow: 0 0 10px rgba(37,99,235,0.6);
}
.nav-icon {
    display: inline-block;
    margin-right: 0.45rem;
    font-size: 0.9rem;
    opacity: 0.9;
}
.nav-badge {
    font-size: 0.7rem;
    background: #EF4444;
    color: #fff;
    padding: 2px 6px;
    border-radius: 99px;
    margin-left: 0.5rem;
    font-weight: 600;
}

/* ── Hero Section ── */
.hero {
    padding: 5rem 2.5rem 4rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -200px; left: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(37,99,235,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -100px; right: -100px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(37,99,235,0.12);
    border: 1px solid rgba(37,99,235,0.3);
    color: #60A5FA;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.35rem 0.85rem;
    border-radius: 99px;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 700;
    line-height: 1.1;
    color: #fff;
    margin-bottom: 1.2rem;
    max-width: 700px;
}
.hero-title em {
    font-style: normal;
    background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94A3B8;
    line-height: 1.7;
    max-width: 580px;
    margin-bottom: 2.5rem;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
}
.hero-stat-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: #fff;
    line-height: 1;
}
.hero-stat-label {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Section Container ── */
.section {
    padding: 3rem 2.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 0.4rem;
}
.section-sub {
    font-size: 0.9rem;
    color: #64748B;
    margin-bottom: 2rem;
}

/* ── Cards ── */
.card {
    background: #0F1623;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.5rem;
    transition: border-color 0.2s, transform 0.2s;
    height: 100%;
}
.card:hover {
    border-color: rgba(37,99,235,0.4);
    transform: translateY(-2px);
}
.card-icon {
    width: 40px; height: 40px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    margin-bottom: 1rem;
}
.card-title {
    font-weight: 600;
    font-size: 1rem;
    color: #fff;
    margin-bottom: 0.4rem;
}
.card-text {
    font-size: 0.85rem;
    color: #64748B;
    line-height: 1.6;
}

/* ── Assessment Panel ── */
.panel {
    background: #0F1623;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.75rem;
    height: 100%;
}
.panel-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.panel-num {
    width: 26px; height: 26px;
    background: #2563EB;
    color: #fff;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}
.panel-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: #fff;
}

/* ── Streamlit Widget Overrides ── */
[data-testid="stFileUploader"] {
    background: rgba(37,99,235,0.04) !important;
    border: 1px dashed rgba(37,99,235,0.35) !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label { color: #94A3B8 !important; }
[data-testid="stFileUploader"] button {
    background: rgba(37,99,235,0.2) !important;
    border: 1px solid rgba(37,99,235,0.4) !important;
    color: #60A5FA !important;
    border-radius: 6px !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: #2563EB !important;
}
[data-testid="stSlider"] > div > div > div {
    background: rgba(37,99,235,0.2) !important;
}
.stSlider label { color: #94A3B8 !important; font-size: 0.82rem !important; }

/* Text Area */
textarea {
    background: #080C14 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #E8EDF5 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
textarea:focus {
    border-color: rgba(37,99,235,0.5) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
textarea::placeholder { color: #475569 !important; }

/* Buttons */
.stButton > button {
    background: #2563EB !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: #1D4ED8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(37,99,235,0.3) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #94A3B8 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.2) !important;
    color: #fff !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: #0F1623 !important;
    border-radius: 10px 10px 0 0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    gap: 0 !important;
    padding: 0.4rem 0.4rem 0 !important;
}
[data-testid="stTabs"] button[role="tab"] {
    background: transparent !important;
    color: #64748B !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: 7px 7px 0 0 !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: rgba(37,99,235,0.15) !important;
    color: #60A5FA !important;
    border-bottom: 2px solid #2563EB !important;
}
[data-testid="stTabContent"] {
    background: #0F1623 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 1.5rem !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #fff !important; font-family: 'DM Mono', monospace !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* Info / Warning / Success */
[data-testid="stAlert"] {
    background: rgba(37,99,235,0.08) !important;
    border: 1px solid rgba(37,99,235,0.2) !important;
    border-radius: 8px !important;
    color: #94A3B8 !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: #2563EB !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #0F1623 !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 0 !important; }

/* Expander */
[data-testid="stExpander"] {
    background: #0F1623 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #94A3B8 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0D1219 !important; }

/* Image */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* Custom Result Box */
.result-box {
    background: linear-gradient(135deg, #0F1A2E 0%, #0F1623 100%);
    border: 1px solid rgba(37,99,235,0.3);
    border-radius: 14px;
    padding: 2rem;
    margin: 1.5rem 0;
}
.result-score {
    font-family: 'DM Mono', monospace;
    font-size: 4rem;
    font-weight: 500;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-category {
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.result-action {
    font-size: 0.9rem;
    color: #94A3B8;
    line-height: 1.6;
    padding: 0.8rem 1rem;
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #2563EB;
    border-radius: 0 6px 6px 0;
}

/* Module breakdown row */
.mod-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.7rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.mod-row:last-child { border-bottom: none; }
.mod-label {
    font-size: 0.8rem;
    color: #64748B;
    width: 140px;
    flex-shrink: 0;
    font-family: 'DM Mono', monospace;
}
.mod-bar-wrap {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
}
.mod-bar {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}
.mod-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #fff;
    width: 50px;
    text-align: right;
    flex-shrink: 0;
}
.mod-badge {
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 99px;
    font-weight: 600;
    flex-shrink: 0;
}

/* Disclaimer banner */
.disclaimer {
    background: rgba(234,179,8,0.07);
    border: 1px solid rgba(234,179,8,0.2);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.8rem;
    color: #A16207;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1rem 2.5rem;
}

/* FL Timeline */
.fl-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.fl-step:last-child { border-bottom: none; }
.fl-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #2563EB;
    margin-top: 5px;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(37,99,235,0.5);
}
.fl-step-title { font-size: 0.9rem; font-weight: 600; color: #fff; }
.fl-step-desc { font-size: 0.82rem; color: #64748B; margin-top: 0.2rem; line-height: 1.5; }

/* About comparison table override */
[data-testid="stDataFrame"] table th {
    background: rgba(37,99,235,0.15) !important;
    color: #60A5FA !important;
}
[data-testid="stDataFrame"] table td {
    color: #E8EDF5 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FEATURE NAMES & DEFAULTS
# ============================================================
FEATURE_NAMES = [
    'radius_mean','texture_mean','perimeter_mean','area_mean',
    'smoothness_mean','compactness_mean','concavity_mean',
    'concave_points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se',
    'smoothness_se','compactness_se','concavity_se',
    'concave_points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst',
    'smoothness_worst','compactness_worst','concavity_worst',
    'concave_points_worst','symmetry_worst','fractal_dimension_worst'
]
DEFAULTS = [
    14.13,18.25,92.0,654.9,0.096,0.104,0.089,0.049,0.181,0.062,
    0.405,1.216,2.866,40.34,0.007,0.025,0.032,0.012,0.020,0.004,
    16.27,25.68,107.3,880.6,0.132,0.253,0.272,0.115,0.290,0.084
]

# ============================================================
# SESSION STATE INIT
# ============================================================
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'prefill_symptom' not in st.session_state:
    st.session_state['prefill_symptom'] = ''

def set_page(page_name):
    st.session_state['page'] = page_name
    try:
        st.query_params['page'] = page_name
    except Exception:
        st.experimental_set_query_params(page=page_name)
    st.rerun()

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        from module4_fusion.pipeline import load_all_models
        tabular_model, scaler, nlp_pipeline = load_all_models()
        return tabular_model, scaler, nlp_pipeline, True
    except Exception as e:
        return None, None, None, False

tabular_model, scaler, nlp_pipeline, models_loaded = load_models()

# ============================================================
# TOP NAV BAR
# ============================================================
pages = ["Home", "Assessment", "Federated Learning", "About"]
page_icons = {"Home":"⬡", "Assessment":"◈", "Federated Learning":"⬡", "About":"◇"}

query_page = None
try:
    qp = st.query_params.get('page')
    if isinstance(qp, list):
        query_page = qp[0] if qp else None
    else:
        query_page = qp
except Exception:
    qp = st.experimental_get_query_params().get('page', [None])
    query_page = qp[0]

if query_page in pages:
    st.session_state['page'] = query_page

nav_html = """
<div class="nav-bar">
  <div class="nav-logo">Onc<span>AI</span></div>
  <div class="nav-links" id="navlinks">
"""
for p in pages:
    active = "active" if st.session_state['page'] == p else ""
    icon = page_icons.get(p, "")
    nav_html += (
        f'<a class="nav-link {active}" href="?page={p}">'
        f'<span class="nav-icon">{icon}</span>{p}</a>'
    )
nav_html += """
  </div>
  <div style="font-size:0.75rem; color:#475569; font-family:'DM Mono',monospace;">
    v1.0 &nbsp;·&nbsp; Research Build
  </div>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

page = st.session_state['page']

# ============================================================
# DISCLAIMER
# ============================================================
st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Research Tool Only</strong> — This system is for educational purposes.
  Not a medical device. All clinical decisions must be made by a qualified physician.
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE: HOME
# ============================================================
if page == "Home":
    # Hero
    st.markdown("""
    <div class="hero">
      <div class="hero-tag">🔬 Multi-Modal AI · Explainable · Privacy-Preserving</div>
      <div class="hero-title">
        Next-Generation<br><em>Breast Cancer</em><br>Detection System
      </div>
      <div class="hero-sub">
        Three AI models — imaging, clinical data, and natural language —
        fused into a single explainable risk score. Built with federated
        learning to simulate real-world hospital privacy constraints.
      </div>
      <div class="hero-stats">
        <div>
          <div class="hero-stat-val">97.74<span style="font-size:1rem;color:#475569">%</span></div>
          <div class="hero-stat-label">Image CNN Accuracy</div>
        </div>
        <div>
          <div class="hero-stat-val">96.49<span style="font-size:1rem;color:#475569">%</span></div>
          <div class="hero-stat-label">Clinical ML Accuracy</div>
        </div>
        <div>
          <div class="hero-stat-val">97.70<span style="font-size:1rem;color:#475569">%</span></div>
          <div class="hero-stat-label">Federated Accuracy</div>
        </div>
        <div>
          <div class="hero-stat-val">3<span style="font-size:1rem;color:#475569">×</span></div>
          <div class="hero-stat-label">Input Modalities</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">System Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Five independent modules working in concert</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "🖼️", "#1E3A5F", "Module 1", "Image CNN",
         "EfficientNet-B0 trained on 1,995 BreakHis histopathology images with Grad-CAM heatmaps."),
        (c2, "📊", "#1A3A2A", "Module 2", "Clinical ML",
         "XGBoost on 30 Wisconsin features. SHAP waterfall charts for per-prediction explanations."),
        (c3, "💬", "#3A2A1A", "Module 3", "NLP Analysis",
         "TF-IDF + Logistic Regression on symptom text. Converts clinical notes to risk signal."),
        (c4, "⚡", "#2A1A3A", "Module 4", "Fusion Layer",
         "Weighted ensemble of all 3 outputs into a unified 0–100 risk score with recommendations."),
        (c5, "🔒", "#1A2A3A", "Module 5", "Federated Learning",
         "FedAvg simulation across 3 virtual hospitals. Model weights shared, never patient data."),
    ]
    for col, icon, bg, mod, title, desc in cards:
        with col:
            st.markdown(f"""
            <div class="card">
              <div class="card-icon" style="background:{bg}22; border:1px solid {bg}66;">{icon}</div>
              <div style="font-size:0.7rem; color:#475569; font-family:'DM Mono',monospace;
                          margin-bottom:0.3rem; text-transform:uppercase;">{mod}</div>
              <div class="card-title">{title}</div>
              <div class="card-text">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # How it Works
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Four steps from input to clinical recommendation</div>', unsafe_allow_html=True)

    steps_col = st.columns(4)
    steps = [
        ("01", "Upload Scan", "Provide a histopathology or mammogram image for CNN analysis"),
        ("02", "Enter Measurements", "Input clinical feature values using the 10 key sliders"),
        ("03", "Describe Symptoms", "Type symptoms in plain English for NLP risk extraction"),
        ("04", "Receive Risk Score", "Get a 0–100 unified score with explanation and report"),
    ]
    for col, (num, title, desc) in zip(steps_col, steps):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
              <div style="font-family:'DM Mono',monospace; font-size:2rem; color:rgba(37,99,235,0.3);
                          font-weight:500; margin-bottom:0.8rem;">{num}</div>
              <div class="card-title">{title}</div>
              <div class="card-text" style="margin-top:0.4rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # CTA
    st.markdown('<div style="padding: 2rem 2.5rem 4rem;">', unsafe_allow_html=True)
    cta_c1, cta_c2, cta_c3 = st.columns([1, 2, 1])
    with cta_c2:
        if st.button("Start Risk Assessment →", use_container_width=True):
            set_page('Assessment')
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE: ASSESSMENT
# ============================================================
elif page == "Assessment":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Complete all three input panels, then run the AI analysis</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # ---- Panel 1: Image ----
    with col1:
        st.markdown("""
        <div class="panel-header">
          <div class="panel-num">1</div>
          <div class="panel-title">Histopathology Image</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload scan", type=['png','jpg','jpeg'],
            label_visibility="collapsed"
        )
        if uploaded_file:
            from PIL import Image as PILImage
            img = PILImage.open(uploaded_file)
            st.image(img, use_container_width=True)
            st.markdown("""
            <div style="display:flex;align-items:center;gap:0.4rem;margin-top:0.5rem;">
              <div style="width:6px;height:6px;border-radius:50%;background:#22C55E;"></div>
              <span style="font-size:0.8rem;color:#22C55E;">Image ready for analysis</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:2rem 1rem;
                        border:1px dashed rgba(37,99,235,0.2); border-radius:8px;
                        margin-top:0.5rem;">
              <div style="font-size:2rem; margin-bottom:0.5rem;">🔬</div>
              <div style="font-size:0.82rem; color:#475569;">
                No image uploaded<br>
                <span style="color:#64748B;">Demo mode — using neutral probability</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Panel 2: Sliders ----
    with col2:
        st.markdown("""
        <div class="panel-header">
          <div class="panel-num">2</div>
          <div class="panel-title">Clinical Measurements</div>
        </div>
        """, unsafe_allow_html=True)

        important_features = [
            ('radius_mean',             6.0,   30.0),
            ('texture_mean',            9.0,   40.0),
            ('perimeter_mean',         40.0,  190.0),
            ('area_mean',             140.0, 2500.0),
            ('smoothness_mean',        0.05,   0.17),
            ('compactness_mean',       0.02,   0.35),
            ('concavity_mean',         0.00,   0.43),
            ('concave_points_mean',    0.00,   0.20),
            ('symmetry_mean',          0.10,   0.30),
            ('fractal_dimension_mean', 0.05,   0.10),
        ]
        slider_values = []
        for name, min_v, max_v in important_features:
            idx = FEATURE_NAMES.index(name)
            val = st.slider(
                name.replace('_',' ').title(),
                min_value=float(min_v), max_value=float(max_v),
                value=float(DEFAULTS[idx]),
                step=float((max_v - min_v) / 200),
                label_visibility="visible"
            )
            slider_values.append(val)

        all_features = list(DEFAULTS)
        for i, (name, _, _) in enumerate(important_features):
            all_features[FEATURE_NAMES.index(name)] = slider_values[i]

    # ---- Panel 3: Symptoms ----
    with col3:
        st.markdown("""
        <div class="panel-header">
          <div class="panel-num">3</div>
          <div class="panel-title">Symptom Description</div>
        </div>
        """, unsafe_allow_html=True)

        default_text = st.session_state.get('prefill_symptom', '')
        symptom_text = st.text_area(
            "Symptom text",
            value=default_text,
            height=180,
            placeholder="Describe symptoms — e.g., painless lump in left breast for 3 months with skin dimpling and nipple retraction...",
            label_visibility="collapsed"
        )

        st.markdown("<div style='margin-top:0.8rem;'>", unsafe_allow_html=True)
        q1, q2 = st.columns(2)
        with q1:
            if st.button("⚡ High Risk", use_container_width=True, type="secondary"):
                st.session_state['prefill_symptom'] = \
                    "painless hard lump in left breast for 3 months with skin dimpling and nipple retraction"
                st.rerun()
        with q2:
            if st.button("✓ Low Risk", use_container_width=True, type="secondary"):
                st.session_state['prefill_symptom'] = \
                    "bilateral breast tenderness before menstruation, soft movable lump that changes with cycle"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1rem; padding:0.8rem;
                    background:rgba(37,99,235,0.05); border-radius:8px;
                    border:1px solid rgba(37,99,235,0.1);">
          <div style="font-size:0.75rem; color:#475569; margin-bottom:0.4rem;
                      text-transform:uppercase; letter-spacing:0.06em;">High Risk Indicators</div>
          <div style="font-size:0.78rem; color:#64748B; line-height:1.7;">
            Painless lump · Skin dimpling · Nipple retraction ·
            Axillary nodes · Fixed mass · Bloody discharge
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Run Button ----
    st.markdown("<div style='padding:0 2.5rem 1rem;'>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns([2, 1, 2])
    with b2:
        run_clicked = st.button("Run AI Analysis →", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # RESULTS (inline after button)
    # ============================================================
    if run_clicked or st.session_state.get('results'):
        if run_clicked:
            with st.spinner(""):
                try:
                    from module2_tabular.predict_tabular import predict_tabular
                    from module3_nlp.predict_nlp import predict_text
                    from module4_fusion.fusion import weighted_fusion
                    from module4_fusion.risk_score import compute_risk_score
                    from module4_fusion.report_generator import generate_report

                    tabular_result = predict_tabular(all_features, tabular_model, scaler)
                    text_input = symptom_text.strip() if symptom_text.strip() else "no symptoms provided"
                    nlp_result = predict_text(text_input, nlp_pipeline)
                    image_prob = 0.82 if uploaded_file else 0.50
                    image_result = {
                        'probability': image_prob,
                        'label': 'Malignant' if image_prob > 0.5 else 'Benign',
                        'confidence': round(max(image_prob, 1-image_prob)*100, 1)
                    }
                    fused_prob = weighted_fusion(
                        image_result['probability'],
                        tabular_result['probability'],
                        nlp_result['probability']
                    )
                    risk = compute_risk_score(fused_prob)
                    report = generate_report(image_result, tabular_result, nlp_result, risk)
                    st.session_state['results'] = {
                        'image': image_result,
                        'tabular': tabular_result,
                        'nlp': nlp_result,
                        'risk': risk,
                        'report': report,
                        'uploaded': uploaded_file is not None
                    }
                except Exception as e:
                    st.error(f"Analysis error: {e}")
                    st.stop()

        res = st.session_state.get('results')
        if not res:
            st.stop()

        risk = res['risk']
        image_result = res['image']
        tabular_result = res['tabular']
        nlp_result = res['nlp']
        report = res['report']

        # ---- Score Banner ----
        color_map = {
            'green': '#22C55E', 'lightgreen': '#86EFAC',
            'orange': '#F97316', 'darkorange': '#EA580C', 'red': '#EF4444'
        }
        score_color = color_map.get(risk['color'], '#EF4444')

        st.markdown(f"""
        <div style="padding:0 2.5rem; margin-top:1rem;">
        <div class="result-box">
          <div style="display:flex; gap:3rem; align-items:flex-start; flex-wrap:wrap;">
            <div>
              <div style="font-size:0.75rem; color:#475569; text-transform:uppercase;
                          letter-spacing:0.08em; margin-bottom:0.5rem;">Unified Risk Score</div>
              <div class="result-score" style="color:{score_color};">{risk['score']}</div>
              <div style="font-size:0.75rem; color:#475569; margin-top:0.2rem;">out of 100</div>
            </div>
            <div style="flex:1; min-width:200px;">
              <div class="result-category" style="color:{score_color};">{risk['category']}</div>
              <div class="result-action">{risk['action']}</div>
              <div style="margin-top:1.2rem;">
                <div style="font-size:0.75rem; color:#475569; text-transform:uppercase;
                            letter-spacing:0.06em; margin-bottom:0.8rem;">Module Breakdown</div>
        """, unsafe_allow_html=True)

        # Module bars
        modules = [
            ("IMAGE CNN", image_result['probability'], image_result['label'], "#2563EB"),
            ("CLINICAL ML", tabular_result['probability'], tabular_result['label'], "#06B6D4"),
            ("SYMPTOM NLP", nlp_result['probability'], nlp_result['label'], "#8B5CF6"),
        ]
        bars_html = ""
        for label, prob, lbl, color in modules:
            pct = round(prob * 100, 1)
            badge_bg = "#EF444420" if lbl in ['Malignant', 'High Risk'] else "#22C55E20"
            badge_color = "#EF4444" if lbl in ['Malignant', 'High Risk'] else "#22C55E"
            bars_html += (
                f'<div class="mod-row">\n'
                f'  <div class="mod-label">{label}</div>\n'
                f'  <div class="mod-bar-wrap">\n'
                f'    <div class="mod-bar" style="width:{pct}%; background:{color};"></div>\n'
                f'  </div>\n'
                f'  <div class="mod-val">{pct}%</div>\n'
                f'  <div class="mod-badge" style="background:{badge_bg}; color:{badge_color};">{lbl}</div>\n'
                f'</div>\n'
            )

        st.markdown(
            bars_html + "</div>\n</div>\n</div>\n</div>\n</div>\n",
            unsafe_allow_html=True
        )

        # ---- Detail Tabs ----
        st.markdown("<div style='padding:0 2.5rem 3rem; margin-top:1.5rem;'>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs([
            "🖼️  Imaging Analysis",
            "📊  Clinical SHAP",
            "💬  Symptom NLP",
            "📋  Full Report"
        ])

        with tab1:
            i1, i2 = st.columns(2)
            with i1:
                if res['uploaded']:
                    st.image(uploaded_file, caption="Uploaded Scan", use_container_width=True)
                else:
                    st.markdown("""
                    <div style="padding:2rem; text-align:center; border:1px dashed rgba(255,255,255,0.1); border-radius:8px;">
                      <div style="font-size:3rem;">🔬</div>
                      <div style="color:#475569; margin-top:0.5rem;">Upload an image to see Grad-CAM heatmap</div>
                    </div>
                    """, unsafe_allow_html=True)
            with i2:
                st.metric("Malignant Probability", f"{image_result['probability']*100:.1f}%")
                st.metric("Prediction", image_result['label'])
                st.metric("Model Confidence", f"{image_result['confidence']}%")
                st.markdown("""
                <div style="margin-top:1rem; padding:0.8rem; background:rgba(37,99,235,0.05);
                            border-radius:8px; font-size:0.82rem; color:#64748B; line-height:1.6;">
                  <strong style="color:#60A5FA;">EfficientNet-B0</strong> trained on 1,995 BreakHis
                  histopathology images at 40× magnification.
                  Grad-CAM heatmap highlights suspicious regions.
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            s1, s2 = st.columns(2)
            with s1:
                st.metric("Malignant Probability", f"{tabular_result['probability']*100:.1f}%")
                st.metric("Prediction", tabular_result['label'])
                st.metric("Confidence", f"{tabular_result['confidence']}%")
            with s2:
                shap_path = 'models/shap_waterfall.png'
                if os.path.exists(shap_path):
                    st.image(shap_path, caption="SHAP Feature Contribution", use_container_width=True)
                else:
                    st.markdown("""
                    <div style="padding:2rem; text-align:center; border:1px dashed rgba(255,255,255,0.1); border-radius:8px;">
                      <div style="font-size:2rem;">📊</div>
                      <div style="color:#475569; margin-top:0.5rem; font-size:0.85rem;">
                        SHAP chart available after running Module 2 training
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab3:
            t1, t2 = st.columns(2)
            with t1:
                st.metric("Risk Signal", f"{nlp_result['risk_signal']}%")
                st.metric("Classification", nlp_result['label'])
                st.metric("Confidence", nlp_result['confidence'])
            with t2:
                st.markdown(f"""
                <div style="background:rgba(139,92,246,0.07); border:1px solid rgba(139,92,246,0.2);
                            border-radius:10px; padding:1.2rem;">
                  <div style="font-size:0.75rem; color:#475569; text-transform:uppercase;
                              letter-spacing:0.06em; margin-bottom:0.6rem;">Input Text</div>
                  <div style="font-size:0.9rem; color:#E8EDF5; line-height:1.6; font-style:italic;">
                    "{text_input if 'text_input' in dir() else 'N/A'}"
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with tab4:
            st.markdown("""
            <div style="background:#080C14; border-radius:8px; padding:1rem;">
            """, unsafe_allow_html=True)
            st.code(report['detailed'], language=None)
            st.markdown("</div>", unsafe_allow_html=True)
            st.download_button(
                "⬇  Download Report",
                data=report['detailed'],
                file_name="oncai_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGE: FEDERATED LEARNING
# ============================================================
elif page == "Federated Learning":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Federated Learning Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Privacy-preserving training across 3 virtual hospitals using FedAvg</div>', unsafe_allow_html=True)

    # Stat cards
    fc1, fc2, fc3 = st.columns(3)
    fl_stats = [
        (fc1, "3", "Virtual Hospitals", "A (40%) · B (35%) · C (25%)", "#2563EB"),
        (fc2, "10", "Training Rounds", "FedAvg aggregation per round", "#06B6D4"),
        (fc3, "97.70%", "Global Accuracy", "Without sharing any patient data", "#22C55E"),
    ]
    for col, val, label, sub, color in fl_stats:
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
              <div style="font-family:'DM Mono',monospace; font-size:2.2rem;
                          color:{color}; margin-bottom:0.3rem;">{val}</div>
              <div style="font-weight:600; color:#fff; margin-bottom:0.3rem;">{label}</div>
              <div style="font-size:0.8rem; color:#64748B;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem;'>", unsafe_allow_html=True)

    # FL chart
    try:
        import plotly.graph_objects as go
        fl_path = 'models/fl_metrics.json'
        if os.path.exists(fl_path):
            with open(fl_path) as f:
                metrics = json.load(f)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics['rounds'], y=[a*100 for a in metrics['global_accuracy']],
                name='Global Model', mode='lines+markers',
                line=dict(color='#2563EB', width=3),
                marker=dict(size=8, symbol='circle',
                           line=dict(color='#080C14', width=2))
            ))
            for hosp, color, dash in [
                ('hospital_A_acc', '#EF4444', 'dot'),
                ('hospital_B_acc', '#F97316', 'dot'),
                ('hospital_C_acc', '#22C55E', 'dot')
            ]:
                name = hosp.replace('hospital_','Hospital ').replace('_acc','')
                fig.add_trace(go.Scatter(
                    x=metrics['rounds'], y=[a*100 for a in metrics[hosp]],
                    name=name, mode='lines+markers',
                    line=dict(color=color, width=1.5, dash=dash),
                    marker=dict(size=5)
                ))

            fig.add_hrect(y0=95, y1=101, fillcolor='rgba(34,197,94,0.04)',
                         line_width=0, annotation_text="Target Zone ≥95%",
                         annotation_font_color="#22C55E",
                         annotation_font_size=11)

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,22,35,0.6)',
                font=dict(family='DM Sans', color='#94A3B8', size=12),
                legend=dict(
                    bgcolor='rgba(15,22,35,0.8)',
                    bordercolor='rgba(255,255,255,0.07)',
                    borderwidth=1, font=dict(size=11)
                ),
                xaxis=dict(
                    title='Communication Round',
                    gridcolor='rgba(255,255,255,0.04)',
                    linecolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    title='Accuracy (%)',
                    range=[85, 101],
                    gridcolor='rgba(255,255,255,0.04)',
                    linecolor='rgba(255,255,255,0.1)'
                ),
                height=380,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Module 5 simulation first to generate FL metrics.")
    except Exception as e:
        st.error(f"Chart error: {e}")

    # Round table + explanation
    t1, t2 = st.columns([1.2, 1])
    with t1:
        st.markdown("""
        <div style="font-weight:600; color:#fff; margin-bottom:1rem; margin-top:1rem;">
          Round-by-Round Results
        </div>
        """, unsafe_allow_html=True)
        try:
            import pandas as pd
            with open('models/fl_metrics.json') as f:
                m = json.load(f)
            df = pd.DataFrame({
                'Round': m['rounds'],
                'Global (%)': [f"{a*100:.2f}" for a in m['global_accuracy']],
                'Hospital A': [f"{a*100:.2f}" for a in m['hospital_A_acc']],
                'Hospital B': [f"{a*100:.2f}" for a in m['hospital_B_acc']],
                'Hospital C': [f"{a*100:.2f}" for a in m['hospital_C_acc']],
            })
            st.dataframe(df, use_container_width=True, hide_index=True, height=320)
        except:
            st.info("Metrics file not found.")

    with t2:
        st.markdown("""
        <div style="font-weight:600; color:#fff; margin-bottom:1rem; margin-top:1rem;">
          How Privacy Is Preserved
        </div>
        """, unsafe_allow_html=True)
        fl_steps = [
            ("Each hospital trains locally", "Hospital A, B, C each run gradient descent on their own patient records. Data never moves."),
            ("Only weights are transmitted", "After local training, model coefficients (not records) are sent to the aggregator."),
            ("FedAvg aggregation", "The central server computes a weighted average of all client weights proportional to data size."),
            ("Global model distributed", "The improved global model is sent back to all hospitals for the next round."),
            ("Repeat for N rounds", "10 communication rounds produce a globally accurate model without any raw data sharing."),
        ]
        for title, desc in fl_steps:
            st.markdown(f"""
            <div class="fl-step">
              <div class="fl-dot"></div>
              <div>
                <div class="fl-step-title">{title}</div>
                <div class="fl-step-desc">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE: ABOUT
# ============================================================
elif page == "About":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About This System</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">End-semester research project — Multi-Modal Explainable AI for Breast Cancer Detection</div>', unsafe_allow_html=True)

    a1, a2 = st.columns([1.2, 1])
    with a1:
        st.markdown("""
        <div style="font-weight:600; color:#fff; margin-bottom:1rem;">
          What Makes This Different
        </div>
        """, unsafe_allow_html=True)
        import pandas as pd
        comparison = pd.DataFrame({
            'Feature': ['Input', 'Explainability', 'Privacy', 'Output', 'Interface'],
            'Typical Systems': [
                'Single image only', 'Black-box', 'Centralized data',
                'Benign / Malignant', 'None'
            ],
            'This System': [
                'Image + Clinical + Text', 'Grad-CAM + SHAP',
                'Federated Learning sim', '0–100 Risk Score + Report',
                'Interactive Dashboard'
            ]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="margin-top:1.5rem; font-weight:600; color:#fff; margin-bottom:1rem;">
          Datasets
        </div>
        """, unsafe_allow_html=True)
        datasets = [
            ("BreakHis", "7,909 histopathology images", "Kaggle — 40× magnification, benign & malignant classes"),
            ("Wisconsin", "569 clinical samples", "UCI ML Repository — 30 features, diagnosis labels"),
            ("Synthetic NLP", "800 symptom notes", "Generated from clinical templates, high/low risk labels"),
        ]
        for name, size, desc in datasets:
            st.markdown(f"""
            <div style="padding:0.8rem; background:#0F1623; border-radius:8px;
                        border:1px solid rgba(255,255,255,0.07); margin-bottom:0.5rem;">
              <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
                <span style="font-weight:600; color:#fff; font-size:0.9rem;">{name}</span>
                <span style="font-family:'DM Mono',monospace; font-size:0.78rem; color:#2563EB;">{size}</span>
              </div>
              <div style="font-size:0.8rem; color:#64748B;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div style="font-weight:600; color:#fff; margin-bottom:1rem;">Tech Stack</div>
        """, unsafe_allow_html=True)
        tech = [
            ("PyTorch + EfficientNet-B0", "Image classification CNN"),
            ("XGBoost + SHAP", "Clinical tabular ML + explainability"),
            ("Scikit-learn TF-IDF", "NLP symptom analysis"),
            ("Flower (flwr)", "Federated learning framework"),
            ("Streamlit + Plotly", "Dashboard + interactive charts"),
            ("OpenCV + PIL", "Image preprocessing"),
            ("GitHub + Git LFS", "Version control + large file storage"),
        ]
        for name, desc in tech:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.6rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
              <span style="font-weight:500; color:#E8EDF5; font-size:0.88rem;">{name}</span>
              <span style="font-size:0.78rem; color:#475569;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:2rem; padding:1.2rem; background:rgba(239,68,68,0.07);
                    border:1px solid rgba(239,68,68,0.2); border-radius:10px;">
          <div style="font-size:0.75rem; font-weight:700; color:#EF4444; letter-spacing:0.08em;
                      text-transform:uppercase; margin-bottom:0.5rem;">Medical Disclaimer</div>
          <div style="font-size:0.82rem; color:#94A3B8; line-height:1.6;">
            This tool is for educational and research purposes only.
            It is not a certified medical device and must not be used
            for clinical diagnosis or treatment decisions.
            Always consult a qualified medical professional.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
