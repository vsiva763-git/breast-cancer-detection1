import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Breast Cancer Detection AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Load all models once at startup ----
@st.cache_resource
def load_models():
    from module4_fusion.pipeline import load_all_models
    tabular_model, scaler, nlp_pipeline = load_all_models()
    return tabular_model, scaler, nlp_pipeline

tabular_model, scaler, nlp_pipeline = load_models()

# ---- Sidebar Navigation ----
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio("Go to", [
    "Home",
    "Risk Assessment",
    "Results",
    "Federated Learning",
    "About"
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.8rem; color:#888;'>
Breast Cancer Detection AI<br>
End-Semester Project<br>
Multi-Modal + Explainable AI
</div>
""", unsafe_allow_html=True)

# ---- Page Routing ----
from module6_dashboard.pages.home import show_home
from module6_dashboard.pages.risk_assessment import show_risk_assessment
from module6_dashboard.pages.results import show_results
from module6_dashboard.pages.federated import show_federated
from module6_dashboard.pages.about import show_about

if page == "Home":
    show_home()
elif page == "Risk Assessment":
    show_risk_assessment()
elif page == "Results":
    show_results(tabular_model, scaler, nlp_pipeline)
elif page == "Federated Learning":
    show_federated()
elif page == "About":
    show_about()
