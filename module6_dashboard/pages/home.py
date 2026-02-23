import streamlit as st
from module6_dashboard.components.charts import render_module_comparison

def show_home():
    st.markdown("""
        <div style='text-align:center; padding: 2rem 0 1rem 0;'>
            <h1 style='color:#1E3A5F; font-size:2.5rem;'>
                Breast Cancer Detection AI
            </h1>
            <p style='color:#555; font-size:1.1rem;'>
                Multi-Modal Explainable AI with Federated Privacy Simulation
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.warning("For Research and Educational Purposes Only — Not a Medical Device")

    st.markdown("---")

    # Architecture overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:#D6E4F0; padding:1.2rem; border-radius:10px; text-align:center;'>
            <h3>3 AI Layers</h3>
            <p>Image CNN + Clinical ML + NLP combined into one risk score</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#D5F5E3; padding:1.2rem; border-radius:10px; text-align:center;'>
            <h3>Explainable AI</h3>
            <p>Grad-CAM heatmaps + SHAP charts show exactly why the AI decides</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:#FDEBD0; padding:1.2rem; border-radius:10px; text-align:center;'>
            <h3>Federated Privacy</h3>
            <p>Simulates 3 hospitals training without sharing patient data</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("How It Works")

    steps = [
        ("1. Upload Scan", "Upload a histopathology image. EfficientNet-B0 classifies it and Grad-CAM shows which regions influenced the decision."),
        ("2. Enter Clinical Data", "Input 30 clinical measurements. XGBoost predicts risk and SHAP explains which features matter most."),
        ("3. Describe Symptoms", "Type symptoms in plain English. NLP model extracts a risk signal from the text."),
        ("4. Get Unified Score", "All 3 outputs are fused into a single 0-100 risk score with a clinical recommendation."),
    ]

    for title, desc in steps:
        with st.expander(title):
            st.write(desc)

    st.markdown("---")
    st.subheader("Module Performance")
    st.plotly_chart(render_module_comparison(), use_container_width=True)
