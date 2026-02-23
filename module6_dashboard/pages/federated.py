import streamlit as st
from module6_dashboard.components.charts import render_fl_chart
import json

def show_federated():
    st.title("Federated Learning Simulation")
    st.markdown("""
    This page shows how the AI was trained using **Federated Learning** —
    a privacy-preserving technique where hospitals collaborate without
    sharing patient data.
    """)

    st.markdown("---")

    # Key concept boxes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:#D6E4F0; padding:1rem; border-radius:8px; text-align:center;'>
            <h4>3 Hospitals</h4>
            <p>A (40%) · B (35%) · C (25%)</p>
            <p>Each keeps data private</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#D5F5E3; padding:1rem; border-radius:8px; text-align:center;'>
            <h4>10 Rounds</h4>
            <p>FedAvg aggregation</p>
            <p>Only weights shared</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:#FDEBD0; padding:1rem; border-radius:8px; text-align:center;'>
            <h4>97.70% Accuracy</h4>
            <p>Global model performance</p>
            <p>No raw data shared</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Global Model Accuracy Across Rounds")
    st.plotly_chart(render_fl_chart(), use_container_width=True)

    st.markdown("---")
    st.subheader("Round-by-Round Metrics")

    try:
        with open('models/fl_metrics.json') as f:
            metrics = json.load(f)

        import pandas as pd
        df = pd.DataFrame({
            'Round': metrics['rounds'],
            'Global Acc (%)': [f"{a*100:.2f}" for a in metrics['global_accuracy']],
            'Hospital A (%)': [f"{a*100:.2f}" for a in metrics['hospital_A_acc']],
            'Hospital B (%)': [f"{a*100:.2f}" for a in metrics['hospital_B_acc']],
            'Hospital C (%)': [f"{a*100:.2f}" for a in metrics['hospital_C_acc']],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
    except:
        st.info("Run Module 5 simulation to generate FL metrics.")

    st.markdown("---")
    st.subheader("How Federated Learning Protects Privacy")
    st.markdown("""
    - **Hospital A, B, C** each train a local model on their own patient data
    - Only **model weights** (not patient records) are sent to the aggregator
    - The aggregator applies **FedAvg** — averaging weights proportional to data size
    - The updated global model is sent back to all hospitals
    - Patient data **never leaves** the hospital at any point
    """)
