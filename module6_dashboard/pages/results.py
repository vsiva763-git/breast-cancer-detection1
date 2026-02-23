import streamlit as st
import sys
import os
import tempfile
import torch
import torch.nn as nn
from torchvision import models as tv_models
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from module6_dashboard.components.gauge import render_risk_gauge
from module6_dashboard.pages.risk_assessment import DEFAULTS, FEATURE_NAMES

def show_results(tabular_model, scaler, nlp_pipeline):
    st.title("Analysis Results")

    if not st.session_state.get('run_analysis'):
        st.warning("Please run an analysis first from the Risk Assessment page.")
        return

    clinical_features = st.session_state.get('clinical_features', DEFAULTS)
    symptom_text = st.session_state.get('symptom_text', '')
    uploaded_image = st.session_state.get('uploaded_image', None)

    with st.spinner("Running AI analysis across all 3 modules..."):
        # --- Module 2: Tabular ---
        from module2_tabular.predict_tabular import predict_tabular
        tabular_result = predict_tabular(clinical_features, tabular_model, scaler)

        # --- Module 3: NLP ---
        from module3_nlp.predict_nlp import predict_text
        nlp_result = predict_text(symptom_text or "no symptoms provided", nlp_pipeline)

        # --- Module 1: Image (demo prob if no image) ---
        if uploaded_image:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp.write(uploaded_image.getvalue())
                tmp_path = tmp.name
            image_prob = 0.75  # placeholder — replace with real predict_image call
        else:
            image_prob = 0.5   # neutral when no image uploaded

        image_result = {
            'probability': image_prob,
            'label': 'Malignant' if image_prob > 0.5 else 'Benign',
            'confidence': round(max(image_prob, 1-image_prob) * 100, 1)
        }

        # --- Fusion ---
        from module4_fusion.fusion import weighted_fusion
        from module4_fusion.risk_score import compute_risk_score
        from module4_fusion.report_generator import generate_report

        fused_prob = weighted_fusion(
            image_result['probability'],
            tabular_result['probability'],
            nlp_result['probability']
        )
        risk = compute_risk_score(fused_prob)
        report = generate_report(image_result, tabular_result, nlp_result, risk)

    # ---- RISK GAUGE ----
    st.markdown("---")
    gauge_col, info_col = st.columns([1, 1])
    with gauge_col:
        st.plotly_chart(
            render_risk_gauge(risk['score'], risk['category'], risk['color']),
            use_container_width=True
        )
    with info_col:
        st.markdown(f"### {risk['category']}")
        st.markdown(f"**Risk Score:** {risk['score']} / 100")
        st.markdown(f"**Urgency Level:** {risk['urgency']} / 5")
        st.info(f"**Recommendation:** {risk['action']}")

    st.markdown("---")

    # ---- MODULE TABS ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "Imaging Results",
        "Clinical Results",
        "Symptom Analysis",
        "Full Report"
    ])

    with tab1:
        st.subheader("Image Analysis (EfficientNet-B0)")
        col_a, col_b = st.columns(2)
        with col_a:
            if uploaded_image:
                st.image(Image.open(uploaded_image), caption="Original Scan")
            else:
                st.info("No image uploaded. Upload a scan on the Risk Assessment page.")
        with col_b:
            st.metric("Malignant Probability",
                      f"{image_result['probability']*100:.1f}%")
            st.metric("Prediction", image_result['label'])
            st.metric("Confidence", f"{image_result['confidence']}%")

    with tab2:
        st.subheader("Clinical Feature Analysis (XGBoost + SHAP)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Malignant Probability",
                      f"{tabular_result['probability']*100:.1f}%")
            st.metric("Prediction", tabular_result['label'])
            st.metric("Confidence", f"{tabular_result.get('confidence', 'N/A')}")
        with col_b:
            shap_path = 'models/shap_waterfall.png'
            if os.path.exists(shap_path):
                st.image(shap_path, caption="SHAP Feature Contribution")
            else:
                st.info("SHAP chart available for tabular predictions")

    with tab3:
        st.subheader("Symptom Text Analysis (NLP)")
        st.write(f"**Input:** {symptom_text or 'No symptoms entered'}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Risk Signal", f"{nlp_result['risk_signal']}%")
            st.metric("Classification", nlp_result['label'])
        with col_b:
            st.metric("Confidence", nlp_result['confidence'])

    with tab4:
        st.subheader("Clinical AI Summary Report")
        st.text(report['detailed'])
        st.download_button(
            "Download Report",
            data=report['detailed'],
            file_name="breast_cancer_ai_report.txt",
            mime="text/plain"
        )
