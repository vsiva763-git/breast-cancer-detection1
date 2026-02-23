import streamlit as st

def show_about():
    st.title("About This Project")

    st.markdown("""
    This project was built as an end-semester college project demonstrating
    a novel multi-modal approach to breast cancer detection using AI.
    """)

    st.markdown("---")
    st.subheader("What Makes This Different")

    data = {
        "Feature": [
            "Input Modalities",
            "Explainability",
            "Privacy",
            "Output",
            "Interface"
        ],
        "Typical Systems": [
            "Single image only",
            "Black-box prediction",
            "Centralized data",
            "Benign / Malignant label",
            "None or basic CLI"
        ],
        "This System": [
            "Image + Clinical + Text",
            "Grad-CAM + SHAP charts",
            "Federated Learning simulation",
            "0-100 risk score + report",
            "Interactive web dashboard"
        ]
    }

    import pandas as pd
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Datasets Used")
    st.markdown("""
    - **BreakHis** — 7909 breast histopathology images (benign / malignant)
    - **Wisconsin Breast Cancer** — 569 clinical samples, 30 features
    - **Synthetic Symptom Data** — 800 generated clinical notes
    """)

    st.subheader("Tech Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - PyTorch + EfficientNet-B0
        - XGBoost + SHAP
        - Scikit-learn TF-IDF
        - Flower (Federated Learning)
        """)
    with col2:
        st.markdown("""
        - Streamlit + Plotly
        - OpenCV + PIL
        - Python 3.10+
        - GitHub + Git LFS
        """)

    st.markdown("---")
    st.warning("This tool is for educational and research purposes only. "
               "It is not a medical device and must not be used for clinical diagnosis.")
