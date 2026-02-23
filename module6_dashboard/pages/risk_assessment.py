import streamlit as st
from PIL import Image
import tempfile
import os

# Wisconsin feature names in order
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Realistic default values (average patient)
DEFAULTS = [
    14.13, 18.25, 92.0, 654.9, 0.096, 0.104, 0.089, 0.049, 0.181, 0.062,
    0.405, 1.216, 2.866, 40.34, 0.007, 0.025, 0.032, 0.012, 0.020, 0.004,
    16.27, 25.68, 107.3, 880.6, 0.132, 0.253, 0.272, 0.115, 0.290, 0.084
]

def show_risk_assessment():
    st.title("AI Risk Assessment")
    st.markdown("Fill in all three sections below then click **Run Analysis**.")
    st.markdown("---")

    col1, col2, col3 = st.columns([1.2, 1.2, 1])

    # ---- COLUMN 1: Image Upload ----
    with col1:
        st.subheader("1. Histopathology Image")
        uploaded_file = st.file_uploader(
            "Upload scan (PNG or JPG)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a breast histopathology or mammogram image"
        )
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded scan", use_column_width=True)
            st.session_state['uploaded_image'] = uploaded_file
            st.success("Image ready")
        else:
            st.info("No image uploaded — demo mode will use a sample probability")

    # ---- COLUMN 2: Clinical Features ----
    with col2:
        st.subheader("2. Clinical Features")
        st.caption("Adjust sliders to match patient measurements")

        features = []
        # Show first 10 most important features as sliders
        important_features = [
            ('radius_mean', 6.0, 30.0),
            ('texture_mean', 9.0, 40.0),
            ('perimeter_mean', 40.0, 190.0),
            ('area_mean', 140.0, 2500.0),
            ('smoothness_mean', 0.05, 0.17),
            ('compactness_mean', 0.02, 0.35),
            ('concavity_mean', 0.0, 0.43),
            ('concave_points_mean', 0.0, 0.20),
            ('symmetry_mean', 0.1, 0.3),
            ('fractal_dimension_mean', 0.05, 0.1),
        ]

        slider_values = []
        for name, min_v, max_v in important_features:
            idx = FEATURE_NAMES.index(name)
            val = st.slider(
                name.replace('_', ' ').title(),
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(DEFAULTS[idx]),
                step=float((max_v - min_v) / 100)
            )
            slider_values.append(val)

        # Fill remaining 20 features with defaults
        all_features = list(DEFAULTS)
        for i, (name, _, _) in enumerate(important_features):
            idx = FEATURE_NAMES.index(name)
            all_features[idx] = slider_values[i]

        st.session_state['clinical_features'] = all_features

    # ---- COLUMN 3: Symptom Text ----
    with col3:
        st.subheader("3. Symptom Description")
        symptom_text = st.text_area(
            "Describe symptoms in plain English",
            height=180,
            placeholder=(
                "e.g., painless lump in left breast "
                "for 3 months with skin dimpling..."
            )
        )
        st.session_state['symptom_text'] = symptom_text

        st.markdown("**Example inputs:**")
        if st.button("High Risk Example"):
            st.session_state['symptom_text'] = \
                "painless hard lump in left breast for 3 months with skin dimpling and nipple retraction"
            st.rerun()
        if st.button("Low Risk Example"):
            st.session_state['symptom_text'] = \
                "bilateral breast tenderness before menstruation, cyclical pain that resolves after period"
            st.rerun()

    st.markdown("---")

    # ---- RUN BUTTON ----
    run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
    with run_col2:
        run_clicked = st.button(
            "Run AI Analysis",
            type="primary",
            use_container_width=True
        )

    if run_clicked:
        st.session_state['run_analysis'] = True
        st.session_state['symptom_text'] = symptom_text
        st.switch_page("pages/results.py")
