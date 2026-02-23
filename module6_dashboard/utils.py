import streamlit as st

def show_disclaimer():
    st.markdown("""
    <div style='background:#FFF3CD; border-left:4px solid #FFC107;
                padding:0.8rem 1rem; border-radius:4px; margin:1rem 0;'>
        <strong>Disclaimer:</strong> This tool is for educational and research
        purposes only. It is not a medical device and must not be used for
        clinical diagnosis. Always consult a qualified physician.
    </div>
    """, unsafe_allow_html=True)

def colored_metric(label, value, color):
    st.markdown(f"""
    <div style='background:{color}22; border-left:4px solid {color};
                padding:0.6rem 1rem; border-radius:4px; margin:0.3rem 0;'>
        <small>{label}</small><br>
        <strong style='font-size:1.3rem;'>{value}</strong>
    </div>
    """, unsafe_allow_html=True)
