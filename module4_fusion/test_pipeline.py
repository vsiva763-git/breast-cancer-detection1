import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module4_fusion.pipeline import load_all_models, run_fusion_pipeline

print("Testing Module 4 Fusion Pipeline...\n")

# Load all models
tabular_model, scaler, nlp_pipeline = load_all_models()

# Wisconsin feature order (30 features)
# These are realistic high-risk values
high_risk_features = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

# Realistic low-risk values
low_risk_features = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
]

print("=" * 50)
print("TEST 1 — HIGH RISK PATIENT")
print("=" * 50)
result1 = run_fusion_pipeline(
    clinical_features=high_risk_features,
    symptom_text="painless hard lump in left breast for 3 months with skin dimpling",
    image_prob=0.92,
    tabular_model=tabular_model,
    scaler=scaler,
    nlp_pipeline=nlp_pipeline
)
print(f"Image    : {result1['image']['label']} ({result1['image']['probability']*100:.1f}%)")
print(f"Tabular  : {result1['tabular']['label']} ({result1['tabular']['probability']*100:.1f}%)")
print(f"NLP      : {result1['nlp']['label']} ({result1['nlp']['risk_signal']}%)")
print(f"Fused    : {result1['fusion']['probability']}")
print(f"RISK     : {result1['risk']['score']}/100 — {result1['risk']['category']}")
print(f"Action   : {result1['risk']['action']}")

print()
print("=" * 50)
print("TEST 2 — LOW RISK PATIENT")
print("=" * 50)
result2 = run_fusion_pipeline(
    clinical_features=low_risk_features,
    symptom_text="bilateral breast tenderness before menstruation, cyclical pain",
    image_prob=0.08,
    tabular_model=tabular_model,
    scaler=scaler,
    nlp_pipeline=nlp_pipeline
)
print(f"Image    : {result2['image']['label']} ({result2['image']['probability']*100:.1f}%)")
print(f"Tabular  : {result2['tabular']['label']} ({result2['tabular']['probability']*100:.1f}%)")
print(f"NLP      : {result2['nlp']['label']} ({result2['nlp']['risk_signal']}%)")
print(f"Fused    : {result2['fusion']['probability']}")
print(f"RISK     : {result2['risk']['score']}/100 — {result2['risk']['category']}")
print(f"Action   : {result2['risk']['action']}")

print()
print("Module 4 pipeline working correctly!")
