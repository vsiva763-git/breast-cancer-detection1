import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module3_nlp.generate_data import generate_symptom_dataset
from module3_nlp.train_nlp import train_nlp_pipeline
from module3_nlp.predict_nlp import load_nlp_model, predict_text

print("Starting Module 3 NLP Training...\n")

# Step 1 - Generate synthetic dataset
print("Generating symptom dataset...")
df = generate_symptom_dataset()

# Step 2 - Train pipeline
print("\nTraining NLP pipeline...")
pipeline, metrics = train_nlp_pipeline()

# Step 3 - Test with sample predictions
print("\n--- PREDICTION TESTS ---")
pipeline = load_nlp_model()

test_cases = [
    ("painless lump in left breast for 3 months", "HIGH RISK expected"),
    ("nipple retraction with skin dimpling noticed", "HIGH RISK expected"),
    ("bilateral breast tenderness before menstruation", "LOW RISK expected"),
    ("soft movable lump that changes with cycle", "LOW RISK expected"),
    ("hard immovable mass in right breast upper outer quadrant", "HIGH RISK expected"),
]

for text, expected in test_cases:
    result = predict_text(text, pipeline)
    print(f"Input    : {text}")
    print(f"Expected : {expected}")
    print(f"Got      : {result['label']} ({result['risk_signal']}%) - Confidence: {result['confidence']}")
    print()

print("Module 3 Complete!")
print(f"Accuracy : {metrics['accuracy']}%")
print(f"AUC-ROC  : {metrics['auc_roc']}")
