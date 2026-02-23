import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module2_tabular.preprocess import load_and_preprocess
from module2_tabular.train_xgboost import train_xgboost
from module2_tabular.shap_explain import generate_shap_explanation, generate_shap_summary

print("Starting Module 2 Training...\n")

# Step 1 - Load data
X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess()

# Step 2 - Train XGBoost
model, metrics = train_xgboost(X_train, X_test, y_train, y_test)

# Step 3 - Generate SHAP charts
print("\nGenerating SHAP explanations...")
generate_shap_explanation(model, X_test[:1], feature_names)
generate_shap_summary(model, X_test, feature_names)

print("\nModule 2 Complete!")
print(f"Accuracy : {metrics['accuracy']}%")
print(f"AUC-ROC  : {metrics['auc_roc']}")
