import shap
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_shap_explanation(model, X_input, feature_names,
                               output_path='models/shap_waterfall.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    # Waterfall plot for single prediction
    plt.figure(figsize=(10, 7))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_input[0],
            feature_names=feature_names
        ),
        show=False
    )
    plt.title('SHAP Feature Contribution', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP waterfall chart saved to {output_path}")
    return shap_values

def generate_shap_summary(model, X_test, feature_names,
                           output_path='models/shap_summary.png'):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        show=False,
        plot_size=None
    )
    plt.title('SHAP Feature Importance Summary', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary chart saved to {output_path}")
