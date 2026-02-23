import joblib
import numpy as np
from utils.helpers import normalize_probability, format_risk_score

def load_tabular_model(model_path='models/xgboost_model.pkl',
                        scaler_path='models/scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_tabular(features: list, model, scaler) -> dict:
    """
    features: list of 30 clinical values in Wisconsin feature order
    Returns: dict with probability, label, confidence
    """
    import numpy as np
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0]
    malignant_prob = normalize_probability(prob[1])
    label = 'Malignant' if malignant_prob > 0.5 else 'Benign'

    return {
        'probability': malignant_prob,
        'label': label,
        'confidence': format_risk_score(max(prob)),
        'benign_prob': normalize_probability(prob[0])
    }
