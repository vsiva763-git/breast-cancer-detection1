import joblib
from utils.helpers import normalize_probability

def load_nlp_model(model_path='models/nlp_pipeline.pkl'):
    return joblib.load(model_path)

def predict_text(symptom_text: str, pipeline) -> dict:
    """
    symptom_text: plain English symptom description
    Returns: dict with probability, label, confidence
    """
    if not symptom_text or len(symptom_text.strip()) < 3:
        return {
            'probability': 0.0,
            'label': 'Insufficient Input',
            'confidence': 'Low',
            'risk_signal': 0.0
        }

    prob = pipeline.predict_proba([symptom_text])[0]
    high_risk_prob = normalize_probability(prob[1])
    label = 'High Risk' if high_risk_prob > 0.5 else 'Low Risk'

    if abs(high_risk_prob - 0.5) > 0.35:
        confidence = 'High'
    elif abs(high_risk_prob - 0.5) > 0.15:
        confidence = 'Medium'
    else:
        confidence = 'Low'

    return {
        'probability': high_risk_prob,
        'label': label,
        'confidence': confidence,
        'risk_signal': round(high_risk_prob * 100, 1)
    }
