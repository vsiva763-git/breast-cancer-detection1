def compute_risk_score(fused_probability: float) -> dict:
    """
    Convert fused probability (0-1) to a 0-100 risk score
    with category, color, and recommended action.
    """
    score = round(fused_probability * 100, 1)

    if score < 25:
        category = 'VERY LOW RISK'
        color = 'green'
        action = 'Routine annual mammogram screening recommended.'
        urgency = 1

    elif score < 45:
        category = 'LOW RISK'
        color = 'lightgreen'
        action = 'Continue regular self-examination. Next screening in 12 months.'
        urgency = 2

    elif score < 60:
        category = 'MODERATE RISK'
        color = 'orange'
        action = 'Follow-up imaging recommended within 6 months.'
        urgency = 3

    elif score < 75:
        category = 'HIGH RISK'
        color = 'darkorange'
        action = 'Specialist referral and biopsy discussion recommended.'
        urgency = 4

    else:
        category = 'VERY HIGH RISK'
        color = 'red'
        action = 'Immediate specialist consultation and biopsy strongly recommended.'
        urgency = 5

    return {
        'score': score,
        'category': category,
        'color': color,
        'action': action,
        'urgency': urgency
    }
