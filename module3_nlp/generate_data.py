import pandas as pd
import random
import os

def generate_symptom_dataset(output_path='data/symptom_data.csv', samples_per_class=400):
    random.seed(42)

    # High risk symptom templates
    high_risk_templates = [
        "painless lump in {side} breast for {duration}",
        "hard immovable mass in {side} breast upper outer quadrant",
        "nipple retraction noticed in {side} breast recently",
        "skin dimpling on {side} breast with lump",
        "bloody nipple discharge from {side} breast",
        "axillary lymph node swelling with {side} breast lump",
        "breast lump with skin thickening and redness",
        "peau d orange skin change on {side} breast",
        "painless hard lump near {side} nipple",
        "nipple inversion with breast asymmetry",
        "swollen lymph nodes under arm with breast mass",
        "breast lump that has grown over {duration}",
        "non tender fixed mass in {side} breast",
        "unilateral nipple discharge with lump",
        "breast skin ulceration with underlying mass",
        "hard irregular lump with nipple discharge",
        "breast asymmetry with visible lump {side}",
        "firm breast lump not moving on palpation",
        "lump with dimpling on compression of {side} breast",
        "nipple crusting with underlying mass in breast",
    ]

    # Low risk symptom templates
    low_risk_templates = [
        "bilateral breast tenderness before menstruation",
        "soft movable lump that changes with menstrual cycle",
        "breast pain during breastfeeding on both sides",
        "symmetrical breast firmness no skin changes",
        "cyclical breast pain relieved after period",
        "tender breast cyst confirmed on ultrasound",
        "fibrocystic breast changes bilateral",
        "breast engorgement during lactation",
        "soft fluctuant lump that is tender to touch",
        "breast pain associated with hormonal changes",
        "mild breast swelling before menstrual cycle",
        "bilateral nipple discharge milky white",
        "breast tenderness with oral contraceptive use",
        "round smooth mobile lump in {side} breast",
        "soft breast lump disappears after period",
        "breast fullness and heaviness premenstrual",
        "nipple discharge clear bilateral both sides",
        "breast pain relieved by wearing support bra",
        "tender mobile lump consistent with fibroadenoma",
        "cyclical mastalgia worse before period",
    ]

    sides = ['left', 'right']
    durations = ['2 weeks', '1 month', '3 months', '6 months', '2 months']

    def fill_template(template):
        text = template
        if '{side}' in text:
            text = text.replace('{side}', random.choice(sides))
        if '{duration}' in text:
            text = text.replace('{duration}', random.choice(durations))
        return text

    data = []

    # Generate high risk samples
    for _ in range(samples_per_class):
        template = random.choice(high_risk_templates)
        text = fill_template(template)
        # Add random noise variations
        if random.random() > 0.7:
            text = text + ", no pain reported"
        if random.random() > 0.8:
            text = text + ", noticed recently"
        data.append({'text': text, 'label': 1})

    # Generate low risk samples
    for _ in range(samples_per_class):
        template = random.choice(low_risk_templates)
        text = fill_template(template)
        if random.random() > 0.7:
            text = text + ", pain resolves after period"
        if random.random() > 0.8:
            text = text + ", bilateral symptoms"
        data.append({'text': text, 'label': 0})

    random.shuffle(data)
    df = pd.DataFrame(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset generated: {len(df)} samples")
    print(f"High Risk (1): {df['label'].sum()}")
    print(f"Low Risk  (0): {(df['label']==0).sum()}")
    print(f"Saved to: {output_path}")
    return df
