import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(*args):
    return os.path.join(ROOT_DIR, *args)

def normalize_probability(prob):
    return float(np.clip(prob, 0.0, 1.0))

def format_risk_score(prob):
    return round(float(prob) * 100, 1)
