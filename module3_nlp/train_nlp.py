from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import pandas as pd
import joblib
import json
import os

def train_nlp_pipeline(csv_path='data/symptom_data.csv',
                        model_path='models/nlp_pipeline.pkl'):

    df = pd.read_csv(csv_path)
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF + Logistic Regression pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=8000,
            min_df=1,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Cross validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

    print("=" * 40)
    print("  MODULE 3 TRAINING RESULTS")
    print("=" * 40)
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  CV Accuracy : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print("=" * 40)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Low Risk', 'High Risk']))

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"NLP pipeline saved to {model_path}")

    # Save metrics
    metrics = {
        'accuracy': round(acc * 100, 2),
        'auc_roc': round(auc, 4),
        'cv_accuracy': round(cv_scores.mean() * 100, 2)
    }
    with open('models/nlp_metrics.json', 'w') as f:
        json.dump(metrics, f)

    return pipeline, metrics
