from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, accuracy_score)
import joblib
import json
import os

def train_xgboost(X_train, X_test, y_train, y_test,
                  model_path='models/xgboost_model.pkl'):

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("=" * 40)
    print("  MODULE 2 TRAINING RESULTS")
    print("=" * 40)
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  AUC-ROC  : {auc:.4f}")
    print("=" * 40)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Benign', 'Malignant']))

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save metrics for dashboard
    metrics = {
        'accuracy': round(acc * 100, 2),
        'auc_roc': round(auc, 4),
        'best_iteration': model.best_iteration
    }
    with open('models/tabular_metrics.json', 'w') as f:
        json.dump(metrics, f)

    return model, metrics
