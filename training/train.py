import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2


def select_threshold(y_true, probabilities):
    """Select threshold maximizing F1 while keeping recall reasonably high."""
    best_threshold = 0.5
    best_f1 = -1.0
    recall_floor = 0.70

    for threshold in np.linspace(0.10, 0.90, 81):
        preds = (probabilities >= threshold).astype(int)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        if recall >= recall_floor and f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    # Fallback when recall floor is too strict for a dataset slice.
    if best_f1 < 0:
        for threshold in np.linspace(0.10, 0.90, 81):
            preds = (probabilities >= threshold).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)

    return best_threshold


def evaluate(y_true, probabilities, threshold):
    preds = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def main():
    data = pd.read_csv("final_data.csv")
    if "class" not in data.columns:
        raise ValueError("Expected a 'class' column in final_data.csv")

    y = data["class"].astype(int)
    X = data.drop(columns=["class"])
    feature_names = X.columns.tolist()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    val_ratio_of_train_val = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_of_train_val,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    candidate_models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    min_samples_leaf=2,
                )),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("model", XGBClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    scale_pos_weight=len(y_train) / sum(y_train),  # Handle class imbalance
                    eval_metric='logloss',
                )),
            ]
        ),
    }

    # Train all models for ensemble
    trained_models = {}
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    # Create ensemble using all models
    ensemble_estimators = [(name, model) for name, model in trained_models.items()]
    ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    ensemble.fit(X_train, y_train)

    # Evaluate individual models and ensemble on validation set
    val_results = {}
    for name, model in trained_models.items():
        val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        val_results[name] = val_auc

    ensemble_val_prob = ensemble.predict_proba(X_val)[:, 1]
    ensemble_val_auc = roc_auc_score(y_val, ensemble_val_prob)
    val_results['ensemble'] = ensemble_val_auc

    # Calibrate the ensemble
    calibrated = CalibratedClassifierCV(estimator=ensemble, method="sigmoid", cv=5)
    calibrated.fit(X_train_val, y_train_val)

    val_prob = calibrated.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_prob)

    test_prob = calibrated.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_prob, threshold)

    artifact = {
        "model": calibrated,
        "threshold": threshold,
        "feature_names": feature_names,
        "horizon_days": 15,
        "ensemble_models": list(trained_models.keys()),
        "individual_validation_aucs": val_results,
        "ensemble_validation_auc": float(ensemble_val_auc),
        "test_metrics": test_metrics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open("model_bundle.pkl", "wb") as f:
        pickle.dump(artifact, f)

    # Backward-compatible output for existing app code paths.
    with open("model.pickle", "wb") as f:
        pickle.dump(calibrated, f)

    print("Training complete")
    print(f"Ensemble models: {list(trained_models.keys())}")
    print(f"Individual validation AUCs: {val_results}")
    print(f"Ensemble validation AUC: {ensemble_val_auc:.4f}")
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("Saved model_bundle.pkl and model.pickle")


if __name__ == "__main__":
    main()
