"""Train ML models (FR4), evaluate (FR8), save best model (FR10)."""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import TARGET, ID_COL, build_preprocessor, clean_dataframe

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "student_model.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "metrics.json")
CM_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "confusion_matrix.png")


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
        "SVM": SVC(probability=True),
    }


def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = pd.read_csv(data_path)
    df = clean_dataframe(df)
    feature_cols = [c for c in df.columns if c not in (TARGET, ID_COL)]
    X = df[feature_cols]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results, best_f1, best_pipe, best_name = {}, -1, None, ""
    for name, clf in get_models().items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
            "f1_weighted": f1,
        }
        print(f"{name}: acc={results[name]['accuracy']:.3f} f1={f1:.3f}")
        if f1 > best_f1:
            best_f1, best_pipe, best_name = f1, pipe, name

    # Final evaluation of best model
    y_pred = best_pipe.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"\nBest model: {best_name}\n{report}")
    metrics = {
        "best_model": best_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "classification_report": report,
        "all_models": results,
        "classes": sorted(y.unique().tolist()),
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "classes": metrics["classes"]}, model_path)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred, labels=metrics["classes"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=metrics["classes"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix ({best_name})")
    fig.tight_layout()
    fig.savefig(CM_PATH)
    print(f"Saved model -> {model_path}, metrics -> {METRICS_PATH}, cm -> {CM_PATH}")
    return metrics


if __name__ == "__main__":
    train()
