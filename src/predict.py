"""Prediction helper (FR5, FR6)."""
import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "student_model.pkl")


def load_model(model_path=MODEL_PATH):
    bundle = joblib.load(model_path)
    return bundle["pipeline"], bundle.get("classes")


def predict_single(data: dict, model_path=MODEL_PATH):
    """data keys: Attendance, Study_Hours, Previous_GPA, Assignment_Score,
    Internal_Score, Previous_Failures, Participation."""
    pipe, _ = load_model(model_path)
    df = pd.DataFrame([data])
    pred = pipe.predict(df)[0]
    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(df)[0]
            labels = list(pipe.classes_)
            proba = dict(zip(labels, [round(float(p) * 100, 1) for p in proba]))
        except Exception:
            proba = None
    return pred, proba


def predict_batch(df: pd.DataFrame, model_path=MODEL_PATH) -> pd.DataFrame:
    pipe, _ = load_model(model_path)
    drop = [c for c in ("Student_ID", "Performance") if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")
    out = df.copy()
    out["Predicted_Performance"] = pipe.predict(X)
    if hasattr(pipe, "predict_proba"):
        try:
            probas = pipe.predict_proba(X)
            for i, cls in enumerate(pipe.classes_):
                out[f"Prob_{cls}"] = (probas[:, i] * 100).round(1)
        except Exception:
            pass
    return out
