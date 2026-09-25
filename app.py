"""Student Performance Prediction System - Streamlit UI (SRS Sec 5.1, 6)."""
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import validate_input
from predict import predict_batch, predict_single
from visualize import (plot_attendance_vs_performance, plot_class_distribution,
                       plot_feature_means)

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

DATA_PATH = "data/student_data.csv"
METRICS_PATH = "results/metrics.json"
CM_PATH = "results/confusion_matrix.png"

st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Batch Predict", "Visualizations", "Model Evaluation"])

# ---------- HOME ----------
if page == "Home":
    st.title("Student Performance Prediction System")
    st.write("ML-based application that predicts **Low / Average / High** performance from "
             "attendance, study hours, previous GPA, assignment & internal scores, failures and participation.")
    st.write("**Flow:** User Interface → Data Validation → Preprocessing → ML Model → Prediction Result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Records (training)", f"{len(pd.read_csv(DATA_PATH))}" if os.path.exists(DATA_PATH) else "—")
    if os.path.exists(METRICS_PATH):
        m = json.load(open(METRICS_PATH))
        col2.metric("Best Model", m.get("best_model", "—"))
        col3.metric("Accuracy", f"{m.get('accuracy', 0):.2%}")
    st.info("Go to **Predict** and press Start Prediction. Batch CSV upload is under **Batch Predict**.")
    if st.button("Start Prediction"):
        st.switch_page if hasattr(st, "switch_page") else None
        st.warning("Use the sidebar → Predict page to enter student details.")

# ---------- PREDICT ----------
elif page == "Predict":
    st.title("Predict Student Performance")
    student_id = st.text_input("Student ID", "ST001")
    attendance = st.number_input("Attendance (%)", 0.0, 100.0, 75.0)
    study_hours = st.number_input("Study Hours (per day)", 0.0, 15.0, 4.0)
    prev_gpa = st.number_input("Previous GPA (0-10)", 0.0, 10.0, 7.0)
    assignment = st.number_input("Assignment Score (0-100)", 0.0, 100.0, 70.0)
    internal = st.number_input("Internal Score (0-100)", 0.0, 100.0, 65.0)
    failures = st.number_input("Previous Failures", 0, 10, 0, step=1)
    participation = st.selectbox("Participation", ["Low", "Medium", "High"])

    if st.button("Predict"):
        errors = validate_input(attendance, study_hours, prev_gpa, assignment, internal, failures)
        if not student_id.strip():
            errors.append("Student ID is required.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                pred, proba = predict_single({
                    "Attendance": attendance, "Study_Hours": study_hours,
                    "Previous_GPA": prev_gpa, "Assignment_Score": assignment,
                    "Internal_Score": internal, "Previous_Failures": failures,
                    "Participation": participation,
                })
                color = {"High": "green", "Average": "orange", "Low": "red"}.get(pred, "blue")
                st.markdown(f"### Result: Performance: :{color}[{pred.upper()}]")
                if proba:
                    st.write("**Prediction Probability (FR6):**")
                    st.bar_chart(pd.Series(proba))
                    for k, v in proba.items():
                        st.write(f"{k}: {v}%")
                # Report (FR10)
                report = (f"STUDENT PERFORMANCE REPORT\nStudent ID: {student_id}\n"
                          f"Attendance: {attendance}%, Study Hours: {study_hours}, GPA: {prev_gpa}\n"
                          f"Assignment: {assignment}, Internal: {internal}, Failures: {failures}, "
                          f"Participation: {participation}\nPredicted Performance: {pred}\n"
                          f"Probabilities: {proba}\n")
                st.download_button("Download Report (.txt)", report, file_name=f"{student_id}_report.txt")
            except FileNotFoundError:
                st.error("Model not found. Run `python src/train.py` first.")
            except Exception as ex:
                st.error(f"Prediction failed: {ex}")

# ---------- BATCH ----------
elif page == "Batch Predict":
    st.title("Batch Prediction (CSV upload)")
    st.write("CSV must contain: Attendance, Study_Hours, Previous_GPA, Assignment_Score, "
             "Internal_Score, Previous_Failures, Participation")
    f = st.file_uploader("Upload CSV", type="csv")
    if f is not None:
        try:
            df = pd.read_csv(f)
            st.dataframe(df.head())
            if st.button("Predict All"):
                out = predict_batch(df)
                st.success(f"Predicted {len(out)} records.")
                st.dataframe(out.head(20))
                st.download_button("Download Predictions (CSV)",
                                   out.to_csv(index=False), file_name="predictions.csv")
        except Exception as ex:
            st.error(f"Failed: {ex}")

# ---------- VISUALIZATIONS ----------
elif page == "Visualizations":
    st.title("Data Visualizations (FR9)")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.pyplot(plot_class_distribution(df))
        st.pyplot(plot_feature_means(df))
        st.pyplot(plot_attendance_vs_performance(df))
    else:
        st.error("Dataset not found.")

# ---------- EVALUATION ----------
elif page == "Model Evaluation":
    st.title("Model Evaluation (FR8)")
    if os.path.exists(METRICS_PATH):
        m = json.load(open(METRICS_PATH))
        st.write(f"**Best model:** {m['best_model']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']:.3f}")
        c2.metric("Precision", f"{m['precision_weighted']:.3f}")
        c3.metric("Recall", f"{m['recall_weighted']:.3f}")
        c4.metric("F1-score", f"{m['f1_weighted']:.3f}")
        st.text(m["classification_report"])
        st.write("**All models:**")
        st.dataframe(pd.DataFrame(m["all_models"]).T)
        if os.path.exists(CM_PATH):
            st.image(CM_PATH, caption="Confusion Matrix")
    else:
        st.error("Metrics not found. Run `python src/train.py` first.")
