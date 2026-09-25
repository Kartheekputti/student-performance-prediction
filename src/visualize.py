"""EDA plots for the Streamlit app (FR9)."""
import matplotlib.pyplot as plt
import pandas as pd


def plot_attendance_vs_performance(df: pd.DataFrame):
    fig, ax = plt.subplots()
    for label in ["Low", "Average", "High"]:
        sub = df[df["Performance"] == label]
        ax.scatter(sub["Attendance"], sub["Internal_Score"], label=label, alpha=0.6)
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Internal Score")
    ax.set_title("Attendance vs Internal Score by Performance")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_means(df: pd.DataFrame):
    means = df.groupby("Performance")[["Attendance", "Study_Hours", "Previous_GPA",
                                       "Assignment_Score", "Internal_Score"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    means.T.plot(kind="bar", ax=ax)
    ax.set_title("Average Feature Values per Performance Category")
    ax.set_ylabel("Mean value")
    fig.tight_layout()
    return fig


def plot_class_distribution(df: pd.DataFrame):
    counts = df["Performance"].value_counts().reindex(["Low", "Average", "High"])
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax, color=["#e74c3c", "#f39c12", "#27ae60"])
    ax.set_title("Performance Category Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig
