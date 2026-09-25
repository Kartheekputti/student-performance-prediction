"""Generate synthetic historical student dataset per SRS Section 7."""
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES = 600


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    attendance = np.clip(rng.normal(75, 12, n_samples), 30, 100).round(1)
    study_hours = np.clip(rng.normal(4.5, 2.2, n_samples), 0, 12).round(1)
    previous_gpa = np.clip(rng.normal(7.0, 1.5, n_samples), 0, 10).round(2)
    assignment = np.clip(rng.normal(72, 15, n_samples), 0, 100).round(1)
    internal = np.clip(rng.normal(68, 16, n_samples), 0, 100).round(1)
    failures = rng.choice([0, 0, 0, 1, 1, 2, 3, 4], size=n_samples)
    participation = rng.choice(["Low", "Medium", "High"], size=n_samples, p=[0.3, 0.45, 0.25])

    # Composite score drives the label (keeps EDA correlations realistic)
    part_map = {"Low": 0, "Medium": 5, "High": 10}
    part_bonus = np.array([part_map[p] for p in participation])
    score = (
        0.25 * attendance
        + 3.0 * study_hours
        + 6.0 * previous_gpa
        + 0.20 * assignment
        + 0.25 * internal
        + part_bonus
        - 8.0 * failures
        + rng.normal(0, 6, n_samples)
    )
    # Quantile-based labelling -> Low / Average / High
    low_thr = np.quantile(score, 0.33)
    high_thr = np.quantile(score, 0.66)
    performance = np.where(score <= low_thr, "Low", np.where(score >= high_thr, "High", "Average"))

    ids = [f"ST{i:03d}" for i in range(1, n_samples + 1)]
    df = pd.DataFrame({
        "Student_ID": ids,
        "Attendance": attendance,
        "Study_Hours": study_hours,
        "Previous_GPA": previous_gpa,
        "Assignment_Score": assignment,
        "Internal_Score": internal,
        "Previous_Failures": failures,
        "Participation": participation,
        "Performance": performance,
    })

    # Inject a few missing values / duplicates to exercise preprocessing (FR3)
    for col in ["Attendance", "Study_Hours", "Assignment_Score"]:
        idx = rng.choice(n_samples, size=5, replace=False)
        df.loc[idx, col] = np.nan
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)  # 3 duplicates
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/student_data.csv", index=False)
    print(f"Saved data/student_data.csv with shape {df.shape}")
    print(df["Performance"].value_counts())
