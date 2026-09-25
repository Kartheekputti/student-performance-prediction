"""Preprocessing pipeline: cleaning + ColumnTransformer (FR3)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "Attendance", "Study_Hours", "Previous_GPA",
    "Assignment_Score", "Internal_Score", "Previous_Failures",
]
CATEGORICAL_FEATURES = ["Participation"]
TARGET = "Performance"
ID_COL = "Student_ID"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values markers, drop duplicates (full cleaning happens in pipeline)."""
    df = df.copy()
    df = df.drop_duplicates()
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def validate_input(attendance, study_hours, prev_gpa, assignment, internal, failures):
    """FR2 data validation. Returns list of error strings (empty = valid)."""
    errors = []
    if not (0 <= attendance <= 100):
        errors.append("Attendance must be between 0-100%.")
    if not (0 <= study_hours <= 15):
        errors.append("Study hours must be between 0-15.")
    if not (0 <= prev_gpa <= 10):
        errors.append("Previous GPA must be between 0-10.")
    if not (0 <= assignment <= 100):
        errors.append("Assignment score must be between 0-100.")
    if not (0 <= internal <= 100):
        errors.append("Internal score must be between 0-100.")
    if not (0 <= failures <= 10):
        errors.append("Previous failures must be between 0-10.")
    return errors
