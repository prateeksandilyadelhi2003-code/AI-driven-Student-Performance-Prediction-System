"""
Data Preprocessing Module for AI-Driven Student Performance Prediction System.

Responsibilities:
1. Loading the dataset with validation and error handling.
2. Checking and handling missing values.
3. Checking and removing duplicate records.
4. Encoding categorical features (e.g., extracurricular_activity).
5. Feature selection and scaling with StandardScaler (avoiding data leakage).
6. Train/Test splitting (80/20) with a reproducible random state.
"""

import os
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Standard list of features used across training and inference
FEATURE_COLUMNS = [
    "attendance",
    "previous_marks",
    "study_hours",
    "assignment_score",
    "internal_marks",
    "completed_assignments",
    "class_participation",
    "extracurricular_activity",
]

TARGET_REGRESSION = "final_marks"
TARGET_CLASSIFICATION = "performance_category"

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV with robust error handling.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("The provided dataset is empty.")
    
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by removing duplicates and imputing any missing values.
    """
    df = df.copy()
    
    # 1. Remove duplicate records
    initial_count = len(df)
    df = df.drop_duplicates()
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"[Preprocessing] Removed {dropped_count} duplicate rows.")
        
    # 2. Check and handle missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"[Preprocessing] Missing values detected:\n{missing_counts[missing_counts > 0]}")
        # Numerical imputation using median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        # Categorical imputation using mode
        cat_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features into numeric format.
    'extracurricular_activity': 'Yes' -> 1, 'No' -> 0
    """
    df = df.copy()
    if "extracurricular_activity" in df.columns:
        mapping = {"Yes": 1, "yes": 1, "Y": 1, "1": 1, 1: 1, "No": 0, "no": 0, "N": 0, "0": 0, 0: 0}
        df["extracurricular_activity"] = df["extracurricular_activity"].map(mapping).fillna(0).astype(int)
            
    return df

def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Prepares train and test splits for both regression and classification.
    Fits StandardScaler ONLY on the training split to avoid data leakage.
    
    Returns a dictionary containing:
        - X_train, X_test (unscaled DataFrames)
        - X_train_scaled, X_test_scaled (scaled numpy arrays)
        - y_train_reg, y_test_reg (numerical marks)
        - y_train_clf, y_test_clf (categorical performance labels)
        - scaler (fitted StandardScaler)
        - feature_names (list of feature names)
    """
    # 1. Clean and encode
    cleaned_df = clean_data(df)
    encoded_df = encode_features(cleaned_df)
    
    # 2. Verify columns
    for col in FEATURE_COLUMNS:
        if col not in encoded_df.columns:
            raise KeyError(f"Expected feature column '{col}' missing from dataset.")
            
    if TARGET_REGRESSION not in encoded_df.columns:
        raise KeyError(f"Regression target '{TARGET_REGRESSION}' missing from dataset.")
    if TARGET_CLASSIFICATION not in encoded_df.columns:
        # Fallback category generation if missing
        encoded_df[TARGET_CLASSIFICATION] = encoded_df[TARGET_REGRESSION].apply(
            lambda m: "Low" if m < 50.0 else ("Average" if m < 75.0 else "High")
        )
        
    X = encoded_df[FEATURE_COLUMNS]
    y_reg = encoded_df[TARGET_REGRESSION]
    y_clf = encoded_df[TARGET_CLASSIFICATION]
    
    # 3. Stratified split based on classification labels to ensure balanced distribution
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf
    )
    
    # 4. Standard Scaling (Fit ONLY on training data to prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_reg": y_train_reg,
        "y_test_reg": y_test_reg,
        "y_train_clf": y_train_clf,
        "y_test_clf": y_test_clf,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS
    }

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_csv = os.path.join(base_dir, "data", "student_performance.csv")
    print(f"[Self-Test] Testing data_preprocessing module with: {test_csv}")
    df_sample = load_data(test_csv)
    data = prepare_train_test_data(df_sample)
    print(f"  Training samples: {len(data['X_train'])}")
    print(f"  Testing samples: {len(data['X_test'])}")
    print(f"  Features: {data['feature_names']}")
    print("  Preprocessing verification SUCCESS!")
