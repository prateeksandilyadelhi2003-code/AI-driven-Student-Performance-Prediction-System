"""
Model Training Script for AI-Driven Student Performance Prediction System.

Trains and compares:
- Regression: Linear Regression vs Random Forest Regressor (MAE, MSE, RMSE, R2)
- Classification: Logistic Regression vs Decision Tree vs Random Forest Classifier (Accuracy, Precision, Recall, F1)

Saves the best-performing models, the scaler, and a comprehensive evaluation metadata file.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Relative import support
try:
    from .data_preprocessing import load_data, prepare_train_test_data, FEATURE_COLUMNS
except ImportError:
    from data_preprocessing import load_data, prepare_train_test_data, FEATURE_COLUMNS


def train_and_evaluate_models(dataset_path: str, models_dir: str) -> dict:
    """
    Executes end-to-end model training, comparative evaluation, and model serialization.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"--> [Step 1] Loading dataset from: {dataset_path}")
    df = load_data(dataset_path)
    
    print("--> [Step 2] Preprocessing and splitting data (80/20 train/test)...")
    data_dict = prepare_train_test_data(df, test_size=0.20, random_state=42)
    
    X_train_scaled = data_dict["X_train_scaled"]
    X_test_scaled = data_dict["X_test_scaled"]
    y_train_reg = data_dict["y_train_reg"]
    y_test_reg = data_dict["y_test_reg"]
    y_train_clf = data_dict["y_train_clf"]
    y_test_clf = data_dict["y_test_clf"]
    scaler = data_dict["scaler"]
    
    # Save the fitted scaler
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"    Saved fitted Scaler to: {scaler_path}")
    
    # =========================================================================
    # 1. REGRESSION MODELS
    # =========================================================================
    print("\n--> [Step 3] Training & Evaluating Regression Models...")
    
    reg_candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    }
    
    reg_results = {}
    best_reg_name = None
    best_reg_r2 = -float("inf")
    best_reg_model = None
    
    for name, model in reg_candidates.items():
        model.fit(X_train_scaled, y_train_reg)
        y_pred = model.predict(X_test_scaled)
        
        mae = float(mean_absolute_error(y_test_reg, y_pred))
        mse = float(mean_squared_error(y_test_reg, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test_reg, y_pred))
        
        reg_results[name] = {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2_Score": round(r2, 4)
        }
        
        print(f"    {name}: R2 = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        
        if r2 > best_reg_r2:
            best_reg_r2 = r2
            best_reg_name = name
            best_reg_model = model
            
    # Save best regression model
    best_reg_path = os.path.join(models_dir, "regression_model.pkl")
    joblib.dump(best_reg_model, best_reg_path)
    print(f"    Selected Best Regressor: '{best_reg_name}' -> Saved to: {best_reg_path}")
    
    # =========================================================================
    # 2. CLASSIFICATION MODELS
    # =========================================================================
    print("\n--> [Step 4] Training & Evaluating Classification Models...")
    
    clf_candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }
    
    clf_results = {}
    best_clf_name = None
    best_clf_f1 = -float("inf")
    best_clf_model = None
    best_conf_matrix = None
    
    labels = ["Low", "Average", "High"]
    
    for name, model in clf_candidates.items():
        model.fit(X_train_scaled, y_train_clf)
        y_pred = model.predict(X_test_scaled)
        
        acc = float(accuracy_score(y_test_clf, y_pred))
        prec = float(precision_score(y_test_clf, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test_clf, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test_clf, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test_clf, y_pred, labels=labels).tolist()
        
        clf_results[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "Confusion_Matrix": cm
        }
        
        print(f"    {name}: Accuracy = {acc:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, F1 = {f1:.4f}")
        
        if f1 > best_clf_f1:
            best_clf_f1 = f1
            best_clf_name = name
            best_clf_model = model
            best_conf_matrix = cm
            
    # Save best classification model
    best_clf_path = os.path.join(models_dir, "classification_model.pkl")
    joblib.dump(best_clf_model, best_clf_path)
    print(f"    Selected Best Classifier: '{best_clf_name}' -> Saved to: {best_clf_path}")
    
    # Feature Importances (if available)
    feature_importances = {}
    if hasattr(best_reg_model, "feature_importances_"):
        feature_importances["regression"] = dict(zip(FEATURE_COLUMNS, best_reg_model.feature_importances_.round(4).tolist()))
    elif hasattr(best_reg_model, "coef_"):
        feature_importances["regression"] = dict(zip(FEATURE_COLUMNS, np.abs(best_reg_model.coef_).round(4).tolist()))
        
    if hasattr(best_clf_model, "feature_importances_"):
        feature_importances["classification"] = dict(zip(FEATURE_COLUMNS, best_clf_model.feature_importances_.round(4).tolist()))
        
    # Assemble metadata
    metadata = {
        "best_regression_model": best_reg_name,
        "regression_results": reg_results,
        "best_classification_model": best_clf_name,
        "classification_results": clf_results,
        "labels": labels,
        "feature_names": FEATURE_COLUMNS,
        "feature_importances": feature_importances,
        "test_sample_count": len(y_test_reg),
        "train_sample_count": len(y_train_reg)
    }
    
    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\n--> [Step 5] Saved model metadata & evaluation summary to: {meta_path}")
    
    return metadata

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data", "student_performance.csv")
    models_dir = os.path.join(base_dir, "models")
    train_and_evaluate_models(dataset_path, models_dir)

if __name__ == "__main__":
    main()
