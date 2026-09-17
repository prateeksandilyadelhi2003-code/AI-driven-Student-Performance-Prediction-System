"""
Prediction Inference Module for AI-Driven Student Performance Prediction System.

Loads trained models, validates input data, applies standard scaling,
and generates both numerical grade predictions and categorical risk assessments.
"""

import os
import json
import joblib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

try:
    from .data_preprocessing import FEATURE_COLUMNS
except ImportError:
    from data_preprocessing import FEATURE_COLUMNS

class StudentPerformancePredictor:
    """
    Inference class for predicting student marks and performance categories.
    """
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
            
        self.models_dir = models_dir
        self.reg_model_path = os.path.join(models_dir, "regression_model.pkl")
        self.clf_model_path = os.path.join(models_dir, "classification_model.pkl")
        self.scaler_path = os.path.join(models_dir, "scaler.pkl")
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")
        
        self.reg_model = None
        self.clf_model = None
        self.scaler = None
        self.metadata = {}
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized models, scaler, and metadata."""
        if not os.path.exists(self.reg_model_path):
            raise FileNotFoundError(f"Regression model not found at: {self.reg_model_path}. Please train models first.")
        if not os.path.exists(self.clf_model_path):
            raise FileNotFoundError(f"Classification model not found at: {self.clf_model_path}. Please train models first.")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at: {self.scaler_path}. Please train models first.")
            
        self.reg_model = joblib.load(self.reg_model_path)
        self.clf_model = joblib.load(self.clf_model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)

    @staticmethod
    def validate_inputs(input_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates academic and behavioral inputs against valid real-world ranges.
        """
        # Attendance validation
        att = input_dict.get("attendance")
        if att is None or not (0.0 <= float(att) <= 100.0):
            return False, "Attendance percentage must be between 0 and 100."
            
        # Previous marks validation
        prev = input_dict.get("previous_marks")
        if prev is None or not (0.0 <= float(prev) <= 100.0):
            return False, "Previous exam marks must be between 0 and 100."
            
        # Study hours validation
        hours = input_dict.get("study_hours")
        if hours is None or float(hours) < 0.0 or float(hours) > 24.0:
            return False, "Daily study hours must be between 0 and 24."
            
        # Assignment score validation
        asg = input_dict.get("assignment_score")
        if asg is None or not (0.0 <= float(asg) <= 100.0):
            return False, "Assignment score must be between 0 and 100."
            
        # Internal marks validation
        internal = input_dict.get("internal_marks")
        if internal is None or not (0.0 <= float(internal) <= 100.0):
            return False, "Internal marks must be between 0 and 100."
            
        # Completed assignments validation
        comp = input_dict.get("completed_assignments")
        if comp is None or not (0 <= int(comp) <= 10):
            return False, "Completed assignments count must be an integer between 0 and 10."
            
        # Class participation validation
        part = input_dict.get("class_participation")
        if part is None or not (1 <= int(part) <= 10):
            return False, "Class participation rating must be an integer between 1 and 10."
            
        # Extracurricular activity validation
        extra = input_dict.get("extracurricular_activity")
        if extra not in ["Yes", "No", 1, 0, "1", "0", True, False]:
            return False, "Extracurricular activity must be 'Yes' or 'No'."
            
        return True, "Valid"

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs dual inference: Numerical regression and performance categorization.
        """
        # 1. Validation
        is_valid, err_msg = self.validate_inputs(input_data)
        if not is_valid:
            return {
                "success": False,
                "error": err_msg
            }
            
        # 2. Prepare feature vector
        extra_val = 1 if input_data["extracurricular_activity"] in ["Yes", "yes", 1, "1", True] else 0
        
        feature_dict = {
            "attendance": float(input_data["attendance"]),
            "previous_marks": float(input_data["previous_marks"]),
            "study_hours": float(input_data["study_hours"]),
            "assignment_score": float(input_data["assignment_score"]),
            "internal_marks": float(input_data["internal_marks"]),
            "completed_assignments": int(input_data["completed_assignments"]),
            "class_participation": int(input_data["class_participation"]),
            "extracurricular_activity": extra_val
        }
        
        df_in = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]
        
        # 3. Standard scaling
        scaled_features = self.scaler.transform(df_in)
        
        # 4. Regression prediction
        raw_pred_marks = self.reg_model.predict(scaled_features)[0]
        # Bounded between 0 and 100
        predicted_marks = round(float(np.clip(raw_pred_marks, 0.0, 100.0)), 1)
        
        # 5. Classification prediction
        predicted_category = self.clf_model.predict(scaled_features)[0]
        
        # Probabilities if available
        category_probabilities = {}
        if hasattr(self.clf_model, "predict_proba"):
            classes = list(self.clf_model.classes_)
            probas = self.clf_model.predict_proba(scaled_features)[0]
            category_probabilities = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probas)}
            
        # 6. Actionable interpretation (viva & academic guidance)
        interpretation = self._generate_interpretation(
            predicted_marks,
            predicted_category,
            feature_dict
        )
        
        return {
            "success": True,
            "predicted_marks": predicted_marks,
            "predicted_category": predicted_category,
            "category_probabilities": category_probabilities,
            "interpretation": interpretation,
            "best_reg_model": self.metadata.get("best_regression_model", "Regression Model"),
            "best_clf_model": self.metadata.get("best_classification_model", "Classification Model")
        }

    def _generate_interpretation(
        self,
        predicted_marks: float,
        predicted_category: str,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Creates practical academic guidance without making unsupported claims.
        """
        suggestions = []
        if features["attendance"] < 75.0:
            suggestions.append("Attendance is below the standard university 75% requirement. Raising attendance can significantly improve internal assessment and grasp of core subjects.")
        if features["study_hours"] < 3.0:
            suggestions.append("Daily self-study is currently low. Allocating 1.5 - 2 more hours daily for structured revision is recommended.")
        if features["completed_assignments"] < 7:
            suggestions.append("Completing pending laboratory and classroom assignments on time will strengthen internal scoring.")
        if features["class_participation"] < 5:
            suggestions.append("Engaging more proactively in classroom discussions will enhance conceptual clarity and teacher feedback.")
            
        if not suggestions:
            suggestions.append("Consistent study habits and high attendance observed. Maintain current disciplined routine.")
            
        if predicted_category == "Low":
            summary = "Based on the entered indicators, the student is categorized in the Low performance tier (< 50 marks) and may need targeted academic intervention and mentoring."
            badge_color = "red"
        elif predicted_category == "Average":
            summary = "Based on the entered information, the predicted academic performance is in the Average category (50 - 74 marks). Regular revision and improved attendance can help push into the High bracket."
            badge_color = "orange"
        else:
            summary = "Based on the entered information, the student demonstrates strong performance indicators (>= 75 marks), placing them in the High achievement category."
            badge_color = "green"
            
        return {
            "summary": summary,
            "badge_color": badge_color,
            "suggestions": suggestions
        }

if __name__ == "__main__":
    print("[Self-Test] Running StudentPerformancePredictor inference verification...")
    predictor = StudentPerformancePredictor()
    sample_input = {
        "attendance": 85.0,
        "previous_marks": 72.0,
        "study_hours": 4.0,
        "assignment_score": 80.0,
        "internal_marks": 75.0,
        "completed_assignments": 8,
        "class_participation": 7,
        "extracurricular_activity": "Yes"
    }
    result = predictor.predict(sample_input)
    print(f"  Input: {sample_input}")
    print(f"  Predicted Final Marks: {result['predicted_marks']} / 100")
    print(f"  Predicted Category: {result['predicted_category']}")
    print(f"  Class Probabilities: {result['category_probabilities']}")
    print(f"  Summary: {result['interpretation']['summary']}")
    print("  Prediction inference verification SUCCESS!")
