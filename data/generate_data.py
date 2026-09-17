"""
Dataset Generation Script for AI-Driven Student Performance Prediction System.

This script creates a realistic synthetic academic dataset (1,200 records)
modeling genuine student academic distributions, behavioral factors, and final grades.

NOTE: This is synthetic benchmark data designed for academic demonstration and reproducible
model evaluation in a BTech CSE final year project.
"""

import os
import numpy as np
import pandas as pd

def generate_student_dataset(num_samples: int = 1200, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic student records with academic and behavioral attributes.
    
    Parameters:
        num_samples (int): Number of student records to generate.
        random_seed (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: Generated dataset.
    """
    np.random.seed(random_seed)
    
    # 1. Student IDs
    student_ids = [f"STD{1000 + i}" for i in range(1, num_samples + 1)]
    
    # 2. Independent Academic & Behavioral Features
    # Attendance percentage (35% to 100%)
    attendance = np.clip(np.random.normal(loc=76.0, scale=12.5, size=num_samples), 35.0, 100.0)
    
    # Previous Exam Marks (30 to 98)
    previous_marks = np.clip(np.random.normal(loc=68.0, scale=14.0, size=num_samples), 30.0, 99.0)
    
    # Daily Study Hours (1.0 to 10.0 hours/day)
    study_hours = np.clip(np.random.normal(loc=4.5, scale=1.8, size=num_samples), 1.0, 10.0)
    
    # Assignment Score (25 to 100)
    assignment_score = np.clip(
        0.55 * previous_marks + 0.35 * (attendance * 0.8 + 20) + np.random.normal(0, 7.5, num_samples),
        25.0, 100.0
    )
    
    # Internal Assessment Marks (25 to 100)
    internal_marks = np.clip(
        0.50 * previous_marks + 0.30 * (study_hours * 8 + 20) + 0.20 * assignment_score + np.random.normal(0, 6.0, num_samples),
        25.0, 100.0
    )
    
    # Completed Assignments (0 to 10)
    completed_assignments = np.clip(
        np.round((assignment_score / 10.0) + np.random.normal(0, 1.0, num_samples)),
        0, 10
    ).astype(int)
    
    # Class Participation (1 to 10 scale)
    class_participation = np.clip(
        np.round((attendance / 12.0) + (study_hours / 3.0) + np.random.normal(0, 1.2, num_samples)),
        1, 10
    ).astype(int)
    
    # Extracurricular Activity Participation (Yes / No)
    # Balanced distribution with slight tendency for active students
    extracurricular_prob = np.clip(0.3 + (class_participation / 30.0), 0.1, 0.9)
    extracurricular_activity = [
        "Yes" if np.random.rand() < prob else "No" for prob in extracurricular_prob
    ]
    
    # Numerical bonus for extracurricular engagement (leadership/discipline factor)
    extra_boost = np.array([2.5 if act == "Yes" else 0.0 for act in extracurricular_activity])
    
    # 3. Target: Final Marks (Continuous 0 - 100)
    # Realistic academic weighted formula + inherent variance/noise
    raw_final_marks = (
        0.28 * previous_marks +
        0.24 * internal_marks +
        0.18 * assignment_score +
        0.14 * attendance +
        0.08 * (study_hours * 8.5) +
        0.05 * (completed_assignments * 8.0) +
        0.03 * (class_participation * 8.0) +
        extra_boost +
        np.random.normal(0, 4.2, num_samples)
    )
    
    final_marks = np.clip(np.round(raw_final_marks, 1), 20.0, 100.0)
    
    # 4. Target: Performance Category (Classification)
    # Defined ranges:
    # Low: < 50
    # Average: 50 <= marks < 75
    # High: marks >= 75
    performance_category = []
    for mark in final_marks:
        if mark < 50.0:
            performance_category.append("Low")
        elif mark < 75.0:
            performance_category.append("Average")
        else:
            performance_category.append("High")
            
    # Assemble DataFrame
    df = pd.DataFrame({
        "student_id": student_ids,
        "attendance": np.round(attendance, 1),
        "previous_marks": np.round(previous_marks, 1),
        "study_hours": np.round(study_hours, 1),
        "assignment_score": np.round(assignment_score, 1),
        "internal_marks": np.round(internal_marks, 1),
        "completed_assignments": completed_assignments,
        "class_participation": class_participation,
        "extracurricular_activity": extracurricular_activity,
        "final_marks": final_marks,
        "performance_category": performance_category
    })
    
    return df

def main():
    # Resolve directory paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, "student_performance.csv")
    print(f"Generating realistic student dataset (1,200 samples)...")
    df = generate_student_dataset(num_samples=1200, random_seed=42)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at: {output_path}")
    print("\nDataset Summary:")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nCategory Distribution:")
    print(df["performance_category"].value_counts())
    print("\nSample Preview:")
    print(df.head(3))

if __name__ == "__main__":
    main()
