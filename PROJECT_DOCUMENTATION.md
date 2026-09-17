# PROJECT DOCUMENTATION
# AI-Driven Student Performance Prediction System

**Degree:** Bachelor of Technology (B.Tech) in Computer Science & Engineering  
**Subject:** Major / Capstone Project  
**Repository:** `student-performance-prediction`  
**Deployment Environment:** Python 3 / Streamlit Web Interface  

---

## TABLE OF CONTENTS
1. [Abstract](#1-abstract)
2. [Problem Statement](#2-problem-statement)
3. [Existing System vs. Proposed System](#3-existing-system-vs-proposed-system)
4. [Project Objectives](#4-project-objectives)
5. [System Architecture](#5-system-architecture)
6. [Dataset Description](#6-dataset-description)
7. [Data Preprocessing Pipeline](#7-data-preprocessing-pipeline)
8. [Exploratory Data Analysis (EDA)](#8-exploratory-data-analysis-eda)
9. [Machine Learning Methodology](#9-machine-learning-methodology)
   - 9.1 Regression Modeling (Continuous Marks)
   - 9.2 Classification Modeling (Risk Tiers)
10. [Model Training, Testing, & Evaluation](#10-model-training-testing--evaluation)
11. [Inference Engine & Real-Time Prediction](#11-inference-engine--real-time-prediction)
12. [User Interface & Dashboard Modules](#12-user-interface--dashboard-modules)
13. [Limitations](#13-limitations)
14. [Future Scope](#14-future-scope)
15. [Conclusion](#15-conclusion)

---

## 1. ABSTRACT

Student academic failure and dropout rates in undergraduate engineering programs pose significant institutional challenges. Traditional evaluation methodologies assess academic output post-mortem—after final end-semester examinations—by which point remedial interventions are ineffective. 

This project presents an **AI-Driven Student Performance Prediction System** that analyzes academic history (previous examination marks, internal assessments, assignment scores) and behavioral discipline indicators (attendance percentage, daily study hours, assignment completion rate, class participation, and extracurricular activities). 

The proposed architecture adopts a **dual machine learning framework**:
1. **Regression Analysis:** Continuous numerical forecasting of final marks using Linear Regression and Random Forest Regressors ($R^2 = 0.8255$).
2. **Classification Analysis:** Academic risk tier categorization into **Low** (< 50 marks), **Average** (50–74 marks), and **High** (≥ 75 marks) using Logistic Regression, Decision Trees, and Random Forest Classifiers (Accuracy = 90.00%).

The trained pipelines are integrated into an interactive, lightweight **Streamlit** web application offering instantaneous inference, actionable feedback recommendations, exploratory data visualizations, and transparent model comparison matrices suitable for university deployment and academic viva evaluation.

---

## 2. PROBLEM STATEMENT

In conventional university environments:
- **Lagging Indicator Dilemma:** Teachers only discover a student's learning deficit after semester examination grades are published.
- **Manual Advisory Bottlenecks:** Faculty advisors manage hundreds of students, making it impractical to manually cross-reference attendance percentages, weekly assignment scores, and mid-term assessments for each student.
- **Subjective Counseling:** Advice given to students often lacks empirical grounding regarding how many additional study hours or attendance classes are needed to move into a secure passing grade.

Consequently, there is an urgent need for an automated, reproducible machine learning system that ingests readily available semester indicators and generates predictive, actionable insights.

---

## 3. EXISTING SYSTEM VS. PROPOSED SYSTEM

| Parameter | Existing System | Proposed AI-Driven System |
| :--- | :--- | :--- |
| **Evaluation Timing** | Reactive (after exams conclude) | Proactive (mid-semester early warning) |
| **Analysis Type** | Rule-based or manual faculty review | Multi-algorithm Machine Learning (Regression + Classification) |
| **Scope of Output** | Pass / Fail outcome | Numerical final score prediction + 3-tier risk classification |
| **Student Guidance** | Generic verbal encouragement | Targeted advice based on specific deficit features (e.g. attendance, study hours) |
| **Transparency** | Black-box or ad-hoc grading | Open evaluation metrics (MAE, RMSE, R², F1-score, Confusion Matrix) |
| **User Experience** | Static spreadsheets or paper registers | Modern, interactive Streamlit web dashboard |

---

## 4. PROJECT OBJECTIVES

1. **Design a clean academic data schema:** Incorporate academic performance indicators and behavioral factors with calibrated realistic distributions.
2. **Eliminate Data Leakage:** Ensure data cleaning, categorical encoding, and feature scaling (`StandardScaler`) are fitted strictly on training data splits.
3. **Implement Comparative Regression:** Evaluate Linear Regression vs. Random Forest Regressor using standard statistical metrics (MAE, MSE, RMSE, $R^2$).
4. **Implement Comparative Classification:** Evaluate Logistic Regression, Decision Trees, and Random Forest Classifiers using Accuracy, Precision, Recall, and F1-score.
5. **Build an Interactive Web Interface:** Develop an intuitive Streamlit interface allowing users to adjust student indicators and view instant predictions with confidence intervals.
6. **Provide Viva-Ready Academic Artifacts:** Include step-by-step Jupyter notebooks, serialized models (`.pkl`), structured metadata (`.json`), and comprehensive documentation.

---

## 5. SYSTEM ARCHITECTURE

```
                      +-----------------------------+
                      |   Student Input Data        |
                      |   (Academic & Behavioral)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Data Preprocessing Pipeline |
                      | - Missing Value Imputation  |
                      | - Categorical Mapping       |
                      | - 80/20 Train-Test Split    |
                      | - StandardScaler Fitting    |
                      +--------------+--------------+
                                     |
                +--------------------+--------------------+
                |                                         |
                v                                         v
+-------------------------------+         +-------------------------------+
|       Regression Task         |         |     Classification Task       |
| Models:                       |         | Models:                       |
| - Linear Regression           |         | - Logistic Regression         |
| - Random Forest Regressor     |         | - Decision Tree Classifier    |
|                               |         | - Random Forest Classifier    |
| Metric: MAE, RMSE, R² Score   |         | Metric: Acc, Prec, Rec, F1    |
+---------------+---------------+         +---------------+---------------+
                |                                         |
                +--------------------+--------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Model Serialization (.pkl)  |
                      | & Metadata Generation (.json|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Streamlit Web Application  |
                      |  - Performance Predictor    |
                      |  - Exploratory Visuals      |
                      |  - Model Benchmark Tables   |
                      |  - Viva Guide Module        |
                      +-----------------------------+
```

---

## 6. DATASET DESCRIPTION

The system utilizes a 1,200-record dataset (`data/student_performance.csv`) generated to mirror real-world university engineering student cohorts:

| Attribute | Data Type | Domain | Realistic Meaning |
| :--- | :--- | :--- | :--- |
| `student_id` | String | STD1001 – STD2200 | Unique primary identifier |
| `attendance` | Continuous | 35.0 – 100.0% | Classroom session presence rate |
| `previous_marks` | Continuous | 30.0 – 100.0 | Past semester grade percentage |
| `study_hours` | Continuous | 1.0 – 10.0 hrs | Daily hours dedicated to independent study |
| `assignment_score`| Continuous | 25.0 – 100.0 | Cumulative assignment evaluation marks |
| `internal_marks` | Continuous | 25.0 – 100.0 | Mid-semester written test score |
| `completed_assignments` | Discrete | 0 – 10 | Completed lab and tutorial problem sets |
| `class_participation` | Discrete | 1 – 10 | Instructor rating of discussion & engagement |
| `extracurricular_activity` | Binary | Yes / No | Engagement in student clubs, sports, or hackathons |
| `final_marks` | Continuous | 20.0 – 100.0 | **Regression Target:** Semester terminal score |
| `performance_category` | Categorical | Low, Average, High | **Classification Target:** Risk level |

### Performance Category Ranges:
- **Low Category:** Marks < 50.0 (Failure threshold / academic intervention mandatory).
- **Average Category:** 50.0 ≤ Marks < 75.0 (Standard competence / satisfactory passing grade).
- **High Category:** Marks ≥ 75.0 (Distinction / exemplary academic standing).

---

## 7. DATA PREPROCESSING PIPELINE

The module `src/data_preprocessing.py` enforces rigorous machine learning standards:

1. **Integrity & Null Checking:** Replaces missing continuous features with median values and categorical features with modes; eliminates duplicate rows.
2. **Categorical Feature Encoding:** Converts `extracurricular_activity` binary strings (`Yes`/`No`) into numeric indicators (`1`/`0`).
3. **Stratified Train-Test Splitting:** An 80/20 train/test split is applied using `random_state=42` and `stratify=y_clf`, ensuring the test set contains identical proportion of Low, Average, and High records.
4. **Data Leakage Elimination:** The `StandardScaler` executes `fit_transform` solely on the 80% training set ($N=960$). The test partition ($N=240$) and subsequent real-time inference samples are strictly normalized via `transform` using parameters ($\mu, \sigma$) calculated on the training partition.

---

## 8. EXPLORATORY DATA ANALYSIS (EDA)

The visualization module (`src/visualization.py`) renders key statistical distributions:

1. **Attendance vs. Final Marks:** Shows a clear positive trend with lower variance among students with attendance above 75%.
2. **Study Hours vs. Final Marks:** Confirms diminishing returns beyond 7–8 hours/day, validating linear-to-moderate non-linear dynamics.
3. **Previous Marks vs. Final Marks:** Demonstrates the highest single-feature correlation ($r \approx 0.78$), serving as a strong baseline predictor.
4. **Distribution of Final Marks:** Bell-shaped Gaussian curve centered around $\mu \approx 67.4$ with standard deviation $\sigma \approx 10.8$.
5. **Correlation Heatmap:** Pearson correlation coefficients verify strong positive associations between internal marks, assignments, and final scores.

---

## 9. MACHINE LEARNING METHODOLOGY

### 9.1 Regression Modeling (Numerical Marks)
- **Linear Regression:** Fits an ordinary least-squares hyper-plane:
  $$y = \beta_0 + \sum_{i=1}^{p} \beta_i X_i + \epsilon$$
  Provides direct interpretability of feature coefficients.
- **Random Forest Regressor:** An ensemble of 100 decorrelated decision trees (`n_estimators=100`, `max_depth=8`) that averages individual tree predictions to minimize variance.

### 9.2 Classification Modeling (Performance Bands)
- **Logistic Regression:** Multinomial cross-entropy model applying softmax over standardized features.
- **Decision Tree Classifier:** Top-down recursive partitioning based on Gini Impurity with `max_depth=5` to prevent branch overfitting.
- **Random Forest Classifier:** Bagging ensemble aggregating 100 classification trees with random feature sub-spacing.

---

## 10. MODEL TRAINING, TESTING, & EVALUATION

Models were evaluated on a strictly held-out test split of **240 samples** (20%).

### 10.1 Regression Evaluation Results
| Model | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | **3.2817** | **16.5145** | **4.0638** | **0.8255** |
| Random Forest Regressor | 3.3887 | 17.4614 | 4.1787 | 0.8155 |

**Outcome:** Linear Regression achieved the lowest prediction error ($RMSE = 4.06$) and highest coefficient of determination ($R^2 = 82.55\%$). Because academic exam scores follow additive institutional guidelines, the linear formulation avoids tree-boundary quantization artifacts.

### 10.2 Classification Evaluation Results
| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **90.00%** | **0.8987** | **0.9000** | **0.8943** |
| Random Forest Classifier | 89.17% | 0.8964 | 0.8917 | 0.8811 |
| Decision Tree Classifier | 83.75% | 0.8296 | 0.8375 | 0.8256 |

**Outcome:** Logistic Regression demonstrated the best generalization with **90.00% accuracy** and an **F1-score of 0.8943**.

---

## 11. INFERENCE ENGINE & REAL-TIME PREDICTION

The `StudentPerformancePredictor` class (`src/prediction.py`):
1. Loads pre-compiled `.pkl` files once at application start.
2. Validates inputs against strict domain constraints (e.g. attendance $\in [0, 100]$, study hours $\in [0, 24]$).
3. Executes standardized vector transformation.
4. Concurrently outputs:
   - Numerical predicted mark (e.g., `75.2 / 100`).
   - Categorical risk tier (e.g., `High`).
   - Class probabilities (e.g., High: 54.6%, Average: 45.4%, Low: 0.0%).
   - Actionable heuristic feedback advising on study hour adjustments and attendance compliance.

---

## 12. USER INTERFACE & DASHBOARD MODULES

The Streamlit interface (`app.py`) provides five structured tabs:
1. **🏠 Home Dashboard:** High-level institutional metrics, summary KPIs, and academic tier definitions.
2. **🎯 Performance Predictor:** Interactive form controls (sliders, drop-downs) generating instant dual predictions and actionable advice.
3. **📊 Data Visualizations (EDA):** Embedded Seaborn figures illustrating relationships and correlations with raw CSV download capabilities.
4. **📈 Model Performance & Comparison:** Transparent tabular display of test set metrics and confusion matrices.
5. **ℹ️ Viva & Architecture Guide:** Curated technical questions and answers detailing algorithmic rationale and design decisions.

---

## 13. LIMITATIONS

1. **Synthetic Data Foundation:** In accordance with institutional student data privacy safeguards, a calibrated synthetic benchmark is utilized rather than live records.
2. **External Confounders:** Unmeasured non-academic variables (e.g., personal health crises, financial stress) are not represented in the input vector.
3. **Static Weights:** Real courses may weigh laboratory work or theory components differently across departments.

---

## 14. FUTURE SCOPE

1. **LMS API Webhooks:** Integration with Canvas/Moodle to pull attendance and quiz submissions automatically.
2. **Explainable AI (SHAP):** Integration of SHAP value force plots to visualize individual feature impact for every student.
3. **Automated Notification Dispatcher:** Automated email alerts to faculty counselors when a student enters the "Low" performance category.

---

## 15. CONCLUSION

The **AI-Driven Student Performance Prediction System** provides a complete, working, and understandable machine learning solution for academic performance estimation. By uniting continuous regression forecasting with discrete risk classification, educators gain both granular grade projections and clear categorization for early intervention. 

The modular architecture, absence of data leakage, local reproducibility, and professional web dashboard fulfill all requirements for a B.Tech Computer Science & Engineering capstone project.
