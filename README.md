# AI-Driven Student Performance Prediction System

🎓 **An End-to-End Machine Learning Web Application for Early Academic Risk Identification and Final Grade Forecasting**

*Designed for B.Tech Computer Science & Engineering (CSE) Capstone / Academic Demonstration.*

---

## 1. Project Overview

The **AI-Driven Student Performance Prediction System** is a full-fledged machine learning platform that evaluates key academic history, behavioral discipline, and classroom engagement parameters to forecast a student's final examination marks and categorize their academic risk band. 

Unlike black-box or non-operational prototypes, this project delivers a **dual-model pipeline**:
1. **Regression Task:** Forecasts expected final marks as a continuous numerical score (0 to 100).
2. **Classification Task:** Categorizes students into three actionable academic tiers: **Low (< 50)**, **Average (50 - 74)**, and **High (≥ 75)**.

The system features an interactive **Streamlit** dashboard designed for educators, academic counselors, and students.

---

## 2. Problem Statement

In contemporary higher education institutions, terminal examination failures and academic probation frequently occur because academic distress is recognized too late in the semester. Traditional assessment systems evaluate performance post-mortem—after final grades are submitted.

There is a critical need for an automated, data-driven early warning system capable of analyzing intermediate indicators (such as internal assessment marks, assignment submission cadence, daily self-study hours, and classroom attendance) to identify students who require targeted remedial tutoring and intervention before it is too late.

---

## 3. Objectives

- **Develop an early risk detection mechanism** to identify at-risk students before terminal examinations.
- **Implement dual machine learning methodologies:** continuous score regression and discrete tier classification.
- **Perform leak-free data preprocessing** with strict train-only standardization (`StandardScaler`) to ensure genuine academic generalization.
- **Evaluate and compare multiple ML models** (Linear Regression vs. Random Forest Regressor; Logistic Regression vs. Decision Tree vs. Random Forest Classifier) on held-out test data.
- **Provide an interactive, responsive web application** using Streamlit with exploratory data analysis (EDA) charts, real-time prediction, and actionable guidance.

---

## 4. Key Features

- **Interactive Multi-Page Streamlit UI:**
  - 🏠 **Home Dashboard:** High-level KPIs, institutional objectives, and risk tier definitions.
  - 🎯 **Performance Predictor:** Form with sliders and controls for real-time inference with dual outputs (score + category) and tailored academic suggestions.
  - 📊 **Exploratory Data Analysis (EDA):** Interactive charts for feature distributions, regression trendlines, and correlation heatmaps.
  - 📈 **Model Performance & Comparison:** Comprehensive evaluation tables (MAE, MSE, RMSE, R², Accuracy, Precision, Recall, F1) calculated on held-out test data, plus the confusion matrix.
  - ℹ️ **Viva & Architecture Guide:** Frequently asked questions for external evaluation and project viva.
- **Offline Model Serialization:** Models and scalers are pre-trained and saved (`.pkl`) so that the application loads instantly without retraining.
- **Robust Input Validation:** Range checks on all input fields (e.g. attendance 0-100%, study hours 0-24 hrs) prevent invalid predictions and runtime crashes.

---

## 5. Technologies Used

- **Programming Language:** Python 3.10+
- **Data Manipulation & Processing:** Pandas, NumPy
- **Machine Learning Library:** Scikit-learn
- **Model Serialization:** Joblib
- **Visualization:** Matplotlib, Seaborn
- **Web Application Framework:** Streamlit

---

## 6. System Architecture

```
[ Student Academic & Behavioral Factors ]
 (Attendance, Study Hours, Prev Marks, Assignments, Internals, Participation, Extracurricular)
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│             Data Preprocessing Pipeline                │
│  - Missing Value Imputation                            │
│  - Categorical Encoding (Extracurricular: Yes/No -> 1/0)│
│  - 80/20 Train-Test Split (random_state=42)           │
│  - StandardScaler (Fitted ONLY on training data)       │
└────────────────────────────┬───────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│    Regression Engine    │       │  Classification Engine  │
│  - Linear Regression    │       │  - Logistic Regression  │
│  - Random Forest Reg.   │       │  - Decision Tree        │
│                         │       │  - Random Forest Clf    │
└───────────┬─────────────┘       └───────────┬─────────────┘
            │                                 │
            ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│   Predicted Final Mark  │       │  Academic Risk Tier     │
│       (e.g., 78.4 / 100)│       │  (Low / Average / High) │
└───────────┬─────────────┘       └───────────┬─────────────┘
            └────────────────┬────────────────┘
                             ▼
┌────────────────────────────────────────────────────────┐
│              Streamlit Web Application                 │
│   - Real-time Interactive Form                         │
│   - Categorical Confidence Percentages                 │
│   - Actionable Academic Guidance / Suggestions         │
│   - Exploratory Data Visualizations & Metrics Table    │
└────────────────────────────────────────────────────────┘
```

---

## 7. Dataset Description

The dataset (`data/student_performance.csv`) consists of 1,200 student records reflecting realistic distributions observed in university engineering programs:

| Feature Name | Type | Valid Range | Description |
| :--- | :--- | :--- | :--- |
| `student_id` | String | STD1001 - STD2200 | Unique student identifier |
| `attendance` | Float | 35.0 - 100.0 | Overall classroom attendance percentage (%) |
| `previous_marks` | Float | 30.0 - 100.0 | Score in previous semester / prerequisite exam (/100) |
| `study_hours` | Float | 1.0 - 10.0 | Daily self-study hours outside classroom lectures |
| `assignment_score` | Float | 25.0 - 100.0 | Average evaluation score for coursework assignments (/100) |
| `internal_marks` | Float | 25.0 - 100.0 | Marks obtained in mid-term internal examinations (/100) |
| `completed_assignments`| Integer | 0 - 10 | Total number of mandatory assignments submitted |
| `class_participation` | Integer | 1 - 10 | Instructor rating of classroom discussion and engagement |
| `extracurricular_activity`| Categorical | Yes / No | Participation in campus societies, sports, or hackathons |
| `final_marks` | Float | 20.0 - 100.0 | **Target (Regression):** Continuous final examination mark |
| `performance_category` | Categorical | Low, Average, High | **Target (Classification):** Risk tier |

### Performance Categories Defined:
- **Low (< 50.0):** Students at academic risk requiring remedial sessions.
- **Average (50.0 - 74.9):** Satisfactory academic progress.
- **High (≥ 75.0):** High academic standing / distinction.

---

## 8. Machine Learning Methodology & Results

### 8.1 Data Preprocessing & Leak Prevention
To guarantee that model evaluation is authentic:
1. An **80% training / 20% testing** split is created using stratified sampling on `performance_category`.
2. The `StandardScaler` is **fitted strictly on the training set** (`X_train`), and then used to transform both `X_train` and `X_test`. Live user inputs in `app.py` use this exact saved scaler.

### 8.2 Actual Evaluation Results (Calculated on 240 Test Samples)

#### Regression Models:
| Algorithm | MAE | MSE | RMSE | R² Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **3.2817** | **16.5145** | **4.0638** | **0.8255** | 🏆 **Best Model** |
| Random Forest Regressor | 3.3887 | 17.4614 | 4.1787 | 0.8155 | Evaluated |

*Why Linear Regression Won:* Academic scoring formulas exhibit strongly additive, linear properties. Linear Regression captured the underlying structure without overfitting.

#### Classification Models:
| Algorithm | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **90.00%** | **0.8987** | **0.9000** | **0.8943** | 🏆 **Best Model** |
| Random Forest Classifier | 89.17% | 0.8964 | 0.8917 | 0.8811 | Evaluated |
| Decision Tree Classifier | 83.75% | 0.8296 | 0.8375 | 0.8256 | Evaluated |

---

## 9. Project Structure

```
student-performance-prediction/
│
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Complete project manual & viva guide
├── PROJECT_DOCUMENTATION.md    # Comprehensive academic report
├── test_prediction_pipeline.py # End-to-end verification test suite
│
├── data/
│   ├── student_performance.csv # Generated realistic student dataset (1,200 records)
│   └── generate_data.py        # Dataset generation script
│
├── models/
│   ├── regression_model.pkl    # Serialized best regression model
│   ├── classification_model.pkl# Serialized best classification model
│   ├── scaler.pkl              # Serialized StandardScaler
│   └── model_metadata.json     # Stored test metrics, parameters & results
│
├── notebooks/
│   └── model_training.ipynb    # Jupyter notebook for viva & experimentation
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Leak-free cleaning, encoding & scaling
│   ├── train_models.py         # Comparative training script
│   ├── prediction.py           # Inference engine with validation & guidance
│   └── visualization.py        # Matplotlib & Seaborn plotting functions
│
└── assets/
    └── screenshots/            # Directory for project presentation screenshots
```

---

## 10. Installation & Setup Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- VS Code or terminal environment.

### Step 1: Navigate to the Project Folder
```bash
cd C:\Users\Prateek\.gemini\antigravity\scratch\student-performance-prediction
```

### Step 2: (Optional) Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Windows Command Prompt:
.\venv\Scripts\activate.bat
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 11. How to Run the Project

### 1. (Optional) Re-generate Dataset & Re-train Models
The repository includes pre-generated data and serialized models. If you wish to retrain:
```bash
# Generate synthetic dataset (1,200 records)
python data/generate_data.py

# Train models, evaluate on held-out test data, and serialize
python src/train_models.py
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

The application will launch locally at `http://localhost:8501`.

---

## 12. How to Use the Streamlit Application

1. **Navigate to the Predictor:** From the sidebar, click on **"🎯 Performance Predictor"**.
2. **Adjust Student Attributes:**
   - Set **Previous Marks** and **Internal Marks** using the sliders.
   - Adjust **Attendance Percentage** (e.g. 82%).
   - Select **Daily Study Hours** and **Completed Assignments**.
   - Pick the **Class Participation** rating and **Extracurricular** status.
3. **Click "🔮 Predict Student Performance":**
   - The predicted final mark (/100) and performance tier (Low / Average / High) will appear immediately.
   - Read the confidence breakdown and actionable guidance suggestions.
4. **Explore EDA & Model Metrics:**
   - Switch to **"📊 Data Visualizations (EDA)"** to review feature correlation heatmaps and trendlines.
   - Switch to **"📈 Model Performance & Comparison"** to review the actual test set metric tables and confusion matrix.

---

## 13. Limitations

- The dataset is a calibrated synthetic academic dataset rather than institutional records protected under privacy laws (FERPA / GDPR).
- External unmeasured factors such as health crises or socioeconomic hardship are not captured in the current feature set.
- Predictions represent probabilistic estimations based on empirical trends and should not be treated as deterministic guarantees.

---

## 14. Future Enhancements

- **Direct ERP / LMS Integration:** Connecting to Moodle or Canvas APIs via OAuth to fetch student submission logs automatically.
- **Explainable AI (XAI):** Integrating SHAP (SHapley Additive exPlanations) or LIME for individual student score attribution.
- **Automated Early Warning Emails:** Automatically emailing academic advisors when a student's risk category turns "Low".
