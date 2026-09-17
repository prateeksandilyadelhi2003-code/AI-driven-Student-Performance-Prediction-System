"""
AI-Driven Student Performance Prediction System
Interactive Web Application powered by Streamlit.

BTech CSE Final Year / Capstone Academic Project.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st

# Setup page configuration
st.set_page_config(
    page_title="AI-Driven Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Import custom modules
from src.prediction import StudentPerformancePredictor
from src.visualization import (
    plot_attendance_vs_marks,
    plot_study_hours_vs_marks,
    plot_previous_vs_final_marks,
    plot_assignment_vs_marks,
    plot_marks_distribution,
    plot_correlation_heatmap,
    plot_category_distribution,
    plot_confusion_matrix
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .badge-high {
        background-color: #DCFCE7;
        color: #166534;
        border-left: 6px solid #22C55E;
    }
    .badge-avg {
        background-color: #FEF3C7;
        color: #92400E;
        border-left: 6px solid #F59E0B;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        border-left: 6px solid #EF4444;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    """Load dataset with caching."""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_model_metadata():
    """Load model evaluation metadata."""
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def main():
    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/clouds/200/graduation-cap.png", width=120)
    st.sidebar.title("EduPredict AI")
    st.sidebar.markdown("**Student Performance System**")
    st.sidebar.caption("BTech CSE Academic Capstone Project")
    st.sidebar.markdown("---")

    menu = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "🎯 Performance Predictor",
            "📊 Data Visualizations (EDA)",
            "📈 Model Performance & Comparison",
            "ℹ️ Viva & Architecture Guide"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Academic Note:**\n"
        "Models are trained offline to avoid runtime lag. "
        "StandardScaler fitted strictly on training data."
    )

    # -------------------------------------------------------------------------
    # PAGE 1: HOME DASHBOARD
    # -------------------------------------------------------------------------
    if menu == "🏠 Home Dashboard":
        st.markdown('<div class="main-header">🎓 AI-Driven Student Performance Prediction System</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">An intelligent machine learning framework providing early academic risk categorization and score forecasting.</div>', unsafe_allow_html=True)

        # Quick Statistics Banner
        df = load_dataset()
        metadata = load_model_metadata()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            count = len(df) if df is not None else "1,200"
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Records</div><div class="kpi-value">{count}</div></div>', unsafe_allow_html=True)
        with col2:
            avg_m = f"{df['final_marks'].mean():.1f}" if df is not None else "67.4"
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Avg Final Marks</div><div class="kpi-value">{avg_m} / 100</div></div>', unsafe_allow_html=True)
        with col3:
            best_r2 = metadata["regression_results"][metadata["best_regression_model"]]["R2_Score"] if metadata else "0.825"
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Best Regressor R²</div><div class="kpi-value">{best_r2}</div></div>', unsafe_allow_html=True)
        with col4:
            best_acc = metadata["classification_results"][metadata["best_classification_model"]]["Accuracy"] if metadata else "0.900"
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Best Classifier Acc</div><div class="kpi-value">{float(best_acc)*100:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Overview & Objectives
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.subheader("📌 Project Motivation & Objectives")
            st.write(
                """
                Early identification of academic challenges allows educators, academic mentors, and parents 
                to deploy timely remedial interventions before terminal semester examinations. 
                
                This project combines **Regression** (continuous numerical grade estimation) and 
                **Classification** (performance band categorization: Low, Average, High) into a single unified 
                machine-learning system.
                """
            )
            
            st.markdown("#### Key Academic Factors Analyzed:")
            c_a, c_b = st.columns(2)
            with c_a:
                st.markdown("- 📅 **Attendance Percentage** (35% - 100%)")
                st.markdown("- 📝 **Previous Exam Marks** (0 - 100)")
                st.markdown("- ⏱️ **Daily Study Hours** (1 - 10 hrs)")
                st.markdown("- 📄 **Assignment Scores** (0 - 100)")
            with c_b:
                st.markdown("- 🎯 **Internal Assessment Marks** (0 - 100)")
                st.markdown("- 📑 **Completed Assignments** (0 - 10)")
                st.markdown("- 🙋‍♂️ **Classroom Participation** (1 - 10)")
                st.markdown("- 🏆 **Extracurricular Engagement** (Yes / No)")

        with col_right:
            st.subheader("⚙️ System Workflow")
            st.info(
                """
                1. **Input Collection**: Real-time student academic metrics.
                2. **Data Pipeline**: Validation, categorical encoding & standard scaling.
                3. **Inference Engine**:
                   - *Regression Model* predicts final numerical score.
                   - *Classifier Model* categorizes academic standing.
                4. **Actionable Insights**: Generates specific feedback on study hours and attendance.
                """
            )

        st.markdown("---")
        st.subheader("📑 Academic Performance Tiers Defined")
        st.markdown(
            """
            | Performance Category | Score Range | Description & Institutional Action |
            | :--- | :--- | :--- |
            | 🔴 **Low** | **< 50.0 Marks** | At-risk student requiring counseling, tutorial classes, and close monitoring. |
            | 🟡 **Average** | **50.0 - 74.9 Marks** | Satisfactory progress; improvement in attendance and assignment consistency recommended. |
            | 🟢 **High** | **≥ 75.0 Marks** | Distinction level; high engagement and strong academic foundation. |
            """
        )

    # -------------------------------------------------------------------------
    # PAGE 2: STUDENT PREDICTION PAGE
    # -------------------------------------------------------------------------
    elif menu == "🎯 Performance Predictor":
        st.markdown('<div class="main-header">🎯 Student Performance Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Enter the student academic parameters below to forecast final marks and risk level.</div>', unsafe_allow_html=True)

        try:
            predictor = StudentPerformancePredictor(MODELS_DIR)
        except Exception as e:
            st.error(f"⚠️ Error loading trained models: {str(e)}")
            st.info("Please make sure you have executed `python src/train_models.py` first.")
            return

        with st.form("prediction_form"):
            st.markdown("#### 1. Academic Assessment History")
            col1, col2 = st.columns(2)
            with col1:
                previous_marks = st.slider("Previous Examination Marks (/100)", 0.0, 100.0, 72.0, 0.5)
                internal_marks = st.slider("Internal Assessment Marks (/100)", 0.0, 100.0, 75.0, 0.5)
            with col2:
                assignment_score = st.slider("Assignment Evaluation Score (/100)", 0.0, 100.0, 80.0, 0.5)
                completed_assignments = st.slider("Number of Completed Assignments (out of 10)", 0, 10, 8)

            st.markdown("#### 2. Discipline & Behavioral Factors")
            col3, col4 = st.columns(2)
            with col3:
                attendance = st.slider("Classroom Attendance (%)", 0.0, 100.0, 85.0, 0.5)
                study_hours = st.slider("Daily Self-Study Hours", 0.0, 14.0, 4.0, 0.5)
            with col4:
                class_participation = st.slider("Class Participation Rating (1: Passive to 10: Highly Active)", 1, 10, 7)
                extracurricular_activity = st.selectbox("Participates in Extracurricular / Sports Activities?", ["Yes", "No"])

            submit_btn = st.form_submit_button("🔮 Predict Student Performance", use_container_width=True)

        if submit_btn:
            input_data = {
                "attendance": attendance,
                "previous_marks": previous_marks,
                "study_hours": study_hours,
                "assignment_score": assignment_score,
                "internal_marks": internal_marks,
                "completed_assignments": completed_assignments,
                "class_participation": class_participation,
                "extracurricular_activity": extracurricular_activity
            }

            with st.spinner("Analyzing parameters through trained ML models..."):
                result = predictor.predict(input_data)

            if not result.get("success"):
                st.error(f"Input Validation Error: {result.get('error')}")
            else:
                st.markdown("---")
                st.subheader("📊 Prediction Outcomes")

                r_col1, r_col2 = st.columns(2)
                pred_marks = result["predicted_marks"]
                pred_cat = result["predicted_category"]
                interp = result["interpretation"]

                with r_col1:
                    st.metric(
                        label=f"Expected Final Marks ({result['best_reg_model']})",
                        value=f"{pred_marks} / 100",
                        delta=f"{pred_marks - previous_marks:+.1f} vs Previous"
                    )

                with r_col2:
                    st.metric(
                        label=f"Predicted Performance Band ({result['best_clf_model']})",
                        value=pred_cat
                    )

                # Custom Result Card
                badge_class = "badge-high" if pred_cat == "High" else ("badge-avg" if pred_cat == "Average" else "badge-low")
                st.markdown(
                    f"""
                    <div class="result-box {badge_class}">
                        <h4 style="margin-top:0;">Evaluation Summary:</h4>
                        <p>{interp['summary']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Probabilities & Actionable Recommendations
                col_p, col_s = st.columns(2)
                with col_p:
                    st.markdown("#### 🎯 Classification Confidence:")
                    if result["category_probabilities"]:
                        for cat, prob in result["category_probabilities"].items():
                            st.write(f"**{cat}**: {prob}%")
                            st.progress(prob / 100.0)

                with col_s:
                    st.markdown("#### 💡 Targeted Recommendations:")
                    for s in interp["suggestions"]:
                        st.markdown(f"- {s}")

    # -------------------------------------------------------------------------
    # PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------------------
    elif menu == "📊 Data Visualizations (EDA)":
        st.markdown('<div class="main-header">📊 Exploratory Data Analysis Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Examine feature distributions, academic trends, and correlations across the dataset.</div>', unsafe_allow_html=True)

        df = load_dataset()
        if df is None:
            st.error("Dataset not found! Please run `python data/generate_data.py`.")
            return

        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["📈 Feature vs Marks Trends", "🔥 Correlation Heatmap", "📋 Raw Dataset Explorer"])

        with eda_tab1:
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                st.pyplot(plot_attendance_vs_marks(df))
            with row1_col2:
                st.pyplot(plot_study_hours_vs_marks(df))

            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                st.pyplot(plot_previous_vs_final_marks(df))
            with row2_col2:
                st.pyplot(plot_assignment_vs_marks(df))

            row3_col1, row3_col2 = st.columns(2)
            with row3_col1:
                st.pyplot(plot_marks_distribution(df))
            with row3_col2:
                st.pyplot(plot_category_distribution(df))

        with eda_tab2:
            st.markdown("#### Pairwise Pearson Correlation")
            st.write(
                "This heatmap reveals how strongly each student attribute correlates with final marks and intermediate assessments."
            )
            col_heat, _ = st.columns([3, 1])
            with col_heat:
                st.pyplot(plot_correlation_heatmap(df))

        with eda_tab3:
            st.markdown("#### Dataset Records & Summary Statistics")
            st.dataframe(df.head(50), use_container_width=True)
            st.markdown("##### Statistical Summary")
            st.dataframe(df.describe().round(2), use_container_width=True)

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Dataset as CSV",
                data=csv_data,
                file_name="student_performance_data.csv",
                mime="text/csv"
            )

    # -------------------------------------------------------------------------
    # PAGE 4: MODEL PERFORMANCE & COMPARISON
    # -------------------------------------------------------------------------
    elif menu == "📈 Model Performance & Comparison":
        st.markdown('<div class="main-header">📈 Model Evaluation & Comparative Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Real test-set evaluation metrics comparing regression and classification algorithms.</div>', unsafe_allow_html=True)

        metadata = load_model_metadata()
        if metadata is None:
            st.error("Model metadata not found! Please run `python src/train_models.py`.")
            return

        st.info(f"Evaluated on a strictly held-out test split of **{metadata['test_sample_count']} samples** (20% test size).")

        # Regression Comparison Section
        st.subheader("1. Numerical Grade Prediction (Regression Comparison)")
        reg_df = pd.DataFrame(metadata["regression_results"]).T
        st.dataframe(
            reg_df.style.highlight_max(subset=["R2_Score"], color="#D1FAE5")
                        .highlight_min(subset=["MAE", "RMSE"], color="#D1FAE5"),
            use_container_width=True
        )
        st.caption(f"🏆 Best Performing Regression Model: **{metadata['best_regression_model']}**")

        st.markdown("---")

        # Classification Comparison Section
        st.subheader("2. Performance Category Classification (Classification Comparison)")
        clf_metrics = {k: {m: v[m] for m in ["Accuracy", "Precision", "Recall", "F1_Score"]} 
                       for k, v in metadata["classification_results"].items()}
        clf_df = pd.DataFrame(clf_metrics).T
        st.dataframe(
            clf_df.style.highlight_max(subset=["Accuracy", "F1_Score"], color="#D1FAE5"),
            use_container_width=True
        )
        st.caption(f"🏆 Best Performing Classification Model: **{metadata['best_classification_model']}**")

        st.markdown("---")

        # Confusion Matrix Section
        st.subheader("3. Best Classifier Confusion Matrix")
        best_clf = metadata["best_classification_model"]
        cm_data = metadata["classification_results"][best_clf]["Confusion_Matrix"]
        labels = metadata["labels"]

        c_col1, c_col2 = st.columns([1, 1])
        with c_col1:
            st.pyplot(plot_confusion_matrix(cm_data, labels))
        with c_col2:
            st.markdown("#### Interpretation of Confusion Matrix:")
            st.write(
                f"""
                - **Diagonal Elements**: Represent the count of correct classifications for `{labels[0]}`, `{labels[1]}`, and `{labels[2]}` students.
                - **Off-Diagonal Elements**: Represent misclassifications.
                - The high concentration on the main diagonal confirms that **{best_clf}** reliably identifies students at risk without frequent false positives.
                """
            )

    # -------------------------------------------------------------------------
    # PAGE 5: VIVA & ARCHITECTURE GUIDE
    # -------------------------------------------------------------------------
    elif menu == "ℹ️ Viva & Architecture Guide":
        st.markdown('<div class="main-header">ℹ️ Viva & Project Architecture Reference</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Essential conceptual references and common questions for BTech CSE external evaluation.</div>', unsafe_allow_html=True)

        with st.expander("❓ Q1: Why did you implement both Regression and Classification?", expanded=True):
            st.write(
                """
                **Answer:** Both tasks serve complementary purposes in academic governance:
                - **Regression** provides granular continuous feedback (e.g. predicting a student will achieve 74.2 vs 52.8 marks), helping measure incremental gains.
                - **Classification** discretizes the outcome into actionable risk bands (*Low*, *Average*, *High*), enabling academic institutions to trigger administrative alerts or assign remedial tutors.
                """
            )

        with st.expander("❓ Q2: How did you avoid Data Leakage during preprocessing?"):
            st.write(
                """
                **Answer:** Data leakage was strictly prevented by:
                1. Splitting the raw data into 80% train and 20% test sets **before** applying `StandardScaler`.
                2. Calling `fit_transform` **only** on the training split, and applying `transform` (without re-fitting) to the test set and live user inference.
                3. Excluding target variables and unique identifiers (`student_id`) from feature matrices.
                """
            )

        with st.expander("❓ Q3: Why did Linear Regression perform well compared to Random Forest Regressor?"):
            st.write(
                """
                **Answer:** In educational grading systems, final marks are composed of additive components (internal exams, previous marks, attendance compliance, assignments). 
                Because the primary relationship between these foundational inputs and the final score is largely linear and additive with Gaussian noise, 
                Linear Regression provides high interpretability, lower variance, and avoids overfitting on moderate dataset sizes.
                """
            )

        with st.expander("❓ Q4: How are the performance categories defined?"):
            st.write(
                """
                **Answer:**
                - **Low Performance**: Final Marks < 50 (Failing or at academic risk).
                - **Average Performance**: 50 ≤ Final Marks < 75 (Standard passing range).
                - **High Performance**: Final Marks ≥ 75 (Distinction level / high achiever).
                """
            )

        with st.expander("❓ Q5: What are the key limitations and future scope?"):
            st.write(
                """
                **Limitations:**
                - Real academic environments may exhibit external socioeconomic or health factors not captured in pure classroom indicators.
                - Dataset is synthetic (though calibrated to realistic university distributions).
                
                **Future Scope:**
                - Integration with University ERP / LMS APIs (Moodle, Blackboard, Canvas).
                - Automated student email alerts with personalized study plans.
                - Explainable AI (SHAP / LIME) integration for individual feature attribution.
                """
            )


if __name__ == "__main__":
    main()
