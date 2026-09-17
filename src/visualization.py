"""
Visualization Module for AI-Driven Student Performance Prediction System.

Generates professional exploratory data analysis (EDA) charts using Matplotlib and Seaborn.
Every function returns a Matplotlib Figure object that integrates cleanly with Streamlit.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent, elegant visual style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

PALETTE = {
    "Low": "#E74C3C",     # Coral Red
    "Average": "#F39C12", # Warm Amber
    "High": "#27AE60"     # Emerald Green
}

def plot_attendance_vs_marks(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot with trendline showing Attendance vs Final Marks."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(
        data=df,
        x="attendance",
        y="final_marks",
        hue="performance_category",
        palette=PALETTE,
        alpha=0.75,
        edgecolor=None,
        s=45,
        ax=ax
    )
    sns.regplot(
        data=df,
        x="attendance",
        y="final_marks",
        scatter=False,
        color="#2C3E50",
        line_kws={"linewidth": 2, "linestyle": "--"},
        ax=ax
    )
    ax.set_title("Attendance Percentage vs Final Marks")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Final Marks (/100)")
    ax.legend(title="Category", loc="upper left")
    return fig

def plot_study_hours_vs_marks(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot with trendline showing Daily Study Hours vs Final Marks."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(
        data=df,
        x="study_hours",
        y="final_marks",
        hue="performance_category",
        palette=PALETTE,
        alpha=0.75,
        s=45,
        ax=ax
    )
    sns.regplot(
        data=df,
        x="study_hours",
        y="final_marks",
        scatter=False,
        color="#2980B9",
        line_kws={"linewidth": 2, "linestyle": "-."},
        ax=ax
    )
    ax.set_title("Daily Study Hours vs Final Marks")
    ax.set_xlabel("Study Hours (per day)")
    ax.set_ylabel("Final Marks (/100)")
    ax.legend(title="Category", loc="upper left")
    return fig

def plot_previous_vs_final_marks(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot showing Previous Exam Marks vs Final Marks."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(
        data=df,
        x="previous_marks",
        y="final_marks",
        hue="performance_category",
        palette=PALETTE,
        alpha=0.75,
        s=45,
        ax=ax
    )
    sns.regplot(
        data=df,
        x="previous_marks",
        y="final_marks",
        scatter=False,
        color="#8E44AD",
        line_kws={"linewidth": 2},
        ax=ax
    )
    ax.set_title("Previous Examination Marks vs Final Marks")
    ax.set_xlabel("Previous Marks (/100)")
    ax.set_ylabel("Final Marks (/100)")
    ax.legend(title="Category", loc="upper left")
    return fig

def plot_assignment_vs_marks(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot showing Assignment Score vs Final Marks."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(
        data=df,
        x="assignment_score",
        y="final_marks",
        hue="performance_category",
        palette=PALETTE,
        alpha=0.75,
        s=45,
        ax=ax
    )
    sns.regplot(
        data=df,
        x="assignment_score",
        y="final_marks",
        scatter=False,
        color="#16A085",
        line_kws={"linewidth": 2},
        ax=ax
    )
    ax.set_title("Assignment Score vs Final Marks")
    ax.set_xlabel("Assignment Score (/100)")
    ax.set_ylabel("Final Marks (/100)")
    ax.legend(title="Category", loc="upper left")
    return fig

def plot_marks_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram with KDE for distribution of Final Marks."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(
        data=df,
        x="final_marks",
        kde=True,
        color="#3498DB",
        bins=25,
        ax=ax
    )
    mean_val = df["final_marks"].mean()
    median_val = df["final_marks"].median()
    ax.axvline(mean_val, color="#E74C3C", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color="#2ECC71", linestyle=":", linewidth=2, label=f"Median: {median_val:.1f}")
    ax.set_title("Distribution of Final Marks")
    ax.set_xlabel("Final Marks (/100)")
    ax.set_ylabel("Student Count")
    ax.legend()
    return fig

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap showing pairwise Pearson correlation of numerical features."""
    fig, ax = plt.subplots(figsize=(8, 6))
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-0.2,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Feature Correlation Matrix (Pearson)")
    return fig

def plot_category_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart showing count and proportion of Performance Categories."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cat_counts = df["performance_category"].value_counts().reindex(["Low", "Average", "High"])
    colors = [PALETTE[cat] for cat in cat_counts.index]
    
    bars = ax.bar(cat_counts.index, cat_counts.values, color=colors, width=0.5, edgecolor="gray")
    
    total = len(df)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 15,
            f"{height} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )
        
    ax.set_title("Academic Performance Category Breakdown")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Students")
    ax.set_ylim(0, cat_counts.max() * 1.18)
    return fig

def plot_confusion_matrix(cm_data: list, labels: list) -> plt.Figure:
    """Plots the confusion matrix for the best classification model."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    cm_arr = np.array(cm_data)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax
    )
    ax.set_title("Classification Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("True Category")
    return fig

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_csv = os.path.join(base_dir, "data", "student_performance.csv")
    print(f"[Self-Test] Testing visualization functions with: {test_csv}")
    test_df = pd.read_csv(test_csv)
    fig1 = plot_attendance_vs_marks(test_df)
    fig2 = plot_study_hours_vs_marks(test_df)
    fig3 = plot_previous_vs_final_marks(test_df)
    fig4 = plot_assignment_vs_marks(test_df)
    fig5 = plot_marks_distribution(test_df)
    fig6 = plot_correlation_heatmap(test_df)
    fig7 = plot_category_distribution(test_df)
    plt.close("all")
    print("  All 7 EDA charts generated successfully!")
    print("  Visualization module verification SUCCESS!")
