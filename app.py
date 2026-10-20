"""Interactive Streamlit dashboard for the liver-patient analysis project."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from main import (
    DEFAULT_DATASET,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    count_iqr_outliers,
    load_and_clean,
    write_report,
)

PROJECT_DIR = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Liver Patient Data Explorer",
    page_icon="🩺",
    layout="wide",
)
sns.set_theme(style="whitegrid")


@st.cache_data(show_spinner="Cleaning and analyzing the dataset...")
def analyze_dataset(file_bytes: bytes | None) -> tuple[pd.DataFrame, dict[str, object]]:
    if file_bytes is None:
        dataset_path = PROJECT_DIR / DEFAULT_DATASET
        return load_and_clean(dataset_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = Path(temp_dir) / "uploaded_dataset.csv"
        dataset_path.write_bytes(file_bytes)
        return load_and_clean(dataset_path)


def show_chart(fig: plt.Figure) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)


st.title("Indian Liver Patient Records")
st.caption("An interactive dashboard for exploring and visualizing the dataset.")
st.info(
    "Educational data analysis only. This dashboard is not a medical diagnostic "
    "tool and should not be used to make healthcare decisions."
)

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader(
        "Optionally upload a CSV",
        type=["csv"],
        help=(
            "By default, the dashboard uses the included full dataset. "
            "An uploaded CSV must contain the patient feature columns."
        ),
    )
    st.caption(
        "The included dataset is loaded automatically when you don't upload a file."
    )

try:
    df, cleaning_summary = analyze_dataset(
        uploaded_file.getvalue() if uploaded_file is not None else None
    )
except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load or clean the dataset: {error}")
    st.stop()

class_available = TARGET_COLUMN in df.columns
unique_classes = df[TARGET_COLUMN].nunique() if class_available else 0

metric_columns = st.columns(4)
metric_columns[0].metric("Records analyzed", f"{len(df):,}")
metric_columns[1].metric(
    "Duplicate rows removed",
    f"{cleaning_summary['duplicate_rows_removed']:,}",
)
metric_columns[2].metric(
    "Missing values filled",
    f"{sum(cleaning_summary['imputed_values'].values()):,}",
)
metric_columns[3].metric(
    "Outcome classes" if class_available else "Outcome column",
    f"{unique_classes}" if class_available else "Not provided",
)

overview_tab, charts_tab, data_tab = st.tabs(
    ["Overview", "Visualizations", "Cleaned data"]
)

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Outcome class counts")
        if class_available:
            class_counts = (
                df[TARGET_COLUMN]
                .value_counts()
                .sort_index()
                .rename(index={1: "Liver disease", 2: "No liver disease"})
            )
            st.bar_chart(class_counts, x_label="Outcome", y_label="Patients")
            st.caption("Dataset codes: 1 = liver disease; 2 = no liver disease.")
        else:
            st.warning(
                "This CSV has no Dataset outcome column. Disease-class comparisons "
                "are unavailable."
            )

    with right:
        st.subheader("Patient gender")
        st.bar_chart(
            df["Gender"].value_counts(),
            x_label="Gender",
            y_label="Patients",
        )

    st.subheader("Numeric feature summary")
    st.dataframe(df[FEATURE_COLUMNS].describe().T.round(2), width="stretch")

    with st.expander("Cleaning details"):
        before = cleaning_summary["missing_before"]
        imputed = cleaning_summary["imputed_values"]
        st.write("Missing values in the source file:")
        st.dataframe(before.rename("Missing values"), width="stretch")
        st.write("Values filled during cleaning:")
        if imputed:
            st.dataframe(
                pd.Series(imputed, name="Values filled"),
                width="stretch",
            )
        else:
            st.write("No values needed to be filled.")
        st.write("Potential outliers using the 1.5 × IQR rule:")
        st.dataframe(
            pd.Series(count_iqr_outliers(df), name="Potential outliers"),
            width="stretch",
        )

with charts_tab:
    first_row = st.columns(2)
    with first_row[0]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df, x="Age", bins=20, kde=True, ax=ax)
        ax.set_title("Age distribution")
        show_chart(fig)
    with first_row[1]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, y="Total_Bilirubin", ax=ax)
        ax.set_title("Total bilirubin and potential outliers")
        show_chart(fig)

    second_row = st.columns(2)
    with second_row[0]:
        fig, ax = plt.subplots(figsize=(7, 4))
        if class_available:
            sns.boxplot(data=df, x=TARGET_COLUMN, y="Albumin", ax=ax)
            ax.set_title("Albumin by disease class")
        else:
            sns.histplot(data=df, x="Albumin", bins=20, kde=True, ax=ax)
            ax.set_title("Albumin distribution")
        show_chart(fig)
    with second_row[1]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(
            data=df,
            x="Age",
            y="Total_Bilirubin",
            hue=TARGET_COLUMN if class_available else None,
            ax=ax,
        )
        ax.set_title("Age vs. total bilirubin")
        show_chart(fig)

    st.subheader("Correlation between numeric measurements")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        df[FEATURE_COLUMNS].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True,
        ax=ax,
    )
    show_chart(fig)

with data_tab:
    st.subheader("Cleaned dataset")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        "Download cleaned CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_liver_dataset.csv",
        mime="text/csv",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = Path(temp_dir) / "analysis_report.txt"
        write_report(
            df,
            cleaning_summary,
            count_iqr_outliers(df),
            [],
            report_path,
        )
        report_text = report_path.read_text(encoding="utf-8")
    st.download_button(
        "Download analysis report",
        data=report_text,
        file_name="analysis_report.txt",
        mime="text/plain",
    )
