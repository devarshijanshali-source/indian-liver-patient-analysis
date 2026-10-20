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
    build_observations,
    count_iqr_outliers,
    load_and_clean,
    numeric_correlation,
    write_report,
)

PROJECT_DIR = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Liver Patient Data Explorer",
    page_icon="🩺",
    layout="wide",
)
sns.set_theme(style="whitegrid")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 90% 0%, rgba(28, 131, 115, .10), transparent 34%),
            #f4f8f7;
    }
    [data-testid="stHeader"] { background: rgba(244, 248, 247, .84); }
    .block-container { max-width: 1320px; padding-top: 2.2rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f1ef 0%, #f4f8f7 100%);
        border-right: 1px solid #dce8e5;
    }
    .hero {
        position: relative;
        overflow: hidden;
        margin: .3rem 0 1.35rem;
        padding: 2.2rem 2.4rem;
        color: #f4fbf9;
        background: linear-gradient(118deg, #123b42 0%, #146c68 58%, #22947d 100%);
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 24px;
        box-shadow: 0 16px 40px rgba(23, 74, 70, .16);
    }
    .hero:after {
        position: absolute;
        top: -105px;
        right: -35px;
        width: 310px;
        height: 310px;
        content: "";
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 50%;
        box-shadow: 0 0 0 34px rgba(255,255,255,.035), 0 0 0 70px rgba(255,255,255,.025);
    }
    .hero-kicker {
        margin-bottom: .8rem;
        color: #b9e5d8;
        font-size: .77rem;
        font-weight: 700;
        letter-spacing: .15em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 0;
        color: #fff;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 750;
        letter-spacing: -.045em;
        line-height: 1.08;
    }
    .hero p {
        max-width: 640px;
        margin: .85rem 0 0;
        color: #d8ece8;
        font-size: 1.03rem;
        line-height: 1.65;
    }
    [data-testid="stMetric"] {
        min-height: 118px;
        padding: 1rem 1.1rem;
        background: rgba(255,255,255,.9);
        border: 1px solid #e0ebe8;
        border-radius: 17px;
        box-shadow: 0 5px 18px rgba(24, 65, 60, .045);
    }
    [data-testid="stMetricLabel"] p { color: #5d7472; font-size: .88rem; }
    [data-testid="stMetricValue"] { color: #153e42; font-weight: 700; }
    [data-testid="stAlert"] { border-radius: 14px; }
    .stTabs [data-baseweb="tab-list"] {
        gap: .45rem;
        padding: .4rem;
        background: #e7efed;
        border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 2.7rem;
        padding: 0 1rem;
        border-radius: 10px;
    }
    .stTabs [aria-selected="true"] { background: #fff; }
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid #e0ebe8;
        border-radius: 16px;
    }
    .stDownloadButton button {
        min-height: 2.7rem;
        border-radius: 10px;
        font-weight: 650;
    }
    @media (max-width: 640px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 1.6rem 1.35rem; border-radius: 18px; }
        .hero:after { right: -210px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Indian Liver Patient Records &nbsp;·&nbsp; Data Explorer</div>
      <h1>Liver health,<br>seen through data.</h1>
      <p>Explore patient measurements, compare outcome groups, and understand
      patterns in the dataset through an interactive visual dashboard.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
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

    st.subheader("What the data shows")
    for observation in build_observations(df):
        st.markdown(f"- {observation}")

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
        if class_available:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.countplot(data=df, x=TARGET_COLUMN, ax=ax)
            ax.set_title("Liver patient vs. non-liver patient count")
            ax.set_xlabel("Dataset code (1 = liver disease, 2 = no liver disease)")
            show_chart(fig)
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.countplot(data=df, x="Gender", ax=ax)
            ax.set_title("Gender distribution")
            show_chart(fig)
    with first_row[1]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df, x="Age", bins=20, kde=True, ax=ax)
        ax.set_title("Age distribution")
        show_chart(fig)

    second_row = st.columns(2)
    with second_row[0]:
        fig, ax = plt.subplots(figsize=(7, 4))
        if class_available:
            sns.countplot(data=df, x="Gender", ax=ax)
            ax.set_title("Gender distribution")
        else:
            sns.histplot(data=df, x="Albumin", bins=20, kde=True, ax=ax)
            ax.set_title("Albumin distribution")
        show_chart(fig)
    with second_row[1]:
        fig, ax = plt.subplots(figsize=(7, 4))
        if class_available:
            sns.boxplot(data=df, x=TARGET_COLUMN, y="Total_Bilirubin", ax=ax)
            ax.set_title("Total bilirubin by patient group")
            ax.set_xlabel("Dataset code (1 = liver disease, 2 = no liver disease)")
        else:
            sns.boxplot(data=df, y="Total_Bilirubin", ax=ax)
            ax.set_title("Total bilirubin and potential outliers")
        show_chart(fig)

    third_row = st.columns(2)
    with third_row[0]:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            numeric_correlation(df),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            ax=ax,
        )
        ax.set_title("Correlation heatmap")
        show_chart(fig)
    with third_row[1]:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            data=df,
            x="Total_Bilirubin",
            y="Direct_Bilirubin",
            hue=TARGET_COLUMN if class_available else None,
            ax=ax,
        )
        ax.set_title("Total bilirubin vs. direct bilirubin")
        show_chart(fig)

    if class_available:
        st.subheader("Albumin distribution by patient group")
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.violinplot(
            data=df,
            x=TARGET_COLUMN,
            y="Albumin",
            inner="quart",
            cut=0,
            ax=ax,
        )
        ax.set_title("Albumin levels by patient group")
        ax.set_xlabel("Dataset code (1 = liver disease, 2 = no liver disease)")
        show_chart(fig)

    st.subheader("Age and bilirubin")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=df,
        x="Age",
        y="Total_Bilirubin",
        hue=TARGET_COLUMN if class_available else None,
        ax=ax,
    )
    ax.set_title("Age vs. total bilirubin")
    show_chart(fig)

with data_tab:
    st.subheader("Cleaned dataset")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        "Download cleaned CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_liver_data.csv",
        mime="text/csv",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = Path(temp_dir) / "analysis_report.txt"
        plot_files = [
            "02_age_distribution.png",
            "03_gender_distribution.png",
            "04_total_bilirubin_by_group.png"
            if class_available
            else "04_total_bilirubin_boxplot.png",
            "05_correlation_heatmap.png",
            "06_bilirubin_relationship.png",
            "07_albumin_violin_by_group.png"
            if class_available
            else "07_albumin_distribution.png",
            "08_age_vs_bilirubin.png",
        ]
        if class_available:
            plot_files.append("01_disease_class_distribution.png")
        write_report(
            df,
            cleaning_summary,
            count_iqr_outliers(df),
            plot_files,
            report_path,
        )
        report_text = report_path.read_text(encoding="utf-8")
    st.download_button(
        "Download analysis report",
        data=report_text,
        file_name="analysis_report.txt",
        mime="text/plain",
    )
