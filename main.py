"""Clean and explore the Indian Liver Patient Records dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURE_COLUMNS = [
    "Age",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]
REQUIRED_COLUMNS = ["Gender", *FEATURE_COLUMNS]
TARGET_COLUMN = "Dataset"
DEFAULT_DATASET = "indian_liver_patient.csv"


def load_and_clean(input_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read the CSV, remove duplicate records, and handle missing values."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {input_path}")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    original_rows = len(df)
    missing_before = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Gender"] = df["Gender"].astype("category")
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("category")

    imputed_values: dict[str, int] = {}
    for column in FEATURE_COLUMNS:
        missing_count = int(df[column].isna().sum())
        if missing_count:
            median = df[column].median()
            if pd.isna(median):
                raise ValueError(
                    f"Column {column!r} has no valid numeric values to impute."
                )
            df[column] = df[column].fillna(median)
            imputed_values[column] = missing_count

    gender_missing = int(df["Gender"].isna().sum())
    if gender_missing:
        modes = df["Gender"].mode()
        if modes.empty:
            raise ValueError("Gender has missing values and no value to impute them with.")
        df["Gender"] = df["Gender"].fillna(modes.iloc[0])
        imputed_values["Gender"] = gender_missing

    missing_after = df.isna().sum()
    summary: dict[str, object] = {
        "original_rows": original_rows,
        "duplicate_rows_removed": duplicate_rows,
        "rows_after_deduplication": len(df),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "imputed_values": imputed_values,
    }
    return df, summary


def count_iqr_outliers(df: pd.DataFrame) -> dict[str, int]:
    """Count feature values outside the 1.5-IQR fences."""
    counts: dict[str, int] = {}
    for column in FEATURE_COLUMNS:
        values = df[column]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        counts[column] = int(
            ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()
        )
    return counts


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> list[str]:
    """Save the exploratory plots and return their filenames."""
    sns.set_theme(style="whitegrid")
    saved: list[str] = []

    def save_plot(filename: str) -> None:
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(filename)

    if TARGET_COLUMN in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=TARGET_COLUMN)
        plt.title("Liver Disease Class Distribution")
        save_plot("01_disease_class_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Age", bins=20, kde=True)
    plt.title("Age Distribution")
    save_plot("02_age_distribution.png")

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Gender")
    plt.title("Gender Distribution")
    save_plot("03_gender_distribution.png")

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, y="Total_Bilirubin")
    plt.title("Total Bilirubin and Potential Outliers")
    save_plot("04_total_bilirubin_boxplot.png")

    plt.figure(figsize=(8, 5))
    if TARGET_COLUMN in df.columns:
        sns.boxplot(data=df, x=TARGET_COLUMN, y="Albumin")
        plt.title("Albumin by Disease Class")
    else:
        sns.histplot(data=df, x="Albumin", bins=20, kde=True)
        plt.title("Albumin Distribution (No Outcome Column Available)")
    save_plot("05_albumin_distribution.png")

    numeric_df = df[FEATURE_COLUMNS]
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True,
    )
    plt.title("Correlation Between Numeric Health Measurements")
    save_plot("06_correlation_heatmap.png")

    plt.figure(figsize=(8, 5))
    if TARGET_COLUMN in df.columns:
        sns.scatterplot(
            data=df,
            x="Age",
            y="Total_Bilirubin",
            hue=TARGET_COLUMN,
        )
    else:
        sns.scatterplot(data=df, x="Age", y="Total_Bilirubin")
    plt.title("Age vs Total Bilirubin")
    save_plot("07_age_vs_bilirubin.png")

    return saved


def write_report(
    df: pd.DataFrame,
    summary: dict[str, object],
    outlier_counts: dict[str, int],
    plot_files: list[str],
    report_path: Path,
) -> None:
    """Write the analysis results as a readable plain-text report."""
    missing_before = summary["missing_before"]
    missing_after = summary["missing_after"]
    imputed_values = summary["imputed_values"]

    lines = [
        "INDIAN LIVER PATIENT RECORDS — DATA ANALYSIS",
        "=" * 48,
        "",
        f"Records before cleaning: {summary['original_rows']}",
        f"Duplicate records removed: {summary['duplicate_rows_removed']}",
        f"Records after cleaning: {len(df)}",
        f"Feature columns: {len(FEATURE_COLUMNS)}",
        f"Outcome column available: {'Yes' if TARGET_COLUMN in df.columns else 'No'}",
        "",
        "DATA CLEANING",
        "-" * 48,
        "Missing values before cleaning:",
        missing_before.to_string(),
        "",
        "Missing values imputed using each feature's median (Gender uses its mode):",
        (
            "\n".join(f"  {column}: {count}" for column, count in imputed_values.items())
            if imputed_values
            else "  None"
        ),
        "",
        "Missing values after cleaning:",
        missing_after.to_string(),
        "",
        "NUMERIC FEATURE SUMMARY",
        "-" * 48,
        df[FEATURE_COLUMNS].describe().round(2).to_string(),
        "",
        "GENDER COUNTS",
        "-" * 48,
        df["Gender"].value_counts(dropna=False).to_string(),
        "",
        "POTENTIAL OUTLIERS (1.5 × IQR RULE)",
        "-" * 48,
        "\n".join(f"{column}: {count}" for column, count in outlier_counts.items()),
        "",
    ]

    if TARGET_COLUMN in df.columns:
        lines.extend(
            [
                "OUTCOME CLASS COUNTS",
                "-" * 48,
                "Dataset codes: 1 = liver disease; 2 = no liver disease.",
                df[TARGET_COLUMN].value_counts(dropna=False).to_string(),
                "",
                "ALBUMIN BY OUTCOME CLASS",
                "-" * 48,
                df.groupby(TARGET_COLUMN, observed=False)["Albumin"]
                .agg(["count", "mean", "median"])
                .round(2)
                .to_string(),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "OUTCOME LIMITATION",
                "-" * 48,
                "The supplied CSV does not contain a Dataset outcome column.",
                "Disease-class comparisons cannot be produced from this file.",
                "This project performs descriptive analysis only and does not train",
                "or present a disease-prediction model.",
                "",
            ]
        )

    lines.extend(
        [
            "VISUALIZATIONS",
            "-" * 48,
            *plot_files,
            "",
            "This is an educational data-analysis project, not a medical diagnostic tool.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Clean and explore Indian Liver Patient Records from a CSV file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / DEFAULT_DATASET,
        help=f"Input CSV path (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "output",
        help="Directory for the cleaned CSV, plots, and plain-text report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, summary = load_and_clean(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = args.output_dir / "cleaned_liver_dataset.csv"
    report_path = args.output_dir / "analysis_report.txt"
    df.to_csv(cleaned_path, index=False)

    outlier_counts = count_iqr_outliers(df)
    plot_files = create_visualizations(df, args.output_dir)
    write_report(df, summary, outlier_counts, plot_files, report_path)

    print(f"Cleaned dataset saved to: {cleaned_path}")
    print(f"Analysis report saved to: {report_path}")
    print(f"Saved {len(plot_files)} visualizations to: {args.output_dir}")
    if TARGET_COLUMN not in df.columns:
        print(
            "Note: no Dataset outcome column was found; disease-class comparisons "
            "were omitted."
        )


if __name__ == "__main__":
    main()