# Indian Liver Patient Records: Data Analysis

A Python project for cleaning and exploring the Indian Liver Patient Records
dataset. It removes duplicate rows, fills missing numeric feature values with
their median, summarizes potential outliers, and saves charts and a text report.

This is an educational data-analysis project. It is not a medical diagnostic
tool and does not provide individual medical advice.

## Requirements

- Python 3.10 or newer
- The packages in `requirements.txt`

## Run from Windows Command Prompt

Open CMD in this project folder, then run:

```cmd
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

The project includes `indian_liver_patient.csv` and uses it by default.

## Run the website locally

From the project folder in CMD:

```cmd
.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit opens the dashboard in your browser (usually at
`http://localhost:8501`). The site shows the analysis, visualizations, cleaned
records, and download buttons. You can optionally upload a CSV with the required
feature columns.

## Publish the website for others

1. Push the project files, including `app.py`, `requirements.txt`, and
   `indian_liver_patient.csv`, to a public GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using
   GitHub and choose **Create app**.
3. Select the repository, `main` branch, and `app.py` as the main file.
4. Deploy. Streamlit provides a shareable public URL for the website.

## Output

The script creates an `output` folder containing:

- `cleaned_liver_dataset.csv` — cleaned records with the disease outcome column
- `analysis_report.txt` — cleaning summary, descriptive statistics, class counts,
  and potential outlier counts
- PNG charts — disease-class distribution, age and gender distributions,
  bilirubin boxplot, albumin by disease class, numeric-feature correlation
  heatmap, and age vs. bilirubin

Generated output and the local Python virtual environment are excluded from Git.
Run the script to recreate them.

## Dataset

The CSV contains 583 records and the `Dataset` outcome column (`1` = liver
disease, `2` = no liver disease). The script removes duplicate rows and fills
missing feature values during cleaning. The dataset is commonly distributed as
the [Indian Liver Patient Records dataset on Kaggle](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records).

## Optional arguments

Use another CSV or choose a different output directory:

```cmd
.venv\Scripts\python.exe main.py --input path\to\data.csv --output-dir results
```
