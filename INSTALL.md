# Model Performance Evaluation Web App

## Description

This project is a simple Streamlit web application designed to evaluate the performance of a risk prediction model. It is intended for use with Mirai and Sybil. The application allows users to upload a results table, run evaluations, and generate performance reports in PDF format.

## Features

- Upload results table in CSV, TSV, XLS, or XLSX format.
- Evaluate model performance based on provided data.
- Generate and download a PDF report of the evaluation.
- Download overall metrics in CSV format.
- Display PDF report within the web app.

## Requirements

- Python 3.11
- Streamlit
- Pandas
- Seaborn
- Matplotlib

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/reginabarzilaygroup/GeneralEvaluation
    cd GeneralEvaluation
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the package, which will install required packages:
    ```sh
    pip install .
    ```

## Install and run

1. Run the Streamlit app:
    ```sh
    streamlit run general_eval/app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Follow the instructions on the web app to upload your results table and run the evaluation.

###  Install and run with pipx

You can also install the package using [pipx](https://pipx.pypa.io/stable/) to run the command-line tool without activating the virtual environment.
```shell
git clone https://github.com/reginabarzilaygroup/GeneralEvaluation
cd GeneralEvaluation
pipx install .
general-eval
```