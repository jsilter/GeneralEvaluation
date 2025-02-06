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

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run general_eval/app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Follow the instructions on the web app to upload your results table and run the evaluation.

## Usage with pipx

You can also install the package using [pipx](https://pipx.pypa.io/stable/) to run the command-line tool without activating the virtual environment.
```shell
git clone https://github.com/reginabarzilaygroup/GeneralEvaluation
cd GeneralEvaluation
pipx install .
general-eval
```

## File Structure

- `general_eval/main.py`: Contains the main evaluation logic and functions. This can also be used as a standalone command-line tool. Run `python general_eval/main.py --help` for more information.
- `general_eval/app.py`: Streamlit web app for user interaction and displaying results.

## Example Data

An example data file is provided in the `data` directory. You can download it from the web app to see the expected format of the input data.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- Streamlit for providing an easy-to-use framework for building web apps.
- The developers of Mirai and Sybil for their contributions to risk prediction models.
