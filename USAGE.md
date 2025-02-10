# Model Performance Evaluation Web App

## Description

This project is a simple Streamlit web application designed to evaluate the performance of a risk prediction model. It is intended for use with Mirai and Sybil. The application allows users to upload a results table, run evaluations, and generate performance reports in PDF format.

For installation instructions, see the [README.md](README.md) file.

# Usage

The following steps assume that the Streamlit app has been installed and launched. See the [Installation](#installation) section for details)

1. Prepare your results from Mirai or Sybil. See the [Input Data Format](#input-data-format) section for details.
2. Upload a results table in CSV, TSV, XLS, or XLSX format.
3. The default recall target is `0.85`. If desired, you may adjust the recall target. 
4. Click the "Run Evaluation" button.
5. Download the PDF report and the CSV file with overall metrics.

## Input Data Format

The input data should be a table with the following columns:
Upload a file containing the results of the model. The file should contain the following columns:   
 - `Days_to_Cancer`: Days between exam and cancer diagnosis. '-1' indicates no cancer diagnosis.  
 - `Days_Followup`: Days between exam and latest follow-up.   
 - `Year1`: Model prediction for year 1.  
 - `Year2`: Model prediction for year 2.  
 - `Year3`: Model prediction for year 3.  
 - `Year4`: Model prediction for year 4.  
 - `Year5`: Model prediction for year 5.  
 - `Year6`: Model prediction for year 6. (optional)

If there are missing columns, you will see an error occur. Any additional columns will be ignored.

See the example data file in the `data` folder for reference.

## Example Data

An example data file is provided in the `data` folder. You can download it from the web app to see the expected format of the input data.

## Methods

The performance is evaluated using multiple metrics. Area-under-the-curve (AUC) and precision-recall curve (PRC) are calculated with [scikit-learn](https://scikit-learn.org/stable/) functions. The "Net Benefit" is calculated according to the definition in [Piovani et al](https://pmc.ncbi.nlm.nih.gov/articles/PMC10454914/).