#!/usr/bin/env python

__doc__ = """
Simple streamlit web app for evaluating the performance of a model
"""

import base64
import os
import tempfile

import streamlit as st

from main import run_full_eval, DIAGNOSIS_DAYS_COL, FOLLOWUP_DAYS_COL

def displayPDF(file_path, name="results"):
    # Opening file from file path
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    height = 800

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" name="{name}" width="1200" height="{height}" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == "__main__":
    st.title("Model Performance Evaluation")
    st.write("This is a simple web app to evaluate the performance of a model. Make sure your data is formatted properly before uploading.")

    ds_name = st.text_input("Dataset Name", value="My Dataset")

    uploaded_file = st.file_uploader("Results table", type=["csv", "tsv", "xls", "xlsx"])
    st.markdown(f"""Upload a file containing the results of the model. The file should contain the following columns:   
                - `{DIAGNOSIS_DAYS_COL}`: Days between exam and cancer diagnosis. '-1' indicates no cancer diagnosis.  
                - `{FOLLOWUP_DAYS_COL}`: Days between exam and last follow-up.   
                - `Year1`: Model prediction for year 1.  
                - `Year2`: Model prediction for year 2.  
                - `Year3`: Model prediction for year 3.  
                - `Year4`: Model prediction for year 4.  
                - `Year5`: Model prediction for year 5.  
                - `Year6`: Model prediction for year 6. (optional)
    """)

    # Example data file
    example_data_file_path = "data/general_eval_demo_data.csv"
    example_data_file_name = example_data_file_path.split("/")[-1]
    with open(example_data_file_path, "rb") as file:
        data_bytes = file.read()

    st.download_button(
        label="Download Example Data File",
        data=data_bytes,
        file_name=example_data_file_name,
        mime="text/plain"
    )

    recall_target = st.number_input("Recall Target", value=0.85, help="The target recall value for the model. ")

    run_button = st.button("Run Evaluation", disabled=uploaded_file is None)

    if run_button and uploaded_file is not None:
        st.write("File uploaded, processing...")

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(suffix=uploaded_file.name, delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        output_file = run_full_eval(ds_name, temp_file_path, recall_target=recall_target)
        st.write("Evaluation complete!")
        output_file_name = output_file.split("/")[-1]

        with open(output_file, "rb") as file:
            st.download_button(
                label="Download PDF",
                data=file,
                file_name=output_file_name,
                mime="application/pdf"
            )

        displayPDF(output_file)

        # Clean up temporary files
        os.remove(temp_file_path)
        os.remove(output_file)
