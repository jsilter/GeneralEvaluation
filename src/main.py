#!/usr/bin/env python

__doc__ = """
Run set of evaluations on collected data validation.
We expect a datasheet of patient and exam information.
"""

import argparse
import datetime
import math
import os
import pprint

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import general_eval_lib as gel
import src.utils as utils
from src.general_eval_lib import plot_roc_prc


def _get_parser():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient_table", default=None, required=True,
                        type=str, help="Path to the file containing information on a per-patient basis "
                                       "(one patient per row). CSV format.")

    parser.add_argument("--exam_table", default=None, required=True,
                        type=str, help="Path to the file containing information on a per-exam basis"
                                       "(one exam per row). CSV format.")

    parser.add_argument("--output_path", default=None,
                        type=str, help="Path to save the output report file.")

    parser.add_argument("--split_col", default=None,
                        type=str, help="Column to use for subdividing data into subsets. "
                                       "Analysis will be repeated across each unique value of this column.")

    return parser

def _to_dt(instr):
    if instr is None or pd.isna(instr):
        return instr
    elif isinstance(instr, (datetime.datetime, pd.Timestamp)):
        return instr

    return utils.parse_date(instr)

def ___subtract_dates(a, b):
    if str(a) == "-1":
        return -1
    if isinstance(a, str):
        a = _to_dt(a)
    if isinstance(b, str):
        b = _to_dt(b)
    year_diff = a.year - b.year
    month_diff = a.month - b.month
    total_diff = 12*year_diff + month_diff
    return total_diff

def _is_positive_diag(days):
    return str(days).lower() not in {"-1", "nan", "nat", "", "none"}

def _is_negative_diag(days):
    return not _is_positive_diag(days)

def _days_to_months(days):
    if _is_negative_diag(days):
        return -1
    return int(days) / 30

def _days_to_years(days):
    if _is_negative_diag(days):
        return -1
    return int(days) / 365

def generate_true_columns(input_df, diagnosis_days_col, true_prefix="true", num_years=5):

    input_df["__interval_months"] = input_df[diagnosis_days_col].apply(_days_to_months)
    input_df["__interval_years"] = input_df[diagnosis_days_col].apply(_days_to_years)

    for _year in range(0, num_years):
        _yp1 = _year + 1
        input_df[f"{true_prefix}_year{_yp1}"] = ((0 <= input_df["__interval_years"]) & (input_df["__interval_years"] < _yp1)).astype(int)

    # TODO Include follow dates and set to -1 if patient no longer available

    return input_df

def create_basic_stats_table(input_df, diagnosis_days_col):
    basic_stats = {"Total": len(input_df),
                   "Positive": int(input_df[diagnosis_days_col].apply(_is_positive_diag).sum())}
    month_intervals = [(0, 3), (3, 12), (12, 24), (24, 36), (36, 48), (48, 60), (60, 72), (72, np.inf)]
    for start, end in month_intervals:
        if start is None or start == -np.inf:
            start_str = "-Inf"
        else:
            start_str = f"{start} months"
        if end is None or end == np.inf:
            end_str = "Inf"
        else:
            end_str = f"{end} months"

        key = f"{start_str} to {end_str}"

        keep_rows = input_df["__interval_months"].between(start, end, inclusive="left")
        basic_stats[key] = int(keep_rows.sum())

    # pprint.pprint(basic_stats)
    basic_stats_df = pd.DataFrame([basic_stats]).T
    basic_stats_df.columns = ["Count"]
    # fraction = basic_stats_df["Count"] / basic_stats_df.loc["Total", "Count"]
    basic_stats_df["Count"] = basic_stats_df["Count"].astype(int)
    basic_stats_df = basic_stats_df.reset_index()
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 2))  # Adjust the size as needed
    # Create the table
    table = plt.table(cellText=basic_stats_df.values,
                      colLabels=basic_stats_df.columns,
                      cellLoc='center', loc='center')
    df = basic_stats_df
    col_width = 1.0 / df.shape[1]
    table.auto_set_font_size(False)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
        cell.set_width(col_width)

    ax.add_table(table)
    ax.axis('tight')
    ax.axis('off')

    return fig, basic_stats_df

def generate_standards_df(all_metrics_df, standards, categories, split_col):
    # Print out table of thresholds and stats

    summary_metrics_by_cat_standard = []
    for standard in standards:
        # print(f"Thresholds for {standard['name']}")
        for cat in categories:
            split_name = cat["name"]
            true_col = cat["true_col"]
            pred_col = cat["pred_col"]
            split_df = all_metrics_df.query(f"{split_col} == '{split_name}'").copy()

            if standard["direction"] == "min_diff":
                split_df["diff"] = np.abs(split_df[standard["metric"]] - standard["target_value"])
                split_df = split_df.sort_values("diff", ascending=True)
                best_row = split_df.iloc[0].copy()
            else:
                split_df = split_df.sort_values(standard["metric"], ascending=standard["direction"] == "min")
                best_row = split_df.iloc[0].copy()

            best_row["standard"] = standard["name"]
            summary_metrics_by_cat_standard.append(best_row)

    summary_metrics_by_cat_standard = pd.DataFrame(summary_metrics_by_cat_standard).reset_index(drop=True)
    return summary_metrics_by_cat_standard

def plot_summary_tables_on_pdf(pdf_pages, summary_metrics_by_cat_standard, split_col):

    standard_names = summary_metrics_by_cat_standard["standard"].unique()
    for standard_name in standard_names:
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(24, 3))
        # ax.axis('tight')
        ax.axis('off')
        side_margin = 0.01
        top_margin = 0.1
        ax.set_position([side_margin, 0.01, 1.-2*side_margin, 1.-top_margin])

        df = summary_metrics_by_cat_standard.query(f"standard == '{standard_name}'")
        df = df.round(4)

        # Re-arrange column order
        custom_column_order = [
            "standard", split_col, "threshold", "f1_score", "balanced_accuracy",
            "PPV", "NPV", "LR+", "N", "TP", "FP", "TN", "FN"
        ]
        df = df[custom_column_order]
        custom_column_labels = list(df.columns)
        cust_mapping = [("f1_score", "F1"), ("balanced_accuracy", "Balanced Acc.")]
        for old, new in cust_mapping:
            custom_column_labels[custom_column_labels.index(old)] = new

        table = plt.table(
            cellText=df.values,
            colLabels=custom_column_labels,
            cellLoc='center',
            loc='center',)

        # Format table for readability
        main_fontsize = 14
        # Leave some margins
        row_height = 1.0/(1.0 + df.shape[0])
        col_width = 1.0/df.shape[1]
        table.auto_set_font_size(False)
        table.set_fontsize(main_fontsize)
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_text_props(fontweight='bold')
            cell.set_height(row_height)
            cell.set_width(col_width)

        ax.add_table(table)

        fig.suptitle(f"Best Thresholds by {standard_name}", fontsize=main_fontsize, fontweight='bold')
        pdf_pages.savefig(fig)
        # plt.show()

def main():
    ds_name = "NLST"
    input_path = "/Users/silterra/chem_home/Sybil/nlst_predictions/sybil_ensemble_calibrated_v2.csv"
    split_col = "Year"

    # Column names
    diagnosis_days_col = "candx_days"
    # followup_days_col = "fup_days"
    num_years = 6

    # Require diagnosis be at least 3 months after exam
    min_months = 3
    min_days = int(min_months*30)
    categories = [
        {"name": "Year 1", "pred_col": "Year1", "true_col": "true_year1"},
        {"name": "Year 2", "pred_col": "Year2", "true_col": "true_year2"},
        {"name": "Year 3", "pred_col": "Year3", "true_col": "true_year3"},
        {"name": "Year 4", "pred_col": "Year4", "true_col": "true_year4"},
        {"name": "Year 5", "pred_col": "Year5", "true_col": "true_year5"},
        {"name": "Year 6", "pred_col": "Year6", "true_col": "true_year6"},
    ]

    ppv_target = 0.85
    standards = [
        {"name": f"PPV (target)", "metric": "PPV", "direction": "min_diff", "target_value": ppv_target},
        {"name": "F1", "metric": "f1_score", "direction": "max"},
        {"name": "Balanced Accuracy", "metric": "balanced_accuracy", "direction": "max"},
    ]

    output_path = os.path.join(os.path.dirname(input_path), f"{ds_name} evaluation report.pdf")

    input_name = os.path.basename(input_path)
    input_df = gel.load_input_df(input_name, input_path, comment="#")

    print(f"Loaded {len(input_df)} rows from {input_path}")
    # print(f"Columns: {input_df.columns}")

    # Keep rows where the diagnosis is at least min_months after the exam
    def _keep_row(row, min_days=min_days):
        # Keep the negatives
        if _is_negative_diag(row[diagnosis_days_col]):
            return True

        # Only include cases where the diagnosis is at least min_days after the exam
        return int(row[diagnosis_days_col]) > min_days

    # Read days until diagnosis, generate binary columns for each year
    generate_true_columns(input_df, diagnosis_days_col, num_years=num_years)

    fig, basic_stats_df = create_basic_stats_table(input_df, diagnosis_days_col)

    pdf_pages = PdfPages(output_path)
    pdf_pages.savefig(fig)
    plt.close(fig)

    if min_months is not None:
        keep_rows = input_df.apply(_keep_row, axis=1)
        print(f"Keeping {keep_rows.sum()} / {len(keep_rows)} rows with valid exam/diagnosis dates")
        input_df = input_df.loc[keep_rows, :]

    curves_by_cat = {}
    stats_by_cat = {}
    all_cat_names = []
    for cat in categories:
        name = cat["name"]
        true_col = cat["true_col"]
        pred_col = cat["pred_col"]
        curves_by_cat[name], stats_by_cat[name] = gel.calculate_roc(input_df, true_col, pred_col)
        all_cat_names.append(name)

        num_true = input_df[true_col].sum()
        num_total = len(input_df[true_col] >= 0)
        print(f"{name}: {num_true} / {num_total} = {num_true / num_total:.2%}")

    sns.set_theme(style="darkgrid")
    # Preconfigured values: {paper, notebook, talk, poster}
    sns.set_context("notebook", font_scale=1.0)

    # ROC Curves
    plot_roc_curves = True
    if plot_roc_curves:
        fig, ax = plot_roc_prc(curves_by_cat, stats_by_cat, f"{ds_name} Validation")
        pdf_pages.savefig(fig)
        plt.close(fig)

    # Plot binary metrics by threshold
    all_metrics_df = gel.calc_all_metrics(curves_by_cat, split_col=split_col)
    for cat in categories:
        split_name = cat["name"]
        true_col = cat["true_col"]
        pred_col = cat["pred_col"]
        fig_name = f"{split_name}"

        metrics_df = all_metrics_df
        if split_col is not None:
            metrics_df = all_metrics_df[all_metrics_df[split_col] == split_name]

        figures = gel.plot_binary_metrics(metrics_df, fig_name)

        for fig in figures:
            pdf_pages.savefig(fig)
            plt.close(fig)

        # Histogram and boxplot of predictions by actual class
        split_df = input_df
        fig = gel.plot_histograms(split_df, true_col, pred_col, fig_name)
        pdf_pages.savefig(fig)
        plt.close(fig)

    summary_metrics_by_cat_standard  = generate_standards_df(all_metrics_df, standards, categories, split_col)
    plot_summary_tables_on_pdf(pdf_pages, summary_metrics_by_cat_standard, split_col)

    pdf_pages.close()

if True and __name__ == "__main__":
    main()

