#!/usr/bin/env python

__doc__ = """
Run set of evaluations on collected data validation.
We expect a datasheet of patient and exam information.
"""

import argparse
import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import general_eval.general_eval_lib as gel
from general_eval.general_eval_lib import plot_roc_prc

DAYS_PER_YEAR = 365

DIAGNOSIS_DAYS_COL = "Days_to_Cancer"
FOLLOWUP_DAYS_COL = "Days_Followup"


def _get_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--ds_name", default="My Dataset", required=False,
                        type=str, help="Name of dataset. Used in figure titles.")

    parser.add_argument("--input_path", default=None, required=True,
                        type=str, help="Path to the file containing information on a per-exam basis"
                                       "(one exam per row). CSV format.")

    parser.add_argument("--output_path", default=None,
                        type=str, help="Path to save the output report file.")

    parser.add_argument("--split_col", default="Year",
                        type=str, help="Column to use for subdividing data into subsets. "
                                       "Analysis will be repeated across each unique value of this column.")

    parser.add_argument("--recall_target", default=0.85,
                        type=str, help="Target recall value for the model.")

    return parser

def _is_positive_diag(days):
    if days is None:
        return False
    elif str(days).lower() in {"-1", "-1.0", "nan", "nat", "", "none"}:
        return False
    else:
        try:
            return float(days) >= 0
        except ValueError:
            return False

def _is_negative_diag(days):
    return not _is_positive_diag(days)

def _days_to_months(days):
    if _is_negative_diag(days):
        return -1
    return int(days) / 30

def _days_to_years(days):
    if _is_negative_diag(days):
        return -1
    return int(days) / DAYS_PER_YEAR

def _gen_year_vec(row, diagnosis_days_col, followup_days_col, max_followup=5):
    """
    Generate a binary vector indicating whether the patient has been diagnosed with cancer within the given year.
    :param row:
    :param diagnosis_days_col:
    :param followup_days_col:
    :param max_followup:
    :return:
        y_seq - Indicator vector of whether the patient has been diagnosed with cancer within the given year.
            -1 indicates that the patient is not available for followup at that time.
            0 indicates that the patient has not been diagnosed with cancer at that time.
            1 indicates that the patient has been diagnosed with cancer at that time.
        y_mask - Indicator vector of whether the patient is available for followup at that time.
    """
    days_to_last_followup = int(row[followup_days_col])
    years_to_last_followup = int(days_to_last_followup // DAYS_PER_YEAR)

    days_to_cancer = int(row[diagnosis_days_col])
    years_to_cancer = int(days_to_cancer // DAYS_PER_YEAR) if days_to_cancer > -1 else 100

    y = years_to_cancer <= max_followup
    y_seq = np.zeros(max_followup)
    if y:
        # time_at_event = years_to_cancer
        y_seq[years_to_cancer:] = 1
        y_mask = np.ones(max_followup)
    else:
        time_at_event = min(years_to_last_followup, max_followup - 1)
        y_mask = np.array([1] * (time_at_event + 1) + [0] * (max_followup - (time_at_event + 1)))

    for ii, yv in enumerate(y_seq):
        y_seq[ii] = y_seq[ii] if y_mask[ii] else -1

    return y_seq, y_mask


def generate_true_columns(input_df, diagnosis_days_col, followup_days_col, true_prefix="true", num_years=5):

    input_df["__interval_months"] = input_df[diagnosis_days_col].apply(_days_to_months)

    year_cols = [f"{true_prefix}_year{y}" for y in range(1, num_years+1)]
    for rn, row in input_df.iterrows():
        y_seq, y_mask = _gen_year_vec(row, diagnosis_days_col, followup_days_col, max_followup=num_years)
        input_df.loc[rn, year_cols] = y_seq

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
    fig, ax = plt.subplots(figsize=(8, 3))  # Adjust the size as needed
    # Create the table
    table = plt.table(cellText=basic_stats_df.values,
                      colLabels=basic_stats_df.columns,
                      cellLoc='center', loc='center')
    df = basic_stats_df
    col_width = 1.0 / df.shape[1]
    col_height = 0.1
    # table.auto_set_font_size(False)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
        cell.set_width(col_width)
        cell.set_height(col_height)

    ax.add_table(table)
    plt.title("Patient Counts by Time Interval", pad=16, fontweight='bold')
    plt.subplots_adjust(top=0.80)
    ax.axis('tight')
    ax.axis('off')

    return fig, basic_stats_df

def create_perf_stats_table(stats_by_cat):
    perf_df = pd.DataFrame(stats_by_cat).T
    perf_df = perf_df[["auc", "pr_auc", "N"]]
    perf_df.columns = ["AUROC", "AUPRC", "N"]
    perf_df = perf_df.round(4)
    perf_df["N"] = perf_df["N"].astype(int)
    perf_df = perf_df.reset_index()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 3))  # Adjust the size as needed
    # Create the table
    df = perf_df
    table = plt.table(cellText=df.values,
                      colLabels=df.columns,
                      cellLoc='center', loc='center')
    col_width = 1.0 / df.shape[1]
    col_height = 0.1
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
        cell.set_width(col_width)
        cell.set_height(col_height)

    ax.add_table(table)
    plt.title("Overall Performance by Year", pad=16, fontweight='bold')
    plt.subplots_adjust(top=0.80)
    ax.axis('tight')
    ax.axis('off')

    return fig, perf_df

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
            "standard", split_col, "threshold", "sensitivity", "precision", "specificity",
            "LR+", "N", "TP", "FP", "TN", "FN"
        ]
        df = df[custom_column_order]
        custom_column_labels = list(df.columns)
        cust_mapping = [("f1_score", "F1"), ("balanced_accuracy", "Balanced Acc."), ("sensitivity", "Sensitivity")]
        for old, new in cust_mapping:
            if old in custom_column_labels:
                custom_column_labels[custom_column_labels.index(old)] = new
        # Capitalize column names
        custom_column_labels = [col.capitalize() if len(col) >= 4 else col.upper() for col in custom_column_labels]

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

def glossary_of_terms():
    # Print definitions of terms used in the report
    fig, ax = plt.subplots()
    ax.axis('off')

    ax.text(0.01, 0.95, "Glossary of Terms", fontsize=16, fontweight='bold')
    other_text = """
    Sensitivity: The proportion of true positive cases that are correctly 
    identified by the model. Also known as "Recall".
    
    Specificity: The proportion of true negative cases that are correctly 
    identified by the model. Also known as "True Negative Rate".
    
    Precision: The proportion of positive cases identified by the model 
    that are actually positive. Also known as "Positive Predictive Value".
    
    AUROC: Area Under the Receiver Operating Characteristic curve.
    A random classifier has an AUROC of 0.5, 
    while a perfect classifier has an AUROC of 1.0.
    
    AUPRC: Area Under the Precision-Recall curve.
    A random classifier has an AUPRC equal to the fraction of 
    positive cases in the dataset, 
    while a perfect classifier has an AUPRC of 1.0.
    
    LR+: The positive likelihood ratio, which is the ratio of the
    true positive rate to the false positive rate.
    """
    ax.text(0.01, 0.9, other_text, fontsize=12, ha="left", va="top")
    plt.tight_layout()

    return fig

def run_full_eval(ds_name, input_path, split_col="Year", sensitivity_target=0.85, output_path=None):
    # Column names
    diagnosis_days_col = DIAGNOSIS_DAYS_COL
    followup_days_col = FOLLOWUP_DAYS_COL
    # num_years = None
    required_columns = [diagnosis_days_col, followup_days_col]

    # Require diagnosis be at least 3 months after exam
    min_months = 0
    min_days = int(min_months*30)
    categories = [
        {"name": "Year 1", "pred_col": "Year1", "true_col": "true_year1"},
        {"name": "Year 2", "pred_col": "Year2", "true_col": "true_year2"},
        {"name": "Year 3", "pred_col": "Year3", "true_col": "true_year3"},
        {"name": "Year 4", "pred_col": "Year4", "true_col": "true_year4"},
        {"name": "Year 5", "pred_col": "Year5", "true_col": "true_year5"},
        {"name": "Year 6", "pred_col": "Year6", "true_col": "true_year6"},
    ]

    standards = [
        {"name": f"Sensitivity", "metric": "sensitivity", "direction": "min_diff", "target_value": sensitivity_target},
        # {"name": "F1", "metric": "f1_score", "direction": "max"},
        # {"name": "Balanced Accuracy", "metric": "balanced_accuracy", "direction": "max"},
    ]

    if output_path is None:
        output_path = os.path.join(os.path.dirname(input_path), f"{ds_name} evaluation report.pdf")

    input_name = os.path.basename(input_path)
    input_df = gel.load_input_df(input_name, input_path, comment="#")
    input_df = input_df.dropna(subset=[followup_days_col])
    # Remove spaces from column names
    input_df.columns = input_df.columns.str.replace(" ", "")

    keep_categories = [cat for cat in categories if cat["pred_col"] in input_df.columns]
    if len(keep_categories) == 0:
        category_names = [cat["pred_col"] for cat in categories]
        raise ValueError(f"Input file does not contain any of the expected categories: {category_names}")
    num_years = len(keep_categories)
    categories = keep_categories

    numeric_cols = [diagnosis_days_col, followup_days_col] + [cat["pred_col"] for cat in categories]
    input_df[numeric_cols] = input_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    for rc in required_columns:
        if rc not in input_df.columns:
            raise ValueError(f"Input file does not contain required column: {rc}")

    # Keep rows where the diagnosis is at least min_days after the exam
    def _keep_row(row, min_days=min_days):
        # Keep the negatives
        if _is_negative_diag(row[diagnosis_days_col]):
            return True

        # Only include cases where the diagnosis is at least min_days after the exam
        return int(row[diagnosis_days_col]) > min_days

    # Read days until diagnosis, generate binary columns for each year
    generate_true_columns(input_df, diagnosis_days_col, followup_days_col, num_years=num_years)

    if min_months is not None:
        keep_rows = input_df.apply(_keep_row, axis=1)
        # print(f"Keeping {keep_rows.sum()} / {len(keep_rows)} rows with valid exam/diagnosis dates")
        input_df = input_df.loc[keep_rows, :]

    # Calculate ROC curves and binary metrics, separated by category (ie year)
    curves_by_cat = {}
    stats_by_cat = {}
    all_cat_names = []
    for cat in categories:
        name = cat["name"]
        true_col = cat["true_col"]
        pred_col = cat["pred_col"]
        cur_df = input_df.query(f"{true_col} >= 0")

        curves_by_cat[name], stats_by_cat[name] = gel.calculate_roc(cur_df, true_col, pred_col)
        all_cat_names.append(name)

    all_metrics_df = gel.calc_all_metrics(curves_by_cat, split_col=split_col)

    # ------------ Plotting ------------ #
    sns.set_theme(style="darkgrid")
    # Preconfigured values: {paper, notebook, talk, poster}
    sns.set_context("notebook", font_scale=1.0)
    pdf_pages = PdfPages(output_path)

    # Basic stats table (counts of patients within specific time windows)
    fig, basic_stats_df = create_basic_stats_table(input_df, diagnosis_days_col)

    pdf_pages.savefig(fig)
    plt.close(fig)

    # Performance statistics (AUC, ROC) by category
    fig, overall_perf_df = create_perf_stats_table(stats_by_cat)
    pdf_pages.savefig(fig)
    plt.close(fig)

    # Tables of thresholds for particular targets
    summary_metrics_by_cat_standard = generate_standards_df(all_metrics_df, standards, categories, split_col)
    plot_summary_tables_on_pdf(pdf_pages, summary_metrics_by_cat_standard, split_col)

    # Plot ROC Curves
    plot_roc_curves = True
    if plot_roc_curves:
        fig, ax = plot_roc_prc(curves_by_cat, stats_by_cat, f"{ds_name} Validation")
        pdf_pages.savefig(fig)
        plt.close(fig)

    fig = glossary_of_terms()
    pdf_pages.savefig(fig)
    plt.close(fig)

    # Plot binary metrics by threshold
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


    pdf_pages.close()

    # print(f"Saved evaluation report to {output_path}")
    return output_path, all_metrics_df


if True and __name__ == "__main__":
    args= _get_parser().parse_args()
    run_full_eval(args.ds_name, args.input_path, split_col=args.split_col, sensitivity_target=args.recall_target,
                  output_path=args.output_path)
