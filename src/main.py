#!/usr/bin/env python

__doc__ = """
Run set of evaluations on collected data validation.
We expect a datasheet of patient and exam information.
"""

import argparse
import os

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import general_eval_lib as gel
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

if __name__ == "__main__":
    input_path = "/Users/silterra/chem_home/Sybil/nlst_predictions/sybil_ensemble_calibrated_v2.csv"
    split_col = "Year"
    input_name = os.path.basename(input_path)
    input_df = gel.load_input_df(input_name, input_path)

    print(f"Loaded {len(input_df)} rows from {input_path}")

    output_path = "nlst_gen_eval_debug.pdf"
    pdf_pages = PdfPages(output_path)

    # TODO Convert to per-year, ie mask properly
    cat_name = "Year"
    categories = [
        {"name": "Year 1", "pred_col": "Year1", "true_col": "golds"},
        {"name": "Year 2", "pred_col": "Year2", "true_col": "golds"},
        {"name": "Year 3", "pred_col": "Year3", "true_col": "golds"},
        {"name": "Year 4", "pred_col": "Year4", "true_col": "golds"},
        {"name": "Year 5", "pred_col": "Year5", "true_col": "golds"},
        {"name": "Year 6", "pred_col": "Year6", "true_col": "golds"}
    ]

    curves_by_cat = {}
    stats_by_cat = {}
    all_cat_names = []
    for cat in categories:
        name = cat["name"]
        true_col = cat["true_col"]
        pred_col = cat["pred_col"]
        curves_by_cat[name], stats_by_cat[name] = gel.calculate_roc(input_df, true_col, pred_col)
        all_cat_names.append(name)

    sns.set_theme(style="darkgrid")
    # Preconfigured values: {paper, notebook, talk, poster}
    sns.set_context("notebook", font_scale=1.0)

    # ROC Curves
    plot_roc_curves = True
    if plot_roc_curves:
        fig, ax = plot_roc_prc(curves_by_cat, stats_by_cat, "NLST")
        pdf_pages.savefig(fig)
        plt.close(fig)

    # Plot binary metrics by threshold
    all_metrics_df = gel.calc_all_metrics(curves_by_cat, split_col=split_col)
    for cat in categories:
        split_name = cat["name"]
        true_col = cat["true_col"]
        pred_col = cat["pred_col"]
        fig_name = f"{split_name}"
        metrics_df = all_metrics_df[all_metrics_df[split_col] == split_name]
        figures = gel.plot_binary_metrics(metrics_df, fig_name)

        for fig in figures:
            pdf_pages.savefig(fig)
            plt.close(fig)

        # split_df = input_df[input_df[split_col] == split_name]
        # Histogram and boxplot of predictions by actual class
        split_df = input_df
        fig = gel.plot_histograms(split_df, true_col, pred_col, fig_name)
        pdf_pages.savefig(fig)
        plt.close(fig)

    # Print out table of thresholds and stats
    standards = [
        {"name": "F1", "metric": "f1_score", "direction": "max"},
        {"name": "Balanced Accuracy", "metric": "balanced_accuracy", "direction": "max"},
    ]

    summary_metrics_by_cat_standard = []
    for standard in standards:
        # print(f"Thresholds for {standard['name']}")
        for cat in categories:
            split_name = cat["name"]
            true_col = cat["true_col"]
            pred_col = cat["pred_col"]
            split_df = all_metrics_df.query(f"{split_col} == '{split_name}'")
            split_df = split_df.sort_values(standard["metric"], ascending=standard["direction"] == "min")
            best_row = split_df.iloc[0].copy()
            best_row["standard"] = standard["name"]
            summary_metrics_by_cat_standard.append(best_row)

    summary_metrics_by_cat_standard = pd.DataFrame(summary_metrics_by_cat_standard).reset_index(drop=True)
    for standard in standards:
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(24, 3))
        # ax.axis('tight')
        ax.axis('off')
        ax.set_position([0, 0, 1, 1])

        df = summary_metrics_by_cat_standard.query(f"standard == '{standard['name']}'")
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
        row_height = 0.95/(1.0 + df.shape[0])
        col_width = 0.95/df.shape[1]
        table.auto_set_font_size(False)
        table.set_fontsize(main_fontsize)
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_text_props(fontweight='bold')
            cell.set_height(row_height)
            cell.set_width(col_width)

        ax.add_table(table)

        plt.title(f"Best Thresholds by {standard['name']}")
        pdf_pages.savefig(fig)
        # plt.show()

    pdf_pages.close()


