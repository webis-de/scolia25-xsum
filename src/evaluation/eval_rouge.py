"""
Script for calculating ROUGE scores between reference and candidate texts.

This module computes ROUGE-1, ROUGE-2, and ROUGE-L metrics (precision, recall, and F1)
to evaluate the quality of text summarization. The script accepts a dataframe
containing reference and candidate text columns and returns the original dataframe
with additional columns for ROUGE scores.
"""

import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm
import os

# Initialize tqdm for pandas apply to show progress bars
tqdm.pandas()


def compute_rouge_scores(
    df: pd.DataFrame, reference_col: str, candidate_col: str
) -> pd.DataFrame:
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1 scores for each row in the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing reference and candidate texts.
        reference_col (str): The name of the column containing reference texts.
        candidate_col (str): The name of the column containing generated/candidate texts.

    Returns:
        pd.DataFrame: The original dataframe with additional columns for ROUGE scores.
    """
    # Initialize the ROUGE scorer with the metrics we want to compute
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Function to compute ROUGE scores for a single row
    def compute_rouge(row):
        reference = row[reference_col]
        generated = row[candidate_col]
        # Calculate all ROUGE scores between the reference and generated text
        scores = scorer.score(reference, generated)
        # Return a dictionary with all individual metrics
        return {
            "rouge1_precision": scores["rouge1"].precision,
            "rouge1_recall": scores["rouge1"].recall,
            "rouge1_fmeasure": scores["rouge1"].fmeasure,
            "rouge2_precision": scores["rouge2"].precision,
            "rouge2_recall": scores["rouge2"].recall,
            "rouge2_fmeasure": scores["rouge2"].fmeasure,
            "rougeL_precision": scores["rougeL"].precision,
            "rougeL_recall": scores["rougeL"].recall,
            "rougeL_fmeasure": scores["rougeL"].fmeasure,
        }

    # Compute ROUGE scores for each row with a progress bar
    rouge_scores = df.progress_apply(compute_rouge, axis=1)

    # Convert the series of dictionaries to a dataframe
    rouge_df = pd.DataFrame(list(rouge_scores))

    # Concatenate the ROUGE scores with the original dataframe
    df = pd.concat([df, rouge_df], axis=1)

    return df


