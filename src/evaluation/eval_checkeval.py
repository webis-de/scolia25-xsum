"""
This script evaluates text summaries using the CheckEval framework with both reference-guided
and criterion-guided evaluation approaches. It processes a dataset of survey sections and their
edited versions, computing quality scores based on specified criteria.

The script contains functions to:
1. Compute reference-guided evaluation scores
2. Compute criterion-guided evaluation scores 
3. Apply both evaluation methods to a dataframe of text pairs
"""

import pandas as pd
import re
from phi3 import Phi3Client
from pathlib import Path
from typing import Optional, Dict
from pydantic import ValidationError
from tqdm import tqdm

# Import the necessary classes from the main check_eval script
from check_eval import Checkeval, Checklist, ChecklistResponse

# Initialize tqdm for pandas progress tracking
tqdm.pandas()


def compute_reference_guided_eval(
    row, checkeval_instance: Checkeval, criterion: str
) -> Optional[float]:
    """
    Computes the Reference-Guided evaluation score for a single row.

    Parameters:
        row (pd.Series): A row from the dataframe containing reference and candidate texts.
        checkeval_instance (Checkeval): An instance of the Checkeval class to perform evaluation.
        criterion (str): The evaluation criterion (e.g., 'consistency').

    Returns:
        Optional[float]: The overall evaluation score as a percentage, or NaN if evaluation failed.
    """
    # Extract the reference and candidate texts from the dataframe row
    reference_text = row.get("section_text_in_survey", "").strip()
    candidate_text = row.get("editor_version", "").strip()

    # Perform Reference-Guided Evaluation using the CheckEval framework
    result = checkeval_instance.reference_guided(
        criterion, reference_text, candidate_text
    )

    # Check if we have valid results before calculating the score
    if result["checklist"] and result["results"]:
        overall_score = result["results"].score() * 100  # Convert to percentage
        return overall_score
    else:
        return float("nan")  # Return NaN if evaluation failed


def compute_criterion_guided_eval(
    row, checkeval_instance: Checkeval, criterion: str
) -> Optional[float]:
    """
    Computes the Criterion-Guided evaluation score for a single row.

    Parameters:
        row (pd.Series): A row from the dataframe containing reference and candidate texts.
        checkeval_instance (Checkeval): An instance of the Checkeval class to perform evaluation.
        criterion (str): The evaluation criterion (e.g., 'consistency').

    Returns:
        Optional[float]: The overall evaluation score as a percentage, or NaN if evaluation failed.
    """
    # Extract the reference and candidate texts from the dataframe row
    reference_text = row.get("section_text_in_survey", "").strip()
    candidate_text = row.get("editor_version", "").strip()

    # Perform Criterion-Guided Evaluation using the CheckEval framework
    result = checkeval_instance.criterion_guided(
        criterion, reference_text, candidate_text
    )

    # Check if we have valid results before calculating the score
    if result["checklist"] and result["results"]:
        overall_score = result["results"].score() * 100  # Convert to percentage
        return overall_score
    else:
        return float("nan")  # Return NaN if evaluation failed


def compute_check_eval_scores(
    df: pd.DataFrame, criterion: str = "consistency"
) -> pd.DataFrame:
    """
    Computes both Reference-Guided and Criterion-Guided evaluation scores for each row in the dataframe.

    Parameters:
        df (pd.DataFrame): The input dataframe containing reference and candidate texts.
        criterion (str): The evaluation criterion (default is 'consistency').

    Returns:
        pd.DataFrame: The original dataframe with additional columns for evaluation scores.
    """
    # Initialize Checkeval instance for text evaluation
    checkeval = Checkeval()

    # Compute Reference-Guided Evaluation Scores with progress bar
    print("\nComputing Reference-Guided Evaluation Scores...")
    df["checkeval_reference_guided"] = df.progress_apply(
        lambda row: compute_reference_guided_eval(row, checkeval, criterion), axis=1
    )

    # Compute Criterion-Guided Evaluation Scores with progress bar
    print("Computing Criterion-Guided Evaluation Scores...")
    df["checkeval_criterion_guided"] = df.progress_apply(
        lambda row: compute_criterion_guided_eval(row, checkeval, criterion), axis=1
    )

    return df

