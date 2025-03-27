"""
This script evaluates the quality of generated texts by calculating the recall score based on
bibliographic references. It detects references in the format BIBREFXXX or [BIBREFXXX] in both
reference and candidate texts, then computes the proportion of references from the reference text 
that appear in the candidate text.
"""

import pandas as pd
import re
from typing import Set, Tuple, Optional
from tqdm import tqdm
import logging
import os

# Initialize tqdm for pandas apply
tqdm.pandas()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_references(text: str) -> Set[str]:
    """
    Extracts all references in the format BIBREFXXX or [BIBREFXXX] from the given text.

    Args:
        text (str): The text from which to extract references.

    Returns:
        Set[str]: A set of unique references found in the text, normalized without brackets.
    """
    # Regular expression pattern to match BIBREFXXX or [BIBREFXXX]
    pattern = r"\[?(BIBREF\d+)\]?"
    references = re.findall(pattern, text)
    return set(references)  # Return unique references


def calculate_recall(reference_text: str, candidate_text: str) -> float:
    """
    Calculates the Recall based on references used in the candidate text
    compared to the reference text.

    Args:
        reference_text (str): The ground truth reference text containing all valid references.
        candidate_text (str): The generated candidate text containing references.

    Returns:
        float: The Recall score between 0.0 and 1.0.
    """
    # Extract references from both texts
    reference_references = extract_references(reference_text)
    candidate_references = extract_references(candidate_text)

    logging.debug(f"Total References in Reference Text: {len(reference_references)}")
    logging.debug(f"Total References in Candidate Text: {len(candidate_references)}")

    # Calculate the number of correctly identified references
    correct_references = reference_references.intersection(candidate_references)
    num_correct = len(correct_references)

    logging.debug(f"Correctly Identified References: {num_correct}")

    # Calculate Recall (avoid division by zero)
    recall = num_correct / len(reference_references) if reference_references else 0.0

    return recall


def compute_reference_f1_recall(
    df: pd.DataFrame, reference_col: str, candidate_col: str
) -> pd.DataFrame:
    """
    Computes the Recall based on references for each row in the dataframe and appends it as a new column.

    Args:
        df (pd.DataFrame): The input dataframe containing reference and candidate texts.
        reference_col (str): The name of the column containing reference texts.
        candidate_col (str): The name of the column containing generated/candidate texts.

    Returns:
        pd.DataFrame: The original dataframe with an additional column for Reference F1 Recall.
    """
    logging.info("Starting Reference F1 Recall computation...")

    def calculate_row_recall(row) -> Optional[float]:
        try:
            # Extract text from specified columns
            reference_text = row.get(reference_col, "").strip()
            candidate_text = row.get(candidate_col, "").strip()

            # Skip computation if reference text is missing
            if not reference_text:
                logging.warning(f"Missing reference text for row index {row.name}.")
                return float("nan")

            # Calculate recall for this row
            recall = calculate_recall(reference_text, candidate_text)
            return recall

        except Exception as e:
            logging.error(f"Error computing recall for row index {row.name}: {e}")
            return float("nan")

    # Apply the recall calculation with a progress bar
    df["reference_f1_recall"] = df.progress_apply(calculate_row_recall, axis=1)

    logging.info(
        "Reference F1 Recall computation completed and added to the dataframe."
    )

    return df


