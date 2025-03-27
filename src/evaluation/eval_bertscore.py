"""
This script evaluates the semantic similarity between reference and candidate texts using BERTScore.
It computes the BERTScore recall metric, which is particularly useful for evaluating text summaries.
The script reads a pandas DataFrame, computes BERTScore metrics, and adds the results as a new column.
"""

import pandas as pd
from bert_score import score
from tqdm import tqdm
import logging
import os

# Initialize tqdm for pandas apply
tqdm.pandas()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def compute_bert_recall(
    df: pd.DataFrame,
    reference_col: str,
    candidate_col: str,
    lang: str = "en",
    model_type: str = "bert-base-uncased",
) -> pd.DataFrame:
    """
    Computes BERTScore recall for each row in the dataframe and appends it as a new column.

    Args:
        df (pd.DataFrame): The input dataframe containing reference and candidate texts.
        reference_col (str): The name of the column containing reference texts.
        candidate_col (str): The name of the column containing generated/candidate texts.
        lang (str): Language of the texts. Default is "en" (English).
        model_type (str): The model type to use for BERTScore. Default is "bert-base-uncased".

    Returns:
        pd.DataFrame: The original dataframe with an additional column for BERTScore recall.
    """
    try:
        logging.info("Extracting candidate and reference texts...")
        cands = df[candidate_col].tolist()  # Extract candidate texts
        refs = df[reference_col].tolist()   # Extract reference texts

        logging.info(
            f"Computing BERTScore for {len(cands)} pairs using model '{model_type}'..."
        )
        # Calculate precision, recall, and F1 scores
        P, R, F = score(cands, refs, lang=lang, model_type=model_type, verbose=True)

        logging.info("Appending BERTScore recall to the dataframe...")
        df["bert_score"] = R.tolist()  # Add recall scores as a new column

        return df

    except Exception as e:
        logging.error(f"An error occurred while computing BERTScore: {e}")
        raise


