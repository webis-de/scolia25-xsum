"""
This script evaluates editor-generated survey content against ground truth texts 
using a Phi3 language model. It computes a score from 1.0 to 5.0 for each piece
of content based on how well it captures the information in the ground truth.

It handles loading data, generating prompts, extracting scores from
model responses, and saving the evaluation results.
"""

import pandas as pd
import re
from phi3 import Phi3Client
from tqdm import tqdm
from typing import Optional

# Initialize tqdm for pandas apply (adds progress bars to dataframe operations)
tqdm.pandas()


def generate_g_eval_prompt(
    survey_title: str, survey_section: str, generated_content: str, ground_truth: str
) -> str:
    """
    Generates the evaluation prompt for g-eval.

    Parameters:
        survey_title (str): The title of the survey.
        survey_section (str): The survey section to evaluate.
        generated_content (str): The generated content to evaluate.
        ground_truth (str): The ground truth text for comparison.

    Returns:
        str: The formatted evaluation prompt.
    """
    prompt = (
        "# CONTEXT\n"
        "You will be given a text for a survey section written by an editor and the ground truth text written by the main author.\n"
        "# INSTRUCTIONS\n"
        "Your task is to rate the content of the survey section on one metric comparing the editor's text with the ground truth which has the maximum score. Please make sure you read and understand the instructions carefully. Please keep the document open while reviewing, and refer to it as needed.\n"
        "# Evaluation Steps.\n"
        "1 - Carefully read the content to identify the main topic and key points.\n"
        "2 - Evaluate whether the content adequately addresses the main topic stated in the title and provides a comprehensive technical description of it.\n"
        "3 - Assign a score to the text on a scale of 1 to 5, where 1 represents the lowest score and 5 represents the highest score, according to the Evaluation Criteria.\n"
        f"Survey Name: {survey_title}\n"
        f"Survey Section: {survey_section}\n"
        f"Editor's Text: {generated_content}\n"
        f"Ground Truth Text: {ground_truth}\n"
        "Evaluation Form (scores ONLY - numbers from 1.0 to 5.0)\n"
        "# Score (1.0 - 5.0):"
    )
    return prompt


def compute_g_eval_score(row, client: Phi3Client, prompt_func) -> float:
    """
    Computes the g-eval score for a single row.

    Parameters:
        row (pd.Series): A row from the dataframe.
        client (Phi3Client): An instance of Phi3Client to interact with the evaluation model.
        prompt_func (callable): A function to generate the evaluation prompt.

    Returns:
        float: The evaluation score between 1.0 and 5.0. Returns NaN if extraction fails.
    """
    # Extract required fields from the dataframe row
    survey_title = row.get("survey_title", "N/A").strip()
    survey_section = row.get("section_title", "N/A").strip()
    generated_content = row.get("editor_version", "N/A").strip()
    ground_truth = row.get("section_text_in_survey", "N/A").strip()

    # Generate the evaluation prompt using the provided function
    prompt = prompt_func(survey_title, survey_section, generated_content, ground_truth)

    try:
        # Send the prompt to Phi3Client and get the response
        response = client.get_response(prompt)

        # Extract the float score from the response text using regex
        match = re.search(r"\d+\.\d+", response)
        if match:
            score = float(match.group())
            return score
        else:
            # Log warning if no score is found in the response
            print(
                f"Warning: No float score found in response for row index {row.name}. Response: {response}"
            )
            return float("nan")
    except Exception as e:
        # Handle any unexpected errors during processing
        print(f"Error processing row index {row.name}: {e}")
        return float("nan")


def compute_g_eval_scores(
    df: pd.DataFrame, client: Optional[Phi3Client] = None
) -> pd.DataFrame:
    """
    Computes g-eval scores for each row in the dataframe and appends them as a new column.

    Parameters:
        df (pd.DataFrame): The input dataframe containing reference and candidate texts.
        client (Optional[Phi3Client]): An instance of Phi3Client. If None, initializes a new client.

    Returns:
        pd.DataFrame: The original dataframe with an additional column for g-eval scores.
    """
    # Initialize Phi3Client if not provided
    if client is None:
        system_prompt = "You are an experienced evaluator reviewing a survey section."
        client = Phi3Client(system_prompt=system_prompt)

    # Define the prompt generation function
    prompt_func = generate_g_eval_prompt

    # Compute g-eval scores for each row with a progress bar
    df["g_eval"] = df.progress_apply(
        lambda row: compute_g_eval_score(row, client, prompt_func), axis=1
    )

    return df
