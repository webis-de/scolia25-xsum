"""
This script generates answers to questions about citations using RAG (Retrieval-Augmented Generation).
It loads data with questions, processes each question through a RAG chatbot, and saves
the resulting answers back to the dataset.
"""

import pandas as pd
import logging
from tqdm import tqdm  
from citation_rag import RAGChatbot

# Use relative paths for better portability
data_path = "PLACEHOLDER_DATA_PATH"  # Placeholder for the actual data path
output_path = "PLACEHOLDER_OUTPUT_PATH"  # Placeholder for the output path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(df_path):
    """
    Load the DataFrame from the specified path.

    Parameters:
    - df_path (str): Path to the DataFrame file.

    Returns:
    - df (pd.DataFrame): Loaded DataFrame.
    """
    try:
        df = pd.read_pickle(df_path)
        logging.info(f"Loaded DataFrame from {df_path}.")
    except FileNotFoundError:
        logging.error(f"DataFrame file not found at {df_path}.")
        return None
    except Exception as e:
        logging.error(f"Error loading DataFrame: {e}")
        return None

    return df


def save_data(df, output_path):
    """
    Save the DataFrame to the specified output path.

    Parameters:
    - df (pd.DataFrame): DataFrame to save.
    - output_path (str): Path to save the DataFrame.
    """
    try:
        df.to_pickle(output_path)
        logging.info(f"Saved DataFrame to {output_path}.")
    except Exception as e:
        logging.error(f"Error saving DataFrame: {e}")


def main():
    """Main execution function for the answer generation pipeline."""
    # Load DataFrame with questions
    df = load_data(data_path)
    if df is None:
        return

    # Initialize RAG chatbot for answering questions
    expert = RAGChatbot()

    # Add a new column to store the generated answers
    df["answers"] = None

    # Process each row in the DataFrame with a progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        questions_dict = row.get("questions", {})

        # Validate questions format
        if not isinstance(questions_dict, dict):
            logging.error(
                f"Expected 'questions' to be a dict at index {index}, but got {type(questions_dict)}."
            )
            df.at[index, "answers"] = {}
            continue

        if not questions_dict:
            logging.warning(
                f"No questions found in 'questions' column at index {index}."
            )
            df.at[index, "answers"] = {}
            continue

        # Initialize answers container for this row
        answers = {}

        # Process each citation's questions
        for citation_id, questions in questions_dict.items():
            if not isinstance(questions, list):
                logging.error(
                    f"Expected a list of questions for citation_id '{citation_id}' at index {index}, but got {type(questions)}."
                )
                answers[citation_id] = {}
                continue

            if not questions:
                logging.warning(
                    f"No questions listed for citation_id '{citation_id}' at index {index}."
                )
                answers[citation_id] = {}
                continue

            answers[citation_id] = {}

            # Generate answers for each question
            for question in questions:
                try:
                    # Use RAG to answer the question
                    answer = expert.answer_question(question, citation_id)
                    answers[citation_id][question] = answer

                    # Log successful answers
                    if (
                        "do not contain" not in answer
                        and "unable to provide" not in answer
                    ):
                        logging.info(
                            f"Successfully answered question for citation '{citation_id}': {question}"
                        )
                    else:
                        logging.warning(
                            f"No answer found for citation '{citation_id}': {question}"
                        )

                except Exception as e:
                    logging.error(
                        f"Error answering question '{question}' for citation_id '{citation_id}' at index {index}: {e}"
                    )
                    answers[citation_id][question] = None

        # Store all answers for this row in the DataFrame
        df.at[index, "answers"] = answers

    # Save the enhanced DataFrame with answers
    save_data(df, output_path)


if __name__ == "__main__":
    main()
