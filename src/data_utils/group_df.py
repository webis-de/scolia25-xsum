"""
This script groups a DataFrame by survey metadata, creating dictionaries for each paper's
content with citation_id as keys. It's designed to pre-process survey data for the
SurveySum dataset.
"""

import pandas as pd
import sys
import os


def load_data(df_path: str) -> pd.DataFrame:
    """
    Load the DataFrame from the specified path.

    Parameters
    ----------
    df_path : str
        Path to the DataFrame file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    try:
        df = pd.read_pickle(df_path)
        print(f"Loaded DataFrame from {df_path}.")
        return df
    except FileNotFoundError:
        print(f"Error: DataFrame file not found at {df_path}.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        sys.exit(1)


def save_data(df: pd.DataFrame, output_path: str):
    """
    Save the DataFrame to the specified output path.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_path : str
        Path to save the DataFrame.
    """
    try:
        df.to_pickle(output_path)
        print(f"Saved DataFrame to {output_path}.")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")
        sys.exit(1)


def group_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group the DataFrame by 'survey_id', 'survey_title', and 'section_title',
    creating dictionaries for specified columns with 'citation_id' as keys.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.

    Returns
    -------
    pd.DataFrame
        New grouped DataFrame with dictionaries.
    """
    # Columns to group into dictionaries with citation_id as keys
    columns_to_group = [
        "abstract",
        "title",
        "paper_id",
        "full_text",
        "questions_core",
        "answers",
    ]

    # Columns to use as grouping criteria
    group_by_columns = ["survey_id", "survey_title", "section_title"]

    # Validate presence of required columns
    missing_group_cols = [col for col in group_by_columns if col not in df.columns]
    if missing_group_cols:
        print(f"Error: Missing grouping columns: {missing_group_cols}")
        sys.exit(1)

    missing_data_cols = [col for col in columns_to_group if col not in df.columns]
    if missing_data_cols:
        print(
            f"Warning: Missing data columns: {missing_data_cols}. These columns will be skipped."
        )

    # Initialize a list to hold grouped data records
    grouped_list = []

    # Group the DataFrame by survey metadata
    grouped = df.groupby(group_by_columns)
    total_groups = len(grouped)
    print(f"Total groups to process: {total_groups}")

    for group_keys, group_df in grouped:
        group_dict = {}

        # Assign group-by column values to the output dictionary
        for idx, col in enumerate(group_by_columns):
            group_dict[col] = group_keys[idx] if len(group_keys) > 1 else group_keys

        # For each data column, create a dictionary with citation_id as keys
        for column in columns_to_group:
            if column not in df.columns:
                continue  # Skip missing columns

            # Ensure citation_id exists for dictionary creation
            if "citation_id" not in group_df.columns:
                print("Error: 'citation_id' column not found in the DataFrame.")
                sys.exit(1)

            # Check for duplicate citation IDs which could cause data loss
            if not group_df["citation_id"].is_unique:
                print(
                    f"Warning: Duplicate 'citation_id' found in group {group_keys}. Overwriting duplicates."
                )

            # Create dictionary mapping citation_id to column values
            column_dict = group_df.set_index("citation_id")[column].to_dict()
            group_dict[f"{column}_dict"] = column_dict
            print(
                f"Grouped column '{column}' into a dictionary for group {group_keys}."
            )

        grouped_list.append(group_dict)

    # Create a new DataFrame from the grouped data
    new_df = pd.DataFrame(grouped_list)
    print("Grouped DataFrame created successfully.")

    return new_df


def main():
    # Define relative paths for input and output files
    base_dir = "data/surveysum/processed"
    data_path = os.path.join(
        base_dir, "sample_source_docs_full_with_questions_answers.pkl"
    )
    output_path = os.path.join(
        base_dir, "sample_source_docs_full_questions_answers_grouped.pkl"
    )

    # Load the original DataFrame
    df = load_data(data_path)

    # Group the DataFrame by survey metadata
    grouped_df = group_dataframe(df)

    # Display a preview of the grouped DataFrame
    print("Grouped DataFrame Preview:")
    print(grouped_df.head())

    # Save the processed grouped DataFrame
    save_data(grouped_df, output_path)

    print("DataFrame grouping completed successfully.")


if __name__ == "__main__":
    main()
