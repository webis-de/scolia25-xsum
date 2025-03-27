"""
This script processes the SurveySum dataset by extracting information about individual
papers cited in survey papers. It transforms a dataset where each row represents a survey
into a dataset where each row represents a cited document with its metadata.

The script reads a DataFrame containing survey papers with abstract collections and
transforms it into a flattened structure focusing on individual source documents.
"""

import os
import pandas as pd

# Define relative paths
BASE_DIR = "/datadrive/projects/autosummary_plus/main"
INPUT_FILE = os.path.join(
    BASE_DIR, "data/surveysum/train_with_abstracts_full_text_with_questions.pkl"
)
OUTPUT_FILE = os.path.join(BASE_DIR, "data/surveysum/processed/source_docs_full.pkl")

# Load the dataset containing surveys with their cited papers
train_with_abstracts_df = pd.read_pickle(INPUT_FILE)

# Uncomment to use a smaller sample for testing
# train_with_abstracts_df = train_with_abstracts_df.head(10)

# Initialize an empty list to collect rows for the new DataFrame
data_for_new_df = []

# Iterate over each survey paper in the original DataFrame
for _, row in train_with_abstracts_df.iterrows():
    # Extract metadata collections from the survey row
    abstracts = row["abstracts"]
    titles = row["titles"]
    paper_ids = row["paper_ids"]
    full_text = row["full_text"]
    survey_id = row["survey_id"]
    survey_title = row["survey_title"]
    section_title = row["section_title"]

    # Process each citation from this survey
    for citation_id in abstracts.keys():
        # Create a dictionary mapping each citation to its metadata and source survey
        entry = {
            "abstract": abstracts.get(citation_id),
            "title": titles.get(citation_id),
            "paper_id": paper_ids.get(citation_id),
            "full_text": full_text.get(citation_id),
            "citation_id": citation_id,
            "survey_id": survey_id,
            "survey_title": survey_title,
            "section_title": section_title,
        }
        # Add this entry to our collection
        data_for_new_df.append(entry)

# Convert the list of dictionaries to a new DataFrame
new_df = pd.DataFrame(data_for_new_df)

# Print the number of source documents collected
print(f"Collected {len(new_df)} source documents")

# Save processed DataFrame to disk
new_df.to_pickle(OUTPUT_FILE)
