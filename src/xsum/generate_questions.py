import pandas as pd
import logging
from gpt4o_mini import GPT4OMini
import re
import os
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def generate_questions_prompt(title: str, abstract: str, topic: str, n: int) -> str:
    """
    Formats the prompt to instruct the AI to generate n questions based on title, abstract, topic, and subtopic.
    """
    prompt = (
        "### CONTEXT ###\n\n"
        f"You are preparing to interview an expert in the field titled `{topic}`.\n\n"
        "You will be shown a title and abstract of a research paper related to the field.\n\n"
        "Keep in mind that the expert is not the author of the paper but is knowledgeable in the field.\n\n"
        "### INSTRUCTIONS ###\n\n"
        f"Prepare {n} relevant questions related to the expert field and subtopic based on the following title and abstract.\n\n"
        "The questions should be general and not too detailed to allow for a broad discussion.\n\n"
        f"Title: {title}\n\n"
        f"Abstract: {abstract}\n\n"
        "List the questions without numbering and avoid adding any additional explanations.\n\n"
        "Avoid subjective questions and focus on objective questions that can be answered by an expert in the field.\n\n"
    )
    return prompt


def parse_questions(response: str) -> list:
    """
    Parses the AI response to extract questions.
    """
    lines = response.strip().split("\n")
    questions = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove numbering if present (e.g., "1. Question")
        line = re.sub(r"^\d+\.\s*", "", line)
        # Remove markdown bold formatting
        line = line.strip("*").strip()
        if line.endswith("?"):
            questions.append(line)

    return questions


def main():
    """
    Main function to process papers and generate questions.
    Loads data, generates questions for each paper, and saves results.
    """
    # Configuration
    df_path = "INPUT_DATA_PATH"
    output_path = "OUTPUT_DATA_PATH"
    n_questions = 5
    save_every = 5

    # Initialize GPT4OMini with an appropriate system prompt
    system_prompt = "You are an well-known expert in formulating questions for academic survey papers."
    gpt = GPT4OMini(system_prompt=system_prompt)

    # Load DataFrame containing paper information
    try:
        df = pd.read_pickle(df_path)
        logger.info(f"Loaded DataFrame from {df_path}.")
    except Exception as e:
        logger.error(f"Error loading DataFrame: {e}")
        return

    # Initialize the 'questions' column as empty dictionaries
    df["questions"] = [{} for _ in range(len(df))]

    # Iterate through each row in the DataFrame with a progress bar
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Questions"):
        titles_dict = row.get("titles", {})
        abstracts_dict = row.get("abstracts", {})
        survey_title = row.get("survey_title", "").strip()

        # Data Validation: Check for missing abstracts
        missing_abstracts = set(titles_dict.keys()) - set(abstracts_dict.keys())
        if missing_abstracts:
            logger.warning(f"Missing abstracts for paper IDs: {missing_abstracts}")

        questions_for_row = {}

        # Process each paper in the current row
        for paper_id in titles_dict:
            title = titles_dict.get(paper_id, "")
            abstract = abstracts_dict.get(paper_id)

            if not title or abstract is None:
                logger.warning(f"Missing title or abstract for paper ID: {paper_id}")
                continue

            topic = survey_title if survey_title else "the relevant field"

            abstract = abstract.strip() if abstract else ""
            prompt = generate_questions_prompt(title, abstract, topic, n_questions)

            try:
                response = gpt.get_response(prompt)
            except Exception as e:
                logger.error(f"Error generating questions for paper ID {paper_id}: {e}")
                continue

            questions = parse_questions(response)

            if questions:
                questions_for_row[paper_id] = questions
                logger.info(f"Generated questions for paper ID {paper_id}.")
            else:
                logger.warning(f"No questions generated for paper ID: {paper_id}")

        df.at[idx, "questions"] = questions_for_row

        # Periodically save progress to avoid data loss
        if (idx + 1) % save_every == 0:
            try:
                df.to_pickle(output_path)
                logger.info(f"Saved progress after processing {idx + 1} rows.")
            except Exception as e:
                logger.error(f"Error saving progress: {e}")

    # Save the final updated DataFrame
    try:
        df.to_pickle(output_path)
        logger.info(f"Final DataFrame saved to {output_path}.")
    except Exception as e:
        logger.error(f"Error saving final DataFrame: {e}")


if __name__ == "__main__":
    main()
