import pandas as pd
import sys
import logging
import time
import random
import re
from tqdm import tqdm
from gpt4o_mini import GPT4OMini

# Configure logging for better debugging and information tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_data(df_path: str) -> pd.DataFrame:
    try:
        df = pd.read_pickle(df_path)
        logger.info(f"Loaded DataFrame from {df_path}.")
        return df
    except FileNotFoundError:
        logger.error(f"DataFrame file not found at {df_path}.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading DataFrame: {e}")
        sys.exit(1)


def save_data(df: pd.DataFrame, output_path: str):
    try:
        df.to_pickle(output_path)
        logger.info(f"Saved DataFrame to {output_path}.")
    except Exception as e:
        logger.error(f"Error saving DataFrame: {e}")
        sys.exit(1)


def sanitize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def generate_final_section_text(topic: str, questions_and_answers: str) -> str:
    prompt = (
        "### CONTEXT ###\n\n"
        f"You are writing the final script of an interview with an expert on the topic `{topic}`.\n\n"
        "The final script should summarize the key insights and findings from the questions and answers provided.\n\n"
        "Keep the target audience in mind, which includes researchers, students, and professionals in the field.\n\n"
        "### QUESTIONS AND ANSWERS ###\n\n"
        f"{questions_and_answers}\n\n"
        "### INSTRUCTIONS ###\n\n"
        "Include the most relevant and important points discussed.\n\n"
        "Be aware of plagiarism, i.e., you should not copy the text, but use them as inspiration.\n\n"
        "Avoid using markdown formatting in the text.\n\n"
        "Avoid splitting into subsections, or creating an introduction and conclusion.\n\n"
        "Avoid introducing new information and focus on summarizing the existing content.\n\n"
        "Always include the citations (e.g., [BIBREF14], [BIBREF16]) mentioned in the answers in the final section.\n\n"
    )
    logger.debug(f"Generated prompt for topic '{topic}'.")
    return prompt


def get_response_with_retry(gpt_instance, prompt, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = gpt_instance.get_response(prompt)
            return response
        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1} failed for prompt. Error: {e}. Retrying after {delay} seconds."
            )
            time.sleep(delay + random.uniform(0, 1))
    logger.error("All retry attempts failed for the prompt.")
    return ""


def main():
    data_path = (
        "../data/surveysum/train_with_abstracts_full_with_questions_and_answers.pkl"
    )
    output_path = "../data/surveysum/train_with_abstracts_full_with_questions_and_answers_editor_version.pkl"

    df = load_data(data_path)

    required_columns = {"survey_id", "survey_title", "answers"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logger.error(f"DataFrame is missing required columns: {missing}")
        sys.exit(1)

    system_prompt = "You are an experienced Editor working on an academic survey article."
    try:
        gpt = GPT4OMini(system_prompt=system_prompt)
        logger.info("Initialized GPT4OMini successfully.")
    except Exception as e:
        logger.error(f"Error initializing GPT4OMini: {e}")
        sys.exit(1)

    if "editor_version" not in df.columns:
        df["editor_version"] = None

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        survey_id = row.get("survey_id", "N/A")
        survey_title = row.get("survey_title", "N/A").strip()
        answers_dict = row.get("answers", {})

        logger.info(f"Processing survey_id {survey_id} at row {index}.")

        if not isinstance(answers_dict, dict):
            logger.warning(
                f"'answers' is not a dictionary for survey_id {survey_id} at row {index}. Skipping."
            )
            continue

        qa_pairs = []
        for citation_id, qa_dict in answers_dict.items():
            if not isinstance(qa_dict, dict):
                logger.warning(
                    f"Answers for citation_id {citation_id} are not in dictionary format. Skipping."
                )
                continue

            for question, answer in qa_dict.items():
                if "do not contain" in answer.lower() or "unable to provide" in answer.lower():
                    logger.info(
                        f"Skipping Q&A for citation_id {citation_id} at row {index} due to filtered answer."
                    )
                    continue

                sanitized_question = sanitize_text(question)
                sanitized_answer = sanitize_text(answer)

                qa_pairs.append(f"Q: {sanitized_question}")
                qa_pairs.append(f"A: {sanitized_answer}")

        if not qa_pairs:
            logger.warning(
                f"No valid Q&A pairs found for survey_id {survey_id} at row {index} after filtering. Skipping."
            )
            df.at[index, "editor_version"] = ""
            continue

        questions_and_answers = "\n".join(qa_pairs)
        prompt = generate_final_section_text(topic=survey_title, questions_and_answers=questions_and_answers)

        try:
            response = get_response_with_retry(gpt, prompt)
        except Exception as e:
            logger.error(f"Error during GPT response for survey_id {survey_id} at row {index}: {e}")
            response = ""
        editor_version_text = response.strip()

        if editor_version_text:
            logger.info(f"Generated editor_version for survey_id {survey_id} at row {index}.")
            logger.debug(f"Editor version text: {editor_version_text}")
        else:
            logger.warning(f"Failed to generate editor_version for survey_id {survey_id} at row {index}.")

        df.at[index, "editor_version"] = editor_version_text

    save_data(df, output_path)
    logger.info("DataFrame processing completed successfully.")


if __name__ == "__main__":
    main()
