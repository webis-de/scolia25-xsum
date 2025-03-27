import pandas as pd
from eval_rouge import compute_rouge_scores

from eval_g_eval import compute_g_eval_scores
from eval_checkeval import compute_check_eval_scores
from eval_bertscore import compute_bert_recall
from eval_ref_f1 import compute_reference_f1_recall


import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    Main function to execute the evaluation pipeline.
    """
    # Paths to the input and output dataframes
    input_pickle_path = "PATH_TO_INPUT_PICKLE"  # Placeholder for the actual input path
    output_pickle_path = "PATH_TO_OUTPUT_PICKLE"  # Placeholder for the actual output path

    # Load the dataframe
    logging.info("Loading dataframe...")
    df = pd.read_pickle(input_pickle_path)
    logging.info(f"Dataframe loaded with {len(df)} records.")

    # -------------------------------
    # Compute ROUGE Recall Scores
    # -------------------------------
    logging.info("\nComputing ROUGE Recall Scores...")
    df = compute_rouge_scores(
        df,
        reference_col="GROUND_TRUTH",
        candidate_col="GENERATED_TEXT",
    )
    logging.info("ROUGE Recall Scores computed and added to the dataframe.")

    # -------------------------------
    # Compute g-eval Scores
    # -------------------------------
    logging.info("\nComputing g-eval Scores...")
    df = compute_g_eval_scores(df)
    logging.info("g-eval Scores computed and added to the dataframe.")

    # -------------------------------
    # Compute Check-Eval Scores
    # -------------------------------
    logging.info("\nComputing Check-Eval Scores...")
    df = compute_check_eval_scores(df, criterion="consistency")
    logging.info("Check-Eval Scores computed and added to the dataframe.")

    # -------------------------------
    # Compute BERTScore Recall
    # -------------------------------
    logging.info("\nComputing BERTScore Recall...")
    df = compute_bert_recall(
        df,
        reference_col="GROUND_TRUTH",
        candidate_col="GENERATED_TEXT",
        lang="en",
        model_type="bert-base-uncased",
    )
    logging.info("BERTScore Recall computed and added to the dataframe.")

    # -------------------------------
    # Compute Reference F1 Recall
    # -------------------------------
    logging.info("\nComputing Reference F1 Recall Scores...")
    df = compute_reference_f1_recall(
        df, reference_col="GROUND_TRUTH", candidate_col="GENERATED_TEXT"
    )
    logging.info("Reference F1 Recall Scores computed and added to the dataframe.")

    # -------------------------------

    # Save the Evaluated Dataframe
    # -------------------------------
    logging.info(f"\nSaving the evaluated dataframe to {output_pickle_path}...")
    df.to_pickle(output_pickle_path)
    logging.info("Evaluation pipeline completed successfully.")

    # -------------------------------
    # Display Average Scores
    # -------------------------------
    logging.info("\nCalculating average evaluation scores...")

    average_scores = {
        "ROUGE-1 Recall": df["rouge1_recall"].mean(),
        "ROUGE-2 Recall": df["rouge2_recall"].mean(),
        "ROUGE-L Recall": df["rougeL_recall"].mean(),
        "g-eval Evaluation Score (%)": df["g_eval"].mean(),
        "Check-Eval Reference-Guided Score (%)": df[
           "checkeval_reference_guided"
        ].mean(),
        "Check-Eval Criterion-Guided Score (%)": df[
           "checkeval_criterion_guided"
        ].mean(),
        "BERTScore Recall": df["bert_score"].mean(),
        "Reference F1 Recall": df["reference_f1_recall"].mean(),
    }

    logging.info("\nAverage Evaluation Scores:")
    for metric, value in average_scores.items():
        logging.info(f"  {metric}: {value:.6f}")


if __name__ == "__main__":
    main()
