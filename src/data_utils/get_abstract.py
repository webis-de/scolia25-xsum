"""
This script fetches abstracts and paper IDs for research papers from arXiv and Semantic Scholar.
It processes a dataframe containing paper citations, retrieves the abstracts using APIs,
caches results for efficiency, and saves the enriched dataset.

The script implements:
1. Logging configuration
2. Abstract retrieval from arXiv and Semantic Scholar
3. Caching mechanisms to avoid redundant API calls
4. Graceful error handling and progress saving
"""

import arxiv
import pandas as pd
from tqdm import tqdm
from functools import lru_cache
import logging
import time
import os
from semanticscholar import SemanticScholar
import diskcache as dc

# ============================
# 1. Configure Logging
# ============================


def setup_logging(log_file_path):
    """
    Configures logging to output to both a file and the console.

    Parameters:
    - log_file_path (str): The path to the log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the desired logging level

    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = logging.FileHandler(log_file_path)  # File handler

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)


# Specify the path to your log file (using relative path)
LOG_FILE_PATH = "logs/abstract_fetch.log"

# Ensure the log directory exists
log_dir = os.path.dirname(LOG_FILE_PATH)
os.makedirs(log_dir, exist_ok=True)

# Setup logging
setup_logging(LOG_FILE_PATH)

# ============================
# 2. Define Fetch Functions
# ============================


@lru_cache(maxsize=None)
def get_arxiv_abstract_cached(title):
    """
    Cached function to fetch abstract and arXiv ID for a given title from arXiv.

    Parameters:
    - title (str): The title of the paper to search for

    Returns:
    - tuple: (abstract, arxiv_id) both can be None if not found
    """
    try:
        # Search for the paper by title on arXiv
        search = arxiv.Search(
            query=f'ti:"{title}"', max_results=1, sort_by=arxiv.SortCriterion.Relevance
        )
        result = next(search.results())
        logging.info(f"Found arXiv entry for title: '{title}'")
        return result.summary, result.get_short_id()
    except StopIteration:
        # No results found
        logging.warning(f"No arXiv results found for title: '{title}'")
        return None, None
    except Exception as e:
        # Handle any other exceptions
        logging.error(f"Error fetching from arXiv for title '{title}': {e}")
        return None, None
    finally:
        time.sleep(3)  # Respect API rate limits


def get_semantic_scholar_abstract(
    title, sch_client, fields=["title", "abstract", "paperId"]
):
    """
    Fetches the abstract and paper ID from Semantic Scholar based on the paper title.

    Parameters:
    - title (str): The exact title of the paper.
    - sch_client (SemanticScholar): An instance of the SemanticScholar client.
    - fields (list, optional): List of fields to return.

    Returns:
    - tuple: (abstract (str or None), paper_id (str or None))
    """
    try:
        # Perform a search for the paper by title
        results = sch_client.search_paper(title, fields=fields, limit=1)
        if results:
            paper = results[0]
            abstract = getattr(paper, "abstract", None)
            paper_id = getattr(paper, "paperId", None)
            logging.info(
                f"Successfully fetched Semantic Scholar entry for title: '{title}'"
            )
            return abstract, paper_id
        else:
            logging.warning(f"No Semantic Scholar match found for title: '{title}'")
            return None, None
    except Exception as e:
        logging.error(f"Error fetching from Semantic Scholar for title '{title}': {e}")
        return None, None
    finally:
        time.sleep(3)  # Respect API rate limits


def fetch_abstracts(citations_dict, sch_client, cache_client):
    """
    Fetch abstracts and paper IDs for all bibrefs in a citations_dict.
    Attempts arXiv first, then Semantic Scholar if arXiv fails.

    Parameters:
    - citations_dict (dict): Dictionary mapping bibref IDs to their details.
    - sch_client (SemanticScholar): An instance of the SemanticScholar client.
    - cache_client (diskcache.Cache): An instance of the DiskCache client.

    Returns:
    - tuple: (abstracts (dict), paper_ids (dict))
    """
    abstracts = {}
    paper_ids = {}
    for bibref, details in citations_dict.items():
        title = details.get("title")
        if title:
            # Check if the title is in cache
            if title in cache_client:
                # Use cached result if available
                abstract, paper_id = cache_client[title]
                logging.info(f"Cache hit for title: '{title}'")
            else:
                # Attempt to fetch from arXiv first
                abstract, arxiv_id = get_arxiv_abstract_cached(title)
                if abstract:
                    paper_id = f"arXiv:{arxiv_id}"
                    cache_client[title] = (abstract, paper_id)
                else:
                    # Fallback to Semantic Scholar if arXiv fails
                    abstract, paper_id = get_semantic_scholar_abstract(
                        title, sch_client
                    )
                    if abstract and paper_id:
                        paper_id = f"S2:{paper_id}"
                        cache_client[title] = (abstract, paper_id)
                    else:
                        paper_id = None
                        cache_client[title] = (abstract, paper_id)
            # Update dictionaries with the results
            abstracts[bibref] = abstract
            paper_ids[bibref] = paper_id
        else:
            abstracts[bibref] = None
            paper_ids[bibref] = None
    return abstracts, paper_ids


# ============================
# 3. Initialize SemanticScholar Client and Cache
# ============================

# Initialize the SemanticScholar client without an API key
sch = SemanticScholar()

# Initialize DiskCache for persistent caching (using relative path)
cache = dc.Cache("cache/arxiv_cache")

# ============================
# 4. Load DataFrame and Initialize Columns
# ============================

# Load the DataFrame (using relative path)
DATA_FILE_PATH = "data/surveysum/train-00000-of-00001-with-titles.pkl"
df = pd.read_pickle(DATA_FILE_PATH)

# Initialize empty columns for abstracts and paper_ids if they don't exist
if "abstracts" not in df.columns:
    df["abstracts"] = [{} for _ in range(len(df))]

if "paper_ids" not in df.columns:
    df["paper_ids"] = [{} for _ in range(len(df))]

# ============================
# 5. Iterate Over DataFrame and Fetch Abstracts
# ============================

try:
    # Iterate over the DataFrame row by row with a progress bar
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        citations_dict = row["citations_non_none"]

        # Fetch abstracts and paper_ids with caching
        abstracts, paper_ids = fetch_abstracts(citations_dict, sch, cache)

        # Update the DataFrame with new data
        df.at[idx, "abstracts"] = abstracts
        df.at[idx, "paper_ids"] = paper_ids

        # Save the DataFrame periodically to ensure progress is saved
        if idx % 100 == 0:
            OUTPUT_FILE_PATH = "data/surveysum/train_with_abstracts.pkl"
            df.to_pickle(OUTPUT_FILE_PATH)
            logging.info(f"Saved progress at index {idx}")
except KeyboardInterrupt:
    # Handle user interruption gracefully
    logging.warning("Script interrupted by user. Saving current progress...")
    OUTPUT_FILE_PATH = "data/surveysum/train_with_abstracts.pkl"
    df.to_pickle(OUTPUT_FILE_PATH)
    logging.info("Progress saved. Exiting.")
    exit()
except Exception as e:
    # Catch any other errors and save progress
    logging.error(f"An unexpected error occurred: {e}")
    OUTPUT_FILE_PATH = "data/surveysum/train_with_abstracts.pkl"
    df.to_pickle(OUTPUT_FILE_PATH)
    logging.info("Progress saved despite the error.")
    exit()

# ============================
# 6. Final Save After Completion
# ============================

# Save the DataFrame after processing all rows (using relative path)
OUTPUT_FILE_PATH = "data/surveysum/train_with_abstracts.pkl"
df.to_pickle(OUTPUT_FILE_PATH)
logging.info("Completed processing all rows and saved the DataFrame.")
