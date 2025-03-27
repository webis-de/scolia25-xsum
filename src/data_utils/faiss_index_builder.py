"""
FAISS Index Builder Module

This module provides functionality to build and manage FAISS indexes from document embeddings.
It handles loading data, processing embeddings, and creating searchable vector indexes
with associated metadata for efficient semantic search operations.
"""

import pandas as pd
import numpy as np
import ast
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from .faiss_index import FaissIndex


class FaissIndexBuilder:
    """
    A class to build and manage a FAISS index with associated metadata.

    This builder loads document embeddings from a pickle file, processes them,
    and constructs a FAISS index for efficient similarity search. The class also
    handles metadata extraction and persistence of both the index and metadata.

    Attributes:
        data_path (str): Path to the input pickle data file containing embeddings.
        index_path (str): Path to save the FAISS index file.
        metadata_path (str): Path to save the metadata file.
        metadata_columns (List[str]): Columns to be used as metadata from the DataFrame.
        batch_size (int): Number of documents to process in each batch.
        faiss_index (Optional[FaissIndex]): Instance of the FAISS index.
        df (Optional[pd.DataFrame]): Loaded data as a pandas DataFrame.
        logger (logging.Logger): Logger instance for this class.
    """

    def __init__(
        self,
        data_path: str,
        index_path: str,
        metadata_path: str,
        metadata_columns: Optional[List[str]] = None,
        batch_size: int = 1000,
    ):
        """
        Initializes the FaissIndexBuilder with necessary paths and configurations.

        Args:
            data_path (str): Path to the input pickle data file.
            index_path (str): Path to save the FAISS index file.
            metadata_path (str): Path to save the metadata file.
            metadata_columns (Optional[List[str]]): Columns to be used as metadata.
                Defaults to a standard set of document metadata fields.
            batch_size (int): Number of documents to process in each batch.
                Defaults to 1000.
        """
        # Initialize paths and configuration
        self.data_path = data_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.metadata_columns = metadata_columns or [
            "abstract",
            "chunk",
            "title",
            "paper_id",
            "citation_id",
            "survey_id",
            "survey_title",
            "section_title",
        ]
        self.batch_size = batch_size
        self.faiss_index: Optional[FaissIndex] = None
        self.df: Optional[pd.DataFrame] = None

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def parse_embedding(embedding: Any) -> np.ndarray:
        """
        Ensures that embeddings are properly formatted as numpy arrays of type float32.

        Handles various input formats including strings, lists, and numpy arrays.

        Args:
            embedding (Any): The embedding data in various possible formats.

        Returns:
            np.ndarray: Numpy array of the embedding with dtype float32.
        """
        # Handle string representations (e.g., from CSV or other text formats)
        if isinstance(embedding, str):
            try:
                return np.array(ast.literal_eval(embedding), dtype=np.float32)
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Failed to parse embedding string. Error: {e}")
                return np.array([], dtype=np.float32)
        # Handle Python lists
        elif isinstance(embedding, list):
            return np.array(embedding, dtype=np.float32)
        # Handle numpy arrays - ensure they're float32 for FAISS compatibility
        elif isinstance(embedding, np.ndarray):
            return embedding.astype(np.float32)
        else:
            # Handle unexpected data types
            logging.warning(
                f"Unexpected embedding type: {type(embedding)}. Setting as empty."
            )
            return np.array([], dtype=np.float32)

    def load_data(self):
        """
        Loads embedding data from the specified pickle file into a pandas DataFrame.

        Validates that the input file exists and can be loaded properly.
        """
        self.logger.info(f"Loading data from: {self.data_path}")
        # Check if file exists
        if not Path(self.data_path).is_file():
            self.logger.error(f"The file {self.data_path} was not found.")
            sys.exit(1)

        # Attempt to load the pickle file
        try:
            self.df = pd.read_pickle(self.data_path)
            self.logger.info("Data loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load data. Error: {e}")
            sys.exit(1)

        # Verify 'chunk' column exists as it's essential for our use case
        if "chunk" not in self.df.columns:
            self.logger.error("Error: 'chunk' column not found in the data.")
            sys.exit(1)

    def inspect_chunks(self):
        """
        Inspects the 'chunk' column for anomalies and logs statistics about chunks.

        Provides information about missing values and sample chunks for debugging.
        """
        self.logger.info("\n--- Inspecting 'chunk' Column ---")
        # Log basic statistics
        self.logger.info(f"Total Rows: {len(self.df)}")
        self.logger.info(
            f"Number of Non-Null 'chunk': {self.df['chunk'].notnull().sum()}"
        )
        self.logger.info(
            f"Number of 'chunk' Equal to 'N/A': {(self.df['chunk'] == 'N/A').sum()}"
        )

        # Show sample values for inspection
        self.logger.info("Sample 'chunk' Values:")
        unique_chunk = self.df["chunk"].dropna().unique()
        sample_unique = unique_chunk[:5] if len(unique_chunk) >= 5 else unique_chunk
        self.logger.info(f"{sample_unique}")
        self.logger.info("----------------------------------\n")

    def handle_missing_chunks(self):
        """
        Fills missing 'chunk' values with 'N/A' placeholder and logs the outcome.

        This ensures all rows have a valid value for the 'chunk' field.
        """
        self.logger.info("Filling missing 'chunk' with 'N/A'...")
        # Replace NaN values with 'N/A'
        self.df["chunk"] = self.df["chunk"].fillna("N/A")

        # Verify and report on N/A values
        num_na_chunk = (self.df["chunk"] == "N/A").sum()
        if num_na_chunk > 0:
            self.logger.warning(f"{num_na_chunk} 'chunk' entries are set to 'N/A'.")
        else:
            self.logger.info("All 'chunk' entries are populated.")

    def process_embeddings(self):
        """
        Parses and validates embeddings, removing any rows with invalid embeddings.

        Ensures all embeddings have consistent dimensions for FAISS compatibility.
        """
        self.logger.info("Processing embeddings...")
        # Parse embeddings to ensure they're properly formatted
        self.df["embedding"] = self.df["embedding"].apply(self.parse_embedding)

        # Remove rows with invalid (empty) embeddings
        initial_count = len(self.df)
        self.df = self.df[self.df["embedding"].apply(lambda x: x.size > 0)]
        removed_count = initial_count - len(self.df)
        if removed_count > 0:
            self.logger.warning(
                f"Removed {removed_count} rows with invalid embeddings."
            )

        # Verify we have at least some valid embeddings
        if self.df.empty:
            self.logger.error("Error: No valid embeddings found.")
            sys.exit(1)

        # Verify that all embeddings have the same dimension
        embedding_dim = self.df["embedding"].iloc[0].shape[0]
        if not all(emb.shape[0] == embedding_dim for emb in self.df["embedding"]):
            self.logger.error("Error: Inconsistent embedding dimensions found.")
            sys.exit(1)
        self.logger.info(f"Embedding dimension: {embedding_dim}")

    def extract_embeddings_and_metadata(
        self,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Extracts embeddings and metadata from the DataFrame for indexing.

        Returns:
            Tuple containing:
                - embeddings (np.ndarray): 2D array of document embeddings.
                - metadata (List[Dict[str, Any]]): List of metadata dictionaries.
        """
        self.logger.info("Extracting embeddings and metadata...")
        # Stack all embeddings into a single 2D array
        embedding = np.vstack(
            self.df["embedding"].values
        )  # Shape: (num_embedding, dim)

        # Verify that all metadata columns exist
        missing_columns = [
            col for col in self.metadata_columns if col not in self.df.columns
        ]
        if missing_columns:
            self.logger.error(f"Error: Missing metadata columns: {missing_columns}")
            sys.exit(1)

        # Convert metadata to a list of dictionaries for easy access
        metadata = self.df[self.metadata_columns].to_dict(orient="records")
        self.logger.info("Embeddings and metadata extracted successfully.")
        return embedding, metadata

    def build_faiss_index(self, embedding: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Initializes the FAISS index and adds embeddings with metadata.

        Args:
            embedding (np.ndarray): 2D array of document embeddings.
            metadata (List[Dict[str, Any]]): List of metadata dictionaries.
        """
        self.logger.info("Initializing FAISS index and adding embeddings...")
        # Get the embedding dimension from the data
        embedding_dim = embedding.shape[1]

        # Create and populate the FAISS index
        self.faiss_index = FaissIndex(dimension=embedding_dim)
        self.faiss_index.add_embeddings(embedding, metadata)
        self.logger.info("FAISS index built successfully.")

    def save_index_and_metadata(self):
        """
        Saves the FAISS index and associated metadata to specified paths.

        This allows the index to be loaded later for search operations.
        """
        if self.faiss_index is None:
            self.logger.error("FAISS index has not been built yet.")
            sys.exit(1)

        self.logger.info(
            f"Saving FAISS index to {self.index_path} and metadata to {self.metadata_path}..."
        )
        try:
            # Save both index and metadata
            self.faiss_index.save(self.index_path, self.metadata_path)
            self.logger.info("FAISS index and metadata have been successfully saved.")
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index or metadata. Error: {e}")
            sys.exit(1)

    def run(self):
        """
        Executes the full pipeline to build and save the FAISS index.

        This is the main entry point that orchestrates the entire indexing process.
        """
        # Load and prepare the data
        self.load_data()
        self.inspect_chunks()
        self.handle_missing_chunks()
        self.process_embeddings()

        # Extract data and build index
        embedding, metadata = self.extract_embeddings_and_metadata()
        self.build_faiss_index(embedding, metadata)

        # Save the results
        self.save_index_and_metadata()


# Example Usage
# if __name__ == "__main__":
#     # Define paths using relative paths
#     DATA_PATH = "../../data/surveysum/processed/sample_source_docs_full_chunks_embed.pkl"
#     INDEX_PATH = "../../data/db/surveysum.bin"
#     METADATA_PATH = "../../data/db/metadata_surveysum.pkl"
#     METADATA_COLUMNS = [
#         "abstract",
#         "chunk",
#         "title",
#         "paper_id",
#         "citation_id",
#         "survey_id",
#         "survey_title",
#         "section_title",
#     ]
#
#     # Initialize the builder
#     builder = FaissIndexBuilder(
#         data_path=DATA_PATH,
#         index_path=INDEX_PATH,
#         metadata_path=METADATA_PATH,
#         metadata_columns=METADATA_COLUMNS,
#     )
#
#     # Run the indexing process
#     builder.run()
