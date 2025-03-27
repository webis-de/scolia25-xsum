"""
This module provides functionality for semantic search using Facebook AI Similarity Search (FAISS).
It combines vector similarity search with metadata filtering and reranking for improved search results.

The main class, FaissSearcher, implements a complete search pipeline:
1. Query embedding generation
2. Metadata filtering
3. FAISS similarity search
4. Result reranking using ColBERT
5. Result formatting and display

This can be used for efficient retrieval of relevant text chunks from a large corpus
based on semantic similarity and additional metadata constraints.

"""

import numpy as np
import faiss
import pickle
import logging
import sys
from typing import List, Dict, Any, Tuple, Optional

from data_loading.embedding_generator import EmbeddingGenerator
from data_loading.faiss_index import FaissIndex
from rerankers import Reranker
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Represents a single search result with ranking, score, and content details."""

    rank: int
    score: float
    chunk: str
    citation_id: Any
    metadata: Dict[str, Any]


class FaissSearcher:
    """
    A class to perform FAISS-based similarity search with metadata filtering and reranking.
    """

    def __init__(
        self,
        faiss_index_path: str,
        metadata_path: str,
        embedding_model_name: str = "allenai/specter2_base",
        embedding_batch_size: int = 16,
        dimension: int = 768,
        top_k_faiss: int = 100,
        top_k_final: int = 20,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the FaissSearcher with necessary paths and configurations.

        Args:
            faiss_index_path (str): Path to the FAISS index file.
            metadata_path (str): Path to the metadata pickle file.
            embedding_model_name (str): Name of the embedding model to use.
            embedding_batch_size (int): Batch size for embedding generation.
            dimension (int): Dimension of the embeddings.
            top_k_faiss (int): Number of top similar results to retrieve from FAISS.
            top_k_final (int): Number of top results to display after reranking.
            metadata_filters (Optional[Dict[str, Any]]): Metadata filters to apply.
        """
        # Store initialization parameters
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name
        self.embedding_batch_size = embedding_batch_size
        self.dimension = dimension
        self.top_k_faiss = top_k_faiss
        self.top_k_final = top_k_final
        self.metadata_filters = metadata_filters or {}

        # Initialize components as None (will be loaded later)
        self.faiss_index: Optional[FaissIndex] = None
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.reranker: Optional[Reranker] = None

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self._load_faiss_index()
        self._initialize_embedding_generator()
        self._initialize_reranker()

    @staticmethod
    def parse_embedding(embedding: Any) -> np.ndarray:
        """
        Ensures that the embedding is a numpy array of type float32.

        Args:
            embedding (Any): The embedding data.

        Returns:
            np.ndarray: Numpy array of the embedding.
        """
        if isinstance(embedding, str):
            try:
                # Convert string representation to numpy array
                return np.array(eval(embedding), dtype=np.float32)
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Failed to parse embedding string. Error: {e}")
                return np.array([], dtype=np.float32)
        elif isinstance(embedding, list):
            # Convert list to numpy array
            return np.array(embedding, dtype=np.float32)
        elif isinstance(embedding, np.ndarray):
            # Ensure correct data type
            return embedding.astype(np.float32)
        else:
            # Handle unexpected data types
            logging.warning(
                f"Unexpected embedding type: {type(embedding)}. Setting as empty."
            )
            return np.array([], dtype=np.float32)

    def _load_faiss_index(self):
        """
        Loads the FAISS index and metadata from the specified files.
        """
        self.logger.info("Loading FAISS index and metadata...")
        try:
            # Load the index using the FaissIndex class
            self.faiss_index = FaissIndex.load(
                self.faiss_index_path, self.metadata_path, dimension=self.dimension
            )
            self.logger.info("FAISS index and metadata loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading FAISS index or metadata: {e}")
            sys.exit(1)

    def _initialize_embedding_generator(self):
        """
        Initializes the EmbeddingGenerator.
        """
        self.logger.info("Initializing EmbeddingGenerator...")
        try:
            # Create embedding generator with specified model
            self.embedding_generator = EmbeddingGenerator(
                model_name=self.embedding_model_name,
                batch_size=self.embedding_batch_size,
            )
            self.logger.info("EmbeddingGenerator initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing EmbeddingGenerator: {e}")
            sys.exit(1)

    def _initialize_reranker(self):
        """
        Initializes the Reranker.
        """
        self.logger.info("Initializing Reranker...")
        try:
            # Initialize ColBERT reranker
            self.reranker = Reranker("colbert")
            self.logger.info("Reranker initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing Reranker: {e}")
            sys.exit(1)

    def generate_query_embedding(self, query_text: str) -> np.ndarray:
        """
        Generates an embedding for the given query text.

        Args:
            query_text (str): The query string.

        Returns:
            np.ndarray: Generated embedding vector.
        """
        self.logger.info("Generating embedding for the query...")
        if not self.embedding_generator:
            self.logger.error("EmbeddingGenerator is not initialized.")
            sys.exit(1)
        try:
            # Generate embedding vector for the query
            embeddings = self.embedding_generator.generate_embeddings([query_text])
            query_embedding = embeddings[0]
            # Normalize the query embedding for cosine similarity
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            self.logger.info("Query embedding generated successfully.")
            return query_embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding for the query: {e}")
            sys.exit(1)

    def _filter_metadata_indices(self) -> List[int]:
        """
        Identifies indices in the FAISS index that match the metadata filters.

        Returns:
            List[int]: List of matching indices.
        """
        self.logger.info("Filtering metadata based on provided filters...")
        if not self.faiss_index:
            self.logger.error("FAISS index is not loaded.")
            sys.exit(1)
        matching_indices = []
        # Iterate through all metadata to find matches
        for idx, meta in enumerate(self.faiss_index.metadata):
            match = True
            for field, value in self.metadata_filters.items():
                # Case-insensitive comparison
                if str(meta.get(field, "")).lower() != str(value).lower():
                    match = False
                    break
            if match:
                matching_indices.append(idx)

        if not matching_indices:
            self.logger.warning("No entries match the specified metadata filters.")
            return []

        self.logger.info(f"Number of entries matching filters: {len(matching_indices)}")
        return matching_indices

    def build_subset_faiss_index(
        self, matching_indices: List[int]
    ) -> Optional[FaissIndex]:
        """
        Builds a temporary FAISS index with embeddings that match the metadata filters.

        Args:
            matching_indices (List[int]): Indices that match the metadata filters.

        Returns:
            Optional[FaissIndex]: Temporary FaissIndex instance with filtered embeddings, or None if no matches.
        """
        if not self.faiss_index:
            self.logger.error("FAISS index is not loaded.")
            sys.exit(1)

        if not matching_indices:
            self.logger.warning("No entries match the specified metadata filters.")
            return None

        try:
            # Extract the matching embeddings
            self.logger.info("Extracting matching embeddings...")
            matching_embeddings = np.vstack(
                [self.faiss_index.index.reconstruct(idx) for idx in matching_indices]
            ).astype("float32")

            # Extract corresponding metadata
            self.logger.info("Extracting corresponding metadata...")
            matching_metadata = [
                self.faiss_index.metadata[idx] for idx in matching_indices
            ]

            # Build a temporary FAISS index with the filtered embeddings
            self.logger.info(
                "Building temporary FAISS index with filtered embeddings..."
            )
            subset_faiss = FaissIndex(dimension=self.dimension)
            subset_faiss.add_embeddings(matching_embeddings, matching_metadata)
            self.logger.info(
                "Temporary FAISS index with filtered embeddings created successfully."
            )

            return subset_faiss
        except Exception as e:
            self.logger.error(f"Error building subset FAISS index: {e}")
            sys.exit(1)

    def perform_faiss_search(
        self, subset_faiss: FaissIndex, query_embedding: np.ndarray
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Performs a FAISS search for the given query embedding.

        Args:
            subset_faiss (FaissIndex): Instance of FaissIndex (subset) to search.
            query_embedding (np.ndarray): Embedding vector for the query.

        Returns:
            List[Tuple[float, Dict[str, Any]]]: List of (distance, metadata) tuples.
        """
        self.logger.info("Performing FAISS search...")
        try:
            # Conduct the similarity search
            distances, indices = subset_faiss.index.search(
                query_embedding.reshape(1, -1), self.top_k_faiss
            )
            results = []
            # Process search results
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue  # Skip invalid indices
                meta = subset_faiss.metadata[idx]
                results.append((distance, meta))
            self.logger.info(
                f"FAISS search completed. Retrieved top {self.top_k_faiss} results."
            )
            return results
        except Exception as e:
            self.logger.error(f"Error performing FAISS search: {e}")
            sys.exit(1)

    def rerank_results(
        self, query: str, faiss_results: List[Tuple[float, Dict[str, Any]]]
    ) -> List[SearchResult]:
        """
        Reranks the FAISS search results using ColBERT.

        Args:
            query (str): The original query string.
            faiss_results (List[Tuple[float, Dict[str, Any]]]): List of (distance, metadata) tuples from FAISS.

        Returns:
            List[SearchResult]: List of reranked SearchResult objects.
        """
        self.logger.info("Reranking FAISS results using ColBERT...")
        if not self.reranker:
            self.logger.error("Reranker is not initialized.")
            sys.exit(1)

        try:
            # Extract documents and doc_ids from FAISS results
            docs = [meta.get("chunk", "") for _, meta in faiss_results]
            doc_ids = [
                meta.get("citation_id", idx)
                for idx, (_, meta) in enumerate(faiss_results)
            ]
            metadata = [meta for _, meta in faiss_results]

            # Perform reranking
            reranked = self.reranker.rank(
                query=query, docs=docs, doc_ids=doc_ids, metadata=metadata
            )
            self.logger.info("Reranking completed successfully.")

            # Convert to SearchResult dataclass
            reranked_results = [
                SearchResult(
                    rank=rank,
                    score=result.score,
                    chunk=result.document.text,
                    citation_id=result.document.doc_id,
                    metadata=result.document.metadata,
                )
                for rank, result in enumerate(reranked.results, start=1)
            ]

            return reranked_results
        except Exception as e:
            self.logger.error(f"Error during reranking: {e}")
            sys.exit(1)

    def display_results(
        self, reranked_results: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        """
        Formats the reranked results as a list of dictionaries.

        Args:
            reranked_results (List[SearchResult]): List of reranked SearchResult objects.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with reranked results.
        """
        self.logger.info(f"Formatting the top {self.top_k_final} reranked results...")
        if not reranked_results:
            self.logger.warning("No reranked results to display.")
            return []

        # Format results for display
        output_list = []
        for result in reranked_results[: self.top_k_final]:
            result_dict = {
                "rank": result.rank,
                "score": result.score,
                "chunk": result.chunk,
                "citation_id": result.citation_id,
                "metadata": result.metadata,
            }
            output_list.append(result_dict)

        self.logger.info(
            f"Top {self.top_k_final} reranked results formatted successfully."
        )
        return output_list

    def search(
        self,
        query_text: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes the full search pipeline: embedding generation, FAISS search, reranking, and result formatting.

        Args:
            query_text (str): The query string.
            metadata_filters (Optional[Dict[str, Any]]): Metadata filters to apply.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with reranked results.
        """
        self.logger.info("Starting search pipeline...")

        # Update metadata filters if provided
        if metadata_filters:
            self.logger.info("Updating metadata filters with provided filters.")
            self.metadata_filters = metadata_filters

        # Generate query embedding
        query_embedding = self.generate_query_embedding(query_text)

        # Filter metadata and get matching indices
        matching_indices = self._filter_metadata_indices()

        if not matching_indices:
            self.logger.warning(
                "No matching entries found after applying metadata filters."
            )
            return []

        # Build subset FAISS index
        subset_faiss = self.build_subset_faiss_index(matching_indices)
        if not subset_faiss:
            self.logger.warning(
                "No subset FAISS index created due to no matching entries."
            )
            return []

        # Perform FAISS search
        faiss_results = self.perform_faiss_search(subset_faiss, query_embedding)

        if not faiss_results:
            self.logger.warning("No results found from FAISS search.")
            return []

        # Rerank results
        reranked_results = self.rerank_results(query_text, faiss_results)

        if not reranked_results:
            self.logger.warning("No reranked results available.")
            return []

        # Format and return final results
        final_results = self.display_results(reranked_results)
        self.logger.info("Search pipeline completed successfully.")
        return final_results


# ------------------------- Example Usage -------------------------

if __name__ == "__main__":
    # Define paths (using relative paths)
    FAISS_INDEX_PATH = "../data/db/surveysum.bin"
    METADATA_PATH = "../data/db/metadata_surveysum.pkl"

    # Query parameters
    QUERY_TEXT = "What are the key differences between the canonical approaches to automatic melody harmonization discussed in the paper?"
    TOP_K_FAISS = 100
    TOP_K_FINAL = 20
    METADATA_FILTERS = {"citation_id": "BIBREF179"}

    # Initialize the FaissSearcher
    searcher = FaissSearcher(
        faiss_index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
        embedding_model_name="allenai/specter2_base",
        embedding_batch_size=16,
        dimension=768,
        top_k_faiss=TOP_K_FAISS,
        top_k_final=TOP_K_FINAL,
        metadata_filters=METADATA_FILTERS,
    )

    # Perform the search
    final_results = searcher.search(
        query_text=QUERY_TEXT, metadata_filters=METADATA_FILTERS
    )

    # Display the final results
    print(f"Top {TOP_K_FINAL} reranked results:")
    for result in final_results:
        print(result)
