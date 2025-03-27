"""
FAISS Index Management Module

This module provides a wrapper class for FAISS (Facebook AI Similarity Search) indices
to store and retrieve vector embeddings efficiently. It supports adding embeddings with
associated metadata, searching for similar embeddings, and saving/loading indices to disk.

"""

import faiss
import numpy as np
import pandas as pd
import pickle


class FaissIndex:
    def __init__(self, dimension):
        """
        Initializes the FAISS index.

        Parameters:
        - dimension (int): Dimension of the embeddings.
        """
        # Initialize the FAISS index with L2 distance metric
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []  # Store metadata corresponding to each embedding

    def add_embeddings(self, embeddings, metadata):
        """
        Adds embeddings to the FAISS index.

        Parameters:
        - embeddings (np.array): Numpy array of shape (num_embeddings, dimension).
        - metadata (list of dict): List of metadata dictionaries for each embedding.
        """
        self.index.add(embeddings)  # Add embeddings to the FAISS index
        self.metadata = metadata  # Store metadata for reference during retrieval

    def search(self, query_embedding, k=5):
        """
        Searches the FAISS index for similar embeddings.

        Parameters:
        - query_embedding (np.array): Query embedding vector of shape (dimension,).
        - k (int): Number of nearest neighbors to retrieve.

        Returns:
        - List of tuples (distance, metadata) for the top-k results.
        """
        # Reshape query to ensure correct dimensions and search the index
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)

        # Combine distances with corresponding metadata
        results = [(distances[0][i], self.metadata[indices[0][i]]) for i in range(k)]
        return results

    def save(self, index_path, metadata_path):
        """
        Saves the FAISS index and metadata to files.

        Parameters:
        - index_path (str): Path to save the FAISS index file.
        - metadata_path (str): Path to save the metadata file.
        """
        # Save FAISS index to disk
        faiss.write_index(self.index, index_path)

        # Save metadata as a pickle file
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print("Index and metadata saved successfully.")

    @classmethod
    def load(cls, index_path, metadata_path, dimension=768):
        """
        Loads the FAISS index and metadata from files.

        Parameters:
        - index_path (str): Path to the FAISS index file.
        - metadata_path (str): Path to the metadata file.
        - dimension (int): Dimension of the embeddings.

        Returns:
        - FaissIndex instance with loaded index and metadata.
        """
        # Load FAISS index from disk
        index = faiss.read_index(index_path)

        # Load metadata from pickle file
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Create a new instance and replace its index and metadata
        faiss_index = cls(dimension)
        faiss_index.index = index
        faiss_index.metadata = metadata

        print("Index and metadata loaded successfully.")
        return faiss_index
