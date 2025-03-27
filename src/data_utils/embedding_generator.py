"""
Embedding Generator Module for Processing Scientific Text

This module provides functionality to generate embeddings from text data
using transformer models like SPECTER2. It's designed for scientific text processing
and supports batch processing for efficiency.

The main class, EmbeddingGenerator, handles the loading of pre-trained models
and conversion of text inputs into numerical embeddings that can be used for
downstream tasks such as document similarity, clustering, or classification.
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path


class EmbeddingGenerator:
    def __init__(self, model_name="allenai/specter2_base", device=None, batch_size=16):
        """
        Initializes the embedding generator with the specified model.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to be loaded, by default "allenai/specter2_base"
        device : str, optional
            The device to run the model on, by default None
        batch_size : int, optional
            The batch size for processing, by default 16
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading model '{model_name}' on device '{self.device}'")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    def generate_embeddings(self, texts):
        """
        Generates embeddings for a list of texts using batch processing.

        Parameters
        ----------
        texts : list of str
            The list of texts to generate embeddings for

        Returns
        -------
        np.ndarray
            The embeddings of the texts
        """
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self.device)
                outputs = self.model(**inputs)
                # Use mean pooling of the last hidden state
                if hasattr(outputs, "last_hidden_state"):
                    batch_embeddings = (
                        outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    )
                else:
                    raise AttributeError(
                        "The model does not have 'last_hidden_state'. Modify the embedding extraction accordingly."
                    )
                embeddings.append(batch_embeddings)
        return np.vstack(embeddings)


# def main():
#     # Define file paths
#     input_data_path = "PATH"
#     output_data_path = "PATH"

#     # Load the input data
#     df = pd.read_pickle(input_data_path)

#     # Initialize the Embedding Generator
#     embedding_gen = EmbeddingGenerator(
#         model_name="allenai/specter2_base", batch_size=32
#     )

#     # generate embeddings for each chunk individually
#     embeddings = []
#     for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating embeddings"):
#         chunk = row.get("chunk", "")
#         if not isinstance(chunk, str) or not chunk.strip():
#             continue
#         embedding = embedding_gen.generate_embeddings([chunk])
#         embeddings.append(embedding)

#     # Add embeddings to the dataframe
#     df["embedding"] = embeddings

#     # Save the dataframe with embeddings
#     df.to_pickle(output_data_path)

# if __name__ == "__main__":
#     main()
