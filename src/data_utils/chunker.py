"""
This module provides a utility class for chunking text into smaller, overlapping segments.
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

# Download NLTK tokenization data if not already present
nltk.download("punkt", quiet=True)


class Chunker:
    """
    A utility class for chunking text into smaller, overlapping segments.

    This class breaks down large text documents into manageable chunks
    with configurable size and overlap, preserving sentence boundaries
    where possible.
    """

    def __init__(self, max_tokens=150, overlap_tokens=20):
        """
        Initialize the Chunker with size parameters.

        Parameters:
            max_tokens (int): Maximum number of tokens per chunk
            overlap_tokens (int): Number of tokens to overlap between adjacent chunks
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def tokenize_sentences(self, text):
        """
        Split text into sentences using NLTK.

        Parameters:
            text (str): The input text to split

        Returns:
            list: List of sentences
        """
        return sent_tokenize(text)

    def tokenize_words(self, text):
        """
        Split text into word tokens using NLTK.

        Parameters:
            text (str): The text to tokenize

        Returns:
            list: List of word tokens
        """
        return word_tokenize(text)

    def split_long_sentence(self, sentence):
        """
        Split a long sentence by commas if it exceeds the max token limit.

        Parameters:
            sentence (str): The sentence to split

        Returns:
            list: List of sub-sentences
        """
        return [sub.strip() for sub in sentence.split(",")]

    def chunk_text(self, text):
        """
        Split text into overlapping chunks without exceeding the token limit.

        Parameters:
            text (str): The full text to chunk

        Returns:
            list: List of text chunks
        """
        sentences = self.tokenize_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            tokens = self.tokenize_words(sentence)
            token_count = len(tokens)

            if token_count > self.max_tokens:
                # Handle sentences that are longer than max_tokens
                sub_sentences = self.split_long_sentence(sentence)
                for sub in sub_sentences:
                    sub_tokens = self.tokenize_words(sub)
                    sub_token_count = len(sub_tokens)

                    if sub_token_count > self.max_tokens:
                        # If sub-sentence is still too long, break it by tokens
                        words = sub_tokens
                        for i in range(0, len(words), self.max_tokens):
                            part = words[i : i + self.max_tokens]
                            part_text = " ".join(part)
                            chunks, current_chunk, current_length = self._add_chunk(
                                part_text, chunks, current_chunk, current_length
                            )
                    else:
                        # Add sub-sentence as is
                        chunks, current_chunk, current_length = self._add_chunk(
                            sub, chunks, current_chunk, current_length
                        )
            else:
                # Check if adding this sentence would exceed max_tokens
                if current_length + token_count > self.max_tokens:
                    # Add current chunk to chunks and create new chunk with overlap
                    chunks.append(" ".join(current_chunk))
                    # Start new chunk with overlap from previous chunk
                    overlap = (
                        current_chunk[-self.overlap_tokens :]
                        if self.overlap_tokens < len(current_chunk)
                        else current_chunk
                    )
                    current_chunk = overlap.copy()
                    current_length = len(self.tokenize_words(" ".join(current_chunk)))

                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += token_count

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _add_chunk(self, text, chunks, current_chunk, current_length):
        """
        Helper method to add text to chunks while managing overlaps.

        Parameters:
            text (str): Text to add
            chunks (list): Existing chunks
            current_chunk (list): Current chunk being built
            current_length (int): Current token count in the chunk

        Returns:
            tuple: Updated (chunks, current_chunk, current_length)
        """
        tokens = self.tokenize_words(text)
        token_count = len(tokens)

        if current_length + token_count > self.max_tokens:
            # Current chunk would exceed max_tokens, finalize it
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Start new chunk with overlap from previous chunk
            overlap = (
                current_chunk[-self.overlap_tokens :]
                if current_chunk and self.overlap_tokens < len(current_chunk)
                else []
            )
            current_chunk = overlap.copy()
            current_length = len(overlap)

        # Add new text to current chunk
        current_chunk.append(text)
        current_length += token_count

        return chunks, current_chunk, current_length


# Example usage:
# if __name__ == "__main__":
#     # Load data
#     input_path = "path/to/input/data.pkl"
#     output_path = "path/to/output/data_chunks.pkl"
#
#     df = pd.read_pickle(input_path)
#
#     # Initialize Chunker
#     chunker = Chunker(max_tokens=150, overlap_tokens=20)
#
#     # Apply chunking to each document
#     df['chunks'] = df['full_text'].apply(chunker.chunk_text)
#
#     # Explode chunks into separate rows
#     df = df.explode('chunks').rename(columns={'chunks': 'chunk'})
#
#     # Save results
#     df.to_pickle(output_path)
