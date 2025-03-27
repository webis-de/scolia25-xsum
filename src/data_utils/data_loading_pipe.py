"""
Data Processing Pipeline for Research Paper Indexing

This script handles the entire pipeline for processing research paper data:
1. Loading raw documents
2. Chunking text into manageable segments
3. Generating embeddings for each chunk
4. Building a FAISS index for efficient similarity search
"""

import os
import pandas as pd
from tqdm import tqdm
from chunker import Chunker
from embedding_generator import EmbeddingGenerator
from faiss_index_builder import FaissIndexBuilder


def main():
    """
    Execute the complete data processing pipeline from raw documents to indexed embeddings.
    """
    # Define base directory (adjust as needed)
    base_dir = os.path.abspath(".")
    
    # Set your file paths using placeholders
    data_dir = os.path.join(base_dir, "<PATH_TO_SOURCE_DOCS>")
    db_dir = os.path.join(base_dir, "<PATH_TO_DB>")
    
    # Ensure output directory exists
    os.makedirs(db_dir, exist_ok=True)

    # Input/output paths with placeholders
    input_data_path = os.path.join(data_dir, "source_docs_full.pkl")
    chunks_path = os.path.join(data_dir, "source_docs_full_chunks.pkl")
    embed_path = os.path.join(data_dir, "source_docs_full_embed.pkl")
    index_path = os.path.join(db_dir, "faiss_index.bin")
    metadata_path = os.path.join(db_dir, "metadata.pkl")

    # Load the input data
    df = pd.read_pickle(input_data_path)

    # Step 1: Text chunking
    # Initialize chunker with specific token limits
    chunker = Chunker(max_tokens=150, overlap_tokens=20)
    df["chunks"] = df["full_text"].apply(lambda x: chunker.chunk_text(x))
    df = df.explode("chunks").rename(columns={"chunks": "chunk"})
    df.to_pickle(chunks_path)
    print(f"Chunked data saved to {chunks_path}")

    # Step 2: Embedding generation
    embedding_gen = EmbeddingGenerator(
        model_name="allenai/specter2_base", batch_size=32
    )
    embeddings = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating embeddings"):
        chunk = row.get("chunk", "")
        if not isinstance(chunk, str) or not chunk.strip():
            continue
        embedding = embedding_gen.generate_embeddings([chunk])
        embeddings.append(embedding)
    df["embedding"] = embeddings
    df.to_pickle(embed_path)
    print(f"Embedded data saved to {embed_path}")

    # Step 3: FAISS index building
    METADATA_COLUMNS = [
        "abstract",
        "chunk",
        "title",
        "paper_id",
        "citation_id",
        "survey_id",
        "survey_title",
        "section_title",
    ]
    builder = FaissIndexBuilder(
        data_path=embed_path,
        index_path=index_path,
        metadata_path=metadata_path,
        metadata_columns=METADATA_COLUMNS,
    )
    builder.run()
    print(f"FAISS index built and saved to {index_path}")


if __name__ == "__main__":
    main()
