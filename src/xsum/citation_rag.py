"""
This module implements a Retrieval-Augmented Generation (RAG) chatbot that searches
through a vector database of scientific papers and generates answers with proper citations.
It uses FAISS for efficient similarity search and GPT4O Mini for response generation.
"""

import numpy as np
import logging
from dotenv import load_dotenv
from gpt4o_mini import GPT4OMini
from faiss_search import FaissSearcher
from llama_index.core.prompts import PromptTemplate
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Paths to the FAISS index and metadata files (using relative paths)
FAISS_INDEX_PATH = "PLACEHOLDER_FAISS_INDEX_PATH"  # Placeholder for the actual FAISS index path
METADATA_PATH = "PLACEHOLDER_METADATA_PATH"  # Placeholder for the actual metadata path

# Dimension of the embeddings
EMB_DIMENSION = 768

# Number of top similar results to retrieve per FAISS query
TOP_K_FAISS = 100

# Number of top results to display after reranking
TOP_K_FINAL = 20

# System prompt for the LLM to ensure it behaves as a retrieval assistant
SYSTEM_PROMPT = """You are a retrieval AI assistant that helps to search for information in a vector database and present the answers in a scientific and neutral style.
DO NOT HALLUCINATE and ONLY use the information given to you without using your own knowledge."""


class RAGChatbot:
    """
    A RAG-based chatbot that retrieves relevant information from a FAISS index
    and generates answers with proper citations using GPT4O Mini.
    """

    def __init__(self):
        """
        Initializes the RAGChatbot with a FaissSearcher and GPT4OMini.
        """
        # Initialize FaissSearcher with the vector database
        try:
            self.searcher = FaissSearcher(
                faiss_index_path=FAISS_INDEX_PATH,
                metadata_path=METADATA_PATH,
                embedding_model_name="allenai/specter2_base",  
                embedding_batch_size=16,
                dimension=EMB_DIMENSION,
                top_k_faiss=TOP_K_FAISS,
                top_k_final=TOP_K_FINAL,
                metadata_filters={},  
            )
            print("FaissSearcher initialized successfully.")
        except Exception as e:
            print(f"Error initializing FaissSearcher: {e}")
            exit(1)

        # Initialize GPT4OMini with the system prompt
        self.gpt = GPT4OMini(system_prompt=SYSTEM_PROMPT)

        print("RAGChatbot initialized successfully.")

    def generate_prompt(self, query, context, existing_answer=None):
        """
        Generates a prompt for the LLM based on the question and retrieved text.

        Parameters:
        - query (str): The question to answer.
        - context (str): The context retrieved from the search.
        - existing_answer (str, optional): An existing answer to refine.

        Returns:
        - str: Generated prompt for the LLM.
        """
        # Generate initial prompt if no existing answer
        if existing_answer is None:
            prompt = f"""Please provide an answer based solely on the provided sources. 
                        When referencing information from a source, cite the appropriate source(s) using their BIBREF given in square brackets. 
                        Every answer should include at least one source citation. 
                        Only cite a source when you are explicitly referencing it. 
                        If none of the sources are helpful, you should indicate that. 

                        For example:

                        Source [BIBREF99]:

                        The sky is red in the evening and blue in the morning.

                        Source [BIBREF87]:

                        Water is wet when the sky is red.

                        Query: When is water wet?

                        Answer: Water will be wet when the sky is red [BIBREF87], which occurs in the evening [BIBREF99].

                        Now it's your turn. Below are several numbered sources of information:

                        ------ 

                        {context}

                        ------ 

                        Query: {query}

                        Answer: """
        else:
            # Generate refinement prompt using existing answer
            prompt = f"""Please provide an answer based solely on the provided sources. 
                        When referencing information from a source, 
                        cite the appropriate source(s) using their BIBREF given in square brackets. 
                        Every answer should include at least one source citation. 
                        Only cite a source when you are explicitly referencing it. 
                        If none of the sources are helpful, you should indicate that. 

                        For example:

                        Source [BIBREF99]:

                        The sky is red in the evening and blue in the morning.

                        Source [BIBREF87]:

                        Water is wet when the sky is red.

                        Query: When is water wet?

                        Answer: Water will be wet when the sky is red [BIBREF87], 
                        which occurs in the evening [BIBREF99].

                        Now it's your turn. 
                        We have provided an existing answer: {existing_answer}
                        Below are several numbered sources of information. 
                        Use them to refine the existing answer. 
                        If the provided sources are not helpful, you will repeat the existing answer.

                        Begin refining!

                        ------ 

                        {context}

                        ------ 

                        Query: {query}

                        Answer: """
        return prompt

    def query_faiss(self, query, citation_id):
        """
        Queries the FaissSearcher with the given query and citation_id.

        Parameters:
        - query (str): The query string.
        - citation_id (str): The citation ID to filter results.

        Returns:
        - List[Dict[str, Any]]: List of reranked search results.
        """
        # Set filter to only return chunks from the specified citation
        metadata_filters = {"citation_id": citation_id}
        # Perform search and reranking
        reranked_results = self.searcher.search(
            query_text=query, metadata_filters=metadata_filters
        )
        return reranked_results

    def answer_question(self, question, citation_id):
        """
        Answers the given question using retrieved and reranked information.

        Parameters:
        - question (str): The question to answer.
        - citation_id (str): The citation ID to filter search results.

        Returns:
        - str: The final answer generated by GPT4OMini.
        """
        # Retrieve reranked search results filtered by citation_id
        query_results = self.query_faiss(question, citation_id)

        # Assemble context from search results
        context = ""
        for result in query_results:
            node_chunk = f"Source [{result['citation_id']}]:\n\n{result['chunk']}\n\n"
            context += node_chunk

        # Generate initial prompt and get first-pass response
        prompt = self.generate_prompt(question, context)
        response = self.gpt.get_response(prompt)

        # Generate refining prompt using the initial response for a second-pass refinement
        prompt = self.generate_prompt(question, context, response)
        final_response = self.gpt.get_response(prompt)

        return final_response



