# XSum
**XSum** is a pipeline that generates extended summaries based on the title, abstract, and full text of multiple papers. This repository contains both the XSum pipeline and various utility scripts for data processing and evaluation on the SurveySum Dataset. This code is the implementation used in the paper: "Ask, Retrieve, Summarize: A Modular Pipeline for Scientific Article Summarization".

---

## Project Overview

- **XSum**: Core pipeline that generates extended summaries.
- **Data Utils**: Scripts to collect source docs, chunk text, build indexes, and more.
- **Evaluation**: Scripts to evaluate summary outputs using multiple metrics and checklists.

---

## Directory Structure

```

├── src/
│   ├── data_utils/
│   │   ├── __init__.py     
│   │   ├── chunker.py      # Splits text into manageable chunks
│   │   ├── collect_all_source_docs.py  # Gathers source documents from the SurveySum DS
│   │   ├── data_loading_pipe.py  # Main pipeline for loading and processing data
│   │   ├── embedding_generator.py  # Generates embeddings from text chunks
│   │   ├── faiss_index_builder.py  # Builds FAISS index for efficient similarity search
│   │   ├── faiss_index.py  # Core FAISS indexing functionality
│   │   ├── get_abstract.py  # Retrieves abstracts for SurveySum Papers
│   │   ├── group_df.py  # Groups data into structured dataframe
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── prompts/      # Prompt templates for evaluation
│   │   ├── check_eval.py  # CheckEval Evaluation
│   │   ├── eval_bertscore.py  # Calculates BERTScore for semantic similarity
│   │   ├── eval_checkeval.py  # Runs CheckEval evaluation
│   │   ├── eval_g_eval.py  # Implements G-Eval LLM-based evaluation
│   │   ├── eval_pipe.py  # Main evaluation pipeline combining metrics
│   │   ├── eval_ref_f1.py  # Calculates F1 scores against references
│   │   └── eval_rouge.py  # Performs ROUGE-based evaluation
│   └── xsum/ 
│       ├── __init__.py
│       ├── citation_rag.py  # Handles retrieval and generation with Citations
│       ├── editor.py  # Refines generated content for final output
│       ├── faiss_search.py  # Performs vector searches using FAISS
│       ├── generate_answers_pipe.py  # Pipeline for generating answers
│       └── generate_questions.py  # Creates questions from paper titles/abstracts
│       
├── __init__.py
├── gpt4o_mini.py
├── phi3.py
├── .gitignore              # Git ignore file
├── README.md               # You're reading it now!
└── requirements.txt        # Python dependencies 
```

---

## Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/XSum.git
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the pipeline**:
   Before starting the pipeline, ensure you have a dataset containing papers with the following fields: Title, Abstract and  Full Text.

   1. Data Loading & Processing Pipeline
   
      Start by running the `data_utils/data_loading_pipe.py` script. This script handles the complete pipeline for processing research paper data and requires you to save your input data using the specified placeholders.

   2. Generate Questions

      After processing your research paper data, move on to running the `generate_questions.py` script. This script uses a GPT-based model to create relevant questions for each paper based on its title and abstract.

   3. Generate Answers Pipeline

      After generating questions, move on to running the `generate_answers_pipe.py` script. This pipeline is responsible for generating extended answers or summaries based on the content of your research papers.

   4. Editor Pipeline

      After generating questions and answers, proceed to run the `editor.py` script. This script refines the Q&A content by creating a final, consolidated section text that summarizes the key insights from the answers.
---
