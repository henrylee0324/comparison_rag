Put the files in the following link into ./embeddings folder:
https://drive.google.com/drive/folders/1Ytee7kna15p-16EvJW5pCs169i3d2s2b?usp=drive_link

# Project Overview

This project is a comprehensive natural language processing (NLP) pipeline that integrates **Classification**, **Extract-Transform-Load (ETL)**, and **Retrieval-Augmented Generation (RAG)** capabilities. It leverages OpenAI's large language models (LLM) and embedding models to classify, process, and retrieve document information effectively.

## Features

### 1. Classification (`classify.py`)
- **Objective:** Categorize documents into two groups based on user-defined conditions.
- **Key Functions:**
  - Dynamically generate group names using LLM.
  - Classify documents by predicting group membership using LLM.
  - Output the classification results in a structured JSON file.

### 2. ETL (`etl.py`)
- **Objective:** Convert documents and glossaries into embeddings for downstream retrieval and analysis.
- **Key Functions:**
  - Chunk documents into smaller sections with overlap.
  - Generate summaries for each chunk using LLM.
  - Create and save embeddings using OpenAI embedding models.

### 3. Retrieval-Augmented Generation (RAG) (`rag.py`)
- **Objective:** Retrieve relevant information from documents and generate AI responses.
- **Key Functions:**
  - Retrieve document chunks based on cosine similarity or other scoring methods.
  - Support queries with glossary and grouped document embeddings.
  - Compare responses between document groups and generate insights.
  - Save query results, retrieved chunks, and comparisons in JSON files.

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file with the following content:
```env
API_KEY=<your_openai_api_key>
LLM_MODEL=<your_openai_llm_model>
EMBEDDING_MODEL=<your_openai_embedding_model>
```

## Usage

### 1. Classification
Run the classifier to categorize documents into two groups:
```bash
python classify.py
```
You will be prompted to enter the classification condition and the folder path containing the documents.

#### Functions
- **`Classifier.get_group_name()`**
  - **Input:** None (uses initialization parameters).
  - **Output:** Returns the names of the two groups.
  - **Usage:**
    ```python
    classifier = Classifier("condition")
    group1, group2 = classifier.get_group_name()
    print(group1, group2)
    ```
- **`Classifier.classify_documents()`**
  - **Input:** Text files in the specified folder.
  - **Output:** A JSON file with classified groups.
  - **Usage:**
    ```python
    classifier = Classifier("condition")
    group_file = classifier.classify_documents()
    print(group_file)
    ```

### 2. ETL (Embedding Creation)
Generate embeddings for documents and glossaries:
```bash
python etl.py
```

#### Functions
- **`Etl.create_document_embeddings()`**
  - **Input:** Text files in the `documents` folder.
  - **Output:** A JSON file containing document embeddings.
  - **Usage:**
    ```python
    etl = Etl()
    etl.create_document_embeddings()
    ```
- **`Etl.create_glossary_embeddings()`**
  - **Input:** Text files in the `glossary` folder.
  - **Output:** A JSON file containing glossary embeddings.
  - **Usage:**
    ```python
    etl = Etl()
    etl.create_glossary_embeddings()
    ```

### 3. Retrieval-Augmented Generation
Perform queries on the document embeddings:
```bash
python rag.py
```

#### Functions
- **`Rag.query(query, embedding_file_name)`**
  - **Input:** A query string and the name of the embedding JSON file.
  - **Output:** Response from LLM and the top document chunks.
  - **Usage:**
    ```python
    rag = Rag()
    response, chunks = rag.query("What is AI?", "documents.json")
    print(response, chunks)
    ```
- **`Rag.query_with_group(query, group_file_name, embedding_file_name)`**
  - **Input:** A query string, group JSON file name, and embedding JSON file name.
  - **Output:** Responses for both groups, their chunks, and a comparison report.
  - **Usage:**
    ```python
    rag = Rag()
    response1, chunks1, response2, chunks2, comparison = rag.query_with_group("What is AI?", "group.json", "documents.json")
    print(response1, response2, comparison)
    ```
- **`Rag.query_with_glossary(query, embedding_file_name)`**
  - **Input:** A query string and the name of the embedding JSON file.
  - **Output:** Response from LLM, document chunks, and glossary chunks.
  - **Usage:**
    ```python
    rag = Rag()
    response, chunks, glossary_chunks = rag.query_with_glossary("What is AI?", "documents.json")
    print(response, chunks, glossary_chunks)
    ```
- **`Rag.query_with_group_and_glossary(query, group_file_name, embedding_file_name)`**
  - **Input:** A query string, group JSON file name, and embedding JSON file name.
  - **Output:** Responses for both groups, document chunks, glossary chunks, and a comparison report.
  - **Usage:**
    ```python
    rag = Rag()
    response1, chunks1, response2, chunks2, comparison = rag.query_with_group_and_glossary("What is AI?", "group.json", "documents.json")
    print(response1, response2, comparison)
    ```

## Output Files
- **Classification:**
  - JSON file in the `group` folder, e.g., `20241126_165421.json`, containing group names and associated documents.
- **ETL:**
  - JSON file in the `embeddings` folder, e.g., `documents.json`, containing embeddings and metadata for chunks.
- **RAG:**
  - JSON files in the `outputs` folder, containing query results, retrieved chunks, and comparisons.

## Folder Structure
```
project/
├── classify.py
├── etl.py
├── rag.py
├── .env
├── documents/       # Input documents
├── glossary/        # Glossary files
├── embeddings/      # Embedding JSON files
├── group/           # Classification result files
├── outputs/         # Query result files
└── requirements.txt # Dependency file
```

## Error Handling
- Implements retry mechanisms for API calls to handle transient errors.
- Logs descriptive error messages for debugging.

## Future Improvements
1. Optimize large file handling with multi-threading or batching.
2. Add support for additional similarity metrics.
3. Enhance error logging and monitoring.
4. Parameterize configurations via YAML or JSON for flexibility.

## License
This project is licensed under the MIT License. See `LICENSE` for more information.

## Contact
For questions or suggestions, please contact [Your Name] at [your.email@example.com].
