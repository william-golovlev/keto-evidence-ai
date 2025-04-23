# Keto Science RAG (KetoSci AI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) AI assistant using Retrieval-Augmented Generation (RAG) to answer questions based on the public Keto Science Zotero Database.

**Goal:** To provide enthusiasts and researchers with quick, evidence-based answers derived directly from studies listed in the [KetoScienceDatabase Zotero Library](https://www.zotero.org/groups/2466685/ketosciencedatabase/library).

## Features

* Retrieves relevant studies and text excerpts from the Zotero library based on user questions.
* Uses a local Large Language Model (LLM) via [Ollama](https://ollama.com/) to generate answers based *only* on the retrieved context.
* Provides source information for generated answers (e.g., citation key).
* Designed to run locally, ensuring data privacy and avoiding API costs.
* Open-source and community-driven.

## How it Works

This project implements a Retrieval-Augmented Generation (RAG) pipeline:

1.  **Data Acquisition:** Scripts scrape metadata and abstracts from the public Zotero library web view, through the Keto Science Database (https://www.zotero.org/groups/2466685/ketosciencedatabase/)
2.  **Indexing:** Extracted text is cleaned, split into chunks, and converted into vector embeddings using a sentence transformer model. These embeddings are stored in a local vector database (e.g., ChromaDB).
3.  **Retrieval:** When a user asks a question, it's embedded, and the vector database is searched for the most relevant text chunks.
4.  **Generation:** The retrieved chunks and the original question are passed to a local LLM (run via Ollama) to generate a synthesized answer based *only* on the provided context.

## Installation

*(Instructions will be added here once the core scripts are developed. This will likely involve:)*

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/keto-science-rag.git](https://github.com/YourUsername/keto-science-rag.git)
    cd keto-science-rag
    ```
2.  **Set up Python environment:** (e.g., using `venv`)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install Ollama:** Follow the instructions at [https://ollama.com/](https://ollama.com/).
5.  **Download an LLM via Ollama:** (e.g., Mistral, Llama 3 8B)
    ```bash
    ollama pull mistral
    ```
6.  **(Optional) Download Embedding Model:** (Instructions if not automatically handled by libraries)

## Usage

1.  **Add API key** Use your API key i.e. OpenAI key to make the calls needed
2.  ** --OR-- Use Ollama** Locally. 

## Contributing

Contributions are welcome! Please feel free to submit Pull Requests or open Issues to suggest improvements, report bugs, or add features.

*(Add more specific contribution guidelines if desired, e.g., coding style, testing requirements)*

## License

This project is licensed under the [MIT License](LICENSE). ## Disclaimer

This tool provides information based on the content within the specified Zotero library. It is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for any health concerns or before making any decisions related to your health or treatment. Scientific literature can be complex and sometimes contradictory; this tool aims to retrieve relevant information but does not guarantee the validity or applicability of the findings to individual circumstances.

