# Keto Science RAG Project

This project uses Retrieval-Augmented Generation (RAG) to answer specific questions about the ketogenic diet and meat, based on a collection of over 500 research papers and abstracts.

## Setup

Follow these steps to set up the project locally.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Willy988/keto-science-rag-chromadb.git](https://github.com/Willy988/keto-science-rag-chromadb.git)
    cd keto-science-rag-chromadb
    ```
2.  **Set up Python environment:** (Python 3.11 or 3.12 recommended for compatibility)
    ```bash
    # Example using Python 3.12
    py -3.12 -m venv .venv

    # Activate the environment (use command for your shell)
    # Windows cmd:
    .venv\Scripts\activate.bat
    # Windows PowerShell:
    .\.venv\Scripts\Activate.ps1
    # macOS/Linux:
    source .venv/bin/activate
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install Ollama:** (Required if using the Ollama provider) Follow instructions at [https://ollama.com/](https://ollama.com/). Ensure the Ollama application/server is running.
5.  **Download Ollama Models:** (Required if using the Ollama provider) Download the embedding model and the LLM you intend to use.
    ```bash
    # Embedding model (e.g., nomic-embed-text, used as default in ingest.py)
    ollama pull nomic-embed-text

    # LLM (e.g., llama3, used as default in ingest.py and query.py)
    ollama pull llama3
    ```
6.  **Configure Environment:**
    * Create a `.env` file in the project root directory (you can copy `.env.example` if provided).
    * Edit the `.env` file and set the following variables:
        * `LLM_PROVIDER`: Choose your provider (e.g., `"ollama"`, `"google"`, `"openai"`).
        * `GOOGLE_API_KEY` / `OPENAI_API_KEY`: Add your API key if using a cloud provider.
        * `OLLAMA_BASE_URL`: Verify if using Ollama (default is `http://localhost:11434`).
        * `DATA_DIR`: Set this only if you plan to run ingestion (Option B below). Point it to your folder containing source PDFs/HTML.
        * `CHROMA_PERSIST_DIR`: Location for the database (default is usually `./chroma_data`).
        * `CHROMA_COLLECTION_NAME`: Name for the database collection (e.g., `research_papers`).

## Usage: Querying the Database

There are two ways to get the necessary ChromaDB vector database:

**Option A: Download Pre-Built Database (Recommended)**

1.  **Download:** Download the `chroma_data.zip` file containing the pre-processed database from the project's Hugging Face Dataset repository:
    * **[Willy988/keto-science-rag-chromadb on Hugging Face Hub](https://huggingface.co/datasets/Willy988/keto-science-rag-chromadb)**
    * Click on the "Files and versions" tab to find the `chroma_data.zip` file.
2.  **Extract:** Unzip the downloaded `chroma_data.zip` file.
3.  **Place:** Move the extracted `chroma_data` folder into the root directory of this project (alongside `query.py`).
4.  **Configure:** Ensure `CHROMA_PERSIST_DIR=./chroma_data` is set in your `.env` file.

**Option B: Build Database Locally (Requires Source Documents & External Tools)**

If you prefer to build the database from source documents yourself using `ingest.py`:

1.  **Place Source Documents:** Put your collection of PDF and HTML files inside a directory (e.g., create a folder named `source_docs`) within the project.
2.  **Update `.env`:** Set the `DATA_DIR` variable in your `.env` file to point to this source document folder (e.g., `DATA_DIR=./source_docs`).
3.  **(IMPORTANT) Install External PDF Tools:** For robust processing of diverse PDF files (especially complex layouts or scanned images), install **Poppler** and **Tesseract OCR** and ensure they are added to your system's PATH environment variable:
    * **Poppler:** Used for reliable text extraction.
        * Windows builds: [oschwartz10612/poppler-windows Releases](https://github.com/oschwartz10612/poppler-windows/releases)
        * *Add the extracted `Library\bin` folder to your system PATH.*
        * *Verify (in a new terminal): `pdftotext -v`*
    * **Tesseract OCR:** Used for extracting text from scanned PDFs.
        * Windows installers: [UB-Mannheim/tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki)
        * *Ensure Tesseract is added to PATH during installation, or add the main installation folder (containing `tesseract.exe`) to PATH manually.*
        * *Verify (in a new terminal): `tesseract --version`*
    * *(Remember to restart your terminal/command prompt after modifying the system PATH for the changes to take effect).*
4.  **Run Ingestion Script:** Execute the ingestion script from your activated virtual environment:
    ```bash
    python ingest.py
    ```
    *(This will process the documents in `DATA_DIR` and create/populate the database in the location specified by `CHROMA_PERSIST_DIR`)*

**Running Queries**

Once the `chroma_data` database is ready (either downloaded or built) and your `.env` file is correctly configured:

1.  Ensure your chosen LLM provider (Ollama server or cloud API) is running and accessible.
2.  Run the query script from your activated virtual environment:
    ```bash
    python query.py
    ```
3.  Enter your questions when prompted. Type `quit` to exit.

## Docker Usage

(Instructions for building and running with Docker will be added here later)

