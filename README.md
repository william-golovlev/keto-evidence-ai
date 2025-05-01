    # Keto Science RAG Project

    Uses research papers (over 500) to answer specific questions to users about the ketogenic diet and meat.

    ## Setup

    1.  **Clone the repository:**
        ```bash
        git clone [https://github.com/Willy988/keto-science-rag-chromadb.git](https://github.com/Willy988/keto-science-rag-chromadb.git)
        cd keto-science-rag-chromadb
        ```
    2.  **Set up Python environment:** (Python 3.11 or 3.12 recommended)
        ```bash
        # Example using Python 3.12
        py -3.12 -m venv .venv
        # Activate (use command for your shell)
        .venv\Scripts\activate.bat
        ```
    3.  **Install Python dependencies:**
        ```bash
        pip install -r requirements.txt
        ```
    4.  **Install Ollama:** (Required if using Ollama provider) Follow instructions at [https://ollama.com/](https://ollama.com/).
    5.  **Download Ollama Models:** (Required if using Ollama provider)
        ```bash
        ollama pull nomic-embed-text
        ollama pull llama3
        ```
    6.  **Configure Environment:** Copy `.env.example` (you should create this) to `.env` and fill in your settings:
        * Set `LLM_PROVIDER` (e.g., "ollama", "google").
        * Add API keys if using cloud providers (`GOOGLE_API_KEY`).
        * Verify `CHROMA_PERSIST_DIR` (usually `./chroma_data`).
        * Verify `CHROMA_COLLECTION_NAME` (e.g., `research_papers`).

    ## Usage: Querying the Database

    There are two ways to get the necessary ChromaDB database:

    **Option A: Download Pre-Built Database (Recommended)**

    1.  Download `chroma_data.zip` from the Hugging Face Hub: [https://huggingface.co/datasets/Willy988/keto-science-rag-chromadb](https://huggingface.co/datasets/Willy988/keto-science-rag-chromadb)
    2.  Extract the zip file.
    3.  Place the resulting `chroma_data` folder in the root of this project directory.
    4.  Ensure `CHROMA_PERSIST_DIR=./chroma_data` is set in your `.env` file.

    **Option B: Build Database Locally (Requires Source Documents & External Tools)**

    If you want to build the database from source documents yourself using `ingest.py`:

    1.  **Place Source Documents:** Put your PDF and HTML files inside a directory (e.g., create a folder named `source_docs`) within the project.
    2.  **Update `.env`:** Set `DATA_DIR` in your `.env` file to point to your source document folder (e.g., `DATA_DIR=./source_docs`).
    3.  **(IMPORTANT) Install External PDF Tools:** For robust processing of diverse PDF files, install **Poppler** and **Tesseract OCR** and add them to your system PATH:
        * **Poppler:** Used for reliable text extraction.
            * Windows builds: [oschwartz10612/poppler-windows Releases](https://github.com/oschwartz10612/poppler-windows/releases)
            * *Add the extracted `Library\bin` folder to PATH.*
            * *Verify (new terminal): `pdftotext -v`*
        * **Tesseract OCR:** Used for extracting text from scanned PDFs.
            * Windows installers: [UB-Mannheim/tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki)
            * *Ensure Tesseract is added to PATH (add install folder containing `tesseract.exe`).*
            * *Verify (new terminal): `tesseract --version`*
        * *(Restart terminal after modifying PATH)*.
    4.  **Run Ingestion Script:**
        ```bash
        python ingest.py
        ```
        *(This will create the database in the location specified by `CHROMA_PERSIST_DIR`)*

    **Running Queries**

    Once the database is ready (either downloaded or built) and your `.env` is configured:

    1.  Ensure your chosen LLM provider (Ollama server or cloud) is running/accessible.
    2.  Run the query script:
        ```bash
        python query.py
        ```
    3.  Enter your questions at the prompt. Type 'quit' to exit.

    ## Docker Usage

    (Instructions for building and running with Docker will be added here later)

    ```
