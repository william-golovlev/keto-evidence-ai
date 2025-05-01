import os
import logging
import glob
import re # Import regular expressions for finding headings
from dotenv import load_dotenv
from bs4 import BeautifulSoup # Import BeautifulSoup for HTML parsing

# --- LangChain Core Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # For creating Document objects manually

# --- LLM & Embedding Imports ---
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- Vector Store Imports ---
import chromadb
from langchain_chroma import Chroma

# --- Optional for progress bar ---
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        logging.warning("tqdm not installed, progress bar disabled. Run 'pip install tqdm'")
        return iterable

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function for HTML Abstract Extraction ---
def find_abstract_or_summary(soup: BeautifulSoup) -> str | None:
    """
    Tries to find and extract text under 'Abstract' or 'Summary' headings.

    Args:
        soup: A BeautifulSoup object representing the parsed HTML.

    Returns:
        The extracted text as a string, or None if not found.
    """
    # Case-insensitive search for headings (h1, h2, h3, h4) containing "Abstract" or "Summary"
    # Adjust tags (e.g., add 'strong', 'b') or keywords if needed based on your HTML structure
    abstract_heading = soup.find(['h1', 'h2', 'h3', 'h4'], string=re.compile(r'^\s*(abstract|summary)\s*$', re.IGNORECASE))

    if abstract_heading:
        # Find the next sibling paragraph(s) after the heading
        content = []
        next_sibling = abstract_heading.find_next_sibling()
        while next_sibling and next_sibling.name == 'p':
            content.append(next_sibling.get_text(separator=' ', strip=True))
            next_sibling = next_sibling.find_next_sibling()

        if content:
            return "\n".join(content)

    # Fallback: Look for common div/section IDs or classes (examples, adjust as needed)
    abstract_div = soup.find(['div', 'section'], id=re.compile(r'abstract', re.IGNORECASE)) or \
                   soup.find(['div', 'section'], class_=re.compile(r'abstract', re.IGNORECASE))
    if abstract_div:
         # Get all paragraph text within this container
         paragraphs = abstract_div.find_all('p')
         if paragraphs:
              return "\n".join([p.get_text(separator=' ', strip=True) for p in paragraphs])

    return None # Return None if no abstract/summary found

# --- Main Function ---
def main():
    load_dotenv()
    logging.info("Loaded environment variables.")

    # --- Configuration ---
    data_dir = os.getenv("DATA_DIR")
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME")
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("MODEL_NAME")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")

    # --- Validate Config ---
    # (Add back the validation checks for data_dir, chroma_persist_dir etc. from previous script)
    if not data_dir or not os.path.isdir(data_dir):
        logging.error(f"DATA_DIR invalid: {data_dir}. Exiting.")
        return
    if not chroma_persist_dir:
        logging.error("CHROMA_PERSIST_DIR not set. Exiting.")
        return
    try:
        os.makedirs(chroma_persist_dir, exist_ok=True)
    except OSError as error:
        logging.error(f"Error creating ChromaDB directory {chroma_persist_dir}: {error}. Exiting.")
        return
    if not chroma_collection_name:
        chroma_collection_name = "research_papers"
    logging.info(f"Config - Data: {data_dir}, DB: {chroma_persist_dir}, Collection: {chroma_collection_name}")


    # --- Initialize Embeddings ---
    embeddings = None
    try:
        logging.info(f"Initializing provider: {provider}")
        # (Keep the if/elif/else block for initializing embeddings based on provider)
        if provider == "google":
            if not google_api_key: raise ValueError("GOOGLE_API_KEY not set")
            embed_model_name = embedding_model_name or "models/embedding-001"
            embeddings = GoogleGenerativeAIEmbeddings(model=embed_model_name, google_api_key=google_api_key)
        elif provider == "openai":
            if not openai_api_key: raise ValueError("OPENAI_API_KEY not set")
            embed_model_name = embedding_model_name or "text-embedding-3-small"
            embeddings = OpenAIEmbeddings(model=embed_model_name, openai_api_key=openai_api_key)
        elif provider == "ollama":
            embed_model_name = embedding_model_name or "nomic-embed-text"
            embeddings = OllamaEmbeddings(base_url=ollama_base_url, model=embed_model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logging.info("Testing embedding model connection...")
        _ = embeddings.embed_query("Test query")
        logging.info("Embedding model connection successful.")
    except Exception as e:
        logging.error(f"Fatal Error initializing embeddings: {e}")
        return

    # --- 1. Load Documents (PDFs and HTML Abstracts) ---
    all_documents = [] # Will hold all loaded pages/abstracts
    skipped_files = 0
    processed_files = 0

    # --- Load PDFs ---
    pdf_files = glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True)
    logging.info(f"Found {len(pdf_files)} PDF files. Starting PDF loading...")
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            pdf_loader = PyPDFLoader(pdf_path, extract_images=False)
            pages = pdf_loader.load()
            if pages:
                 all_documents.extend(pages)
                 processed_files += 1
            else:
                 logging.warning(f"No pages loaded from PDF {pdf_path}. Skipping.")
                 skipped_files += 1
        except Exception as e:
            logging.warning(f"Error loading PDF {pdf_path}: {type(e).__name__} - {e}. Skipping.")
            skipped_files += 1

    # --- Load HTML Abstracts ---
    html_files = glob.glob(os.path.join(data_dir, "**/*.html"), recursive=True)
    html_files.extend(glob.glob(os.path.join(data_dir, "**/*.htm"), recursive=True)) # Also find .htm
    logging.info(f"Found {len(html_files)} HTML files. Starting HTML abstract extraction...")
    processed_html = 0
    for html_path in tqdm(html_files, desc="Processing HTML"):
        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'lxml') # Use lxml parser
            abstract_text = find_abstract_or_summary(soup)

            if abstract_text:
                # Create a LangChain Document object for the abstract
                metadata = {"source": html_path, "content_type": "abstract"}
                doc = Document(page_content=abstract_text, metadata=metadata)
                all_documents.append(doc)
                processed_files += 1
                processed_html += 1
            else:
                # logging.info(f"No abstract/summary found in {html_path}. Skipping.") # Optional: log skips
                skipped_files += 1

        except Exception as e:
            logging.warning(f"Error processing HTML file {html_path}: {type(e).__name__} - {e}. Skipping.")
            skipped_files += 1

    if not all_documents:
        logging.error("No documents (PDF pages or HTML abstracts) were successfully loaded. Exiting.")
        return

    logging.info(f"Successfully loaded content: {len(all_documents)} items (PDF pages + HTML abstracts) from {processed_files} files ({skipped_files} files skipped).")
    logging.info(f"Extracted abstracts from {processed_html} HTML files.")

    # --- 2. Chunk Documents ---
    logging.info("Splitting combined documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    try:
        split_docs = text_splitter.split_documents(all_documents) # Split the combined list
        total_chunks = len(split_docs)
        if total_chunks == 0:
            logging.warning("No text chunks generated after splitting. Exiting.")
            return
        logging.info(f"Split {len(all_documents)} items into {total_chunks} chunks.")
    except Exception as e:
        logging.error(f"Fatal Error during document splitting: {e}")
        return

    # --- 3. Embed and Store in ChromaDB (BATCH PROCESSING) ---
    logging.info(f"Initializing ChromaDB client to persist data at: {chroma_persist_dir}")
    logging.info(f"Using collection name: {chroma_collection_name}")
    try:
        persistent_client = chromadb.PersistentClient(path=chroma_persist_dir)
        vector_store = Chroma(
            client=persistent_client,
            collection_name=chroma_collection_name,
            embedding_function=embeddings,
        )

        batch_size = 500
        logging.info(f"Starting to add {total_chunks} chunks to ChromaDB in batches of {batch_size}...")
        for i in tqdm(range(0, total_chunks, batch_size), desc="Embedding Batches"):
            batch = split_docs[i:i + batch_size]
            if not batch: continue
            start_index = i
            end_index = min(i + batch_size, total_chunks)
            try:
                vector_store.add_documents(documents=batch)
            except Exception as batch_error:
                logging.error(f"Error adding batch {start_index+1}-{end_index}: {batch_error}")
                logging.warning("Continuing to next batch despite error.")

        logging.info("Finished processing all batches.")
        count = vector_store._collection.count()
        logging.info(f"Collection '{chroma_collection_name}' now contains approximately {count} documents.")

    except Exception as e:
        logging.error(f"Fatal Error during ChromaDB processing: {e}")
        return

    logging.info("Ingestion pipeline completed successfully.")

# --- Standard Python entry point ---
if __name__ == "__main__":
    main()
