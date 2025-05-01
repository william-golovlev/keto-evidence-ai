import os
import logging
from dotenv import load_dotenv

# --- LangChain Core Imports ---
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- LLM & Embedding Imports ---
# Using community imports as determined during previous debugging
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
# Keep others for potential future use if provider changes
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- Vector Store Imports ---
import chromadb # Import the chromadb client library
from langchain_chroma import Chroma # Import the LangChain Chroma wrapper

# --- Basic Logging Setup ---
# Less verbose for querying, set to WARNING or ERROR for cleaner output
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("chromadb").setLevel(logging.ERROR) # Suppress noisy ChromaDB logs

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    # --- Load Environment Variables ---
    load_dotenv()
    logging.info("Loaded environment variables from .env file.")

    # --- Configuration from Environment Variables ---
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME")
    provider = os.getenv("LLM_PROVIDER", "ollama").lower() # Default to ollama
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("MODEL_NAME") # Optional override for LLM
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME") # Optional override for Embeddings

    # --- Validate Essential Configuration ---
    if not chroma_persist_dir or not os.path.isdir(chroma_persist_dir):
        print(f"ERROR: ChromaDB persist directory not found or not specified in .env: {chroma_persist_dir}")
        return
    if not chroma_collection_name:
        print("ERROR: CHROMA_COLLECTION_NAME not set in .env file.")
        return

    print(f"--- Configuration ---")
    print(f"Using LLM Provider: {provider}")
    print(f"ChromaDB Path: {chroma_persist_dir}")
    print(f"Collection Name: {chroma_collection_name}")
    print(f"--------------------")

    # --- Initialize Core Components ---
    llm = None
    embeddings = None
    try:
        # 1. Initialize Embedding Model (MUST match ingestion)
        logging.info(f"Initializing embeddings for provider: {provider}")
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
        logging.info(f"Using Embedding Model: {embed_model_name}")

        # 2. Connect to Existing Vector Store
        logging.info("Connecting to existing ChromaDB...")
        persistent_client = chromadb.PersistentClient(path=chroma_persist_dir)
        vector_store = Chroma(
            client=persistent_client,
            collection_name=chroma_collection_name,
            embedding_function=embeddings, # Use the same embedding function!
        )
        # Check if collection exists and has items (optional but good)
        try:
             count = vector_store._collection.count()
             if count == 0:
                  print(f"WARNING: ChromaDB collection '{chroma_collection_name}' exists but is empty.")
             else:
                  print(f"Connected to ChromaDB collection '{chroma_collection_name}' with {count} items.")
        except Exception as e:
             print(f"ERROR: Could not get collection '{chroma_collection_name}'. Did ingestion complete successfully? Error: {e}")
             return

        # 3. Create Retriever
        # k=4 means retrieve the top 4 most relevant chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        logging.info(f"Created retriever to fetch top {retriever.search_kwargs['k']} relevant chunks.")

        # 4. Initialize LLM (for generation)
        logging.info(f"Initializing LLM for provider: {provider}")
        if provider == "google":
            llm_model_name = model_name or "gemini-1.5-pro-latest"
            llm = ChatGoogleGenerativeAI(model=llm_model_name, google_api_key=google_api_key, temperature=0.1) # Slightly creative temp
        elif provider == "openai":
            llm_model_name = model_name or "gpt-4o"
            llm = ChatOpenAI(model=llm_model_name, openai_api_key=openai_api_key, temperature=0.1)
        elif provider == "ollama":
            llm_model_name = model_name or "llama3"
            llm = Ollama(base_url=ollama_base_url, model=llm_model_name, temperature=0.1)
        # No need to test connection here, will happen during first query

        logging.info(f"Using LLM: {llm_model_name}")

    except Exception as e:
        print(f"ERROR: Failed to initialize components: {e}")
        return

    # --- Define RAG Chain using LCEL ---

    # Prompt Template: Defines how to structure the input to the LLM
    template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # LangChain Expression Language (LCEL) Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # Explanation of the chain:
    # 1. Input is the user's question (string).
    # 2. It's passed simultaneously to:
    #    a) The 'retriever' (which embeds the question, searches Chroma, gets docs)
    #       followed by `format_docs` to combine the retrieved docs into a single string.
    #       The result is assigned to the 'context' variable for the prompt.
    #    b) `RunnablePassthrough()` which just passes the original question through.
    #       The result is assigned to the 'question' variable for the prompt.
    # 3. The 'context' and 'question' are fed into the 'prompt' template.
    # 4. The formatted prompt is sent to the 'llm'.
    # 5. The LLM's output (which might be a ChatMessage object) is parsed into a simple string
    #    by 'StrOutputParser()'.

    print("\n--- RAG Query Ready ---")
    print("Enter your question, or type 'quit' to exit.")

    # --- User Interaction Loop ---
    while True:
        try:
            user_question = input("\nQuestion: ")
            if user_question.lower().strip() == 'quit':
                break
            if not user_question:
                continue

            print("Thinking...")
            # Invoke the RAG chain with the user's question
            answer = rag_chain.invoke(user_question)

            print("\nAnswer:")
            print(answer)

        except Exception as e:
            print(f"\nAn error occurred during query processing: {e}")
            # Optionally add more robust error handling or retries

    print("\nExiting.")


# --- Standard Python entry point ---
if __name__ == "__main__":
    main()
