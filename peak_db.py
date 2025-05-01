import os
import chromadb
import logging
import argparse # Import argparse for command-line arguments
from dotenv import load_dotenv

# --- Basic Logging Setup ---
# Set to WARNING or ERROR for cleaner output during peeking
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("chromadb").setLevel(logging.ERROR) # Suppress noisy ChromaDB logs

def peek_database(persist_directory: str, collection_name: str, offset: int = 0, limit: int = 5):
    """
    Connects to a persistent ChromaDB collection and retrieves a sample of items
    within a specified range (offset and limit).

    Args:
        persist_directory: The path to the directory where ChromaDB data is stored.
        collection_name: The name of the collection to inspect.
        offset: The starting position (0-based index) from which to retrieve items.
        limit: The maximum number of items to retrieve starting from the offset.
    """
    if not persist_directory or not os.path.isdir(persist_directory):
        logging.error(f"ChromaDB persist directory not found or not specified: {persist_directory}")
        print(f"ERROR: ChromaDB persist directory not found: {persist_directory}")
        return

    if not collection_name:
        logging.error("ChromaDB collection name not specified.")
        print("ERROR: ChromaDB collection name not specified.")
        return

    logging.info(f"Connecting to ChromaDB at: {persist_directory}")
    try:
        # Create a persistent Chroma client pointing to the directory
        client = chromadb.PersistentClient(path=persist_directory)

        logging.info(f"Getting collection: {collection_name}")
        # Get the existing collection
        collection = client.get_collection(name=collection_name)

        total_items = collection.count()
        logging.info(f"Collection '{collection_name}' contains {total_items} items.")

        if offset >= total_items and total_items > 0:
             logging.warning(f"Offset ({offset}) is greater than or equal to the total number of items ({total_items}). No items to retrieve.")
             print(f"Offset ({offset}) is beyond the total number of items ({total_items}).")
             return
        elif total_items == 0:
             logging.warning(f"Collection '{collection_name}' is empty.")
             print(f"Collection '{collection_name}' is empty.")
             return


        print(f"\nPeeking at {limit} items starting from offset {offset}...")
        print("-" * 30)

        # Retrieve items from the collection using offset and limit
        results = collection.get(
            offset=offset,
            limit=limit,
            include=["metadatas", "documents"] # Specify what data to include
        )

        if not results or not results.get("ids"):
            logging.warning(f"No results retrieved for the specified offset/limit in collection '{collection_name}'.")
            print("No results found for this range.")
            return

        logging.info(f"Retrieved {len(results['ids'])} items.")

        # Loop through the retrieved items and print them
        for i, item_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i] if results.get("metadatas") else {}
            document = results["documents"][i] if results.get("documents") else "N/A"

            print(f"Item #{offset + i + 1} (ID: {item_id})") # Show overall index
            print(f"  Metadata: {metadata}")
            # Print only the first few hundred characters of the document for brevity
            print(f"  Document: {document[:300]}...")
            print("-" * 30)

    except chromadb.errors.CollectionNotFoundError:
         logging.error(f"Collection '{collection_name}' not found in the database at {persist_directory}.")
         print(f"ERROR: Collection '{collection_name}' not found. Did ingestion run correctly?")
    except Exception as e:
        logging.error(f"An error occurred while trying to peek into ChromaDB: {e}")
        print(f"ERROR: An error occurred: {e}")
        print("Please ensure the persist directory and collection name are correct and the database is not corrupted.")

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Peek into a ChromaDB collection with offset and limit.")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting offset (0-based index) to retrieve items from. Default is 0."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of items to retrieve. Default is 5."
    )
    args = parser.parse_args()

    # --- Load Environment Variables ---
    load_dotenv()
    logging.info("Loaded environment variables.")

    # --- Get Configuration ---
    persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME")

    # --- Call the Peeking Function ---
    peek_database(
        persist_directory=persist_dir,
        collection_name=collection_name,
        offset=args.offset,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
