import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import os
import pandas as pd
from utils import timer_logger, logger

CHROMA_DIR = "chromadb"

def chunk_text(text, max_length=500):
    """
    Manually chunk text into pieces of strictly <= max_length characters.
    Attempts to split by sentences, but enforces a hard limit for excessively long blocks.
    """
    if not text or not isinstance(text, str):
        return []
        
    # Split by periods to get sentences
    sentences = text.replace('\n', ' ').split('. ')
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        sentence += ". "
        
        # If the sentence itself is larger than the max_length limit, 
        # we MUST slice it up forcibly into smaller sub-chunks so it doesn't crash Ollama
        if len(sentence) > max_length:
            # First, save whatever current chunk we were building if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                
            # Now, force-split the massive sentence into strict max_length sized chunks
            for i in range(0, len(sentence), max_length):
                sub_chunk = sentence[i:i+max_length]
                if sub_chunk:
                    chunks.append(sub_chunk.strip())
            continue
            
        # If adding the new sentence keeps us under the limit, add it
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            # We reached the limit. Save current chunk.
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start the new chunk with the sentence that didn't fit
            current_chunk = sentence
            
    # Add any leftover text in the final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

@timer_logger
def store_articles_in_vector_db(df):
    """
    Takes the SQLite dataframe, chunks the text natively, 
    and inserts it into a local ChromaDB instance via Ollama.
    """
    if df.empty:
        logger.warning("[VectorDB] No articles provided to store.")
        return

    logger.info("[VectorDB] Initializing ChromaDB persistent client...")
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR)
        
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    logger.info("[VectorDB] Configuring native Ollama Embedding Function (nomic-embed-text)...")
    # This natively communicates with your local Ollama instance running on port 11434
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="qwen3-embedding:0.6B",
    )
    
    # Get or create the table (collection)
    collection = client.get_or_create_collection(
        name="news_articles", 
        embedding_function=ollama_ef
    )

    documents = []
    metadatas = []
    ids = []
    
    chunk_counter = 0

    logger.info("[VectorDB] Chunking articles manually and preparing for insertion...")
    for index, row in df.iterrows():
        # Using SQLite auto-increment ID if available
        article_id = str(row.get('id', index))
        title = str(row.get('title', ''))
        content = str(row.get('content', ''))
        
        # Combine title and content as they both hold valuable context
        full_text = f"{title}. {content}"
        
        # Call our manual Python chunker
        chunks = chunk_text(full_text, max_length=500)
        
        for i, chunk in enumerate(chunks):
            if chunk: 
                chunk_id = f"{article_id}_chunk_{i}"
                documents.append(chunk)
                
                # Attach metadata so future RAG queries can mathematically link back to the SQLite row!
                metadatas.append({
                    "original_article_id": article_id,
                    "source": str(row.get('source', '')),
                    "url": str(row.get('link', '')),
                    "published_at": str(row.get('published_at', ''))
                })
                ids.append(chunk_id)
                chunk_counter += 1

    if documents:
        logger.info(f"[VectorDB] Sending {chunk_counter} chunked items to Ollama & saving to ChromaDB...")
        # Upsert adds new items or updates existing items based on chunk_id
        # We process in batches of 50 to ensure local Ollama isn't overwhelmed instantly
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            end_idx = i + batch_size
            collection.upsert(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        logger.info("[VectorDB] Insertion complete! Your articles are now embedded.")
    else:
        logger.warning("[VectorDB] No textual chunks created.")
