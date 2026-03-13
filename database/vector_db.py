import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import os
from utils import logger
import logging

class VectorDatabase:
    def __init__(self, chroma_dir="chromadb", model_name="qwen3-embedding:0.6B", collection_name="news_articles"):
        self.chroma_dir = chroma_dir
        self.model_name = model_name
        self.collection_name = collection_name
        
    def chunk_text_generator(self, text, max_length=500):
        """
        Yields chunks of text strictly <= max_length characters using a generator pattern.
        """
        if not text or not isinstance(text, str):
            return
            
        sentences = text.replace('\n', ' ').split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue
            sentence += ". "
            
            if len(sentence) > max_length:
                if current_chunk:
                    yield current_chunk.strip()
                    current_chunk = ""
                    
                for i in range(0, len(sentence), max_length):
                    sub_chunk = sentence[i:i+max_length]
                    if sub_chunk:
                        yield sub_chunk.strip()
                continue
                
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence
            else:
                if current_chunk:
                    yield current_chunk.strip()
                current_chunk = sentence
                
        if current_chunk:
            yield current_chunk.strip()

    def store_articles_in_vector_db(self, articles_iterator):
        """
        Takes an iterator of SQLite dictionaries, chunks the text natively via generator, 
        and inserts it into a local ChromaDB instance via Ollama.
        """
        logger.info("[VectorDB] Initializing ChromaDB persistent client...")
        if not os.path.exists(self.chroma_dir):
            os.makedirs(self.chroma_dir)
            
        client = chromadb.PersistentClient(path=self.chroma_dir)
        
        logger.info("[VectorDB] Configuring native Ollama Embedding Function...")
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=self.model_name,
        )
        
        collection = client.get_or_create_collection(
            name=self.collection_name, 
            embedding_function=ollama_ef
        )

        documents = []
        metadatas = []
        ids = []
        
        chunk_counter = 0

        logger.info("[VectorDB] Consuming DB Iterator, Chunking logic, and preparing for insertion...")
        
        for row in articles_iterator:
            article_id = str(row.get('id', ''))
            title = str(row.get('title', ''))
            content = str(row.get('content', ''))
            
            full_text = f"{title}. {content}"
            
            # Consume the chunk generator for this specific article
            for i, chunk in enumerate(self.chunk_text_generator(full_text, max_length=500)):
                if chunk: 
                    chunk_id = f"{article_id}_chunk_{i}"
                    documents.append(chunk)
                    
                    metadatas.append({
                        "original_article_id": article_id,
                        "source": str(row.get('source', '')),
                        "url": str(row.get('link', '')),
                        "published_at": str(row.get('published_at', ''))
                    })
                    ids.append(chunk_id)
                    chunk_counter += 1

                # Dynamic batch yielding/upsert
                if len(documents) >= 50:
                    collection.upsert(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    documents = []
                    metadatas = []
                    ids = []

        # Flush any remaining chunks
        if documents:
            logger.info(f"[VectorDB] Sending final {len(documents)} chunked items to Ollama & saving to ChromaDB...")
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
        logger.info(f"[VectorDB] Insertion complete! {chunk_counter} textual chunks are now embedded.")
