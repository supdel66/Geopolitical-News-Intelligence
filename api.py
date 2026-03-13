import os
import sqlite3

from typing import List, Optional
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from fastapi.staticfiles import StaticFiles
import requests
import json

# Config
CHROMA_DIR = "chromadb"
DB_PATH = os.path.join("sqlite_databases", "news.db")
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

app = FastAPI(title="Geopolitical News RAG API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust if frontend runs on different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the EDA HTML report and assets statically
if os.path.exists("eda_output"):
    app.mount("/report", StaticFiles(directory="eda_output", html=True), name="report")


# --- Helper Models -ls
#--
class ChatRequest(BaseModel):
    query: str

class ArticleInfo(BaseModel):
    id: int
    title: str
    source: str
    url: Optional[str] = None
    img_link: Optional[str] = None
    published_at: Optional[str] = None
    content_snippet: Optional[str] = None
    similarity_distance: Optional[float] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[ArticleInfo]




# --- Helper Functions ---

def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="SQLite Database not found.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def query_chromadb(query_sentence: str, top_n: int = 5):
    """
    Search ChromaDB for relevant chunks and deduplicate by article ID.
    Returns a list of dicts with article IDs and their closest match distance.
    """
    if not os.path.exists(CHROMA_DIR):
         raise HTTPException(status_code=500, detail="ChromaDB not found. Please run the pipeline first to index data.")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url=f"{OLLAMA_URL}",
            model_name=EMBEDDING_MODEL,
        )
        
        collection = client.get_collection(
            name="news_articles",
            embedding_function=ollama_ef
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to ChromaDB or Ollama Embedding: {str(e)}")

    results = collection.query(
        query_texts=[query_sentence],
        n_results=top_n * 3,
        include=["metadatas", "distances", "documents"]
    )

    if not results or not results["metadatas"] or not results["metadatas"][0]:
        return [], []

    seen_ids = set()
    unique_articles = []
    context_chunks = []

    for metadata, distance, document in zip(results["metadatas"][0], results["distances"][0], results["documents"][0]):
        article_id = metadata.get("original_article_id")
        
        # We always want to collect context for the LLM
        context_chunks.append(document)

        if article_id and article_id not in seen_ids:
            seen_ids.add(article_id)
            unique_articles.append({
                "article_id": int(article_id),
                "distance": distance
            })

        if len(unique_articles) >= top_n:
             # Just break the unique articles collection, keep collecting context if needed? 
             # No, if we break we stop collecting context too. That's fine for top 5 articles.
            break
            
    return unique_articles, context_chunks

def generate_llm_response(query: str, context: List[str]) -> str:
    """
    Calls local Ollama instance to generate a response based on the query and context.
    """
    prompt = f"""You are a geopolitical intelligence analyst assistant. 
    Use the following retrieved news article context to answer the user's question.
    If the answer isn't firmly in the context, do your best based on the context provided, 
    but acknowledge limitations if necessary. Do not hallucinate facts.
    
    Context:
    {' --- '.join(context)}
    
    User Question: {query}
    
    Answer:"""

    try:
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
        )
        return response.response
    except Exception as e:
        print(f"Ollama error: {e}")
        return "I'm sorry, I encountered an error communicating with the local LLM. Please make sure Ollama is running and the model is loaded."


# --- API Endpoints ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    RAG Endpoint: 
    1. Embeds query & searches Vector DB.
    2. Fetches full article info from SQLite.
    3. Calls LLM with context.
    4. Returns Answer + Sources.
    """
    query = request.query
    
    # 1. Retrieve similar chunks and deduplicated article IDs
    unique_articles, context_chunks = query_chromadb(query, top_n=20)
    
    # 2. Fetch full article info for the sources sidebar
    sources = []
    if unique_articles:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        for item in unique_articles:
            cursor.execute(
                "SELECT id, title, source, link as url, img_link, published_at, content FROM articles WHERE id = ?",
                (item["article_id"],)
            )
            row = cursor.fetchone()
            if row:
                # Add a content snippet so the frontend can show a preview
                snippet = row["content"][:200] + "..." if row["content"] and len(row["content"]) > 200 else row["content"]
                
                sources.append(ArticleInfo(
                    id=row["id"],
                    title=row["title"],
                    source=row["source"],
                    url=row["url"],
                    img_link=row["img_link"],
                    published_at=row["published_at"],
                    content_snippet=snippet,
                    similarity_distance=item["distance"]
                ))
        conn.close()

    # 3. Generate LLM Answer
    if context_chunks:
        answer = generate_llm_response(query, context_chunks)
    else:
        answer = "I couldn't find any relevant geopolitical news articles in your database to answer that question."

    # 4. Return results
    return ChatResponse(
        answer=answer,
        sources=sources
    )




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
