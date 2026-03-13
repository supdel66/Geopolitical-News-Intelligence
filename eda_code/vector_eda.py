import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import timer_logger, logger

CHROMA_DIR = "chromadb"
EDA_DIR = "eda_output"

THEMES = {
    "WW3 / Escalation": "world war 3 ww3 nuclear escalation global conflict third thermonuclear armageddon doomsday nato article 5 mutual assured destruction",
    "US-Israel": "us israel united states biden trump american iron dome us aid military idf israel lobby",
    "Israel-Iran": "israel iran attack strike tehran idf iranian drone missile mossad nuclear deal jcpoa khamenei",
    "Iran-US": "iran us america sanction us strike iran proxy hezbollah hamas houthi irgc strait of hormuz persian gulf",
    "Middle East Conflict": "gaza west bank beirut damascus iraq militia red sea tanker attack drone swarm ballistic missile",
    "Europe/Russia Context": "russia ukraine putin moscow kiev nato expansion eastern europe border poland baltic black sea",
    "Global Economy & Oil": "oil price brent crude opec shipping route supply chain inflation global economy market trade route"
}

@timer_logger
def run_vector_eda():
    if not os.path.exists(CHROMA_DIR):
        logger.warning(f"[Vector EDA] ChromaDB directory '{CHROMA_DIR}' not found. Cannot run Vector EDA.")
        return {}
        
    if not os.path.exists(EDA_DIR):
        os.makedirs(EDA_DIR)
        
    logger.info("[Vector EDA] Initializing ChromaDB persistent client...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    )
    
    try:
        collection = client.get_collection(name="news_articles", embedding_function=ollama_ef)
    except Exception as e:
        logger.error(f"[Vector EDA] Could not get collection: {e}")
        return {}

    stats = {
        "theme_counts": {},
        "theme_distances": {}
    }
    
    logger.info("[Vector EDA] Querying themes...")
    
    for theme_name, theme_query in THEMES.items():
        try:
            results = collection.query(
                query_texts=[theme_query],
                n_results=100  # Pull top 100 chunks matching the theme
            )
            
            # Distance measures how "far" the semantic meaning is from the query (Lower = better)
            distances = results['distances'][0] if 'distances' in results and results['distances'] else []
            metadatas = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] else []
            
            # Filter matches by a reasonable distance threshold for nomic-embed-text (e.g., < 1.0 to 1.2)
            # You may need to tune this threshold based on your embedding model
            DISTANCE_THRESHOLD = 1.0 
            
            valid_articles = set()
            valid_distances = []
            
            for dist, meta in zip(distances, metadatas):
                if dist <= DISTANCE_THRESHOLD:
                    article_id = meta.get("original_article_id")
                    if article_id:
                        valid_articles.add(article_id)
                        valid_distances.append(dist)
            
            stats["theme_counts"][theme_name] = len(valid_articles)
            if valid_distances:
                stats["theme_distances"][theme_name] = sum(valid_distances) / len(valid_distances)
            else:
                stats["theme_distances"][theme_name] = 0.0
                
        except Exception as e:
            logger.error(f"[Vector EDA] Error querying theme '{theme_name}': {e}")
            stats["theme_counts"][theme_name] = 0
            stats["theme_distances"][theme_name] = 0.0

    logger.info("[Vector EDA] Generating Vector DB Charts...")
    sns.set_theme(style="whitegrid")
    
    # === 1. Top Themes by Article Count ===
    counts = stats["theme_counts"]
    if any(counts.values()):
        # Sort desc
        sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(sorted_counts.values()), y=list(sorted_counts.keys()), hue=list(sorted_counts.keys()), palette="crest", legend=False)
        plt.title('Top Geopolitical Themes by Article Volume (Semantic Search)', fontsize=14)
        plt.xlabel('Number of Unique Articles')
        plt.ylabel('Theme')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, "vector_themes_volume.png"))
        plt.close()
        logger.info("Saved vector theme volume chart.")
    
    # === 2. Average Semantic Distance (Relevance) ===
    dists = stats["theme_distances"]
    # Filter out 0s (no matches)
    valid_dists = {k: v for k, v in dists.items() if v > 0}
    if valid_dists:
        # Sort asc (lower distance is better)
        sorted_dists = dict(sorted(valid_dists.items(), key=lambda item: item[1]))
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(sorted_dists.values()), y=list(sorted_dists.keys()), hue=list(sorted_dists.keys()), palette="rocket_r", legend=False)
        plt.title('Average Semantic Distance per Theme (Lower = Closer Match)', fontsize=14)
        plt.xlabel('Average L2/Cosine Distance')
        plt.ylabel('Theme')
        plt.axvline(DISTANCE_THRESHOLD, color='r', linestyle='dashed', linewidth=1, label=f'Threshold ({DISTANCE_THRESHOLD})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, "vector_themes_relevance.png"))
        plt.close()
        logger.info("Saved vector theme relevance chart.")
        
    logger.info("[Vector EDA] Execution completed.")
    return stats
