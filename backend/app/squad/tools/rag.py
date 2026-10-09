"""
RAG Tool
--------
Handles vector search in Supabase for Medical and Legal documents.
"""

from sentence_transformers import SentenceTransformer
from app.core.database import supabase

# Load the same embedding model used during ingestion
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def search_documents(query: str, match_threshold: float = 0.3, match_count: int = 3) -> str:
    """
    Search the vector database for chunks relevant to the user's query.
    """
    # Generate embedding for the user's query
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
    
    # Convert list to string format for pgvector (this prevents serialization errors)
    query_embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    # Call the Supabase match_chunks function
    response = supabase.rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding_str,
            "match_threshold": match_threshold,
            "match_count": match_count,
        },
    ).execute()

    if not response.data:
        return "No relevant information found in the documents."

    # Combine the top chunks into a single context string
    context = "\n\n".join([chunk["content"] for chunk in response.data])
    return context