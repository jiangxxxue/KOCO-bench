
"""
BGE Reranker v2-m3 Function for LightRAG
Uses BAAI/bge-reranker-v2-m3 - optimized for RTX 4060 (8GB VRAM)
Latest 2024 model with best performance-to-size ratio
"""

from sentence_transformers import CrossEncoder
import torch
from typing import List, Tuple

# Initialize reranker model (lazy loading)
_reranker_model = None

def get_reranker_model():
    """Get or initialize the reranker model"""
    global _reranker_model

    if _reranker_model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading BGE Reranker v2-m3 on {device.upper()}...")
        _reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device, max_length=8192)
        print(f"   ✓ BGE Reranker v2-m3 loaded successfully (1.06GB VRAM)")

    return _reranker_model


def bge_rerank(query: str, documents: List[str], top_k: int = None) -> List[Tuple[str, float]]:
    """
    Rerank documents using BGE reranker
    
    Args:
        query: Search query
        documents: List of document texts to rerank
        top_k: Number of top documents to return (None = return all)
        
    Returns:
        List of (document, score) tuples, sorted by relevance
    """
    if not documents:
        return []
    
    # Get reranker model
    model = get_reranker_model()
    
    # Create query-document pairs
    pairs = [[query, doc] for doc in documents]
    
    # Get relevance scores
    scores = model.predict(pairs, show_progress_bar=False)
    
    # Combine documents with scores and sort
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k if specified
    if top_k is not None:
        doc_scores = doc_scores[:top_k]
    
    return doc_scores


# For LightRAG compatibility - async version
async def bge_rerank_async(query: str, documents: List[str], top_k: int = None) -> List[Tuple[str, float]]:
    """Async wrapper for BGE reranking"""
    return bge_rerank(query, documents, top_k)
