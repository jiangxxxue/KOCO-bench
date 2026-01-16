#!/usr/bin/env python3
"""
Setup BGE Reranker for LightRAG on RTX 3060
This adds a reranking model to improve retrieval quality
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("="*70)
print("SETTING UP BGE RERANKER FOR RTX 3060")
print("="*70)

print("\n[1/4] Checking GPU...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   ✓ CUDA Version: {torch.version.cuda}")
    else:
        print("   ✗ No GPU detected - reranker will run on CPU (slower)")
except Exception as e:
    print(f"   [ERROR] {e}")

print("\n[2/4] Installing sentence-transformers (if needed)...")
try:
    from sentence_transformers import CrossEncoder
    print("   ✓ sentence-transformers already installed")
except ImportError:
    print("   Installing sentence-transformers...")
    os.system("pip install sentence-transformers")

print("\n[3/4] Creating reranker function...")

# BGE reranker model - perfect for RTX 3060 (350MB, fast inference)
RERANKER_MODEL = "BAAI/bge-reranker-base"

reranker_code = f'''
"""
BGE Reranker Function for LightRAG
Uses BAAI/bge-reranker-base - optimized for RTX 3060
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
        print(f"Loading BGE reranker on {{device.upper()}}...")
        _reranker_model = CrossEncoder('{RERANKER_MODEL}', device=device, max_length=512)
        print(f"   ✓ BGE reranker loaded successfully")
    
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
'''

# Save reranker function
reranker_file = "./src/bge_reranker.py"
with open(reranker_file, 'w', encoding='utf-8') as f:
    f.write(reranker_code)

print(f"   ✓ Reranker function saved to: {reranker_file}")

print("\n[4/4] Testing reranker model...")
try:
    # Test loading the model
    from sentence_transformers import CrossEncoder
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Loading {RERANKER_MODEL} on {device.upper()}...")
    
    model = CrossEncoder(RERANKER_MODEL, device=device, max_length=512)
    
    # Test reranking
    query = "What is machine learning?"
    docs = [
        "Machine learning is a subset of AI that enables computers to learn from data.",
        "The weather today is sunny and warm.",
        "Deep learning uses neural networks with multiple layers."
    ]
    
    pairs = [[query, doc] for doc in docs]
    scores = model.predict(pairs, show_progress_bar=False)
    
    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Model size: ~350MB")
    print(f"   ✓ Max sequence length: 512 tokens")
    
    print(f"\n   Test reranking results:")
    for doc, score in zip(docs, scores):
        print(f"      Score {score:.4f}: {doc[:60]}...")
    
except Exception as e:
    print(f"   [ERROR] Failed to test reranker: {e}")
    print(f"   You may need to install: pip install sentence-transformers")

print("\n" + "="*70)
print("SETUP COMPLETE!")
print("="*70)

print("\n✓ BGE Reranker is ready to use!")
print(f"✓ Model: {RERANKER_MODEL}")
print(f"✓ Size: ~350MB")
print(f"✓ Device: {'GPU (RTX 3060)' if torch.cuda.is_available() else 'CPU'}")

print("\nTo use the reranker in your code:")
print("```python")
print("from src.bge_reranker import bge_rerank")
print("")
print("# Rerank documents")
print("results = bge_rerank(query, documents, top_k=10)")
print("for doc, score in results:")
print("    print(f'Score: {score:.4f} - {doc}')")
print("```")

print("\nThe reranker will automatically:")
print("✓ Use GPU if available (RTX 3060)")
print("✓ Fall back to CPU if needed")
print("✓ Cache the model after first load")
print("✓ Provide relevance scores for better ranking")
