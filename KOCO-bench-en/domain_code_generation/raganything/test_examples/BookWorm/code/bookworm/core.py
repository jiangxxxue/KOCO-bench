"""
BookWorm Core Components - Compatibility Layer

This module provides backward compatibility by re-exporting the refactored components.
The original monolithic core.py has been broken down into focused modules:

- models/: Data models (ProcessedDocument, MindmapResult)
- processors/: Document processing and description generation  
- knowledge/: Knowledge graph management
- generators/: Content generation (mindmaps, etc.)
- library/: Document library management
- utils/: Configuration and utilities

For new code, import directly from the specific modules.
This file maintains compatibility for existing imports.
"""

# Re-export models
from .models import ProcessedDocument, MindmapResult

# Re-export processors  
from .processors import DocumentProcessor, DocumentDescriptionGenerator

# Re-export knowledge components
from .knowledge import DocumentKnowledgeGraph, KnowledgeGraph

# Re-export generators
from .generators import MindmapGenerator

# Maintain backward compatibility
__all__ = [
    'ProcessedDocument',
    'MindmapResult', 
    'DocumentProcessor',
    'DocumentDescriptionGenerator',
    'DocumentKnowledgeGraph',
    'KnowledgeGraph',
    'MindmapGenerator'
]
