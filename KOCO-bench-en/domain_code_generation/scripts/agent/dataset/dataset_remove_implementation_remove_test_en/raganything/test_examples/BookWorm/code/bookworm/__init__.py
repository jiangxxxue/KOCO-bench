"""
BookWorm: Advanced Document/Knowledge Ingestion System

Refactored modular architecture with focused responsibilities:
- models: Data models and schemas
- processors: Document processing and description generation
- knowledge: Knowledge graph management
- generators: Content generation (mindmaps, etc.)
- library: Document library management
- utils: Configuration and utilities
"""

__version__ = "0.1.0"
__author__ = "BookWorm Team"
__description__ = "Advanced document/knowledge ingestion system with LightRAG and mindmap generation"

from .models import ProcessedDocument, MindmapResult
from .processors import DocumentProcessor, DocumentDescriptionGenerator
from .knowledge import DocumentKnowledgeGraph, KnowledgeGraph

from .library import LibraryManager, DocumentStatus
from .utils import BookWormConfig, load_config, setup_logging
from bookworm.pipeline import BookWormPipeline

# Maintain backward compatibility - these are the main classes users need
__all__ = [
    # main pipeline
    'BookWormPipeline',

    # Core models
    'ProcessedDocument',
    'MindmapResult', 
    
    # Processing components
    'DocumentProcessor',
    'DocumentDescriptionGenerator',
    
    # Knowledge graph components
    'DocumentKnowledgeGraph',
    'KnowledgeGraph',

    
    # Library management
    'LibraryManager',
    'DocumentStatus',
    
    # Configuration
    'BookWormConfig',
    'load_config',
    'setup_logging',
    'utils'
]
