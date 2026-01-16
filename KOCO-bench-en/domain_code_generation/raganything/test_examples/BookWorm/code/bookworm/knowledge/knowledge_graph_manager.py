"""
Knowledge graph manager for multiple document graphs
Rebuilt from scratch based on LightRAGManager and lightrag_ex.py
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# LightRAG components
try:
    from lightrag import LightRAG
    from lightrag.utils import set_verbose_debug
except ImportError:
    LightRAG = None
    set_verbose_debug = None

from .document_knowledge_graph import DocumentKnowledgeGraph
from ..models import ProcessedDocument
from ..library import LibraryManager, DocumentStatus
from ..utils import BookWormConfig


class KnowledgeGraph:
    """
    Manages multiple document knowledge graphs - one graph per document
    Based on LightRAGManager patterns with multi-document support
    """
    
    def __init__(self, config: BookWormConfig, library_manager: Optional[LibraryManager] = None):
        self.config = config
        self.logger = logging.getLogger("bookworm.knowledge_graph_manager")
        self.library_manager = library_manager or LibraryManager(config)
        self.document_graphs: Dict[str, DocumentKnowledgeGraph] = {}
        
        if not LightRAG:
            raise ImportError("LightRAG not available. Install lightrag-hku.")
    
    async def initialize(self) -> None:
        """Initialize knowledge graph manager"""
        # Turn on verbose LightRAG logging if available
        try:
            if set_verbose_debug:
                set_verbose_debug(True)
        except Exception:
            pass
        
        self.logger.info("Initializing KnowledgeGraph manager...")
        graphs_dir = Path("./workspace/knowledge_graphs")
        graphs_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("KnowledgeGraph manager ready")
    
    async def finalize(self) -> None:
        """Finalize all document graphs and release resources"""
        self.logger.info("Finalizing KnowledgeGraph manager and all document graphs...")
        for doc_id, doc_kg in list(self.document_graphs.items()):
            try:
                await doc_kg.cleanup()
                self.logger.info(f"✅ Finalized KG for {doc_id[:8]}")
            except Exception as e:
                self.logger.warning(f"Failed to finalize KG {doc_id[:8]}: {e}")

    async def add_document(self, document: ProcessedDocument) -> Tuple[DocumentKnowledgeGraph, Optional[str]]:
        """Create a document graph and ingest its content - based on LightRAGManager.add_document"""
        return await self.create_document_graph(document)

    async def batch_add_documents(self, documents: List[ProcessedDocument]) -> List[Tuple[str, Optional[str]]]:
        """Batch add multiple documents to the knowledge graph system"""
        results: List[Tuple[str, Optional[str]]] = []
        for doc in documents:
            try:
                doc_kg, lib_id = await self.create_document_graph(doc)
                results.append((doc_kg.document_id, lib_id))
            except Exception as e:
                self.logger.warning(f"Failed to add document {doc.id[:8]}: {e}")
        return results

    async def create_document_graph(self, document: ProcessedDocument) -> Tuple[DocumentKnowledgeGraph, Optional[str]]:
        """Create a new knowledge graph for a specific document"""
        self.logger.info(f"Creating knowledge graph for document {document.id[:8]}...")
        
        # Create document-specific knowledge graph
        doc_kg = DocumentKnowledgeGraph(self.config, document.id, self.library_manager)
        await doc_kg.initialize()
        
        # Add document content to the graph (preserve original file path for metadata)
        try:
            original_path = document.original_path if document.original_path else None
            file_paths = [original_path] if original_path else None
        except Exception:
            file_paths = None
        
        self.logger.info(f"Adding document content for {document.id[:8]}...")
        await doc_kg.add_document_content(document.text_content, file_paths=file_paths)
        
        # Store reference to the graph
        self.document_graphs[document.id] = doc_kg
        
        # Add to library manager
        library_doc_id = None
        try:
            # Determine if this is a directory or file
            original_path = Path(document.original_path)
            is_directory = document.metadata.get("is_directory", False) or original_path.is_dir()
            
            if original_path.exists():
                # Check if document already exists in library
                search_name = original_path.name
                existing_docs = self.library_manager.find_documents(filename=search_name)
                
                if not existing_docs:
                    # Add new document to library
                    library_doc_id = self.library_manager.add_document(filepath=document.original_path)
                    self.logger.info(f"📚 Document {library_doc_id} added to library index")
                    
                    # If it's a directory, update the document record with directory metadata
                    if is_directory and document.metadata.get("sub_files"):
                        self.library_manager.update_document_metadata(
                            library_doc_id,
                            {
                                "is_directory": True,
                                "sub_files": document.metadata.get("sub_files", []),
                                "file_count": document.metadata.get("file_count", 0),
                                "processing_type": document.metadata.get("processing_type", "directory_collection")
                            }
                        )
                        self.logger.info(f"📁 Directory metadata saved for {search_name}")
                else:
                    # Update existing document status
                    existing_doc = existing_docs[0]
                    library_doc_id = existing_doc.id
                    self.logger.info(f"📚 Found existing document {existing_doc.id}")
                
                # Always update the status to processed and knowledge graph path for both new and existing documents
                if library_doc_id:
                    self.library_manager.update_document_status(
                        library_doc_id, 
                        DocumentStatus.PROCESSED,
                        processed_file_path=document.processed_path
                    )
                    self.library_manager.update_document_metadata(
                        library_doc_id, 
                        {"knowledge_graph_id": str(doc_kg.doc_working_dir)}
                    )
                    self.logger.info(f"📚 Document {library_doc_id} status updated to PROCESSED")
                    self.logger.info(f"📚 Knowledge graph path saved to library: {doc_kg.doc_working_dir}")
            else:
                self.logger.warning(f"Original file/directory {document.original_path} not found, skipping library update")
        except Exception as e:
            self.logger.warning(f"Failed to update library for document {document.id}: {e}")
        
        return doc_kg, library_doc_id
    
    async def get_document_graph(self, document_id: str) -> Optional[DocumentKnowledgeGraph]:
        """Get the knowledge graph for a specific document"""
        if document_id in self.document_graphs:
            return self.document_graphs[document_id]
        
        # Try to load from disk if not in memory
        doc_working_dir = Path("./workspace/knowledge_graphs") / document_id
        if doc_working_dir.exists():
            self.logger.info(f"Loading existing knowledge graph for document {document_id[:8]}...")
            doc_kg = DocumentKnowledgeGraph(self.config, document_id, self.library_manager)
            await doc_kg.initialize()
            self.document_graphs[document_id] = doc_kg
            return doc_kg
        
        return None
    
    async def query(self, query: str, mode: str = "hybrid", **kwargs) -> str:
        """
        High-level query across available document graphs
        Based on LightRAGManager.query() with multi-document support
        """
        # Prefer loaded graphs; if none loaded, attempt to load all existing from disk
        if not self.document_graphs:
            graphs_dir = Path("./workspace/knowledge_graphs")
            if graphs_dir.exists():
                for doc_dir in graphs_dir.iterdir():
                    if doc_dir.is_dir():
                        try:
                            await self.get_document_graph(doc_dir.name)
                        except Exception:
                            continue
        
        if not self.document_graphs:
            self.logger.warning("No document graphs available to query")
            return ""
        
        # If only one graph, return its response directly
        if len(self.document_graphs) == 1:
            only_doc = next(iter(self.document_graphs.values()))
            self.logger.info(f"Querying single available document graph: {only_doc.document_id[:8]}")
            return await only_doc.query(query, mode=mode, **kwargs)
        
        # Otherwise, query all and aggregate
        self.logger.info(f"Querying {len(self.document_graphs)} document graphs...")
        results = await self.query_all_documents(query, mode=mode, **kwargs)
        combined = []
        for doc_id, answer in results.items():
            if answer:
                combined.append(f"[Document {doc_id[:8]}]\n{answer}")
        return "\n\n".join(combined)
    
    async def query_document(self, document_id: str, query: str, mode: str = "hybrid", **kwargs) -> Optional[str]:
        """Query a specific document's knowledge graph"""
        doc_kg = await self.get_document_graph(document_id)
        if doc_kg:
            return await doc_kg.query(query, mode=mode, **kwargs)
        self.logger.warning(f"Could not find or load document graph for ID: {document_id}")
        return None
    
    async def query_all_documents(self, query: str, mode: str = "hybrid", **kwargs) -> Dict[str, str]:
        """Query all document knowledge graphs and return results"""
        results = {}
        
        # Load all existing graphs
        graphs_dir = Path("./workspace/knowledge_graphs")
        if graphs_dir.exists():
            for doc_dir in graphs_dir.iterdir():
                if doc_dir.is_dir():
                    document_id = doc_dir.name
                    try:
                        self.logger.debug(f"Querying document graph: {document_id[:8]}")
                        doc_kg = await self.get_document_graph(document_id)
                        if doc_kg:
                            result = await doc_kg.query(query, mode=mode, **kwargs)
                            results[document_id] = result
                    except Exception as e:
                        self.logger.warning(f"Failed to query document {document_id[:8]}: {e}")
        
        return results
    
    def list_document_graphs(self) -> List[str]:
        """List all available document knowledge graphs"""
        graphs = []
        graphs_dir = Path("./workspace/knowledge_graphs")
        if graphs_dir.exists():
            for doc_dir in graphs_dir.iterdir():
                if doc_dir.is_dir():
                    graphs.append(doc_dir.name)
        return graphs
