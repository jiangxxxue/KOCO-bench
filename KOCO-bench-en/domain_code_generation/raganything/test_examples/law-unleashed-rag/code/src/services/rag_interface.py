"""
Abstract interface for RAG implementations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime


class RAGInterface(ABC):
    """Abstract base class for RAG implementations"""
    
    def __init__(self, firebase_manager, gcs_manager, auth_service):
        self.firebase_manager = firebase_manager
        self.gcs_manager = gcs_manager
        self.auth_service = auth_service
    
    @abstractmethod
    async def process_document(
        self,
        job_id: str,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_path: str,
        parser: str = "mineru",
        parse_method: str = "auto",
        model: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single document
        
        Args:
            job_id: Processing job ID
            user_id: User ID
            project_id: Project ID
            workspace_id: Workspace ID
            gcs_path: GCS path to the document
            parser: Parser to use
            parse_method: Parse method
            model: LLM model to use
            config: Additional configuration
            
        Returns:
            Processing result
        """
        pass
    
    @abstractmethod
    async def process_folder(
        self,
        job_id: str,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_folder_path: str,
        file_extensions: List[str] = None,
        parser: str = "mineru",
        parse_method: str = "auto",
        model: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process all documents in a folder
        
        Args:
            job_id: Processing job ID
            user_id: User ID
            project_id: Project ID
            workspace_id: Workspace ID
            gcs_folder_path: GCS path to the folder
            file_extensions: List of file extensions to process
            parser: Parser to use
            parse_method: Parse method
            model: LLM model to use
            config: Additional configuration
            
        Returns:
            Processing result
        """
        pass
    
    @abstractmethod
    async def get_processing_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processing status for a job
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status information or None if not found
        """
        pass
    
    @abstractmethod
    def get_approach_name(self) -> str:
        """Get the name of this RAG approach"""
        pass
    
    @abstractmethod
    def get_supported_parsers(self) -> List[str]:
        """Get list of supported parsers for this approach"""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Get list of supported models for this approach"""
        pass
    
    @abstractmethod
    async def query_documents(
        self,
        user_id: str,
        project_id: str,
        query: str,
        model: str = "gpt-4o-mini",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query processed documents
        
        Args:
            user_id: User ID
            project_id: Project ID
            query: Query to ask about the documents
            model: LLM model to use for answering
            config: Additional configuration
            
        Returns:
            Query result with answer, sources, and metadata
        """
        pass
