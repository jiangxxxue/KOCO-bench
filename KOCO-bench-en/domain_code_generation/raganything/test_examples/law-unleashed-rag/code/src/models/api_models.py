"""
API request and response models for RAG-Anything service
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ProcessDocumentRequest(BaseModel):
    """Request model for processing a single document"""
    user_id: str = Field(..., description="User ID")
    project_id: str = Field(..., description="Project ID")
    workspace_id: str = Field(..., description="Workspace ID")
    gcs_path: str = Field(..., description="GCS path to the document")
    rag_approach: str = Field(default="raganything", description="RAG approach: raganything, evidence_sweep, or rag_vertex")
    parser: str = Field(default="mineru", description="Parser to use: mineru, docling, llamaparse, or simple")
    parse_method: str = Field(default="auto", description="Parse method: auto, ocr, or txt")
    model: str = Field(default="gpt-4", description="LLM model to use")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")


class ProcessFolderRequest(BaseModel):
    """Request model for processing all documents in a folder"""
    user_id: str = Field(..., description="User ID")
    project_id: str = Field(..., description="Project ID")
    workspace_id: str = Field(..., description="Workspace ID")
    gcs_folder_path: str = Field(..., description="GCS path to the folder")
    rag_approach: str = Field(default="raganything", description="RAG approach: raganything, evidence_sweep, or rag_vertex")
    parser: str = Field(default="mineru", description="Parser to use: mineru, docling, llamaparse, or simple")
    parse_method: str = Field(default="auto", description="Parse method: auto, ocr, or txt")
    model: str = Field(default="gpt-4", description="LLM model to use")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")
    file_extensions: List[str] = Field(
        default=[".pdf", ".doc", ".docx", ".txt", ".md", ".jpg", ".png", ".bmp", ".tiff"],
        description="File extensions to process"
    )


class ProcessingJobResponse(BaseModel):
    """Response model for processing job creation"""
    job_id: str = Field(..., description="Unique job ID")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Job creation time")


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status"""
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Current status")
    message: Optional[str] = Field(default=None, description="Status message")
    progress: Dict[str, Any] = Field(default_factory=dict, description="Processing progress")
    created_at: datetime = Field(..., description="Job creation time")
    updated_at: datetime = Field(..., description="Last update time")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Processing result")
    corpus_info: Optional[Dict[str, Any]] = Field(default=None, description="RAG corpus information for Vertex AI")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    services: Optional[Dict[str, str]] = Field(default=None, description="Dependent services status")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class RAGApproachInfo(BaseModel):
    """Information about a RAG approach"""
    name: str = Field(..., description="RAG approach name")
    supported_parsers: List[str] = Field(..., description="Supported parsers")
    supported_models: List[str] = Field(..., description="Supported models")
    description: Optional[str] = Field(default=None, description="Description of the approach")


class RAGApproachesResponse(BaseModel):
    """Response model for available RAG approaches"""
    approaches: Dict[str, RAGApproachInfo] = Field(..., description="Available RAG approaches")
    default_approach: str = Field(default="raganything", description="Default RAG approach")


class QueryRequest(BaseModel):
    """Request model for querying processed documents"""
    user_id: str = Field(..., description="User ID")
    project_id: str = Field(..., description="Project ID")
    rag_approach: str = Field(..., description="RAG approach to use: raganything, evidence_sweep, or rag_vertex")
    query: str = Field(..., description="Query to ask about the documents")
    model: str = Field(default="gpt-4o-mini", description="LLM model to use for answering")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")
    corpus_info: Optional[Dict[str, Any]] = Field(default=None, description="RAG corpus information for Vertex AI")


class QueryResponse(BaseModel):
    """Response model for query results"""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Answer to the query")
    rag_approach: str = Field(..., description="RAG approach used")
    model: str = Field(..., description="Model used")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents and chunks used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processing_time: float = Field(..., description="Time taken to process the query in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Query timestamp")
