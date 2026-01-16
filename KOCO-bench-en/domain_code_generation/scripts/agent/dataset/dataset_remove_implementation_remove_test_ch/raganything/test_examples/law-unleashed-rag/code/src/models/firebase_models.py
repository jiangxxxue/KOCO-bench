"""
Firebase data models for RAG-Anything service
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class WorkspaceMembership(BaseModel):
    """Workspace membership model"""
    workspaceId: str = Field(..., description="Workspace ID")
    role: str = Field(..., description="User role in workspace")
    joinedAt: datetime = Field(default_factory=datetime.utcnow, description="When user joined workspace")


class User(BaseModel):
    """User model from Firebase"""
    uid: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    displayName: Optional[str] = Field(default=None, description="User display name")
    memberships: List[WorkspaceMembership] = Field(default_factory=list, description="Workspace memberships")
    tier: str = Field(default="PILOT", description="User tier")
    premiumStatus: str = Field(default="none", description="Premium status")
    isAdmin: bool = Field(default=False, description="Admin status")
    hasNoLimit: bool = Field(default=False, description="No limit status")
    customTokenLimit: Optional[int] = Field(default=None, description="Custom token limit")
    customRequestLimit: Optional[int] = Field(default=None, description="Custom request limit")
    customCostLimit: Optional[float] = Field(default=None, description="Custom cost limit")
    isDisabled: bool = Field(default=False, description="Disabled status")
    createdAt: datetime = Field(default_factory=datetime.utcnow, description="User creation time")
    updatedAt: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class Workspace(BaseModel):
    """Workspace model from Firebase"""
    id: str = Field(..., description="Workspace ID")
    name: str = Field(..., description="Workspace name")
    description: Optional[str] = Field(default=None, description="Workspace description")
    ownerId: str = Field(..., description="Workspace owner ID")
    createdAt: datetime = Field(default_factory=datetime.utcnow, description="Workspace creation time")
    updatedAt: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class Project(BaseModel):
    """Project model from Firebase"""
    id: str = Field(..., description="Project ID")
    workspaceId: str = Field(..., description="Workspace ID")
    title: str = Field(..., description="Project title")
    description: str = Field(..., description="Project description")
    userId: str = Field(..., description="Project owner ID")
    createdAt: datetime = Field(default_factory=datetime.utcnow, description="Project creation time")
    updatedAt: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    reportsCount: Optional[int] = Field(default=0, description="Number of reports")
    caseFilesCount: Optional[int] = Field(default=0, description="Number of case files")
    graphProcessingStatus: Optional[str] = Field(default=None, description="Graph processing status")
    graphProcessingMessage: Optional[str] = Field(default=None, description="Graph processing message")
    graphProcessingUpdatedAt: Optional[datetime] = Field(default=None, description="Graph processing update time")
    graphProcessingMetadata: Optional[Dict[str, Any]] = Field(default=None, description="Graph processing metadata")


class RAGProcessingJob(BaseModel):
    """RAG processing job model"""
    id: str = Field(..., description="Job ID")
    projectId: str = Field(..., description="Project ID")
    userId: str = Field(..., description="User ID")
    workspaceId: str = Field(..., description="Workspace ID")
    jobType: str = Field(..., description="Job type: document or folder")
    gcsPath: str = Field(..., description="GCS path to process")
    status: str = Field(default="pending", description="Job status")
    createdAt: datetime = Field(default_factory=datetime.utcnow, description="Job creation time")
    updatedAt: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    progress: Dict[str, Any] = Field(default_factory=dict, description="Processing progress")
    errorMessage: Optional[str] = Field(default=None, description="Error message if failed")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Processing result")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Processing configuration")


class RAGProcessingMetadata(BaseModel):
    """RAG processing metadata"""
    totalDocuments: int = Field(default=0, description="Total number of documents")
    processedDocuments: int = Field(default=0, description="Number of processed documents")
    failedDocuments: int = Field(default=0, description="Number of failed documents")
    processingTime: float = Field(default=0.0, description="Processing time in seconds")
    model: str = Field(default="gpt-4", description="LLM model used")
    parser: str = Field(default="mineru", description="Parser used")
    parseMethod: str = Field(default="auto", description="Parse method used")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Processing configuration")
    lastProcessedAt: datetime = Field(default_factory=datetime.utcnow, description="Last processing time")
    storageInfo: Optional[Dict[str, Any]] = Field(default=None, description="Storage information")
