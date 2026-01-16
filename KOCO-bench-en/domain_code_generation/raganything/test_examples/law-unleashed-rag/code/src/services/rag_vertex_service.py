"""
Vertex AI RAG service implementation using Google Cloud Platform
Based on the GCP generative AI RAG engine approach from:
https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/refs/heads/main/gemini/rag-engine/intro_rag_engine.ipynb
"""

import logging
import os
import tempfile
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import google.generativeai as genai
from google.cloud import aiplatform
from google.cloud import storage
from vertexai import generative_models
from vertexai.generative_models import Tool, GenerationConfig
from vertexai import rag
from vertexai.rag import Retrieval, VertexRagStore, RagResource, RagRetrievalConfig, Filter

from src.services.rag_interface import RAGInterface

logger = logging.getLogger(__name__)


class RAGVertexService(RAGInterface):
    """Service for processing documents with Vertex AI RAG engine"""
    
    def __init__(self, firebase_manager, gcs_manager, auth_service):
        super().__init__(firebase_manager, gcs_manager, auth_service)
        
        # Configuration - use existing environment variables
        self.project_id = os.getenv("FIREBASE_PROJECT_ID")
        self.region = os.getenv("GOOGLE_REGION", "us-east4")
        self.gcs_bucket = os.getenv("GCS_BUCKET_NAME")
        
        if not self.project_id:
            raise ValueError("FIREBASE_PROJECT_ID environment variable is required")
        if not self.gcs_bucket:
            raise ValueError("GCS_BUCKET_NAME environment variable is required")
        
        # Initialize Vertex AI and RAG components
        self._initialize_vertex_ai()
        
        # Storage configuration
        self.temp_dir = os.getenv("RAG_TEMP_DIR", "./temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # RAG corpus storage
        self.rag_corpus = None
        self.rag_retrieval_tool = None
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI components"""
        try:
            # Initialize Vertex AI
            aiplatform.init(project=self.project_id, location=self.region)
            
            # Configure Google GenAI for Vertex AI
            genai.configure(api_key=None)  # Will use default credentials
            
            # Initialize storage client
            self.storage_client = storage.Client(project=self.project_id)
            
            logger.info("Vertex AI initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    async def _create_or_get_rag_corpus(self, project_id: str):
        """Create or get existing RAG corpus"""
        try:
            if self.rag_corpus:
                return self.rag_corpus
            
            # Create RAG corpus using the vertexai.rag module
            corpus_display_name = f"rag-corpus-{project_id}"
            
            self.rag_corpus = rag.create_corpus(
                display_name=corpus_display_name,
                description=f"RAG corpus for project {project_id}"
            )
            
            logger.info(f"Created RAG corpus: {self.rag_corpus.name}")
            
            # Store corpus information in the project configuration
            await self._store_corpus_info(project_id, self.rag_corpus.name)
            
            return self.rag_corpus
            
        except Exception as e:
            logger.error(f"Failed to create RAG corpus: {e}")
            raise
    
    async def _store_corpus_info(self, project_id: str, corpus_name: str):
        """Store RAG corpus information in the project configuration"""
        try:
            # Update the project status in Firestore with corpus information
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'completed',
                f'RAG corpus created successfully: {corpus_name}',
                {
                    'rag_corpus_name': corpus_name,
                    'corpus_created_at': datetime.utcnow().isoformat()
                }
            )
            logger.info(f"Stored corpus info for project {project_id}: {corpus_name}")
        except Exception as e:
            logger.error(f"Failed to store corpus info: {e}")
            # Don't raise here as this is not critical for the main flow
    
    async def _load_corpus_info(self, project_id: str) -> Optional[str]:
        """Load RAG corpus information from the project configuration"""
        try:
            # Get project document from Firestore
            project_doc = await self.firebase_manager.get_project_document(project_id)
            if project_doc and 'graphProcessingMetadata' in project_doc:
                metadata = project_doc['graphProcessingMetadata']
                if 'rag_corpus_name' in metadata:
                    corpus_name = metadata['rag_corpus_name']
                    logger.info(f"Loaded corpus info for project {project_id}: {corpus_name}")
                    return corpus_name
            
            logger.warning(f"No corpus info found for project {project_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to load corpus info: {e}")
            return None
    
    
    async def _import_files_to_rag_corpus(self, gcs_path: str, project_id: str, config: Optional[Dict[str, Any]] = None):
        """Import files from GCS to RAG corpus using the exact method from the notebook with LLM parser"""
        try:
            # Default LLM parser configuration
            llm_parser_config = None
            if config and config.get("use_llm_parser", True):
                # Configure LLM parser as per the documentation
                llm_parser_config = rag.LlmParserConfig(
                    model_name=config.get("llm_parser_model", "gemini-2.0-flash-001"),
                    max_parsing_requests_per_min=config.get("max_parsing_requests_per_min", 60),  # Reduced from 900 to 60
                    custom_parsing_prompt=config.get("custom_parsing_prompt", None)
                )
                logger.info(f"Using LLM parser with model: {llm_parser_config.model_name}")
            
            # Import files using the vertexai.rag module
            response = rag.import_files(
                corpus_name=self.rag_corpus.name,
                paths=[gcs_path],
                # LLM parser configuration
                llm_parser=llm_parser_config,
                # Optional transformation config (used when LLM parser is not available)
                transformation_config=rag.TransformationConfig(
                    chunking_config=rag.ChunkingConfig(
                        chunk_size=config.get("chunk_size", 1024) if config else 1024, 
                        chunk_overlap=config.get("chunk_overlap", 100) if config else 100
                    )
                ),
                max_embedding_requests_per_min=config.get("max_embedding_requests_per_min", 60) if config else 60,  # Reduced from 900 to 60
            )
            
            # Log detailed import results
            logger.info(f"Imported files to RAG corpus with LLM parser:")
            logger.info(f"  - imported_rag_files_count: {response.imported_rag_files_count}")
            logger.info(f"  - failed_rag_files_count: {response.failed_rag_files_count}")
            
            # Log failed files if any
            if hasattr(response, 'failed_rag_files') and response.failed_rag_files:
                logger.warning("Failed files:")
                for failed_file in response.failed_rag_files:
                    logger.warning(f"  - {failed_file}")
            
            # Log successful files if any
            if hasattr(response, 'imported_rag_files') and response.imported_rag_files:
                logger.info("Successfully imported files:")
                for imported_file in response.imported_rag_files:
                    logger.info(f"  - {imported_file}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to import files to RAG corpus: {e}")
            raise
    
    
    async def _create_rag_retrieval_tool(self, project_id: str, corpus_info: Optional[Dict[str, Any]] = None):
        """Create RAG retrieval tool using the official Google Cloud documentation approach"""
        try:
            # Use provided corpus info or try to load from stored data
            corpus_name = None
            if corpus_info and 'corpus_name' in corpus_info:
                corpus_name = corpus_info['corpus_name']
                logger.info(f"Using provided corpus info: {corpus_name}")
            else:
                corpus_name = await self._load_corpus_info(project_id)
            
            if not corpus_name:
                logger.error("No corpus information found. Please process documents first.")
                raise ValueError("RAG corpus not available. Please process documents first to create the corpus.")
            
            # Create RAG retrieval tool using the exact method from the official docs
            self.rag_retrieval_tool = Tool.from_retrieval(
                retrieval=rag.Retrieval(
                    source=rag.VertexRagStore(
                        rag_resources=[
                            rag.RagResource(
                                rag_corpus=corpus_name,
                            )
                        ],
                        rag_retrieval_config=rag.RagRetrievalConfig(
                            top_k=10,
                            filter=rag.utils.resources.Filter(vector_distance_threshold=0.5),
                        ),
                    ),
                )
            )
            
            logger.info(f"Created RAG retrieval tool successfully using corpus: {corpus_name}")
            return self.rag_retrieval_tool
            
        except Exception as e:
            logger.error(f"Failed to create RAG retrieval tool: {e}")
            raise
    
    async def process_document(
        self,
        job_id: str,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_path: str,
        parser: str = "llm_parser",
        parse_method: str = "auto",
        model: str = "gemini-2.0-flash-001",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a single document with Vertex AI RAG using the exact GCP RAG engine method"""
        start_time = datetime.utcnow()
        local_file_path = None
        
        try:
            # Update job status to processing
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'processing',
                'progress': {
                    'totalDocuments': 1,
                    'processedDocuments': 0,
                    'failedDocuments': 0,
                    'overallProgress': 0,
                    'currentDocument': gcs_path
                }
            })
            
            # Set up config based on parser type
            if config is None:
                config = {}
            
            # Configure LLM parser based on parser type
            if parser.startswith("llm_parser"):
                config["use_llm_parser"] = True
            else:
                config["use_llm_parser"] = False
            
            # Create or get RAG corpus
            await self._create_or_get_rag_corpus(project_id)
            
            # Import files directly from GCS path to RAG corpus using the exact method from the notebook
            logger.info(f"Importing file from GCS to RAG corpus: {gcs_path}")
            import_response = await self._import_files_to_rag_corpus(gcs_path, project_id, config)
            
            result = {
                'success': True,
                'source': 'vertex_rag_processed',
                'corpus_name': self.rag_corpus.name,
                'gcs_path': gcs_path,
                'import_response': str(import_response),
                'document_count': 1
            }
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update job status to completed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'completed',
                'progress': {
                    'totalDocuments': 1,
                    'processedDocuments': 1,
                    'failedDocuments': 0,
                    'overallProgress': 100,
                    'currentDocument': gcs_path
                },
                'result': {
                    'processing_time': processing_time,
                    'document_path': gcs_path,
                    'rag_result': result,
                    'storage_info': {
                        'corpus_name': self.rag_corpus.name,
                        'gcs_bucket': self.gcs_bucket,
                        'gcs_path': gcs_path
                    }
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'completed',
                f'Document processed successfully with Vertex AI RAG: {Path(gcs_path).name}',
                {
                    'processing_time': processing_time,
                    'document_count': 1,
                    'model': model,
                    'parser': parser,
                    'parse_method': parse_method,
                    'corpus_name': self.rag_corpus.name
                }
            )
            
            logger.info(f"Successfully processed document with Vertex AI RAG: {gcs_path}")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'document_path': gcs_path,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error processing document {gcs_path}: {e}")
            
            # Update job status to failed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'failed',
                'errorMessage': str(e),
                'progress': {
                    'totalDocuments': 1,
                    'processedDocuments': 0,
                    'failedDocuments': 1,
                    'overallProgress': 0,
                    'currentDocument': gcs_path
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'failed',
                f'Document processing failed: {str(e)}'
            )
            
            raise
            
        finally:
            # No cleanup needed since we're not downloading files
            pass
    
    async def process_folder(
        self,
        job_id: str,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_folder_path: str,
        file_extensions: List[str] = None,
        parser: str = "llm_parser",
        parse_method: str = "auto",
        model: str = "gemini-2.0-flash-001",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process all documents in a folder with Vertex AI RAG using the exact GCP RAG engine method"""
        start_time = datetime.utcnow()
        downloaded_files = {}
        
        try:
            # Set default file extensions
            if file_extensions is None:
                file_extensions = [".pdf", ".doc", ".docx", ".txt", ".md"]
            
            # List files in folder
            logger.info(f"Listing files in folder: {gcs_folder_path}")
            file_paths = await self.gcs_manager.list_files_in_folder(gcs_folder_path, file_extensions)
            
            if not file_paths:
                raise ValueError(f"No files found in folder {gcs_folder_path} with extensions {file_extensions}")
            
            total_files = len(file_paths)
            logger.info(f"Found {total_files} files to process")
            
            # Update job status with total file count
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'processing',
                'progress': {
                    'totalDocuments': total_files,
                    'processedDocuments': 0,
                    'failedDocuments': 0,
                    'overallProgress': 0,
                    'currentDocument': None
                }
            })
            
            # Set up config based on parser type
            if config is None:
                config = {}
            
            # Configure LLM parser based on parser type
            if parser.startswith("llm_parser"):
                config["use_llm_parser"] = True
            else:
                config["use_llm_parser"] = False
            
            # Create or get RAG corpus
            await self._create_or_get_rag_corpus(project_id)
            
            # Import files directly from GCS paths to RAG corpus using the exact method from the notebook
            logger.info("Importing files from GCS to RAG corpus")
            import_response = await self._import_files_to_rag_corpus(gcs_folder_path, project_id, config)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                'success': True,
                'source': 'vertex_rag_processed',
                'corpus_name': self.rag_corpus.name,
                'gcs_folder_path': gcs_folder_path,
                'import_response': str(import_response),
                'document_count': total_files
            }
            
            # Update job status to completed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'completed',
                'progress': {
                    'totalDocuments': total_files,
                    'processedDocuments': total_files,
                    'failedDocuments': 0,
                    'overallProgress': 100,
                    'currentDocument': None
                },
                'result': {
                    'processing_time': processing_time,
                    'total_files': total_files,
                    'processed_files': total_files,
                    'failed_files': 0,
                    'folder_path': gcs_folder_path,
                    'rag_result': result,
                    'storage_info': {
                        'corpus_name': self.rag_corpus.name,
                        'gcs_bucket': self.gcs_bucket,
                        'gcs_folder_path': gcs_folder_path
                    }
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'completed',
                f'Folder processed successfully with Vertex AI RAG: {total_files} documents processed',
                {
                    'processing_time': processing_time,
                    'total_files': total_files,
                    'processed_files': total_files,
                    'failed_files': 0,
                    'model': model,
                    'parser': parser,
                    'parse_method': parse_method,
                    'corpus_name': self.rag_corpus.name
                }
            )
            
            logger.info(f"Successfully processed folder with Vertex AI RAG: {gcs_folder_path} ({total_files} documents)")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'total_files': total_files,
                'processed_files': total_files,
                'failed_files': 0,
                'folder_path': gcs_folder_path,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error processing folder {gcs_folder_path}: {e}")
            
            # Update job status to failed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'failed',
                'errorMessage': str(e),
                'progress': {
                    'totalDocuments': total_files,
                    'processedDocuments': 0,
                    'failedDocuments': total_files,
                    'overallProgress': 0,
                    'currentDocument': None
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'failed',
                f'Folder processing failed: {str(e)}'
            )
            
            raise
            
        finally:
            # No cleanup needed since we're not downloading files
            pass
    
    async def get_processing_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get processing status for a job"""
        try:
            return await self.firebase_manager.get_rag_processing_job(job_id)
        except Exception as e:
            logger.error(f"Error getting processing status for job {job_id}: {e}")
            return None
    
    async def query_documents(
        self,
        user_id: str,
        project_id: str,
        query: str,
        model: str = "gemini-2.0-flash-001",
        config: Optional[Dict[str, Any]] = None,
        corpus_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query processed documents using Vertex AI RAG with the exact GCP RAG engine method"""
        try:
            # Create retrieval tool if it doesn't exist
            if not self.rag_retrieval_tool:
                logger.info("RAG retrieval tool not found, creating it...")
                await self._create_rag_retrieval_tool(project_id, corpus_info)
            
            # Generate content with Gemini using RAG Retrieval Tool
            model_instance = generative_models.GenerativeModel(model)
            response = model_instance.generate_content(
                query,
                tools=[self.rag_retrieval_tool],
                generation_config=GenerationConfig()
            )
            
            # Extract answer and sources from response
            answer = response.text if response.text else "No answer generated"
            
            # Extract sources from the response if available
            sources = []
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'function_call') and part.function_call:
                                    # Extract retrieval information if available
                                    if part.function_call.name == "retrieval":
                                        sources.append({
                                            "content": str(part.function_call.args),
                                            "type": "retrieval_result"
                                        })
            
            # Also check for citation metadata in the response
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    # Check citation_metadata
                    if hasattr(candidate, 'citation_metadata') and candidate.citation_metadata:
                        logger.info(f"Found citation_metadata: {candidate.citation_metadata}")
                        if hasattr(candidate.citation_metadata, 'citation_sources'):
                            for citation_source in candidate.citation_metadata.citation_sources:
                                sources.append({
                                    "content": f"Citation from {getattr(citation_source, 'uri', 'unknown source')}",
                                    "type": "citation",
                                    "metadata": {
                                        "uri": getattr(citation_source, 'uri', None),
                                        "start_index": getattr(citation_source, 'start_index', None),
                                        "end_index": getattr(citation_source, 'end_index', None)
                                    }
                                })
                    
                    # Check grounding_metadata (this is where RAG sources are typically stored)
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        logger.info(f"Found grounding_metadata: {candidate.grounding_metadata}")
                        if hasattr(candidate.grounding_metadata, 'grounding_chunks'):
                            for chunk in candidate.grounding_metadata.grounding_chunks:
                                # Extract content from retrieved_context.text
                                content = "No content available"
                                uri = None
                                title = None
                                
                                if hasattr(chunk, 'retrieved_context') and chunk.retrieved_context:
                                    content = getattr(chunk.retrieved_context, 'text', 'No content available')
                                    uri = getattr(chunk.retrieved_context, 'uri', None)
                                    title = getattr(chunk.retrieved_context, 'title', None)
                                
                                sources.append({
                                    "content": content,
                                    "type": "grounding_chunk",
                                    "metadata": {
                                        "uri": uri,
                                        "title": title,
                                        "confidence_score": None  # Will be extracted from grounding_supports
                                    }
                                })
                        
                        # Also extract grounding_supports for confidence scores and text segments
                        if hasattr(candidate.grounding_metadata, 'grounding_supports'):
                            for i, support in enumerate(candidate.grounding_metadata.grounding_supports):
                                if i < len(sources):  # Match with corresponding grounding_chunk
                                    if hasattr(support, 'confidence_scores') and support.confidence_scores:
                                        sources[i]["metadata"]["confidence_score"] = support.confidence_scores[0] if support.confidence_scores else None
                                    if hasattr(support, 'segment') and support.segment:
                                        sources[i]["metadata"]["segment_text"] = getattr(support.segment, 'text', None)
            
            # Log the response structure for debugging
            logger.info(f"Response structure: {type(response)}")
            logger.info(f"Response attributes: {dir(response)}")
            if hasattr(response, 'candidates'):
                logger.info(f"Number of candidates: {len(response.candidates) if response.candidates else 0}")
                if response.candidates:
                    logger.info(f"First candidate attributes: {dir(response.candidates[0])}")
            
            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'query': query,
                'model': model,
                'response': str(response)
            }
                
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }
    
    def get_approach_name(self) -> str:
        """Get the name of this RAG approach"""
        return "Vertex AI RAG"
    
    def get_supported_parsers(self) -> List[str]:
        """Get list of supported parsers for this approach"""
        return ["vertex", "auto", "llm_parser"]
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models for this approach"""
        return ["gemini-2.0-flash-001", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
