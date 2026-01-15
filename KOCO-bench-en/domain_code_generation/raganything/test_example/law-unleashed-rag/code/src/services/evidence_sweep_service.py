"""
EvidenceSweep RAG service implementation - Custom approach with page selection, evidence pass, and synthesis
"""

import logging
import os
import tempfile
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from openai import AsyncOpenAI
from llama_parse import LlamaParse

from src.services.rag_interface import RAGInterface

logger = logging.getLogger(__name__)


class EvidenceSweepService(RAGInterface):
    """Service for processing documents with EvidenceSweep approach"""
    
    def __init__(self, firebase_manager, gcs_manager, auth_service):
        super().__init__(firebase_manager, gcs_manager, auth_service)
        
        # Storage configuration
        self.storage_dir = os.getenv("EVIDENCE_SWEEP_STORAGE_DIR", "./storage/evidence_sweep")
        self.temp_dir = os.getenv("RAG_TEMP_DIR", "./temp")
        
        # Ensure directories exist
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize components
        self._initialize_evidence_sweep()
    
    def _initialize_evidence_sweep(self):
        """Initialize EvidenceSweep components"""
        try:
            # Initialize OpenAI client
            self.openai_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
            
            # Initialize LlamaParse if available
            self.llama_parse = None
            if os.getenv("LLAMA_CLOUD_API_KEY"):
                self.llama_parse = LlamaParse(
                    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                    result_type="markdown",
                    verbose=True
                )
            
            logger.info("EvidenceSweep initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EvidenceSweep: {e}")
            raise
    
    async def _extract_pages_with_metadata(self, document_path: str) -> List[Dict[str, Any]]:
        """Extract pages with metadata from document"""
        try:
            if self.llama_parse:
                # Use LlamaParse for better page extraction
                documents = await self.llama_parse.aload_data(document_path)
                pages = []
                for i, doc in enumerate(documents):
                    pages.append({
                        'page_number': i + 1,
                        'content': doc.text,
                        'metadata': doc.metadata
                    })
                return pages
            else:
                # Fallback to simple text extraction
                with open(document_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple page splitting (this could be improved)
                page_size = 2000  # characters per page
                pages = []
                for i in range(0, len(content), page_size):
                    page_content = content[i:i + page_size]
                    pages.append({
                        'page_number': len(pages) + 1,
                        'content': page_content,
                        'metadata': {'source': document_path}
                    })
                return pages
                
        except Exception as e:
            logger.error(f"Error extracting pages from {document_path}: {e}")
            raise
    
    async def _select_relevant_pages(self, pages: List[Dict[str, Any]], query: str = None) -> List[Dict[str, Any]]:
        """Select relevant pages based on content analysis"""
        try:
            if not pages:
                return []
            
            # If no specific query, select pages with substantial content
            if not query:
                # Simple heuristic: select pages with more than 500 characters
                relevant_pages = [page for page in pages if len(page['content']) > 500]
                return relevant_pages[:10]  # Limit to top 10 pages
            
            # Use LLM to score and select relevant pages
            page_scores = []
            for page in pages:
                try:
                    response = await self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a document analysis expert. Rate the relevance of a page to a query on a scale of 0-10, where 10 is highly relevant. Respond with only a number."
                            },
                            {
                                "role": "user",
                                "content": f"Query: {query}\n\nPage content: {page['content'][:1000]}..."
                            }
                        ],
                        max_tokens=10,
                        temperature=0
                    )
                    
                    score = float(response.choices[0].message.content.strip())
                    page_scores.append((page, score))
                    
                except Exception as e:
                    logger.warning(f"Error scoring page {page['page_number']}: {e}")
                    page_scores.append((page, 5.0))  # Default score
            
            # Sort by score and return top pages
            page_scores.sort(key=lambda x: x[1], reverse=True)
            relevant_pages = [page for page, score in page_scores[:10] if score > 3.0]
            
            return relevant_pages
            
        except Exception as e:
            logger.error(f"Error selecting relevant pages: {e}")
            # Fallback: return first few pages
            return pages[:5]
    
    async def _evidence_pass(self, pages: List[Dict[str, Any]], query: str = None) -> List[Dict[str, Any]]:
        """Perform evidence pass to extract key information from selected pages"""
        try:
            evidence_results = []
            
            for page in pages:
                try:
                    # Extract key evidence from the page
                    evidence_prompt = f"""
                    Analyze the following page content and extract key evidence, facts, and important information.
                    Focus on:
                    1. Key facts and data points
                    2. Important dates, names, and entities
                    3. Legal or technical terms
                    4. Relationships between concepts
                    5. Any contradictions or inconsistencies
                    
                    Page {page['page_number']}:
                    {page['content']}
                    
                    Provide a structured summary of the evidence found.
                    """
                    
                    response = await self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert at extracting and analyzing evidence from documents."},
                            {"role": "user", "content": evidence_prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.1
                    )
                    
                    evidence = response.choices[0].message.content
                    
                    evidence_results.append({
                        'page_number': page['page_number'],
                        'original_content': page['content'],
                        'evidence': evidence,
                        'metadata': page['metadata']
                    })
                    
                except Exception as e:
                    logger.warning(f"Error in evidence pass for page {page['page_number']}: {e}")
                    evidence_results.append({
                        'page_number': page['page_number'],
                        'original_content': page['content'],
                        'evidence': "Error extracting evidence",
                        'metadata': page['metadata']
                    })
            
            return evidence_results
            
        except Exception as e:
            logger.error(f"Error in evidence pass: {e}")
            return []
    
    async def _synthesize_evidence(self, evidence_results: List[Dict[str, Any]], query: str = None) -> Dict[str, Any]:
        """Synthesize evidence into final result"""
        try:
            if not evidence_results:
                return {"synthesis": "No evidence found", "confidence": 0.0}
            
            # Combine all evidence
            combined_evidence = "\n\n".join([
                f"Page {result['page_number']} Evidence:\n{result['evidence']}"
                for result in evidence_results
            ])
            
            synthesis_prompt = f"""
            Based on the following evidence extracted from document pages, provide a comprehensive synthesis.
            
            Evidence:
            {combined_evidence}
            
            Please provide:
            1. A comprehensive summary of the key findings
            2. Important patterns or themes identified
            3. Any contradictions or inconsistencies found
            4. Confidence level (0-100%) in the analysis
            5. Key recommendations or conclusions
            
            Structure your response clearly and cite specific page numbers where relevant.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at synthesizing evidence from multiple sources into coherent insights."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            synthesis = response.choices[0].message.content
            
            # Extract confidence score (simple heuristic)
            confidence = 85.0  # Default confidence
            if "confidence" in synthesis.lower():
                try:
                    # Try to extract confidence percentage
                    import re
                    confidence_match = re.search(r'(\d+)%', synthesis)
                    if confidence_match:
                        confidence = float(confidence_match.group(1))
                except:
                    pass
            
            return {
                "synthesis": synthesis,
                "confidence": confidence,
                "evidence_count": len(evidence_results),
                "pages_analyzed": [result['page_number'] for result in evidence_results]
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing evidence: {e}")
            return {
                "synthesis": f"Error in synthesis: {str(e)}",
                "confidence": 0.0,
                "evidence_count": 0,
                "pages_analyzed": []
            }
    
    async def process_document(
        self,
        job_id: str,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_path: str,
        parser: str = "llamaparse",
        parse_method: str = "auto",
        model: str = "gpt-4o",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a single document with EvidenceSweep approach"""
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
            
            # Download file from GCS
            logger.info(f"Downloading document from GCS: {gcs_path}")
            local_file_path = await self.gcs_manager.download_file(gcs_path)
            
            # Step 1: Extract pages with metadata
            logger.info("Step 1: Extracting pages with metadata")
            pages = await self._extract_pages_with_metadata(local_file_path)
            
            # Update progress
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'progress': {
                    'totalDocuments': 1,
                    'processedDocuments': 0,
                    'failedDocuments': 0,
                    'overallProgress': 25,
                    'currentDocument': gcs_path,
                    'currentStep': 'Page extraction completed'
                }
            })
            
            # Step 2: Select relevant pages
            logger.info("Step 2: Selecting relevant pages")
            query = config.get('query') if config else None
            relevant_pages = await self._select_relevant_pages(pages, query)
            
            # Update progress
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'progress': {
                    'totalDocuments': 1,
                    'processedDocuments': 0,
                    'failedDocuments': 0,
                    'overallProgress': 50,
                    'currentDocument': gcs_path,
                    'currentStep': f'Selected {len(relevant_pages)} relevant pages'
                }
            })
            
            # Step 3: Evidence pass
            logger.info("Step 3: Performing evidence pass")
            evidence_results = await self._evidence_pass(relevant_pages, query)
            
            # Update progress
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'progress': {
                    'totalDocuments': 1,
                    'processedDocuments': 0,
                    'failedDocuments': 0,
                    'overallProgress': 75,
                    'currentDocument': gcs_path,
                    'currentStep': 'Evidence extraction completed'
                }
            })
            
            # Step 4: Synthesize evidence
            logger.info("Step 4: Synthesizing evidence")
            synthesis = await self._synthesize_evidence(evidence_results, query)
            
            result = {
                'success': True,
                'source': 'evidence_sweep_processed',
                'total_pages': len(pages),
                'relevant_pages': len(relevant_pages),
                'evidence_results': evidence_results,
                'synthesis': synthesis
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
                    'currentDocument': gcs_path,
                    'currentStep': 'Processing completed'
                },
                'result': {
                    'processing_time': processing_time,
                    'document_path': gcs_path,
                    'local_path': local_file_path,
                    'rag_result': result,
                    'storage_info': {
                        'evidence_storage_path': os.path.join(self.storage_dir, f"evidence_{project_id}"),
                        'synthesis_storage_path': os.path.join(self.storage_dir, f"synthesis_{project_id}")
                    }
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'completed',
                f'Document processed successfully with EvidenceSweep: {Path(gcs_path).name}',
                {
                    'processing_time': processing_time,
                    'document_count': 1,
                    'model': model,
                    'parser': parser,
                    'parse_method': parse_method,
                    'total_pages': len(pages),
                    'relevant_pages': len(relevant_pages)
                }
            )
            
            logger.info(f"Successfully processed document with EvidenceSweep: {gcs_path}")
            
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
            # Clean up temporary files
            if local_file_path and os.path.exists(local_file_path):
                try:
                    await self.gcs_manager.cleanup_temp_files([local_file_path])
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary file {local_file_path}: {e}")
    
    async def process_folder(
        self,
        job_id: str,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_folder_path: str,
        file_extensions: List[str] = None,
        parser: str = "llamaparse",
        parse_method: str = "auto",
        model: str = "gpt-4o",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process all documents in a folder with EvidenceSweep approach"""
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
            
            # Download all files
            logger.info("Downloading files from GCS")
            downloaded_files = await self.gcs_manager.download_files(file_paths)
            
            if not downloaded_files:
                raise ValueError("Failed to download any files from GCS")
            
            # Process each document
            processed_count = 0
            failed_count = 0
            processing_results = []
            all_evidence = []
            
            for i, (gcs_path, local_path) in enumerate(downloaded_files.items()):
                try:
                    logger.info(f"Processing document {i+1}/{total_files}: {gcs_path}")
                    
                    # Update progress
                    progress = int((i / total_files) * 100)
                    await self.firebase_manager.update_rag_processing_job(job_id, {
                        'progress': {
                            'totalDocuments': total_files,
                            'processedDocuments': processed_count,
                            'failedDocuments': failed_count,
                            'overallProgress': progress,
                            'currentDocument': gcs_path
                        }
                    })
                    
                    # Process document with EvidenceSweep
                    pages = await self._extract_pages_with_metadata(local_path)
                    query = config.get('query') if config else None
                    relevant_pages = await self._select_relevant_pages(pages, query)
                    evidence_results = await self._evidence_pass(relevant_pages, query)
                    
                    # Store evidence for cross-document synthesis
                    for evidence in evidence_results:
                        evidence['source_document'] = gcs_path
                        all_evidence.append(evidence)
                    
                    processed_count += 1
                    processing_results.append({
                        'gcs_path': gcs_path,
                        'local_path': local_path,
                        'success': True,
                        'total_pages': len(pages),
                        'relevant_pages': len(relevant_pages),
                        'evidence_count': len(evidence_results)
                    })
                    
                    logger.info(f"Successfully processed: {gcs_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {gcs_path}: {e}")
                    failed_count += 1
                    processing_results.append({
                        'gcs_path': gcs_path,
                        'local_path': local_path,
                        'success': False,
                        'error': str(e)
                    })
                    continue
            
            # Cross-document synthesis
            logger.info("Performing cross-document synthesis")
            cross_document_synthesis = await self._synthesize_evidence(all_evidence, config.get('query') if config else None)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update job status to completed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'completed',
                'progress': {
                    'totalDocuments': total_files,
                    'processedDocuments': processed_count,
                    'failedDocuments': failed_count,
                    'overallProgress': 100,
                    'currentDocument': None
                },
                'result': {
                    'processing_time': processing_time,
                    'total_files': total_files,
                    'processed_files': processed_count,
                    'failed_files': failed_count,
                    'folder_path': gcs_folder_path,
                    'processing_results': processing_results,
                    'cross_document_synthesis': cross_document_synthesis,
                    'total_evidence_items': len(all_evidence),
                    'storage_info': {
                        'evidence_storage_path': os.path.join(self.storage_dir, f"evidence_{project_id}"),
                        'synthesis_storage_path': os.path.join(self.storage_dir, f"synthesis_{project_id}")
                    }
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'completed',
                f'Folder processed successfully with EvidenceSweep: {processed_count}/{total_files} documents processed',
                {
                    'processing_time': processing_time,
                    'total_files': total_files,
                    'processed_files': processed_count,
                    'failed_files': failed_count,
                    'model': model,
                    'parser': parser,
                    'parse_method': parse_method,
                    'total_evidence_items': len(all_evidence)
                }
            )
            
            logger.info(f"Successfully processed folder with EvidenceSweep: {gcs_folder_path} ({processed_count}/{total_files} documents)")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'total_files': total_files,
                'processed_files': processed_count,
                'failed_files': failed_count,
                'folder_path': gcs_folder_path,
                'results': processing_results,
                'cross_document_synthesis': cross_document_synthesis
            }
            
        except Exception as e:
            logger.error(f"Error processing folder {gcs_folder_path}: {e}")
            
            # Update job status to failed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'failed',
                'errorMessage': str(e),
                'progress': {
                    'totalDocuments': len(downloaded_files) if downloaded_files else 0,
                    'processedDocuments': 0,
                    'failedDocuments': len(downloaded_files) if downloaded_files else 0,
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
            # Clean up temporary files
            if downloaded_files:
                try:
                    local_paths = list(downloaded_files.values())
                    await self.gcs_manager.cleanup_temp_files(local_paths)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary files: {e}")
    
    async def get_processing_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get processing status for a job"""
        try:
            return await self.firebase_manager.get_rag_processing_job(job_id)
        except Exception as e:
            logger.error(f"Error getting processing status for job {job_id}: {e}")
            return None
    
    def get_approach_name(self) -> str:
        """Get the name of this RAG approach"""
        return "EvidenceSweep"
    
    def get_supported_parsers(self) -> List[str]:
        """Get list of supported parsers for this approach"""
        parsers = ["simple"]
        if self.llama_parse:
            parsers.append("llamaparse")
        return parsers
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models for this approach"""
        return ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
