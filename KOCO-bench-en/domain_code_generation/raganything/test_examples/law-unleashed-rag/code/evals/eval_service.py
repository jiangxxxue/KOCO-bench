"""
Evaluation service for comparing RAG approaches
"""

import logging
import uuid
import asyncio
import time
import psutil
import os
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime

from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .eval_models import (
    EvaluationCase, EvaluationSuite, EvaluationCaseResult, EvaluationRun, MetricResult,
    EvaluationStatus, MetricType, EvaluationType
)
from src.utils.firebase_utils import FirebaseManager

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for running evaluations of RAG approaches"""
    
    def __init__(self, firebase_manager: FirebaseManager, gcs_manager, auth_service):
        self.firebase_manager = firebase_manager
        self.gcs_manager = gcs_manager
        self.auth_service = auth_service
        
        # API configuration
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.api_timeout = 300  # 5 minutes timeout for API calls
        
        # Initialize evaluation components
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_transformer = None
    
    async def run_evaluation(
        self,
        evaluation_run: EvaluationRun,
        evaluation_suite: EvaluationSuite
    ) -> EvaluationRun:
        """Run a complete evaluation across all evaluation cases and RAG approaches"""
        
        logger.info(f"Starting evaluation run: {evaluation_run.id}")
        
        try:
            # Update evaluation status
            evaluation_run.status = EvaluationStatus.RUNNING
            evaluation_run.started_at = datetime.utcnow()
            evaluation_run.total_evaluation_cases = len(evaluation_suite.evaluation_cases) * len(evaluation_run.rag_approaches)
            
            await self._update_evaluation_run(evaluation_run)
            
            # Run evaluation cases for each RAG approach
            for rag_approach in evaluation_run.rag_approaches:
                logger.info(f"Evaluating RAG approach: {rag_approach}")
                
                for evaluation_case in evaluation_suite.evaluation_cases:
                    # Skip if evaluation case specifies a different RAG approach
                    if evaluation_case.rag_approach and evaluation_case.rag_approach != rag_approach:
                        continue
                    
                    try:
                        result = await self._run_evaluation_case(evaluation_case, rag_approach, evaluation_run.user_id)
                        evaluation_run.evaluation_case_results.append(result)
                        
                        # Update progress
                        evaluation_run.completed_evaluation_cases += 1
                        if result.status == EvaluationStatus.FAILED:
                            evaluation_run.failed_evaluation_cases += 1
                        
                        await self._update_evaluation_run(evaluation_run)
                        
                    except Exception as e:
                        logger.error(f"Error running evaluation case {evaluation_case.id} with {rag_approach}: {e}")
                        
                        # Create failed result
                        failed_result = EvaluationCaseResult(
                            evaluation_case_id=evaluation_case.id,
                            rag_approach=rag_approach,
                            status=EvaluationStatus.FAILED,
                            error_message=str(e),
                            executed_by=evaluation_run.user_id
                        )
                        evaluation_run.evaluation_case_results.append(failed_result)
                        evaluation_run.failed_evaluation_cases += 1
                        evaluation_run.completed_evaluation_cases += 1
                        
                        await self._update_evaluation_run(evaluation_run)
            
            # Calculate summary statistics
            await self._calculate_summary_statistics(evaluation_run)
            
            # Mark as completed
            evaluation_run.status = EvaluationStatus.COMPLETED
            evaluation_run.completed_at = datetime.utcnow()
            
            await self._update_evaluation_run(evaluation_run)
            
            logger.info(f"Completed evaluation run: {evaluation_run.id}")
            return evaluation_run
            
        except Exception as e:
            logger.error(f"Error in evaluation run {evaluation_run.id}: {e}")
            evaluation_run.status = EvaluationStatus.FAILED
            evaluation_run.completed_at = datetime.utcnow()
            await self._update_evaluation_run(evaluation_run)
            raise
    
    async def _run_evaluation_case(
        self,
        evaluation_case: EvaluationCase,
        rag_approach: str,
        user_id: str
    ) -> EvaluationCaseResult:
        """Run a single evaluation case with a specific RAG approach"""
        
        logger.info(f"Running evaluation case {evaluation_case.id} with {rag_approach}")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = EvaluationCaseResult(
            evaluation_case_id=evaluation_case.id,
            rag_approach=rag_approach,
            status=EvaluationStatus.RUNNING,
            executed_by=user_id
        )
        
        try:
            # Process documents based on evaluation case type using API calls
            if evaluation_case.evaluation_type == EvaluationType.DOCUMENT_PROCESSING:
                actual_outputs = await self._evaluate_document_processing(
                    evaluation_case, rag_approach, user_id
                )
            elif evaluation_case.evaluation_type == EvaluationType.QUERY_ANSWERING:
                actual_outputs = await self._evaluate_query_answering(
                    evaluation_case, rag_approach, user_id
                )
            elif evaluation_case.evaluation_type == EvaluationType.CROSS_DOCUMENT_ANALYSIS:
                actual_outputs = await self._evaluate_cross_document_analysis(
                    evaluation_case, rag_approach, user_id
                )
            else:
                raise ValueError(f"Unsupported evaluation type: {evaluation_case.evaluation_type}")
            
            result.actual_outputs = actual_outputs
            
            # Calculate metrics
            metrics = await self._calculate_metrics(evaluation_case, actual_outputs)
            result.metrics = metrics
            
            # Calculate overall score
            result.overall_score = self._calculate_overall_score(metrics, evaluation_case.evaluation_criteria)
            
            # Record performance metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            result.processing_time = end_time - start_time
            result.memory_usage = end_memory - start_memory
            result.status = EvaluationStatus.COMPLETED
            
            logger.info(f"Completed evaluation case {evaluation_case.id} with score: {result.overall_score}")
            
        except Exception as e:
            logger.error(f"Error in evaluation case {evaluation_case.id}: {e}")
            result.status = EvaluationStatus.FAILED
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
        
        return result
    
    async def _evaluate_document_processing(
        self,
        evaluation_case: EvaluationCase,
        rag_approach: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Evaluate document processing capabilities using API calls"""
        
        # Process each document using API calls
        processing_results = []
        for doc_path in evaluation_case.document_paths:
            # Process the document via API
            result = await self._process_document_via_api(
                user_id=user_id,
                project_id=evaluation_case.project_id,
                workspace_id=evaluation_case.project_id,
                gcs_path=doc_path,
                rag_approach=rag_approach,
                parser=evaluation_case.parser,
                model=evaluation_case.model,
                config=evaluation_case.config
            )
            
            processing_results.append({
                'document_path': doc_path,
                'result': result
            })
        
        return {
            'processing_results': processing_results,
            'total_documents': len(evaluation_case.document_paths),
            'successful_documents': len([r for r in processing_results if r['result'].get('success', False)])
        }
    
    async def _evaluate_query_answering(
        self,
        evaluation_case: EvaluationCase,
        rag_approach: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Evaluate query answering capabilities using API calls"""
        
        if not evaluation_case.query:
            raise ValueError("Query is required for query answering evaluation")
        
        # First process documents via API
        processing_results = []
        for doc_path in evaluation_case.document_paths:
            result = await self._process_document_via_api(
                user_id=user_id,
                project_id=evaluation_case.project_id,
                workspace_id=evaluation_case.project_id,
                gcs_path=doc_path,
                rag_approach=rag_approach,
                parser=evaluation_case.parser,
                model=evaluation_case.model,
                config=evaluation_case.config
            )
            processing_results.append(result)
        
        # Query the processed documents via API
        query_result = await self._query_documents_via_api(
            user_id=user_id,
            project_id=evaluation_case.project_id,
            query=evaluation_case.query,
            rag_approach=rag_approach,
            model=evaluation_case.model,
            config=evaluation_case.config
        )
        
        return {
            'query': evaluation_case.query,
            'processing_results': processing_results,
            'answer': query_result.get('answer', ''),
            'sources': query_result.get('sources', []),
            'metadata': query_result.get('metadata', {})
        }
    
    async def _evaluate_cross_document_analysis(
        self,
        evaluation_case: EvaluationCase,
        rag_approach: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Evaluate cross-document analysis capabilities"""
        
        # Process all documents using API calls
        if len(evaluation_case.document_paths) > 1:
            # Use folder processing for multiple documents via API
            result = await self._process_folder_via_api(
                user_id=user_id,
                project_id=evaluation_case.project_id,
                workspace_id=evaluation_case.project_id,
                gcs_folder_path=os.path.dirname(evaluation_case.document_paths[0]) + "/",
                rag_approach=rag_approach,
                parser=evaluation_case.parser,
                model=evaluation_case.model,
                config=evaluation_case.config
            )
        else:
            # Single document processing via API
            result = await self._process_document_via_api(
                user_id=user_id,
                project_id=evaluation_case.project_id,
                workspace_id=evaluation_case.project_id,
                gcs_path=evaluation_case.document_paths[0],
                rag_approach=rag_approach,
                parser=evaluation_case.parser,
                model=evaluation_case.model,
                config=evaluation_case.config
            )
        
        return {
            'cross_document_analysis': result,
            'document_count': len(evaluation_case.document_paths)
        }
    
    async def _calculate_metrics(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> List[MetricResult]:
        """Calculate metrics for an evaluation case"""
        
        metrics = []
        
        for metric_type in evaluation_case.evaluation_criteria.metrics:
            try:
                if metric_type == MetricType.PROCESSING_TIME:
                    # This is calculated in the evaluation case execution
                    continue
                elif metric_type == MetricType.MEMORY_USAGE:
                    # This is calculated in the evaluation case execution
                    continue
                elif metric_type == MetricType.SEMANTIC_SIMILARITY:
                    value = await self._calculate_semantic_similarity(evaluation_case, actual_outputs)
                elif metric_type == MetricType.COMPLETENESS:
                    value = await self._calculate_completeness(evaluation_case, actual_outputs)
                elif metric_type == MetricType.RELEVANCE:
                    value = await self._calculate_relevance(evaluation_case, actual_outputs)
                elif metric_type == MetricType.LLM_COMPLETENESS:
                    value = await self._calculate_llm_completeness(evaluation_case, actual_outputs)
                elif metric_type == MetricType.LLM_ACCURACY:
                    value = await self._calculate_llm_accuracy(evaluation_case, actual_outputs)
                elif metric_type == MetricType.LLM_COHERENCE:
                    value = await self._calculate_llm_coherence(evaluation_case, actual_outputs)
                else:
                    # Default to 0 for unimplemented metrics
                    value = 0.0
                
                threshold = evaluation_case.evaluation_criteria.thresholds.get(metric_type)
                passed = value >= threshold if threshold is not None else None
                
                metrics.append(MetricResult(
                    metric_type=metric_type,
                    value=value,
                    threshold=threshold,
                    passed=passed
                ))
                
            except Exception as e:
                logger.warning(f"Error calculating metric {metric_type}: {e}")
                metrics.append(MetricResult(
                    metric_type=metric_type,
                    value=0.0,
                    threshold=evaluation_case.evaluation_criteria.thresholds.get(metric_type),
                    passed=False
                ))
        
        return metrics
    
    async def _calculate_semantic_similarity(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> float:
        """Calculate semantic similarity between expected and actual outputs"""
        
        if not self.sentence_transformer:
            return 0.0
        
        try:
            # Extract text from expected outputs
            expected_texts = []
            for expected in evaluation_case.expected_outputs:
                if isinstance(expected.content, str):
                    expected_texts.append(expected.content)
                elif isinstance(expected.content, list):
                    expected_texts.extend([str(item) for item in expected.content])
            
            # Extract text from actual outputs
            actual_texts = []
            if 'processing_results' in actual_outputs:
                for result in actual_outputs['processing_results']:
                    if 'result' in result and 'rag_result' in result['result']:
                        rag_result = result['result']['rag_result']
                        if 'synthesis' in rag_result:
                            actual_texts.append(str(rag_result['synthesis']))
            
            if not expected_texts or not actual_texts:
                return 0.0
            
            # Calculate embeddings
            expected_embeddings = self.sentence_transformer.encode(expected_texts)
            actual_embeddings = self.sentence_transformer.encode(actual_texts)
            
            # Calculate similarity
            similarities = cosine_similarity(expected_embeddings, actual_embeddings)
            return float(np.max(similarities))
            
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    async def _calculate_completeness(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> float:
        """Calculate completeness of actual outputs compared to expected"""
        
        try:
            expected_count = len(evaluation_case.expected_outputs)
            if expected_count == 0:
                return 1.0
            
            # Count how many expected outputs have corresponding actual outputs
            found_count = 0
            
            for expected in evaluation_case.expected_outputs:
                if self._find_expected_in_actual(expected, actual_outputs):
                    found_count += 1
            
            return found_count / expected_count
            
        except Exception as e:
            logger.warning(f"Error calculating completeness: {e}")
            return 0.0
    
    async def _calculate_relevance(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> float:
        """Calculate relevance of actual outputs to the evaluation case"""
        
        try:
            if not evaluation_case.query:
                return 1.0  # No query to compare against
            
            # Use LLM to evaluate relevance
            actual_text = str(actual_outputs)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Rate the relevance of the actual output to the given query on a scale of 0-1, where 1 is highly relevant. Respond with only a number."
                    },
                    {
                        "role": "user",
                        "content": f"Query: {evaluation_case.query}\n\nActual Output: {actual_text[:1000]}..."
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            relevance_score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, relevance_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Error calculating relevance: {e}")
            return 0.0
    
    async def _calculate_llm_completeness(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> float:
        """Calculate completeness using LLM-as-a-Judge for unstructured content"""
        
        try:
            if not evaluation_case.expected_outputs:
                return 1.0
            
            # Prepare expected outputs for LLM evaluation
            expected_descriptions = []
            for expected in evaluation_case.expected_outputs:
                expected_descriptions.append(f"- {expected.description or expected.type}: {expected.content}")
            
            expected_text = "\n".join(expected_descriptions)
            actual_text = str(actual_outputs)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Rate how completely the actual output covers the expected information on a scale of 0-1, where 1 means all expected information is present. Consider that the actual output may express the same information in different words. Respond with only a number."
                    },
                    {
                        "role": "user",
                        "content": f"Expected Information:\n{expected_text}\n\nActual Output: {actual_text[:1500]}..."
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            completeness_score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, completeness_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Error calculating LLM completeness: {e}")
            return 0.0
    
    async def _calculate_llm_accuracy(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> float:
        """Calculate accuracy using LLM-as-a-Judge for unstructured content"""
        
        try:
            if not evaluation_case.expected_outputs:
                return 1.0
            
            # Prepare expected outputs for LLM evaluation
            expected_descriptions = []
            for expected in evaluation_case.expected_outputs:
                expected_descriptions.append(f"- {expected.description or expected.type}: {expected.content}")
            
            expected_text = "\n".join(expected_descriptions)
            actual_text = str(actual_outputs)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Rate the accuracy of the actual output compared to the expected information on a scale of 0-1, where 1 means the information is completely accurate. Consider that the actual output may express the same information in different words. Respond with only a number."
                    },
                    {
                        "role": "user",
                        "content": f"Expected Information:\n{expected_text}\n\nActual Output: {actual_text[:1500]}..."
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            accuracy_score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, accuracy_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Error calculating LLM accuracy: {e}")
            return 0.0
    
    async def _calculate_llm_coherence(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> float:
        """Calculate coherence using LLM-as-a-Judge"""
        
        try:
            actual_text = str(actual_outputs)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Rate the coherence and logical flow of the actual output on a scale of 0-1, where 1 means the output is well-structured, logical, and easy to follow. Respond with only a number."
                    },
                    {
                        "role": "user",
                        "content": f"Actual Output: {actual_text[:1500]}..."
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            coherence_score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, coherence_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Error calculating LLM coherence: {e}")
            return 0.0
    
    def _find_expected_in_actual(
        self,
        expected: Any,
        actual_outputs: Dict[str, Any]
    ) -> bool:
        """Check if expected output is found in actual outputs"""
        
        try:
            expected_str = str(expected.content).lower()
            actual_str = str(actual_outputs).lower()
            
            # Simple substring matching for now
            return expected_str in actual_str
            
        except Exception:
            return False
    
    def _calculate_overall_score(
        self,
        metrics: List[MetricResult],
        criteria
    ) -> float:
        """Calculate overall weighted score from individual metrics"""
        
        if not metrics:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric in metrics:
            weight = criteria.weights.get(metric.metric_type, 1.0)
            weighted_sum += metric.value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    async def _calculate_summary_statistics(self, evaluation_run: EvaluationRun):
        """Calculate summary statistics for the evaluation run"""
        
        # Group results by RAG approach
        approach_results = {}
        for result in evaluation_run.evaluation_case_results:
            if result.rag_approach not in approach_results:
                approach_results[result.rag_approach] = []
            approach_results[result.rag_approach].append(result)
        
        # Calculate average scores by approach
        for approach, results in approach_results.items():
            scores = [r.overall_score for r in results if r.overall_score is not None]
            if scores:
                evaluation_run.average_scores[approach] = sum(scores) / len(scores)
    
    async def _update_evaluation_run(self, evaluation_run: EvaluationRun):
        """Update evaluation run in storage"""
        
        try:
            # Store in Firebase or local storage
            # For now, we'll just log the update
            logger.info(f"Updated evaluation run {evaluation_run.id}: {evaluation_run.status}")
            
        except Exception as e:
            logger.error(f"Error updating evaluation run: {e}")
    
    async def get_evaluation_run(self, evaluation_run_id: str) -> Optional[EvaluationRun]:
        """Get an evaluation run by ID"""
        
        # TODO: Implement retrieval from storage
        return None
    
    async def get_evaluation_suite(self, evaluation_suite_id: str) -> Optional[EvaluationSuite]:
        """Get an evaluation suite by ID"""
        
        # TODO: Implement retrieval from storage
        return None
    
    async def _process_document_via_api(
        self,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_path: str,
        rag_approach: str,
        parser: str = "mineru",
        model: str = "gpt-4o-mini",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a document via API call"""
        
        async with httpx.AsyncClient(timeout=self.api_timeout) as client:
            payload = {
                "user_id": user_id,
                "project_id": project_id,
                "workspace_id": workspace_id,
                "gcs_path": gcs_path,
                "rag_approach": rag_approach,
                "parser": parser,
                "model": model,
                "config": config or {}
            }
            
            response = await client.post(
                f"{self.api_base_url}/process-document",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API call failed: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Wait for processing to complete
            job_id = result["job_id"]
            return await self._wait_for_job_completion(client, job_id, user_id)
    
    async def _process_folder_via_api(
        self,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_folder_path: str,
        rag_approach: str,
        parser: str = "mineru",
        model: str = "gpt-4o-mini",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a folder via API call"""
        
        async with httpx.AsyncClient(timeout=self.api_timeout) as client:
            payload = {
                "user_id": user_id,
                "project_id": project_id,
                "workspace_id": workspace_id,
                "gcs_folder_path": gcs_folder_path,
                "rag_approach": rag_approach,
                "parser": parser,
                "model": model,
                "config": config or {}
            }
            
            response = await client.post(
                f"{self.api_base_url}/process-folder",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API call failed: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Wait for processing to complete
            job_id = result["job_id"]
            return await self._wait_for_job_completion(client, job_id, user_id)
    
    async def _query_documents_via_api(
        self,
        user_id: str,
        project_id: str,
        query: str,
        rag_approach: str,
        model: str = "gpt-4o-mini",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query documents via API call"""
        
        async with httpx.AsyncClient(timeout=self.api_timeout) as client:
            payload = {
                "user_id": user_id,
                "project_id": project_id,
                "query": query,
                "rag_approach": rag_approach,
                "model": model,
                "config": config or {}
            }
            
            response = await client.post(
                f"{self.api_base_url}/query",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API call failed: {response.status_code} - {response.text}")
            
            return response.json()
    
    async def _wait_for_job_completion(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        user_id: str,
        max_wait_time: int = 300
    ) -> Dict[str, Any]:
        """Wait for a processing job to complete"""
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = await client.get(
                f"{self.api_base_url}/processing-status/{job_id}",
                params={"user_id": user_id}
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to get job status: {response.status_code} - {response.text}")
            
            status_data = response.json()
            status = status_data["status"]
            
            if status == "completed":
                return status_data.get("result", {})
            elif status == "failed":
                error_msg = status_data.get("error_message", "Unknown error")
                raise Exception(f"Job failed: {error_msg}")
            
            # Wait before checking again
            await asyncio.sleep(5)
        
        raise Exception(f"Job {job_id} did not complete within {max_wait_time} seconds")
