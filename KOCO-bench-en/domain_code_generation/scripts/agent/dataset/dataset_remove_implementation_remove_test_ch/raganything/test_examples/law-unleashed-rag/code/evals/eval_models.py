"""
Evaluation framework models for RAG approach comparison
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class EvaluationType(str, Enum):
    """Types of evaluations that can be performed"""
    DOCUMENT_PROCESSING = "document_processing"
    QUERY_ANSWERING = "query_answering"
    CROSS_DOCUMENT_ANALYSIS = "cross_document_analysis"
    PERFORMANCE = "performance"
    END_TO_END = "end_to_end"


class MetricType(str, Enum):
    """Types of metrics that can be calculated"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    PROCESSING_TIME = "processing_time"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    # LLM-based metrics for unstructured content
    LLM_COMPLETENESS = "llm_completeness"
    LLM_ACCURACY = "llm_accuracy"
    LLM_COHERENCE = "llm_coherence"


class EvaluationStatus(str, Enum):
    """Status of an evaluation case execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExpectedOutput(BaseModel):
    """Expected output for a test case - optimized for unstructured content evaluation"""
    type: str = Field(..., description="Type of expected output (entities, facts, summary, etc.)")
    content: Union[str, List[str], Dict[str, Any]] = Field(..., description="Expected content - can be unstructured text, lists, or any format")
    weight: float = Field(default=1.0, description="Weight for this output in scoring")
    description: Optional[str] = Field(default=None, description="Human-readable description of what should be found - used by LLM-as-a-Judge")
    # For unstructured content, description is often more important than structured content


class EvaluationCriteria(BaseModel):
    """Criteria for evaluating a test case"""
    metrics: List[MetricType] = Field(..., description="Metrics to calculate")
    thresholds: Dict[MetricType, float] = Field(default_factory=dict, description="Minimum thresholds for each metric")
    weights: Dict[MetricType, float] = Field(default_factory=dict, description="Weights for each metric in overall score")


class EvaluationCase(BaseModel):
    """Individual evaluation case definition"""
    id: str = Field(..., description="Unique evaluation case ID")
    name: str = Field(..., description="Human-readable evaluation case name")
    description: str = Field(..., description="Description of what this evaluation case evaluates")
    evaluation_type: EvaluationType = Field(..., description="Type of evaluation")
    
    # Input data
    project_id: str = Field(..., description="Project ID for the evaluation case")
    document_paths: List[str] = Field(..., description="GCS paths to documents to process")
    query: Optional[str] = Field(default=None, description="Query to test (for query answering)")
    
    # Expected outputs and criteria
    expected_outputs: List[ExpectedOutput] = Field(..., description="Expected outputs")
    evaluation_criteria: EvaluationCriteria = Field(..., description="Evaluation criteria")
    
    # Configuration
    rag_approach: Optional[str] = Field(default=None, description="Specific RAG approach to test (None for all)")
    parser: str = Field(default="mineru", description="Parser to use")
    model: str = Field(default="gpt-4o-mini", description="Model to use")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    difficulty: str = Field(default="medium", description="Difficulty level (easy, medium, hard)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationSuite(BaseModel):
    """Collection of related evaluation cases"""
    id: str = Field(..., description="Unique evaluation suite ID")
    name: str = Field(..., description="Human-readable evaluation suite name")
    description: str = Field(..., description="Description of the evaluation suite")
    evaluation_cases: List[EvaluationCase] = Field(..., description="Evaluation cases in this suite")
    
    # Configuration
    default_rag_approaches: List[str] = Field(default_factory=list, description="Default RAG approaches to evaluate")
    default_parser: str = Field(default="mineru", description="Default parser")
    default_model: str = Field(default="gpt-4o-mini", description="Default model")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MetricResult(BaseModel):
    """Result of a single metric calculation"""
    metric_type: MetricType = Field(..., description="Type of metric")
    value: float = Field(..., description="Calculated metric value")
    threshold: Optional[float] = Field(default=None, description="Threshold for this metric")
    passed: Optional[bool] = Field(default=None, description="Whether threshold was met")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional metric details")


class EvaluationCaseResult(BaseModel):
    """Result of executing a single evaluation case"""
    evaluation_case_id: str = Field(..., description="ID of the evaluation case")
    rag_approach: str = Field(..., description="RAG approach that was evaluated")
    status: EvaluationStatus = Field(..., description="Execution status")
    
    # Results
    actual_outputs: Optional[Dict[str, Any]] = Field(default=None, description="Actual outputs from the RAG approach")
    metrics: List[MetricResult] = Field(default_factory=list, description="Calculated metrics")
    overall_score: Optional[float] = Field(default=None, description="Overall weighted score")
    
    # Performance data
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    memory_usage: Optional[float] = Field(default=None, description="Memory usage in MB")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Metadata
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    executed_by: str = Field(..., description="User or system that executed the test")


class EvaluationRun(BaseModel):
    """Complete evaluation run across multiple test cases and RAG approaches"""
    id: str = Field(..., description="Unique evaluation run ID")
    name: str = Field(..., description="Human-readable evaluation run name")
    description: Optional[str] = Field(default=None, description="Description of the evaluation run")
    
    # Configuration
    evaluation_suite_id: str = Field(..., description="Evaluation suite being evaluated")
    rag_approaches: List[str] = Field(..., description="RAG approaches to evaluate")
    user_id: str = Field(..., description="User running the evaluation")
    project_id: str = Field(..., description="Project context for the evaluation")
    
    # Results
    evaluation_case_results: List[EvaluationCaseResult] = Field(default_factory=list, description="Results for each evaluation case")
    status: EvaluationStatus = Field(default=EvaluationStatus.PENDING, description="Overall evaluation status")
    
    # Summary statistics
    total_evaluation_cases: int = Field(default=0, description="Total number of evaluation cases")
    completed_evaluation_cases: int = Field(default=0, description="Number of completed evaluation cases")
    failed_evaluation_cases: int = Field(default=0, description="Number of failed evaluation cases")
    average_scores: Dict[str, float] = Field(default_factory=dict, description="Average scores by RAG approach")
    
    # Metadata
    started_at: Optional[datetime] = Field(default=None, description="When evaluation started")
    completed_at: Optional[datetime] = Field(default=None, description="When evaluation completed")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationRequest(BaseModel):
    """Request to run an evaluation"""
    evaluation_suite_id: str = Field(..., description="Evaluation suite to evaluate")
    rag_approaches: List[str] = Field(..., description="RAG approaches to evaluate")
    user_id: str = Field(..., description="User running the evaluation")
    project_id: str = Field(..., description="Project context")
    name: str = Field(..., description="Name for this evaluation run")
    description: Optional[str] = Field(default=None, description="Description of the evaluation")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")


class EvaluationResponse(BaseModel):
    """Response from starting an evaluation"""
    evaluation_run_id: str = Field(..., description="ID of the created evaluation run")
    status: str = Field(..., description="Initial status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Creation timestamp")


class EvaluationStatusResponse(BaseModel):
    """Response for evaluation status check"""
    evaluation_run_id: str = Field(..., description="Evaluation run ID")
    status: EvaluationStatus = Field(..., description="Current status")
    progress: Dict[str, Any] = Field(default_factory=dict, description="Progress information")
    results_summary: Optional[Dict[str, Any]] = Field(default=None, description="Summary of results so far")
    started_at: Optional[datetime] = Field(default=None, description="Start time")
    completed_at: Optional[datetime] = Field(default=None, description="Completion time")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class EvaluationResultsResponse(BaseModel):
    """Response with detailed evaluation results"""
    evaluation_run: EvaluationRun = Field(..., description="Complete evaluation run data")
    detailed_results: List[EvaluationCaseResult] = Field(..., description="Detailed evaluation case results")
    comparison_summary: Dict[str, Any] = Field(..., description="Comparison summary across approaches")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations based on results")
