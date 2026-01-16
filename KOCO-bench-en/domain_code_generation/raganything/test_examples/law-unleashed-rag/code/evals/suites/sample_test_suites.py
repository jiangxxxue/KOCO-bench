"""
Sample evaluation suites for RAG approach evaluation
"""

from typing import Dict
from ..eval_models import (
    EvaluationCase, EvaluationSuite, ExpectedOutput, EvaluationCriteria,
    EvaluationType, MetricType
)


def create_legal_document_evaluation_suite() -> EvaluationSuite:
    """Create an evaluation suite for legal document processing"""
    
    evaluation_cases = [
        EvaluationCase(
            id="legal_contract_analysis",
            name="Contract Analysis",
            description="Test ability to extract key terms and clauses from legal contracts",
            evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
            project_id="legal_test_project",
            document_paths=["gs://test-bucket/contracts/sample_contract.pdf"],
            expected_outputs=[
                ExpectedOutput(
                    type="entities",
                    content=["parties", "payment terms", "termination clause", "liability"],
                    weight=1.0,
                    description="Key legal entities and concepts"
                ),
                ExpectedOutput(
                    type="facts",
                    content=["contract duration", "payment amount", "deliverables"],
                    weight=1.0,
                    description="Important factual information"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY],
                thresholds={MetricType.COMPLETENESS: 0.7, MetricType.SEMANTIC_SIMILARITY: 0.6},
                weights={MetricType.COMPLETENESS: 0.6, MetricType.SEMANTIC_SIMILARITY: 0.4}
            ),
            tags=["legal", "contracts", "entity-extraction"],
            difficulty="medium"
        ),
        
        EvaluationCase(
            id="legal_research_query",
            name="Legal Research Query",
            description="Test ability to answer legal research questions from case law",
            evaluation_type=EvaluationType.QUERY_ANSWERING,
            project_id="legal_test_project",
            document_paths=["gs://test-bucket/case_law/supreme_court_cases.pdf"],
            query="What are the key principles established in contract law cases?",
            expected_outputs=[
                ExpectedOutput(
                    type="answer",
                    content="Key principles include offer and acceptance, consideration, capacity, and legality",
                    weight=1.0,
                    description="Comprehensive answer to legal research question"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.RELEVANCE, MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY],
                thresholds={MetricType.RELEVANCE: 0.8, MetricType.COMPLETENESS: 0.7, MetricType.SEMANTIC_SIMILARITY: 0.7},
                weights={MetricType.RELEVANCE: 0.4, MetricType.COMPLETENESS: 0.3, MetricType.SEMANTIC_SIMILARITY: 0.3}
            ),
            tags=["legal", "research", "query-answering"],
            difficulty="hard"
        )
    ]
    
    return EvaluationSuite(
        id="legal_documents",
        name="Legal Document Processing",
        description="Test suite for evaluating RAG approaches on legal documents",
        evaluation_cases=evaluation_cases,
        default_rag_approaches=["raganything", "evidence_sweep", "rag_vertex"],
        tags=["legal", "documents", "comprehensive"]
    )


def create_medical_document_evaluation_suite() -> EvaluationSuite:
    """Create a test suite for medical document processing"""
    
    evaluation_cases = [
        EvaluationCase(
            id="medical_record_analysis",
            name="Medical Record Analysis",
            description="Test ability to extract medical information from patient records",
            evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
            project_id="medical_test_project",
            document_paths=["gs://test-bucket/medical/patient_records.pdf"],
            expected_outputs=[
                ExpectedOutput(
                    type="medical_entities",
                    content=["diagnosis", "medications", "vital signs", "treatment plan"],
                    weight=1.0,
                    description="Key medical entities and concepts"
                ),
                ExpectedOutput(
                    type="temporal_info",
                    content=["dates", "timeline", "follow-up appointments"],
                    weight=0.8,
                    description="Temporal information in medical records"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.PROCESSING_TIME],
                thresholds={MetricType.COMPLETENESS: 0.8, MetricType.SEMANTIC_SIMILARITY: 0.7, MetricType.PROCESSING_TIME: 30.0},
                weights={MetricType.COMPLETENESS: 0.5, MetricType.SEMANTIC_SIMILARITY: 0.3, MetricType.PROCESSING_TIME: 0.2}
            ),
            tags=["medical", "records", "entity-extraction"],
            difficulty="medium"
        ),
        
        EvaluationCase(
            id="medical_cross_document_analysis",
            name="Cross-Document Medical Analysis",
            description="Test ability to analyze multiple medical documents for comprehensive patient understanding",
            evaluation_type=EvaluationType.CROSS_DOCUMENT_ANALYSIS,
            project_id="medical_test_project",
            document_paths=[
                "gs://test-bucket/medical/patient_records.pdf",
                "gs://test-bucket/medical/lab_results.pdf",
                "gs://test-bucket/medical/imaging_reports.pdf"
            ],
            expected_outputs=[
                ExpectedOutput(
                    type="comprehensive_summary",
                    content=["patient condition", "treatment history", "current status", "recommendations"],
                    weight=1.0,
                    description="Comprehensive patient summary across documents"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.RELEVANCE],
                thresholds={MetricType.COMPLETENESS: 0.8, MetricType.SEMANTIC_SIMILARITY: 0.7, MetricType.RELEVANCE: 0.8},
                weights={MetricType.COMPLETENESS: 0.4, MetricType.SEMANTIC_SIMILARITY: 0.3, MetricType.RELEVANCE: 0.3}
            ),
            tags=["medical", "cross-document", "synthesis"],
            difficulty="hard"
        )
    ]
    
    return EvaluationSuite(
        id="medical_documents",
        name="Medical Document Processing",
        description="Test suite for evaluating RAG approaches on medical documents",
        evaluation_cases=evaluation_cases,
        default_rag_approaches=["raganything", "evidence_sweep", "rag_vertex"],
        tags=["medical", "documents", "healthcare"]
    )


def create_performance_evaluation_suite() -> EvaluationSuite:
    """Create a test suite focused on performance metrics"""
    
    evaluation_cases = [
        EvaluationCase(
            id="large_document_processing",
            name="Large Document Processing",
            description="Test processing performance on large documents",
            evaluation_type=EvaluationType.PERFORMANCE,
            project_id="performance_test_project",
            document_paths=["gs://test-bucket/large_documents/100_page_report.pdf"],
            expected_outputs=[
                ExpectedOutput(
                    type="processing_success",
                    content=["successful_processing", "no_errors"],
                    weight=1.0,
                    description="Successful processing without errors"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.PROCESSING_TIME, MetricType.MEMORY_USAGE, MetricType.THROUGHPUT],
                thresholds={MetricType.PROCESSING_TIME: 60.0, MetricType.MEMORY_USAGE: 1000.0, MetricType.THROUGHPUT: 1.0},
                weights={MetricType.PROCESSING_TIME: 0.4, MetricType.MEMORY_USAGE: 0.3, MetricType.THROUGHPUT: 0.3}
            ),
            tags=["performance", "large-documents", "scalability"],
            difficulty="medium"
        ),
        
        EvaluationCase(
            id="batch_processing",
            name="Batch Document Processing",
            description="Test processing performance on multiple documents",
            evaluation_type=EvaluationType.PERFORMANCE,
            project_id="performance_test_project",
            document_paths=[
                "gs://test-bucket/batch_docs/doc1.pdf",
                "gs://test-bucket/batch_docs/doc2.pdf",
                "gs://test-bucket/batch_docs/doc3.pdf",
                "gs://test-bucket/batch_docs/doc4.pdf",
                "gs://test-bucket/batch_docs/doc5.pdf"
            ],
            expected_outputs=[
                ExpectedOutput(
                    type="batch_success",
                    content=["all_documents_processed", "consistent_quality"],
                    weight=1.0,
                    description="All documents processed successfully with consistent quality"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.PROCESSING_TIME, MetricType.THROUGHPUT, MetricType.COMPLETENESS],
                thresholds={MetricType.PROCESSING_TIME: 120.0, MetricType.THROUGHPUT: 0.5, MetricType.COMPLETENESS: 0.9},
                weights={MetricType.PROCESSING_TIME: 0.3, MetricType.THROUGHPUT: 0.4, MetricType.COMPLETENESS: 0.3}
            ),
            tags=["performance", "batch-processing", "throughput"],
            difficulty="hard"
        )
    ]
    
    return EvaluationSuite(
        id="performance_tests",
        name="Performance Evaluation",
        description="Test suite focused on performance and scalability metrics",
        evaluation_cases=evaluation_cases,
        default_rag_approaches=["raganything", "evidence_sweep", "rag_vertex"],
        tags=["performance", "scalability", "benchmarking"]
    )


def create_comprehensive_evaluation_suite() -> EvaluationSuite:
    """Create a comprehensive test suite covering all evaluation types"""
    
    evaluation_cases = [
        EvaluationCase(
            id="comprehensive_document_processing",
            name="Comprehensive Document Processing",
            description="Test comprehensive document processing across multiple document types",
            evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
            project_id="comprehensive_test_project",
            document_paths=[
                "gs://test-bucket/mixed_docs/contract.pdf",
                "gs://test-bucket/mixed_docs/report.docx",
                "gs://test-bucket/mixed_docs/presentation.pptx",
                "gs://test-bucket/mixed_docs/spreadsheet.xlsx"
            ],
            expected_outputs=[
                ExpectedOutput(
                    type="content_extraction",
                    content=["text_content", "tables", "images", "structure"],
                    weight=1.0,
                    description="Comprehensive content extraction from various document types"
                ),
                ExpectedOutput(
                    type="metadata_extraction",
                    content=["document_type", "creation_date", "author", "keywords"],
                    weight=0.8,
                    description="Document metadata extraction"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.PROCESSING_TIME],
                thresholds={MetricType.COMPLETENESS: 0.8, MetricType.SEMANTIC_SIMILARITY: 0.7, MetricType.PROCESSING_TIME: 90.0},
                weights={MetricType.COMPLETENESS: 0.5, MetricType.SEMANTIC_SIMILARITY: 0.3, MetricType.PROCESSING_TIME: 0.2}
            ),
            tags=["comprehensive", "multi-format", "document-processing"],
            difficulty="hard"
        ),
        
        EvaluationCase(
            id="end_to_end_workflow",
            name="End-to-End Workflow",
            description="Test complete end-to-end workflow from document ingestion to knowledge extraction",
            evaluation_type=EvaluationType.END_TO_END,
            project_id="comprehensive_test_project",
            document_paths=["gs://test-bucket/workflow_docs/complete_case_study.pdf"],
            query="Provide a comprehensive analysis of the case study including key findings and recommendations",
            expected_outputs=[
                ExpectedOutput(
                    type="comprehensive_analysis",
                    content=["key_findings", "recommendations", "supporting_evidence", "conclusions"],
                    weight=1.0,
                    description="Comprehensive analysis with supporting evidence"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.RELEVANCE, MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.COHERENCE],
                thresholds={MetricType.RELEVANCE: 0.8, MetricType.COMPLETENESS: 0.8, MetricType.SEMANTIC_SIMILARITY: 0.7, MetricType.COHERENCE: 0.7},
                weights={MetricType.RELEVANCE: 0.3, MetricType.COMPLETENESS: 0.3, MetricType.SEMANTIC_SIMILARITY: 0.2, MetricType.COHERENCE: 0.2}
            ),
            tags=["end-to-end", "workflow", "comprehensive"],
            difficulty="hard"
        )
    ]
    
    return EvaluationSuite(
        id="comprehensive_tests",
        name="Comprehensive Evaluation",
        description="Comprehensive test suite covering all aspects of RAG evaluation",
        evaluation_cases=evaluation_cases,
        default_rag_approaches=["raganything", "evidence_sweep", "rag_vertex"],
        tags=["comprehensive", "end-to-end", "multi-domain"]
    )


def get_all_sample_evaluation_suites() -> Dict[str, EvaluationSuite]:
    """Get all sample evaluation suites"""
    
    return {
        "legal_documents": create_legal_document_evaluation_suite(),
        "medical_documents": create_medical_document_evaluation_suite(),
        "performance_tests": create_performance_evaluation_suite(),
        "comprehensive_tests": create_comprehensive_evaluation_suite()
    }
