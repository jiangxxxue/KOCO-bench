"""
Real-world evaluation suites based on actual use cases
"""

from typing import Dict
from ..eval_models import (
    EvaluationCase, EvaluationSuite, ExpectedOutput, EvaluationCriteria,
    EvaluationType, MetricType
)


def create_chiropractic_records_evaluation_suite() -> EvaluationSuite:
    """Create a test suite for chiropractic medical records processing"""
    
    evaluation_cases = [
        EvaluationCase(
            id="chiropractic_records_analysis",
            name="Chiropractic Medical Records Analysis",
            description="Test ability to extract key information from chiropractic medical records",
            evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
            project_id="1lUOSTzmKN7GC5cjgiLI",
            document_paths=[
                "workspaces/RahGZZ7sQXX6De61oAl4/projects/1lUOSTzmKN7GC5cjgiLI/case_files/1754682622870_Nuce, Colby - Records - Thomaston Chiropractic Clinic.pdf"
            ],
            expected_outputs=[
                ExpectedOutput(
                    type="patient_info",
                    content=["patient_name", "date_of_birth", "patient_id", "contact_info"],
                    weight=1.0,
                    description="Basic patient identification information"
                ),
                ExpectedOutput(
                    type="medical_history",
                    content=["chief_complaint", "symptoms", "pain_level", "duration"],
                    weight=1.0,
                    description="Primary medical complaints and symptoms"
                ),
                ExpectedOutput(
                    type="treatment_info",
                    content=["diagnosis", "treatment_plan", "medications", "follow_up"],
                    weight=1.0,
                    description="Diagnosis and treatment information"
                ),
                ExpectedOutput(
                    type="chiropractic_specific",
                    content=["spinal_assessment", "adjustments", "range_of_motion", "posture_analysis"],
                    weight=0.8,
                    description="Chiropractic-specific assessment and treatment details"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.PROCESSING_TIME],
                thresholds={
                    MetricType.COMPLETENESS: 0.8,
                    MetricType.SEMANTIC_SIMILARITY: 0.7,
                    MetricType.PROCESSING_TIME: 60.0
                },
                weights={
                    MetricType.COMPLETENESS: 0.6,
                    MetricType.SEMANTIC_SIMILARITY: 0.3,
                    MetricType.PROCESSING_TIME: 0.1
                }
            ),
            tags=["medical", "chiropractic", "records", "real-world"],
            difficulty="medium"
        ),
        
        EvaluationCase(
            id="chiropractic_query_answering",
            name="Chiropractic Records Query Answering",
            description="Test ability to answer specific questions about chiropractic treatment",
            evaluation_type=EvaluationType.QUERY_ANSWERING,
            project_id="1lUOSTzmKN7GC5cjgiLI",
            document_paths=[
                "workspaces/RahGZZ7sQXX6De61oAl4/projects/1lUOSTzmKN7GC5cjgiLI/case_files/1754682622870_Nuce, Colby - Records - Thomaston Chiropractic Clinic.pdf"
            ],
            query="What was the patient's chief complaint and what treatment was recommended?",
            expected_outputs=[
                ExpectedOutput(
                    type="query_answer",
                    content="Patient's primary complaint and recommended treatment plan",
                    weight=1.0,
                    description="Comprehensive answer about patient's condition and treatment"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.RELEVANCE, MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY],
                thresholds={
                    MetricType.RELEVANCE: 0.8,
                    MetricType.COMPLETENESS: 0.7,
                    MetricType.SEMANTIC_SIMILARITY: 0.7
                },
                weights={
                    MetricType.RELEVANCE: 0.4,
                    MetricType.COMPLETENESS: 0.3,
                    MetricType.SEMANTIC_SIMILARITY: 0.3
                }
            ),
            tags=["medical", "chiropractic", "query-answering", "real-world"],
            difficulty="medium"
        ),
        
        EvaluationCase(
            id="chiropractic_evidence_extraction",
            name="Chiropractic Evidence Extraction",
            description="Test ability to extract evidence for legal/insurance purposes",
            evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
            project_id="1lUOSTzmKN7GC5cjgiLI",
            document_paths=[
                "workspaces/RahGZZ7sQXX6De61oAl4/projects/1lUOSTzmKN7GC5cjgiLI/case_files/1754682622870_Nuce, Colby - Records - Thomaston Chiropractic Clinic.pdf"
            ],
            expected_outputs=[
                ExpectedOutput(
                    type="legal_evidence",
                    content=["treatment_dates", "provider_credentials", "billing_codes", "medical_necessity"],
                    weight=1.0,
                    description="Information relevant for legal or insurance claims"
                ),
                ExpectedOutput(
                    type="clinical_evidence",
                    content=["objective_findings", "subjective_symptoms", "functional_limitations", "prognosis"],
                    weight=1.0,
                    description="Clinical evidence supporting treatment decisions"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.RELEVANCE],
                thresholds={
                    MetricType.COMPLETENESS: 0.85,
                    MetricType.SEMANTIC_SIMILARITY: 0.75,
                    MetricType.RELEVANCE: 0.8
                },
                weights={
                    MetricType.COMPLETENESS: 0.4,
                    MetricType.SEMANTIC_SIMILARITY: 0.3,
                    MetricType.RELEVANCE: 0.3
                }
            ),
            tags=["medical", "chiropractic", "legal", "evidence", "real-world"],
            difficulty="hard"
        )
    ]
    
    return EvaluationSuite(
        id="chiropractic_records",
        name="Chiropractic Medical Records",
        description="Test suite for evaluating RAG approaches on chiropractic medical records",
        evaluation_cases=evaluation_cases,
        default_rag_approaches=["raganything", "evidence_sweep", "rag_vertex"],
        tags=["medical", "chiropractic", "real-world", "legal"]
    )


def create_law_unleashed_evaluation_suite() -> EvaluationSuite:
    """Create a comprehensive test suite for law-unleashed use cases"""
    
    evaluation_cases = [
        EvaluationCase(
            id="medical_records_comprehensive",
            name="Comprehensive Medical Records Analysis",
            description="Test comprehensive analysis of medical records for legal purposes",
            evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
            project_id="1lUOSTzmKN7GC5cjgiLI",
            document_paths=[
                "workspaces/RahGZZ7sQXX6De61oAl4/projects/1lUOSTzmKN7GC5cjgiLI/case_files/1754682622870_Nuce, Colby - Records - Thomaston Chiropractic Clinic.pdf"
            ],
            expected_outputs=[
                ExpectedOutput(
                    type="patient_summary",
                    content=["demographics", "medical_history", "current_condition", "treatment_timeline"],
                    weight=1.0,
                    description="Comprehensive patient summary"
                ),
                ExpectedOutput(
                    type="legal_relevant_info",
                    content=["injuries", "causation", "damages", "treatment_costs", "prognosis"],
                    weight=1.0,
                    description="Information relevant for legal proceedings"
                ),
                ExpectedOutput(
                    type="medical_entities",
                    content=["diagnoses", "procedures", "medications", "providers", "facilities"],
                    weight=0.8,
                    description="Medical entities and terminology"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.RELEVANCE, MetricType.PROCESSING_TIME],
                thresholds={
                    MetricType.COMPLETENESS: 0.8,
                    MetricType.SEMANTIC_SIMILARITY: 0.7,
                    MetricType.RELEVANCE: 0.8,
                    MetricType.PROCESSING_TIME: 90.0
                },
                weights={
                    MetricType.COMPLETENESS: 0.3,
                    MetricType.SEMANTIC_SIMILARITY: 0.2,
                    MetricType.RELEVANCE: 0.3,
                    MetricType.PROCESSING_TIME: 0.2
                }
            ),
            tags=["legal", "medical", "comprehensive", "real-world"],
            difficulty="hard"
        ),
        
        EvaluationCase(
            id="legal_case_analysis",
            name="Legal Case Analysis",
            description="Test ability to analyze medical records for legal case preparation",
            evaluation_type=EvaluationType.QUERY_ANSWERING,
            project_id="1lUOSTzmKN7GC5cjgiLI",
            document_paths=[
                "workspaces/RahGZZ7sQXX6De61oAl4/projects/1lUOSTzmKN7GC5cjgiLI/case_files/1754682622870_Nuce, Colby - Records - Thomaston Chiropractic Clinic.pdf"
            ],
            query="What evidence supports the patient's claim of injury and what are the key facts for the legal case?",
            expected_outputs=[
                ExpectedOutput(
                    type="legal_analysis",
                    content="Comprehensive analysis of injury evidence, causation, damages, and key legal facts",
                    weight=1.0,
                    description="Legal analysis suitable for case preparation"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.RELEVANCE, MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.COHERENCE],
                thresholds={
                    MetricType.RELEVANCE: 0.85,
                    MetricType.COMPLETENESS: 0.8,
                    MetricType.SEMANTIC_SIMILARITY: 0.75,
                    MetricType.COHERENCE: 0.8
                },
                weights={
                    MetricType.RELEVANCE: 0.3,
                    MetricType.COMPLETENESS: 0.3,
                    MetricType.SEMANTIC_SIMILARITY: 0.2,
                    MetricType.COHERENCE: 0.2
                }
            ),
            tags=["legal", "case-analysis", "real-world"],
            difficulty="hard"
        )
    ]
    
    return EvaluationSuite(
        id="law_unleashed_comprehensive",
        name="Law Unleashed Comprehensive",
        description="Comprehensive test suite for law-unleashed legal document processing",
        evaluation_cases=evaluation_cases,
        default_rag_approaches=["raganything", "evidence_sweep", "rag_vertex"],
        tags=["legal", "medical", "comprehensive", "real-world", "law-unleashed"]
    )


def get_real_world_evaluation_suites() -> dict:
    """Get all real-world evaluation suites"""
    
    return {
        "chiropractic_records": create_chiropractic_records_evaluation_suite(),
        "law_unleashed_comprehensive": create_law_unleashed_evaluation_suite()
    }
