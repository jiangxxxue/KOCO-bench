"""
Test suites optimized for unstructured content evaluation using LLM-as-a-Judge
"""

from typing import Dict
from ..eval_models import (
    EvaluationCase, EvaluationSuite, ExpectedOutput, EvaluationCriteria,
    EvaluationType, MetricType
)


def create_unstructured_medical_records_suite() -> EvaluationSuite:
    """Create a test suite for unstructured medical records evaluation"""
    
    evaluation_cases = [
        EvaluationCase(
            id="unstructured_patient_summary",
            name="Unstructured Patient Summary",
            description="Test ability to extract and summarize patient information from unstructured medical records",
            evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
            project_id="1lUOSTzmKN7GC5cjgiLI",
            document_paths=[
                "workspaces/RahGZZ7sQXX6De61oAl4/projects/1lUOSTzmKN7GC5cjgiLI/case_files/1754682622870_Nuce, Colby - Records - Thomaston Chiropractic Clinic.pdf"
            ],
            expected_outputs=[
                ExpectedOutput(
                    type="patient_narrative",
                    content="Patient presented with chronic lower back pain following a motor vehicle accident. The pain has been persistent for several weeks and affects daily activities.",
                    weight=1.0,
                    description="A natural language summary describing the patient's condition, cause of injury, and impact on daily life"
                ),
                ExpectedOutput(
                    type="treatment_plan",
                    content="Treatment plan includes chiropractic adjustments, physical therapy exercises, and pain management strategies.",
                    weight=1.0,
                    description="Description of the recommended treatment approach and interventions"
                ),
                ExpectedOutput(
                    type="key_facts",
                    content=[
                        "Patient was involved in a motor vehicle accident",
                        "Experiencing lower back pain since the accident",
                        "Pain affects daily activities and mobility",
                        "Treatment includes chiropractic care and physical therapy"
                    ],
                    weight=0.8,
                    description="Important facts that should be mentioned in any comprehensive summary"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[
                    MetricType.LLM_COMPLETENESS, 
                    MetricType.LLM_ACCURACY, 
                    MetricType.LLM_COHERENCE,
                    MetricType.RELEVANCE,
                    MetricType.PROCESSING_TIME
                ],
                thresholds={
                    MetricType.LLM_COMPLETENESS: 0.8,
                    MetricType.LLM_ACCURACY: 0.8,
                    MetricType.LLM_COHERENCE: 0.7,
                    MetricType.RELEVANCE: 0.8,
                    MetricType.PROCESSING_TIME: 60.0
                },
                weights={
                    MetricType.LLM_COMPLETENESS: 0.3,
                    MetricType.LLM_ACCURACY: 0.3,
                    MetricType.LLM_COHERENCE: 0.2,
                    MetricType.RELEVANCE: 0.1,
                    MetricType.PROCESSING_TIME: 0.1
                }
            ),
            tags=["medical", "unstructured", "narrative", "llm-judge"],
            difficulty="medium"
        ),
        
        EvaluationCase(
            id="unstructured_legal_analysis",
            name="Unstructured Legal Case Analysis",
            description="Test ability to provide unstructured legal analysis from medical records",
            evaluation_type=EvaluationType.QUERY_ANSWERING,
            project_id="1lUOSTzmKN7GC5cjgiLI",
            document_paths=[
                "workspaces/RahGZZ7sQXX6De61oAl4/projects/1lUOSTzmKN7GC5cjgiLI/case_files/1754682622870_Nuce, Colby - Records - Thomaston Chiropractic Clinic.pdf"
            ],
            query="Provide a comprehensive legal analysis of this case, including evidence of injury, causation, and potential damages.",
            expected_outputs=[
                ExpectedOutput(
                    type="legal_analysis",
                    content="This case involves a patient who sustained injuries in a motor vehicle accident, resulting in chronic lower back pain that has significantly impacted their daily activities and quality of life. The medical records provide clear evidence of the injury, its causation by the accident, and the ongoing need for treatment.",
                    weight=1.0,
                    description="A comprehensive legal analysis that addresses injury evidence, causation, and damages in natural language"
                ),
                ExpectedOutput(
                    type="evidence_summary",
                    content="The medical records document objective findings of injury, subjective patient reports of pain and functional limitations, and ongoing treatment requirements that support the claim for damages.",
                    weight=0.9,
                    description="Summary of evidence that supports the legal claim"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[
                    MetricType.LLM_COMPLETENESS, 
                    MetricType.LLM_ACCURACY, 
                    MetricType.LLM_COHERENCE,
                    MetricType.RELEVANCE
                ],
                thresholds={
                    MetricType.LLM_COMPLETENESS: 0.85,
                    MetricType.LLM_ACCURACY: 0.85,
                    MetricType.LLM_COHERENCE: 0.8,
                    MetricType.RELEVANCE: 0.9
                },
                weights={
                    MetricType.LLM_COMPLETENESS: 0.25,
                    MetricType.LLM_ACCURACY: 0.25,
                    MetricType.LLM_COHERENCE: 0.25,
                    MetricType.RELEVANCE: 0.25
                }
            ),
            tags=["legal", "unstructured", "analysis", "llm-judge"],
            difficulty="hard"
        ),
        
        EvaluationCase(
            id="unstructured_insurance_summary",
            name="Unstructured Insurance Claim Summary",
            description="Test ability to create unstructured summaries for insurance purposes",
            evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
            project_id="1lUOSTzmKN7GC5cjgiLI",
            document_paths=[
                "workspaces/RahGZZ7sQXX6De61oAl4/projects/1lUOSTzmKN7GC5cjgiLI/case_files/1754682622870_Nuce, Colby - Records - Thomaston Chiropractic Clinic.pdf"
            ],
            expected_outputs=[
                ExpectedOutput(
                    type="insurance_narrative",
                    content="The patient sustained injuries in a motor vehicle accident that resulted in chronic lower back pain requiring ongoing chiropractic treatment and physical therapy. The medical records demonstrate the necessity and appropriateness of the treatment provided.",
                    weight=1.0,
                    description="A narrative summary suitable for insurance claim documentation"
                ),
                ExpectedOutput(
                    type="treatment_justification",
                    content="The treatment provided is medically necessary and appropriate for the patient's condition, with documented improvement in symptoms and functional capacity.",
                    weight=0.8,
                    description="Justification for the medical necessity of the treatment"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[
                    MetricType.LLM_COMPLETENESS, 
                    MetricType.LLM_ACCURACY, 
                    MetricType.LLM_COHERENCE
                ],
                thresholds={
                    MetricType.LLM_COMPLETENESS: 0.8,
                    MetricType.LLM_ACCURACY: 0.8,
                    MetricType.LLM_COHERENCE: 0.7
                },
                weights={
                    MetricType.LLM_COMPLETENESS: 0.4,
                    MetricType.LLM_ACCURACY: 0.4,
                    MetricType.LLM_COHERENCE: 0.2
                }
            ),
            tags=["insurance", "unstructured", "narrative", "llm-judge"],
            difficulty="medium"
        )
    ]
    
    return EvaluationSuite(
        id="unstructured_medical_records",
        name="Unstructured Medical Records Evaluation",
        description="Test suite for evaluating RAG approaches on unstructured medical record processing using LLM-as-a-Judge",
        evaluation_cases=evaluation_cases,
        default_rag_approaches=["raganything", "evidence_sweep", "rag_vertex"],
        tags=["medical", "unstructured", "narrative", "llm-judge", "legal", "insurance"]
    )


def get_unstructured_evaluation_suites() -> dict:
    """Get all unstructured evaluation suites"""
    
    return {
        "unstructured_medical_records": create_unstructured_medical_records_suite()
    }

