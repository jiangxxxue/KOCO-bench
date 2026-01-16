# RAG Evaluation Framework

This directory contains the evaluation framework for comparing different RAG approaches.

## Requirements

Install the evaluation framework dependencies:

```bash
# Install evaluation framework dependencies
pip install -r evals/requirements.txt

# Or install individually
pip install httpx sentence-transformers scikit-learn openai pydantic psutil
```

### Dependencies

- **httpx**: HTTP client for API calls to the RAG service
- **sentence-transformers**: For semantic similarity calculations
- **scikit-learn**: For machine learning metrics and similarity calculations
- **openai**: For LLM-based evaluation metrics
- **pydantic**: For data validation and serialization
- **psutil**: For system resource monitoring during evaluations

## Directory Structure

```
evals/
├── __init__.py                 # Main evals module
├── README.md                   # This file
├── eval_models.py             # Evaluation data models
├── evaluation_service.py      # Evaluation execution service
├── evals_cli.py               # Command-line interface
├── results_manager.py         # Results storage and organization
├── suites/                    # Test suite definitions
│   ├── __init__.py
│   ├── sample_test_suites.py  # Sample test suites
│   └── real_world_test_suites.py  # Real-world test suites
├── runs/                      # Evaluation run configurations
└── results/                   # Evaluation results storage
    ├── YYYY-MM-DD/           # Results organized by date
    │   ├── evaluation_run_id/ # Individual evaluation results
    │   └── comparisons/       # Comparison reports
    └── ...
```

## Evaluation Suites

### Sample Evaluation Suites
- **Legal Documents**: Contract analysis, legal research
- **Medical Documents**: Medical records, cross-document analysis
- **Performance Tests**: Large documents, batch processing
- **Comprehensive Tests**: Multi-format, end-to-end workflows

### Real-World Evaluation Suites
- **Chiropractic Records**: Based on your actual chiropractic medical records
- **Law Unleashed Comprehensive**: Comprehensive legal document processing

## Usage

### Command Line Interface

```bash
# List available evaluation suites
python evals/evals_cli.py list-suites

# Get detailed evaluation suite information
python evals/evals_cli.py get-suite chiropractic_records

# Start an evaluation
python evals/evals_cli.py start-eval chiropractic_records \
  --approaches raganything evidence_sweep rag_vertex \
  --user-id 7CtdhckRcxOIjU3Dh7Ao3jvigg13 \
  --project-id 1lUOSTzmKN7GC5cjgiLI \
  --name "Chiropractic Records Comparison"

# Monitor evaluation progress
python evals/evals_cli.py monitor {evaluation_id} --user-id 7CtdhckRcxOIjU3Dh7Ao3jvigg13

# Get evaluation results
python evals/evals_cli.py results {evaluation_id} --user-id 7CtdhckRcxOIjU3Dh7Ao3jvigg13

# List local results
python evals/evals_cli.py list-results

# Compare multiple evaluations
python evals/evals_cli.py compare {eval_id1} {eval_id2} {eval_id3}
```

### Programmatic Usage

```python
from evals import get_all_evaluation_suites
from evals.results_manager import EvaluationResultsManager

# Get all evaluation suites
evaluation_suites = get_all_evaluation_suites()

# Manage results
results_manager = EvaluationResultsManager()
runs = results_manager.list_evaluation_runs()
```

## Results Organization

Evaluation results are automatically organized in the `results/` directory:

```
results/
├── 2024-01-15/                    # Date-based organization
│   ├── eval_12345/               # Individual evaluation run
│   │   ├── evaluation_results.json  # Complete results
│   │   ├── summary.json          # Summary information
│   │   ├── README.md             # Human-readable summary
│   │   └── test_cases/           # Individual test case results
│   │       ├── test_case_1_raganything.json
│   │       ├── test_case_1_evidence_sweep.json
│   │       └── test_case_1_rag_vertex.json
│   └── comparisons/              # Comparison reports
│       └── comparison_143022.md
└── 2024-01-16/
    └── ...
```

## Creating Custom Evaluation Suites

1. **Create a new evaluation suite file** in `suites/`:

```python
from evals.eval_models import EvaluationCase, EvaluationSuite, ExpectedOutput, EvaluationCriteria
from evals.eval_models import EvaluationType, MetricType

def create_my_custom_suite() -> EvaluationSuite:
    evaluation_cases = [
        EvaluationCase(
            id="my_evaluation_case",
            name="My Custom Evaluation",
            description="Evaluate my specific use case",
            evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
            project_id="my_project",
            document_paths=["gs://my-bucket/my-document.pdf"],
            expected_outputs=[
                ExpectedOutput(
                    type="custom_output",
                    content=["expected_item1", "expected_item2"],
                    weight=1.0
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY],
                thresholds={MetricType.COMPLETENESS: 0.8}
            )
        )
    ]
    
    return EvaluationSuite(
        id="my_custom_suite",
        name="My Custom Evaluation Suite",
        description="Custom evaluation suite for my use case",
        evaluation_cases=evaluation_cases
    )
```

2. **Add to the main module** by updating `__init__.py`:

```python
from .suites.my_custom_suite import create_my_custom_suite

def get_all_evaluation_suites():
    # ... existing suites ...
    custom_suites = {"my_custom_suite": create_my_custom_suite()}
    return {**existing_suites, **custom_suites}
```

## Evaluation Metrics

The framework calculates multiple metrics:

- **Completeness**: How much of expected content was found
- **Semantic Similarity**: How similar actual outputs are to expected
- **Relevance**: How relevant outputs are to the query/task
- **Processing Time**: Speed of document processing
- **Memory Usage**: Resource consumption
- **Overall Score**: Weighted combination of all metrics

## Best Practices

1. **Start with existing evaluation suites** to understand the framework
2. **Use real documents** from your domain for meaningful results
3. **Define clear expected outputs** with specific, measurable criteria
4. **Set appropriate thresholds** based on your requirements
5. **Save results locally** for future comparison and analysis
6. **Document your evaluation cases** with clear descriptions and rationale

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running from the project root
2. **Service not running**: Start the RAG service with `./start_rag_service.sh`
3. **Authentication errors**: Check your Firebase and GCS credentials
4. **Document not found**: Verify document paths exist in your GCS bucket

### Getting Help

- Check the main project README for setup instructions
- Review the API documentation at `http://localhost:8000/docs`
- Look at example evaluation suites for reference implementations
- Check the evaluation guide in the project root
