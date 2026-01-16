# RAG Test Bed Service

A FastAPI-based service for testing and comparing multiple RAG (Retrieval-Augmented Generation) approaches. This service provides a unified interface for processing documents using different RAG implementations with progress tracking via Firestore.

## Features

- **Multiple RAG Approaches**: 
  - **RAGAnything**: Original implementation using RAG-Anything framework
  - **Vertex AI RAG**: Google Cloud's native RAG engine with Gemini models
  - **EvidenceSweep**: Custom approach with page selection, evidence pass, and synthesis
- **Document Processing**: Process individual documents from Google Cloud Storage
- **Folder Processing**: Process all documents in a GCS folder
- **Multiple Parsers**: Support for MinerU, Docling, LlamaParse, and simple parsers
- **Progress Tracking**: Real-time progress updates via Firestore
- **Authentication**: User and workspace access control
- **Flexible Storage**: Approach-specific storage backends
- **Docker Support**: Containerized deployment ready
- **Monitoring**: Prometheus metrics and health checks

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  RAG Factory    │    │   Firestore     │
│                 │    │                 │    │                 │
│ • /process-doc  │───▶│ • RAGAnything   │───▶│ • Progress      │
│ • /process-folder│   │ • Vertex AI RAG │    │   Tracking      │
│ • /rag-approaches│   │ • EvidenceSweep │    │ • Job Status    │
│ • /status       │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GCS Manager   │    │  Auth Service   │    │   Prometheus    │
│                 │    │                 │    │                 │
│ • File Download │    │ • User Access   │    │ • Metrics       │
│ • Folder List   │    │ • Workspace     │    │ • Health Checks │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud Storage bucket
- Firebase project with Firestore
- OpenAI API key
- LibreOffice (for Office documents)
- Tesseract (for OCR)
- Llama Cloud API key (optional, for LlamaParse)
- Google Cloud project with Vertex AI enabled (for Vertex AI RAG)

### Local Development

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd law-unleashed-rag
   cp env.example .env
   ```

2. **Configure environment**:
   Edit `.env` with your configuration:
   ```bash
   FIREBASE_PROJECT_ID=your-firebase-project-id
   GCS_BUCKET_NAME=your-gcs-bucket-name
   OPENAI_API_KEY=your-openai-api-key
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   
   # Optional: Vertex AI configuration
   GOOGLE_PROJECT_ID=your-google-project-id
   GOOGLE_REGION=us-central1
   GOOGLE_VECTOR_STORE_ID=your-vector-store-id
   ```

3. **Start the service**:
   ```bash
   ./start_rag_service.sh
   ```

4. **Access the API**:
   - Service: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - RAG Approaches: http://localhost:8000/rag-approaches

### Docker Deployment

1. **Build the image**:
   ```bash
   docker build -t rag-anything-service .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     --name rag-service \
     -p 8000:8000 \
     -e FIREBASE_PROJECT_ID=your-project-id \
     -e GCS_BUCKET_NAME=your-bucket \
     -e OPENAI_API_KEY=your-api-key \
     -v /path/to/service-account.json:/app/credentials.json \
     -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
     rag-anything-service
   ```

## RAG Approaches

### RAGAnything
- **Description**: Original implementation using the RAG-Anything framework
- **Strengths**: Comprehensive document processing, graph-based knowledge representation
- **Parsers**: MinerU, Docling
- **Storage**: Local JSON, NanoVector, Faiss, NetworkX
- **Best for**: Complex documents with tables, images, and structured content

### Vertex AI RAG
- **Description**: Google Cloud's native RAG engine using Gemini models and Vertex AI
- **Strengths**: Native GCP integration, advanced LLM parsing, cloud-native vector storage
- **Parsers**: LLM Parser (Gemini), Auto
- **Models**: Gemini-2.0-flash-001, Gemini-1.5-pro, Gemini-1.5-flash, Gemini-1.0-pro
- **Best for**: GCP-native deployments, advanced document understanding

### EvidenceSweep
- **Description**: Custom approach with page selection, evidence pass, and synthesis
- **Strengths**: Focused analysis, evidence-based reasoning, cross-document synthesis
- **Parsers**: LlamaParse, Simple
- **Storage**: Evidence and synthesis storage
- **Best for**: Research and analysis tasks requiring deep document understanding

## API Endpoints

### Get Available RAG Approaches

Get information about all available RAG approaches:

```bash
curl "http://localhost:8000/rag-approaches"
```

**Response**:
```json
{
  "approaches": {
    "raganything": {
      "name": "RAGAnything",
      "supported_parsers": ["mineru", "docling"],
      "supported_models": ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
      "description": "RAG approach using RAGAnything"
    },
    "rag_vertex": {
      "name": "Vertex AI RAG",
      "supported_parsers": ["llm_parser", "auto"],
      "supported_models": ["gemini-2.0-flash-001", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
      "description": "RAG approach using Vertex AI"
    },
    "evidence_sweep": {
      "name": "EvidenceSweep",
      "supported_parsers": ["simple", "llamaparse"],
      "supported_models": ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
      "description": "RAG approach using EvidenceSweep"
    }
  },
  "default_approach": "raganything"
}
```

### Process Document

Process a single document from GCS with a specific RAG approach:

```bash
curl -X POST "http://localhost:8000/process-document" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "project_id": "project456",
    "workspace_id": "workspace789",
    "gcs_path": "gs://bucket/path/to/document.pdf",
    "rag_approach": "rag_vertex",
    "parser": "llm_parser",
    "parse_method": "auto",
    "model": "gpt-4o-mini"
  }'
```

**Response**:
```json
{
  "job_id": "uuid-here",
  "status": "pending",
  "message": "Document processing job created successfully with rag_vertex",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Process Folder

Process all documents in a GCS folder with a specific RAG approach:

```bash
curl -X POST "http://localhost:8000/process-folder" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "project_id": "project456",
    "workspace_id": "workspace789",
    "gcs_folder_path": "gs://bucket/path/to/folder/",
    "rag_approach": "evidence_sweep",
    "file_extensions": [".pdf", ".doc", ".docx", ".txt"],
    "parser": "llamaparse",
    "parse_method": "auto",
    "model": "gpt-4o",
    "config": {
      "query": "What are the key findings in these documents?"
    }
  }'
```

### Check Processing Status

Get the status of a processing job:

```bash
curl "http://localhost:8000/processing-status/{job_id}?user_id=user123"
```

**Response**:
```json
{
  "job_id": "uuid-here",
  "status": "processing",
  "message": "Processing document...",
  "progress": {
    "totalDocuments": 1,
    "processedDocuments": 0,
    "failedDocuments": 0,
    "overallProgress": 0,
    "currentDocument": "gs://bucket/document.pdf"
  },
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:01:00Z"
}
```

### Get User Jobs

Get all processing jobs for a user:

```bash
curl "http://localhost:8000/user-jobs/user123?limit=50"
```

### Get Project Jobs

Get all processing jobs for a project:

```bash
curl "http://localhost:8000/project-jobs/project456?user_id=user123"
```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `FIREBASE_PROJECT_ID` | Firebase project ID | Yes | - |
| `GCS_BUCKET_NAME` | GCS bucket name | Yes | - |
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `HOST` | Service host | No | `0.0.0.0` |
| `PORT` | Service port | No | `8000` |
| `LOG_LEVEL` | Logging level | No | `info` |
| `RAG_STORAGE_DIR` | RAGAnything storage directory | No | `/app/storage` |
| `VERTEX_AI_REGION` | Vertex AI region | No | `us-central1` |
| `EVIDENCE_SWEEP_STORAGE_DIR` | EvidenceSweep storage directory | No | `/app/storage/evidence_sweep` |
| `RAG_TEMP_DIR` | Temp directory | No | `/app/temp` |
| `PARSER` | Default parser | No | `mineru` |
| `PARSE_METHOD` | Default parse method | No | `auto` |
| `LLAMA_CLOUD_API_KEY` | Llama Cloud API key (for LlamaParse) | No | - |
| `GOOGLE_PROJECT_ID` | Google Cloud project ID (for Vector Store) | No | - |
| `GOOGLE_VECTOR_STORE_ID` | Google Vector Store ID | No | - |

### Storage Configuration

The service uses approach-specific storage backends:

#### RAGAnything
- **JsonKVStorage**: Key-value storage in local JSON files
- **NanoVectorDBStorage**: Vector database for embeddings
- **FaissVectorDBStorage**: Alternative vector storage (GPU optional)
- **NetworkXStorage**: Graph storage for relationships
- **JsonDocStatusStorage**: Document processing status

#### Vertex AI RAG
- **Vertex AI RAG Store**: Cloud-native RAG corpus and vector storage
- **Gemini Models**: Native integration with Google's Gemini models
- **LlamaParse**: Advanced document parsing service

#### EvidenceSweep
- **Evidence Storage**: Structured evidence extraction results
- **Synthesis Storage**: Cross-document synthesis results
- **Page Metadata**: Document page analysis results

### Parser Options

#### MinerU Parser
- Supports PDF, images, Office documents
- Powerful OCR and table extraction
- GPU acceleration support
- Configuration options:
  - `lang`: Document language (e.g., "ch", "en", "ja")
  - `device`: Inference device ("cpu", "cuda", "cuda:0")
  - `formula`: Enable formula parsing
  - `table`: Enable table parsing

#### Docling Parser
- Optimized for Office documents and HTML
- Better document structure preservation
- Native multi-format support

#### LlamaParse Parser
- Advanced document parsing with AI
- Better table and image extraction
- Structured output in markdown format
- Requires Llama Cloud API key

#### Simple Parser
- Basic text extraction
- Fallback option for unsupported formats
- Fast processing for simple documents

## Monitoring

### Health Checks

- **Basic**: `GET /health`
- **Detailed**: `GET /health/detailed` (includes service status)

### Metrics

Prometheus metrics available at `GET /metrics`:

- `rag_anything_http_requests_total`: HTTP request counter
- `rag_anything_http_request_duration_seconds`: Request duration histogram
- `rag_anything_document_processing_duration_seconds`: Processing duration

### Logging

Structured logging with configurable levels:
- Request/response logging
- Processing progress updates
- Error tracking with context

## Development

### Project Structure

```
law-unleashed-rag/
├── src/
│   ├── models/           # Pydantic models
│   ├── services/         # Business logic
│   ├── utils/           # Utilities
│   └── main.py          # FastAPI app
├── evals/               # Evaluation framework
│   ├── eval_models.py  # Evaluation data models
│   ├── evaluation_service.py # Evaluation execution service
│   ├── suites/         # Test suite definitions
│   ├── runs/          # Evaluation run configurations
│   ├── results/       # Evaluation results storage
│   ├── evals_cli.py   # Evaluation CLI
│   └── results_manager.py # Results organization
├── requirements.txt     # Dependencies
├── Dockerfile          # Container config
├── env.example         # Environment template
├── start_rag_service.sh # Startup script
└── README.md           # This file
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run tests
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

1. **Firebase Authentication Error**:
   - Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to valid service account key
   - Verify Firebase project ID is correct

2. **GCS Access Denied**:
   - Check service account has Storage Object Viewer permissions
   - Verify bucket name and file paths

3. **RAG-Anything Processing Fails**:
   - Ensure LibreOffice is installed for Office documents
   - Check Tesseract installation for OCR
   - Verify OpenAI API key is valid

4. **Memory Issues**:
   - Reduce batch size for large documents
   - Use CPU-only processing if GPU memory is limited
   - Monitor temp directory usage

### Logs

Check service logs for detailed error information:

```bash
# Docker logs
docker logs rag-service

# Local development
tail -f server.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Evaluation Framework

The RAG Test Bed includes a comprehensive evaluation framework for comparing different RAG approaches objectively.

### Overview

The evaluation framework allows you to:
- **Define Test Cases**: Create standardized test cases with expected outputs
- **Run Comparative Evaluations**: Test multiple RAG approaches on the same test cases
- **Calculate Metrics**: Measure accuracy, completeness, relevance, and performance
- **Generate Reports**: Get detailed comparison reports with recommendations

### Test Case Structure

Each test case includes:
- **Input Documents**: GCS paths to documents to process
- **Expected Outputs**: What the RAG approach should extract or generate
- **Evaluation Criteria**: Metrics to calculate and thresholds to meet
- **Configuration**: Parser, model, and other settings

### Evaluation Types

1. **Document Processing**: Test document parsing and knowledge extraction
2. **Query Answering**: Test retrieval and response quality
3. **Cross-Document Analysis**: Test multi-document reasoning
4. **Performance**: Test processing speed and resource usage
5. **End-to-End**: Test complete workflows

### Metrics

The framework calculates multiple metrics:
- **Accuracy Metrics**: Precision, recall, F1-score
- **Quality Metrics**: Semantic similarity, coherence, completeness
- **Performance Metrics**: Processing time, memory usage, throughput
- **User Experience Metrics**: Response relevance, clarity, actionability

### Sample Test Suites

The framework includes pre-built test suites:

#### Legal Documents
- Contract analysis and entity extraction
- Legal research query answering
- Case law analysis

#### Medical Documents
- Medical record analysis
- Cross-document patient understanding
- Clinical decision support

#### Performance Tests
- Large document processing
- Batch processing throughput
- Resource utilization

#### Comprehensive Tests
- Multi-format document processing
- End-to-end workflow evaluation

### Using the Evaluation Framework

#### 1. List Available Test Suites

```bash
curl "http://localhost:8000/test-suites"
```

#### 2. Get Test Suite Details

```bash
curl "http://localhost:8000/test-suites/legal_documents"
```

#### 3. Start an Evaluation

```bash
curl -X POST "http://localhost:8000/evaluations" \
  -H "Content-Type: application/json" \
  -d '{
    "test_suite_id": "legal_documents",
    "rag_approaches": ["raganything", "evidence_sweep", "rag_vertex"],
    "user_id": "user123",
    "project_id": "project456",
    "name": "Legal Document Comparison",
    "description": "Comparing RAG approaches on legal documents"
  }'
```

#### 4. Monitor Evaluation Progress

```bash
curl "http://localhost:8000/evaluations/{evaluation_run_id}/status?user_id=user123"
```

#### 5. Get Results

```bash
curl "http://localhost:8000/evaluations/{evaluation_run_id}/results?user_id=user123"
```

### CLI Tool

Use the command-line interface for easier evaluation management:

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
python evals/evals_cli.py monitor {evaluation_run_id} --user-id 7CtdhckRcxOIjU3Dh7Ao3jvigg13

# Get evaluation results
python evals/evals_cli.py results {evaluation_run_id} --user-id 7CtdhckRcxOIjU3Dh7Ao3jvigg13

# List local results
python evals/evals_cli.py list-results

# Compare multiple evaluations
python evals/evals_cli.py compare {eval_id1} {eval_id2}
```

### Creating Custom Evaluation Suites

You can create custom evaluation suites by defining evaluation cases:

```python
from evals.eval_models import EvaluationCase, EvaluationSuite, ExpectedOutput, EvaluationCriteria
from evals.eval_models import EvaluationType, MetricType

# Create a test case
test_case = TestCase(
    id="custom_test",
    name="Custom Document Test",
    description="Test custom document processing",
    evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
    project_id="your_project",
    document_paths=["gs://your-bucket/test-doc.pdf"],
    expected_outputs=[
        ExpectedOutput(
            type="entities",
            content=["key_terms", "important_concepts"],
            weight=1.0
        )
    ],
    evaluation_criteria=EvaluationCriteria(
        metrics=[MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY],
        thresholds={MetricType.COMPLETENESS: 0.8, MetricType.SEMANTIC_SIMILARITY: 0.7}
    )
)

# Create test suite
test_suite = TestSuite(
    id="custom_suite",
    name="Custom Test Suite",
    description="Your custom test suite",
    test_cases=[test_case]
)
```

### Evaluation Results

The framework provides comprehensive results including:

- **Overall Scores**: Weighted average scores for each RAG approach
- **Detailed Metrics**: Individual metric scores for each test case
- **Performance Data**: Processing times and resource usage
- **Comparison Summary**: Side-by-side comparison of approaches
- **Recommendations**: Suggestions based on results

### Best Practices

1. **Start Small**: Begin with simple test cases and gradually add complexity
2. **Use Real Data**: Test with actual documents from your domain
3. **Set Clear Expectations**: Define specific, measurable expected outputs
4. **Iterate**: Refine test cases based on evaluation results
5. **Document**: Keep detailed records of test case rationale and results

## Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `/docs`
- Check the evaluation framework documentation
