# law-unleashed-rag: RAG Testing Platform Service

## 1. System Overview

`law-unleashed-rag` is a comprehensive, FastAPI-based testing platform service designed for testing and comparing multiple Retrieval-Augmented Generation (RAG) solutions. The system provides a unified API interface that allows users to process documents stored on Google Cloud Storage (GCS) using different RAG implementations (such as `RAGAnything`, `Vertex AI RAG`, `EvidenceSweep`) and track the processing in real-time through Firestore.

The core value of this platform lies in its built-in comprehensive evaluation framework, which can objectively compare different RAG methods across multiple dimensions such as accuracy, performance, and cost, helping developers select the optimal technology stack for specific scenarios (such as legal and medical document analysis).

## 2. Workflow

The system's workflow is designed around API calls and background asynchronous processing. Users submit a document processing task through the API, the system completes processing in the background, and allows users to query task status at any time.

```
Algorithm: RAG Test Bed Service Pipeline
Input: GCS document path, User ID, chosen RAG approach, parser, etc.
Output: A job ID for tracking, and processed knowledge in the backend.

// 1. User initiates a processing job via API
1: POST /process-document with {gcs_path, rag_approach, parser, ...}
2:
3: // 2. FastAPI Backend handles the request
4: Authenticate user and workspace
5: Create a job ID and record the job in Firestore with "pending" status
6: Return job_id to the user immediately
7.
8: // 3. Background task starts processing
9: Download the document from GCS
10: Select the RAG implementation using the RAG Factory (e.g., RAGAnything, Vertex AI)
11: Use the selected RAG approach to process the document (parse, chunk, embed, index)
12: Update the job status and progress in Firestore throughout the process
13:
14: // 4. User checks status
15: GET /processing-status/{job_id} to monitor progress
```

Key components in the pipeline:
- **FastAPI Endpoints**: Provides interfaces for asynchronous task creation (`/process-document`, `/process-folder`) and status querying (`/processing-status`).
- **RAG Factory**: A core component that dynamically selects and initializes the corresponding RAG implementation (such as `RAGAnythingService`, `VertexAIRAGService`) based on user request parameters (`rag_approach`).
- **GCS Manager**: Responsible for interacting with Google Cloud Storage to securely download source files.
- **Firestore Service**: Responsible for creating, updating, and querying the status and progress of processing tasks in the Firestore database.
- **RAG Implementations**: Specific implementations of various RAG solutions, each encapsulating its own document processing, indexing, and storage logic.

## 3. Application Scenarios

### RAG Method Comparison and Evaluation
- **Input**: A set of standardized test documents (such as legal contracts, medical records) and a series of preset evaluation metrics.
- **Output**: Detailed comparison reports of different RAG methods (`RAGAnything`, `Vertex AI RAG`, `EvidenceSweep`) in terms of accuracy, performance, cost, etc.
- **Purpose**: To select the most suitable RAG technology stack for specific business scenarios, providing data-driven decision support.

### Scalable Asynchronous Document Processing Service
- **Input**: Massive documents stored on GCS.
- **Output**: A structured knowledge base accessible via API calls, and task IDs for monitoring processing progress.
- **Purpose**: To provide an enterprise-level, monitorable, asynchronous document processing pipeline that decouples task submission from actual processing, suitable for large-scale document knowledge scenarios.

