# RAG Testing Platform System Architecture and Module Design

## 1. RAG Platform Module File Structure

### 1.1 File Organization

The core logic of the system is located in the `src/` directory, adopting a clear service-oriented and modular structure.

```
src/
├── main.py              # FastAPI application main entry and API routes
├── models/              # Pydantic data models (API request and response structures)
├── services/            # Core business logic services
│   ├── rag_interface.py # Abstract base class interface for RAG services
│   ├── rag_factory.py   # RAG service factory for creating specific RAG service instances
│   ├── rag_anything_service.py  # Specific implementation of RAGAnything approach
│   ├── rag_vertex_service.py    # Specific implementation of Vertex AI RAG approach
│   ├── evidence_sweep_service.py # Specific implementation of EvidenceSweep approach
│   └── auth_service.py  # User authentication and workspace authorization service
└── utils/               # Utility classes
    ├── firebase_utils.py  # Firebase/Firestore database interaction
    └── gcs_utils.py       # Google Cloud Storage file operations
```

### 1.2 Module Responsibility Division

- **main.py**: The application gateway. Responsible for defining all API endpoints, handling HTTP requests, calling corresponding services through dependency injection, and managing background tasks.
- **services/rag_interface.py**: Defines the "contract" (interface) that all RAG implementations must follow, ensuring different RAG approaches can be called uniformly.
- **services/rag_factory.py**: Implements the factory design pattern. Dynamically creates and returns a specific RAG service instance based on the `rag_approach` parameter specified in the API request.
- **services/*_service.py**: Specific implementations of each RAG approach. They encapsulate their respective data processing, index building, and query logic.
- **services/auth_service.py**: Responsible for user authentication and permission checking, ensuring users can only access workspaces and projects they have permission to access.
- **utils/**: Provides low-level encapsulation for interacting with external services (Firebase and GCS), making core business logic clearer.

## 2. RAG Platform Module Design

### 2.1 FastAPI Application (main.py)

- **Core Functionality**: Acts as the system's master controller, responsible for API routing, dependency injection, background task management, and monitoring metrics exposure.
- **Main Interfaces**: `/process-document`, `/process-folder`, `/query`, `/processing-status`.
- **Implementation Details**: 
    - Uses `lifespan` context manager to initialize all singleton services (such as `RAGFactory`, `FirebaseManager`) at application startup.
    - Through FastAPI's `Depends` system, injects `RAGFactory` and `AuthService` on-demand when processing requests.
    - For time-consuming document processing tasks, uses `BackgroundTasks` to implement asynchronous execution, immediately returning a task ID to the client, avoiding long polling.

### 2.2 RAG Service Factory (rag_factory.py)

- **Core Class**: `RAGFactory`
- **Core Functionality**: Dynamically creates and returns a specific RAG service instance based on the incoming `approach` string (such as `"raganything"`).
- **Design Pattern**: Factory Pattern.
- **Implementation Details**: The factory internally maintains an `_implementations` dictionary, which dynamically scans and registers all available RAG services during initialization. This design greatly improves system extensibility; when supporting a new RAG method is needed, only a new service implementation file needs to be added without modifying any existing core code.

### 2.3 RAG Service Interface (rag_interface.py)

- **Core Class**: `RAGInterface` (abstract base class)
- **Core Functionality**: Defines unified methods that all RAG implementations must follow, such as `process_document`, `process_folder`, `query_documents`.
- **Design Pattern**: Core of the Strategy Pattern, separating algorithm implementation from usage.
- **Purpose**: Ensures that API routing logic in `main.py` can uniformly handle any RAG approach without concerning itself with internal specific implementations.

### 2.4 Specific RAG Services (rag_anything_service.py, etc.)

- **Core Classes**: `RAGAnythingService`, `RAGVertexService`, `EvidenceSweepService`.
- **Core Functionality**: Encapsulates the complete logic of specific RAG technology stacks, including initialization, document processing, and querying.
- **Implementation Details**: Each service class inherits from `RAGInterface` and implements all its abstract methods. For example:
    - `RAGAnythingService` is responsible for calling the `RAG-Anything` library to create and manage indexes on the local file system.
    - `RAGVertexService` is responsible for calling Google Cloud APIs to create and manage RAG Corpus in the cloud.
    - `EvidenceSweepService` executes a custom multi-step LLM call chain (page filtering -> evidence extraction -> comprehensive analysis).

## 3. Data Protocol and Module Interaction

### 3.1 API Data Protocol (models/api_models.py)

The system uses Pydantic models to define clear, strongly-typed API data structures. For example, the request body for processing a single document is defined by the `ProcessDocumentRequest` model:

```python
class ProcessDocumentRequest(BaseModel):
    user_id: str
    project_id: str
    workspace_id: str
    gcs_path: str
    rag_approach: str
    parser: Optional[str] = "mineru"
    model: Optional[str] = "gpt-4o-mini"
    config: Optional[Dict[str, Any]] = None
```

### 3.2 Asynchronous Processing Flow

1.  Client sends a POST request to `/process-document`.
2.  `main.py` receives the request and immediately calls `AuthService` to verify user permissions.
3.  After verification passes, `FirebaseManager` creates a task record with `pending` status in the database.
4.  `main.py` immediately returns the task ID to the client (HTTP 200 OK).
5.  A `BackgroundTask` is triggered in the background, which obtains the correct RAG service instance through `RAGFactory` and calls its `process_document` method.
6.  During execution, the RAG service continuously updates task progress and final results in the database through `FirebaseManager`.
7.  The client can use the task ID to poll the `/processing-status/{job_id}` endpoint to get real-time progress.

### 3.3 Data Flow

- **Input Data**: File paths (strings) on Google Cloud Storage (GCS).
- **Intermediate Data**: Services download files from GCS to local temporary directories for processing.
- **Processing Results**: Each RAG approach stores its indexes and metadata in their respective backends (`RAGAnything` uses local file system, `Vertex AI` uses Google Cloud services).
- **Status Data**: All task metadata, status, and progress are stored as JSON objects in Firestore.
- **Output Data**: API returns task IDs, status, or query results in JSON format.

### 3.4 Module Interaction Diagram

```mermaid
graph TD
    A[User via API] --> B[FastAPI App (main.py)];
    B -- validates with --> C[AuthService];
    B -- creates job & returns job_id --> D[FirebaseManager];
    subgraph Background Task
        B -- triggers --> E{RAG Factory};
        E -- creates --> F[RAG Service (e.g., RAGAnythingService)];
        F -- downloads from --> G[GCS Manager];
        F -- processes & indexes --> H[RAG Backend Storage];
        F -- updates status --> D;
    end
    A -- polls status --> B;
    B -- gets status --> D;
```

