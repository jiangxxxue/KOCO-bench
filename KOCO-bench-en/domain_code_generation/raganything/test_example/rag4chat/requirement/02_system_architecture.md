# rag4chat: System Architecture and Module Design

## 1. rag4chat Module File Structure

### 1.1 File Organization

The project source code is mainly located in the `code/` directory, with functional divisions around core directories such as `services`, `utils`, and `models`.

```
code/
├── app.py                 # Streamlit Web UI application main entry
├── app_api.py             # Flask backend API service main entry
├── services/              # Core business services
│   ├── build_database.py  # Core engine for knowledge base construction and management
│   ├── vector_store.py    # Upper-level encapsulation of vector storage service
│   ├── get_top_from_rag.py# Low-level implementation for executing retrieval from vector database
│   └── weather_tools.py   # Weather query tool implementation
├── utils/                 # General utility modules
│   ├── document_processor.py # Document parsing and chunking tools
│   ├── chat_history.py    # Session history manager
│   └── ui_components.py   # Streamlit UI component encapsulation
├── models/                # Agent and model definitions
│   └── agent.py           # Defines RAGAgent for interacting with LLM
└── config/                # Configuration files
    └── settings.py        # Stores global configurations such as model names, API Keys
```

### 1.2 Module Responsibility Division

- **app.py / app_api.py**: The two main entry points of the system. `app.py` is responsible for providing a complete, interactive Streamlit graphical interface; `app_api.py` provides a headless Flask API service for third-party system integration.
- **services/build_database.py**: Defines `RAGDatabaseManager`, the data processing core of the system. It calls the `RAGAnything` library to complete document parsing, vectorization, and index construction.
- **services/vector_store.py**: Defines `VectorStoreService`, acting as a "middleman" between the application layer and data layer. It provides a simple interface for upper-level applications (such as `search_documents`) and delegates specific database operations to `RAGDatabaseManager` and `get_top_from_rag`.
- **models/agent.py**: Defines `RAGAgent`, the "agent" for communicating with Large Language Models (LLMs). It is responsible for constructing final prompts based on retrieved context and obtaining answers from LLMs.
- **utils/document_processor.py**: Defines `DocumentProcessor`, responsible for text extraction and chunking of various uploaded documents (PDF, DOCX, etc.) in the early stages of knowledge base construction.

## 2. rag4chat Module Design

### 2.1 Application Layer (app.py & app_api.py)

- **Core Functionality**: As the two gateways of the system, they respectively meet the needs of UI interaction and API integration.
- **`app.py` (Streamlit)**: Orchestrates all backend services, manages UI state (such as "thinking"), renders chat interface, and completes a full RAG process through `VectorStoreService` and `RAGAgent`.
- **`app_api.py` (Flask)**: Follows OpenAI's API format, providing the `/v1/chat/completions` interface. It receives JSON format requests, calls core logic, and can return results in streaming or non-streaming mode.

### 2.2 Knowledge Base Construction Engine (services/build_database.py)

- **Core Class**: `RAGDatabaseManager`
- **Core Functionality**: Encapsulates the complexity of the `RAGAnything` library, serving as the execution engine for knowledge base construction.
- **Main Methods**: 
    - `_create_rag_instance`: Responsible for configuring and instantiating the `RAGAnything` object, including setting up calling functions for LLM (Ollama), embedding model (Alibaba Cloud Bailian), and vision model (Llava).
    - `add_document`: Receives a single file path and calls `RAGAnything`'s `process_document_complete` method to complete the full processing and indexing process for that file.

### 2.3 Retrieval Service (services/vector_store.py)

- **Core Class**: `VectorStoreService`
- **Core Functionality**: Provides a simple, stable data retrieval interface for upper-level applications, isolating the specific underlying implementation (whether Faiss, Chroma, or others).
- **Main Methods**: 
    - `create_vector_store`: Calls `RAGDatabaseManager` to build or rebuild the entire vector database from a batch of documents.
    - `search_documents`: Receives user queries, calls the `query_and_find_topk` function in `get_top_from_rag.py`, executes vector similarity search in the `vdb_chunks.json` file, and returns the most relevant document segments.

### 2.4 LLM Interaction Agent (models/agent.py)

- **Core Class**: `RAGAgent`
- **Core Functionality**: As the last checkpoint for communicating with LLMs, responsible for constructing prompts and parsing LLM responses.
- **Implementation Details**: 
    - It uses the `agno` library to create an Agent and configures it with the Ollama-driven `qwen3:8b` model.
    - It defines tools such as `query_weather`, demonstrating the system's function calling capability.
    - Its `run` method is the core, dynamically constructing different prompt templates based on whether there is `context` (context obtained from retrieval service), ensuring LLMs can fully utilize retrieved information when answering.

## 3. Data Protocol and Module Interaction

### 3.1 API Data Protocol

The request body received by Flask API (`app_api.py`) follows a structure similar to OpenAI, with core fields including:

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "isRAG": true,       // Whether to enable RAG
  "similarity": 0.5,   // Similarity threshold
  "stream": false      // Whether to return in streaming mode
}
```

### 3.2 Knowledge Base Construction Flow

1.  User uploads one or more documents on the `app.py` UI interface.
2.  `UIComponents` passes files to `DocumentProcessor` for preliminary processing.
3.  `VectorStoreService`'s `create_vector_store` method is called.
4.  `VectorStoreService` delegates to `RAGDatabaseManager`'s `add_documents` method.
5.  `RAGDatabaseManager` calls `RAGAnything`'s `process_document_complete` for each document, which completes the entire process of **parsing, chunking, calling embedding model, generating vectors, writing to local JSON file**.

### 3.3 RAG Query Flow

1.  User submits a question through UI or API.
2.  `app.py` or `app_api.py` receives the question and calls `VectorStoreService.search_documents`.
3.  `VectorStoreService` calls `get_top_from_rag.query_and_find_topk`, which reads `vdb_chunks.json`, calculates and returns the most relevant document chunks.
4.  `app.py` formats these document chunks into a `context` string and passes it along with the original question to `RAGAgent.run` method.
5.  `RAGAgent` constructs the final prompt and obtains an answer from Ollama.
6.  The answer is returned to the frontend and displayed.

### 3.4 Module Interaction Diagram

```mermaid
graph TD
    subgraph Entrypoints
        A[UI - app.py] --> C{VectorStoreService};
        B[API - app_api.py] --> G{agent_response};
    end

    subgraph Knowledge Base Pipeline
        C -- create_vector_store --> D[RAGDatabaseManager];
        D -- uses --> E[RAGAnything Library];
        E -- uses --> F[Embedding Model];
    end

    subgraph Query Pipeline
        G -- calls --> C;
        C -- search_documents --> H[query_and_find_topk];
        H -- reads --> I[vdb_chunks.json];
        C -- returns context --> G;
        G -- runs --> J[RAGAgent];
        J -- calls --> K[LLM (Ollama)];
        K -- returns answer --> G;
        G -- returns to --> B;
    end
```

