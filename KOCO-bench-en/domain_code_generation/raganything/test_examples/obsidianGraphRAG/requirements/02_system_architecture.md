# ObsidianGraphRAG System Architecture and Module Design

## 1. ObsidianGraphRAG Module File Structure

### 1.1 File Organization

The project adopts a clear functional partitioning structure, with core logic, UI, and execution scripts separated for easy maintenance and extension.

```
obsidianGraphRAG/
├── src/                   # Core system logic
│   ├── simple_raganything.py    # Main RAG orchestration class, responsible for initialization, processing, and querying
│   ├── obsidian_chunker.py      # Obsidian-specific note chunker
│   ├── vault_monitor.py         # Vault change monitoring and incremental sync logic
│   ├── gemini_llm.py            # Wrapper for Google Gemini API
│   └── bge_reranker.py          # Wrapper for BGE Reranker model
├── ui/                    # Web user interface
│   ├── app.py                   # FastAPI backend, providing WebSocket chat and API
│   └── static/                  # HTML/CSS/JS frontend static files
├── run_obsidian_raganything.py  # Execution script for initial knowledge base construction
├── run_incremental_sync.py      # Script for executing incremental sync
├── run_ui.py                  # Script for launching Web UI
└── requirements.txt           # Python dependencies
```

### 1.2 Module Responsibilities

- **simple_raganything.py**: The system's "brain", responsible for initializing and orchestrating all core components. It manages the database lifecycle (create or load), calls the chunker to process documents, and executes query logic.
- **obsidian_chunker.py**: The "preprocessor" optimized for Obsidian. It not only chunks notes into pieces suitable for model processing, but more importantly, it injects rich metadata (such as bidirectional links, tags, file relationships) into each chunk, maximizing the preservation of knowledge context.
- **vault_monitor.py**: The core of the "incremental update" functionality. It efficiently detects additions, modifications, and deletions in the Vault by comparing file hashes, and calls the RAG engine to perform corresponding updates, avoiding full index rebuilds each time.
- **gemini_llm.py / bge_reranker.py**: Model "connectors". They wrap the external Gemini API and local BGE Reranker model into standard interfaces that comply with the `LightRAG` framework requirements, enabling plug-and-play model integration.
- **ui/app.py**: The "server side" of the user interface. Built with FastAPI, it loads the RAG system on application startup, provides real-time streaming chat service to the frontend via WebSocket, and offers system status queries and manual sync management functions through REST API.

## 2. ObsidianGraphRAG Module Design

### 2.1 RAG Orchestrator (SimpleRAGAnything)

- **Core Class**: `SimpleRAGAnything`
- **Core Functions**: 
    1.  **Initialization (`initialize`)**: Checks for existing knowledge base and decides whether to load the old database or create a new one based on user choice (or automatically in web mode). It is responsible for loading the embedding model, configuring the RAG engine, and the reranker.
    2.  **Full Processing (`process_vault`)**: Calls `ObsidianChunker` to process the entire Vault, then batch-inserts the chunked, metadata-rich text blocks into the `RAGAnything` engine for indexing.
    3.  **Query (`query`)**: Receives questions, calls the `RAGAnything` query interface, ensures the reranker is activated, and finally returns the LLM-generated answer.

### 2.2 Obsidian Chunker (ObsidianChunker)

- **Core Class**: `ObsidianChunker`
- **Core Function**: Chunks `.md` files into text blocks of approximately 2000 tokens. Its advanced feature is that it doesn't simply cut text, but through the `_enhance_chunk_with_metadata` method, adds structured metadata to the header of each chunk, such as:
    - **File Information**: Note name, path, creation/modification date.
    - **Knowledge Connections**: Explicitly lists all `[[Wikilinks]]` and `#Tags` contained in the chunk.
    - **Document Structure**: Marks which chunk this is in the file, total number of chunks, and the IDs of the previous and next chunks.

### 2.3 Incremental Sync Module (VaultMonitor)

- **Core Classes**: `VaultMonitor`, `IncrementalVaultUpdater`
- **Core Functions**: 
    - `VaultMonitor`: Responsible for "detection". It maintains a `vault_tracking.json` file that records the hash value of each processed file. During each scan, it compares with the current file system state to quickly identify "new, modified, deleted" changes.
    - `IncrementalVaultUpdater`: Responsible for "execution". It receives detection results from `VaultMonitor` and performs specific operations on the `RAGAnything` database: calls `insert_content_list` for new files, calls `adelete_by_doc_id` then `insert_content_list` for modified files, and calls `adelete_by_doc_id` for deleted files.

### 2.4 Web UI Backend (ui/app.py)

- **Core Framework**: FastAPI
- **Core Functions**: 
    - **Startup Loading**: Creates a global `SimpleRAGAnything` instance and completes initialization when the server starts, ensuring the system is always ready.
    - **Real-time Chat**: Through the `/ws/chat` WebSocket endpoint, receives user questions, calls the RAG system's `query` method, and pushes the LLM-returned answer to the frontend in streaming (word-by-word) fashion, providing an excellent interactive experience.
    - **System Management**: Through REST API endpoints like `/api/sync`, allows users to manually trigger incremental sync from the frontend interface, or query current system configuration and status.

## 3. Data Protocol and Module Interaction

### 3.1 Core Data Structure: Enhanced Chunk

The most core data unit in the system is the "enhanced chunk" generated by `ObsidianChunker`. It is a dictionary structure containing rich metadata and original text, ensuring that even when text is chunked, its context and knowledge associations are completely preserved and fed into the RAG system.

### 3.2 Core Flows

- **Indexing Flow**: `run_obsidian_raganything.py` -> `SimpleRAGAnything.process_vault` -> `ObsidianChunker` -> `RAGAnything.insert_content_list` -> Local storage.
- **Query Flow**: `UI` -> `WebSocket` -> `SimpleRAGAnything.query` -> `RAGAnything.aquery` -> Retrieval -> `BGE Reranker` -> `Gemini LLM` -> `UI`.
- **Sync Flow**: `UI Sync Button` -> `/api/sync` -> `VaultMonitor.scan_vault` -> `IncrementalVaultUpdater.sync_vault` -> RAG database update.

### 3.3 Data Flow

- **Input Data**: `.md` files in the user's local file system.
- **Intermediate Data**: Metadata-rich chunks generated by `ObsidianChunker` (processed in memory).
- **Persistent Data**: All knowledge is stored in the `rag_storage/` directory, mainly including:
    - `LightRAG` database files (such as GraphML graph database, JSON vector database, etc.).
    - `vault_tracking.json`: File status tracking file for incremental sync.
- **Output Data**: Streaming text transmitted via WebSocket, ultimately aggregated into complete answers in the frontend UI.

### 3.4 Module Interaction Diagram

```mermaid
graph TD
    subgraph User Interaction
        A[User] <--> B[Web UI (static/index.html)]
    end

    subgraph Backend
        B <-->|WebSocket| C[FastAPI App (ui/app.py)]
        C -- query --> D[SimpleRAGAnything]
        C -- sync --> E[IncrementalVaultUpdater]
        E -- scans with --> F[VaultMonitor]
        D -- uses --> G[RAGAnything Library]
        G -- gets context from --> H[Knowledge DB (rag_storage/)]
        G -- reranks with --> I[BGE Reranker]
        G -- generates with --> J[Gemini LLM]
    end

    subgraph Initial Build
        K[run_obsidian_raganything.py] --> D
        D -- chunks with --> L[ObsidianChunker]
        D -- builds --> H
    end
```
