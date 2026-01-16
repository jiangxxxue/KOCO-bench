# RAG-Anything Service Core Function Description

This document provides detailed descriptions of two core initialization functions in `RAGAnythingService`. These two functions are respectively responsible for creating a persistent RAG instance for "writing" (document processing) at service startup, and creating a temporary RAG instance for "reading" during queries to ensure data consistency and real-time accuracy.

---

# FUNCTION: _initialize_rag_storage

## Function Overview
This function is called when the `RAGAnythingService` service is first initialized. Its core responsibility is to create a long-lived, shared `RAGAnything` instance (`self.rag_anything`) for processing and indexing documents.

## Function Signature

def _initialize_rag_storage(self) -> None

## Input Parameters
- `self`: Instance object of `RAGAnythingService`.

(This function does not receive direct parameters but reads all configurations from environment variables through `os.getenv`)

## Detailed Description
Reads configuration from environment variables, creates RAG engine configuration and defines LLM, vision, and embedding model functions, initializes RAGAnything instance for document processing and indexing.

## Output
- **No direct return value**.

## Function Implementation
`code\src\services\rag_anything_service.py:line 46-154`

## Test Code
`code\test_code\my_test_initialize_rag_storage.py`

---

# FUNCTION: _create_rag_instance

## Function Overview
This function is used to create a brand new, temporary `RAGAnything` instance each time a query (`query_documents`) is executed. Its core purpose is to ensure the query engine can load and use the latest state of the knowledge base on disk, avoiding data inconsistency issues caused by long-running services.

## Function Signature

def _create_rag_instance(self) -> RAGAnything

## Input Parameters
- `self`: Instance object of `RAGAnythingService`.

## Detailed Description
Redefines model functions, loads existing knowledge base data from storage directory into LightRAG instance, creates new RAGAnything instance and injects data, returns instance containing the latest knowledge base state.

## Output
- **Return Value**: `rag_instance` - A brand new `RAGAnything` object instance ready to execute queries.

## Function Implementation
`code\src\services\rag_anything_service.py:line 157-271`

## Test Code
`code\test_code\my_test_create_rag_instance.py`

