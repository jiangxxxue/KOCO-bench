# RAGAnything Core Function Description

# FUNCTION: process_with_rag

## Function Signature

async def process_with_rag(file_path: str, output_dir: str, api_key: str, base_url: str = None, working_dir: str = None, parser: str = None) -> None

## Function Overview
This function is an end-to-end example of the `RAGAnything` library, fully demonstrating the complete process of processing a single document and performing various query modes. It first configures and initializes the `RAGAnything` engine, then processes the specified document to build a knowledge base, and finally executes a series of preset text and multimodal queries to validate and demonstrate the system's core capabilities.

## Input Parameters
- `file_path: str`: Path to the document to be processed.
- `output_dir: str`: Output directory for RAG processing results.
- `api_key: str`: API key for accessing Large Language Model (LLM) services.
- `base_url: str`: Base URL for the LLM API, defaults to `http://brainmachine:11434`.
- `working_dir: str`: Working directory for RAG persistent storage (such as vector indexes, caches, etc.), defaults to `./rag_storage`.
- `parser: str`: Specifies the document parser, defaults to `minerus`.

## Detailed Description

Configures the working directory and document parsing method, sets up text, image, and embedding models, initializes the RAG system, processes the document, and finally validates document understanding capabilities using three methods: pure text, tables, and equations.


## Output
This function has no explicit return value.

## Function Implementation
code\raganything_example.py:line 90-245

## Test Code
code\test_code\my_test_process_with_rag.py
