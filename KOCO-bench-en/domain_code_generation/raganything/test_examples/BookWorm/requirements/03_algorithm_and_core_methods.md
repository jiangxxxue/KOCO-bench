# RAGAnything Core Function Description

# FUNCTION: process_with_rag

## Function Signature

async def process_with_rag(file_path: str, output_dir: str, api_key: str, base_url: str = None, working_dir: str = None, parser: str = None) -> str

## Function Overview
This function is an end-to-end example of the `RAGAnything` library, fully demonstrating the complete process of processing a single document and performing various modes of queries. It first configures and initializes the `RAGAnything` engine, then processes the specified document to build a knowledge base, and finally executes a series of preset text and multimodal queries to verify and demonstrate the core capabilities of the system.

## Input Parameters
- `file_path: str`: The path to the document to be processed.
- `output_dir: str`: The output directory for RAG processing results.
- `api_key: str`: The API key for accessing Large Language Model (LLM) services.
- `base_url: str`: The base URL for the LLM API, defaults to `http://brainmachine:11434`.
- `working_dir: str`: The working directory for RAG persistent storage (such as vector indexes, caches, etc.), defaults to `./rag_storage`.
- `parser: str`: Specifies the document parser, defaults to `minerus`.

## Detailed Description

Configure the working directory and document parsing method, set up text, image, and embedding models, initialize the RAG system, process the document, and finally verify the document understanding capability using three methods: pure text, tables, and formulas.


## Output
This function has no explicit return value (`return` None).

## Function Implementation
code/raganything_example.py:line 90-245

## Test Code
code/test_code/test_process_with_rag.py
