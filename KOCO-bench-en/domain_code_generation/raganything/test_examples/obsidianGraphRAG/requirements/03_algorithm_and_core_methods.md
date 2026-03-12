# ObsidianGraphRAG Core Function Description

# FUNCTION: SimpleRAGAnything.initialize

## Function Overview
`initialize` is the most core initialization method in the `SimpleRAGAnything` class. It is responsible for completing all preparation work at system startup, including checking database status, loading local embedding models, configuring the RAG engine, and correctly instantiating the `RAGAnything` object based on the situation (new or load), preparing for subsequent document processing and querying.

## Function Signature

async def initialize(self) -> None

## Input Parameters
- `self`: Instance object of `SimpleRAGAnything`.

(This function does not receive direct parameters, but accesses instance properties like `working_dir` through `self`, and reads configuration from environment variables via `os.getenv`.)

## Detailed Description
Checks existing database status, handles database options based on interaction mode, loads local embedding model, configures RAG engine and instantiates RAGAnything object, and finally configures reranker functionality.

## Output
- **No direct return value**.

## Function Implementation
code/src/simple_raganything.py:line 433-570`

## Test Code
code/test_code/test_initialize.py
