# RAGAnything Core Function Description


# FUNCTION: MCPGeminiRAGClient.__init__

## Function Overview

Creates a unified intelligent client that integrates local document knowledge base, Gemini artificial intelligence model, and external tool calling capabilities. Establishes a complete intelligent Q&A system that can both retrieve knowledge from local documents and call external tools to obtain real-time information.

## Function Signature

def __init__(self, gemini_api_key: str) -> RAGAnything

## Input Parameters

- `gemini_api_key: str`: API key for accessing Gemini API services

## Detailed Description

This function initializes session management, connects to the Gemini model, configures document processing and text vectorization capabilities, sets up language understanding models, and initializes the knowledge retrieval engine to build a multi-layered intelligent service system.

## Output

- **Return Value**: `rag` - A fully configured knowledge retrieval and Q&A engine instance (RAGAnything type)

## Function Implementation

code\rag-web-ui\backend\app\services\my_rag.py:line 15-153

## Test Code

code\test_code\my_test_MCPGeminiRAGClient_init.py

---

# FUNCTION: load_documents

## Function Overview

Integrates new documents into the knowledge base system, making their content retrievable and queryable. Converts raw documents into structured knowledge representations and establishes corresponding retrieval indexes.

## Function Signature

async def load_documents(self, file_path: str) -> None

## Input Parameters

- `file_path: str`: Path to the document file to be loaded

## Detailed Description

This function performs structured parsing of the specified document (including text, images, and other elements), marks the loading status, and provides feedback on processing results, achieving intelligent document ingestion.

## Output

This function has no explicit return value

## Function Implementation

code\rag-web-ui\backend\app\services\my_rag.py:line 155-164

## Test Code

code\test_code\my_test_load_documents.py

---

# FUNCTION: process_query

## Function Overview

This function is the core processing engine of the intelligent Q&A system, which needs to intelligently determine when to use the local knowledge base and when external real-time information is needed. Through a strategy of first retrieving from local documents and then evaluating whether external support is needed, it ensures providing users with the most accurate and up-to-date answers.

## Function Signature

async def process_query(self, query: str) -> str

## Input Parameters

- `query: str`: User's query question

## Detailed Description

This function first clarifies currently available external tools and services, maintains dialogue context in real-time to connect multi-turn interactions, precisely searches for relevant information from the local knowledge base through hybrid retrieval strategy, intelligently evaluates the sufficiency and completeness of retrieval results, calls external tools to supplement if local knowledge is insufficient or real-time data is needed, finally integrates information to generate answers and synchronously updates dialogue history to provide continuous support for subsequent interactions.


## Output

- **Return Value**: `str` - Intelligent answer to the user's question, which may be based on local knowledge base, external real-time information, or a combination of both

## Function Implementation

code\rag-web-ui\backend\app\services\my_rag.py:line 217-248

## Test Code

code\test_code\my_test_process_query.py

