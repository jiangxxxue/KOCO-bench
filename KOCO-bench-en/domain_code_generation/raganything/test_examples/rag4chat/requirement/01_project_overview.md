# rag4chat: Local Intelligent Customer Service Construction

## 1. System Overview

`rag4chat` is a comprehensive, locally deployable intelligent customer service chat application. This project aims to provide a flexible framework where users can either interact directly with an AI assistant through its built-in **Streamlit**-based Web graphical interface, or integrate intelligent Q&A capabilities into existing business systems through its provided **Flask**-based API service.

The core of the system is its powerful Retrieval-Augmented Generation (RAG) capability. Users can conveniently build a vector knowledge base through the interface or code using product manuals, FAQs, and knowledge base documents (supporting multiple formats such as md, docx, txt, pdf, etc.). When answering questions, the system can first retrieve the most relevant information from the knowledge base, then pass this information as context to a Large Language Model (LLM) to generate accurate and reliable answers. Additionally, the system has certain intent recognition capabilities (such as transferring to human agents, weather queries) and multimodal dialogue capabilities.

## 2. Workflow

The core workflow of `rag4chat` revolves around two main processes: "knowledge base construction" and "RAG Q&A". The following uses its Streamlit application as an example to demonstrate its main processing flow.

```
Algorithm: rag4chat RAG Pipeline (Streamlit App)
Input: User query, Uploaded documents (optional)
Output: Answer from the chatbot

// Part 1: Knowledge Base Construction (via UI)
1: User uploads documents (PDF, DOCX, MD, etc.) through the web interface.
2: DocumentProcessor parses the files.
3: VectorStoreService creates embeddings and indexes the documents.
4: The knowledge base is now ready.

// Part 2: Chat Interaction (RAG Enabled)
5: procedure HandleUserQuery(query_text)
6:    // Step 1: Retrieve
7:    // Search for document segments most relevant to the question from the vector database
8:    relevant_docs = VectorStoreService.search_documents(query_text)
9:
10:   // Step 2: Augment
11:   // Integrate retrieved document segments into context provided to LLM
12:   context = VectorStoreService.get_context(relevant_docs)
13:
14:   // Step 3: Generate
15:   // Call RAG agent, send question and context together to LLM
16:   agent = RAGAgent(selected_model)
17:   answer = agent.run(prompt=query_text, context=context)
18:
19:   // Step 4: Display
20:   Display answer in the chat UI.
21:   Add query and answer to chat history.
22: end procedure
```

Key components in the pipeline:
- **app.py**: Main Streamlit-based application, responsible for rendering frontend UI, managing session state, and orchestrating the entire Q&A process.
- **app_api.py**: Flask-based backend service, providing API interfaces such as `/v1/chat/completions`, making RAG functionality callable by other programs.
- **DocumentProcessor**: Responsible for processing user-uploaded documents, including file reading and text parsing.
- **VectorStoreService**: Core vector storage service. It manages knowledge base construction (vectorizing and indexing documents) and retrieval (performing similarity searches based on questions).
- **RAGAgent**: Agent that directly interacts with Large Language Models (LLMs). It receives user questions and context retrieved from the vector database, and generates final answers.
- **ChatHistoryManager**: Session history manager, used to support multi-turn dialogue.

## 3. Application Scenarios

### Enterprise Intelligent Customer Service
- **Input**: Enterprise internal documents, such as product manuals, operation guides, regulations, FAQ lists, etc.
- **Output**: An intelligent customer service robot that can be online 24/7, accurately answering various questions for customers or internal employees.
- **Purpose**: Automate handling of common questions, relieve pressure on human customer service, provide standardized, high-quality information services.

### Personal Knowledge Base Assistant
- **Input**: Any personal accumulated electronic documents, such as study notes, research materials, meeting minutes, e-books, etc.
- **Output**: A completely private, intelligent assistant that fully understands the user's personal knowledge system.
- **Purpose**: Help users quickly retrieve information, summarize viewpoints, and inspire ideas in their accumulated knowledge ocean, making knowledge "come alive".

