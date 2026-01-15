# ObsidianGraphRAG: Chat with Your Notes

## 1. System Overview

ObsidianGraphRAG is a system designed to transform personal Obsidian note vaults into queryable, conversational local knowledge bases. It aims to solve core pain points in personal knowledge management: numerous note contents leading to information forgetting and broken thought connections. Through this system, users can interact with all their notes as if conversing with a knowledgeable personal assistant, discovering forgotten knowledge and hidden deep connections between notes.

The core design philosophy of the system is "local-first" and "data privacy". All knowledge parsing, vector embedding, and relevance reranking processes run on the user's local device, only calling external Gemini API (free tier) at the final answer generation stage, thus achieving powerful functionality while maximizing user data privacy and security with almost no running costs.

## 2. Workflow

The system's core workflow is divided into two main phases: **knowledge base construction (initial build and incremental sync)** and **query answering**.

```
Algorithm: ObsidianGraphRAG Pipeline
Input: Obsidian Vault Path, User Query
Output: Answer based on user's notes

// Part 1: Indexing (Initial Build or Incremental Sync)
1: procedure RunIndexing(vault_path, is_incremental)
2:    if is_incremental then
3:        // Incremental sync: only get modified or new notes
4:        notes_to_process = vault_monitor.get_changed_notes(vault_path)
5:    else
6:        // Initial build: get all notes
7:        notes_to_process = vault_monitor.get_all_notes(vault_path)
8:    end if
9:
10:   for each note in notes_to_process do
11:       // Process notes using Obsidian-optimized chunker
12:       chunks = obsidian_chunker.chunk(note)
13:       // Add text chunks and generate vector embeddings
14:       rag_system.add_chunks(chunks)
15:       // Extract entities and relationships, update knowledge graph
16:       rag_system.extract_and_add_graph_elements(chunks)
17:   end for
18:   // Update file status for next incremental sync
19:   vault_monitor.update_sync_status(processed_notes)
20: end procedure
21:
// Part 2: Querying
22: procedure HandleQuery(query_text)
23:   // Step 1: Hybrid retrieval
24:   retrieved_chunks = rag_system.vector_search(query_text)
25:   retrieved_graph_context = rag_system.graph_search(query_text)
26:
27:   // Step 2: Result reranking
28:   reranked_chunks = bge_reranker.rerank(query_text, retrieved_chunks)
29:
30:   // Step 3: Context construction and answer generation
31:   context = format_prompt(reranked_chunks, retrieved_graph_context)
32:   answer = gemini_llm.generate(query_text, context)
33:
34:   return answer
35: end procedure
```

Key components in the pipeline:
- **run_obsidian_raganything.py**: Script for initial execution, which completely processes the entire Obsidian note vault and builds the initial knowledge base.
- **run_incremental_sync.py**: Incremental sync script for daily use. It only processes new or modified notes by comparing file hashes, greatly improving update efficiency.
- **run_ui.py**: Starts a local Web server based on FastAPI and WebSocket, providing a ChatGPT-like interactive interface.
- **obsidian_chunker.py**: A text chunker customized for Obsidian notes, capable of understanding and preserving wikilinks between notes, thus maintaining contextual associations during chunking.
- **bge_reranker.py**: BGE reranking model, which performs secondary sorting of results after retrieval, placing the most relevant content at the front, effectively improving answer accuracy (15-30%).
- **gemini_llm.py**: Encapsulates interaction logic with Google Gemini API, responsible for sending final context and questions to LLM and returning generated answers in streaming mode.

## 3. Application Scenarios

### Personal Knowledge Management and Review
- **Input**: An Obsidian vault containing long-term accumulated, numerous notes, and a natural language question about note content (e.g., "What ideas have I recorded about 'decision models'?").
- **Output**: A comprehensive answer based on the user's own notes, which may reveal hidden connections between notes from different times and different topics.
- **Purpose**: Help users combat forgetting, efficiently review, integrate, and reuse knowledge they have recorded, truly achieving knowledge compounding.

### Creativity and Research Assistance
- **Input**: A vague question about research topics or creative directions (e.g., "In my notes, what cross-content exists between 'psychology' and 'marketing'?").
- **Output**: The system finds all relevant concepts, notes, and connections from the knowledge base, providing a structured overview or summary as a starting point for innovation.
- **Purpose**: Assist research and creative processes, inspire new ideas by deeply mining the inherent potential of existing notes, rather than starting from scratch.

