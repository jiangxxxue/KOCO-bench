# ObsidianGraphRAG: Chat with Your Notes

## 1. System Overview

ObsidianGraphRAG is a system designed to transform your personal Obsidian Vault into a queryable, conversational local knowledge base. It addresses a core pain point in personal knowledge management: the abundance of notes leading to forgotten information and broken connections between ideas. Through this system, users can interact with all their notes as if conversing with a knowledgeable personal assistant, uncovering forgotten knowledge and discovering deep connections hidden between notes.

The system's core design philosophy is "local-first" and "data privacy". All knowledge parsing, vector embedding, and reranking processes run on the user's local device. Only the final answer generation stage calls the external Gemini API (free tier), thereby achieving powerful functionality while maximizing user data privacy and security with virtually no running costs.

## 2. Workflow

The system's core workflow is divided into two main phases: **Knowledge Base Construction (Initial Build and Incremental Sync)** and **Query Answering**.

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
13:       // Add chunks and generate vector embeddings
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
- **run_obsidian_raganything.py**: Script for initial execution that fully processes the entire Obsidian Vault and builds the initial knowledge base.
- **run_incremental_sync.py**: Daily incremental sync script. It compares file hashes and only processes new or modified notes, greatly improving update efficiency.
- **run_ui.py**: Launches a local web server based on FastAPI and WebSocket, providing a ChatGPT-like interactive interface.
- **obsidian_chunker.py**: A text chunker customized for Obsidian notes that understands and preserves wikilinks between notes, maintaining contextual associations during chunking.
- **bge_reranker.py**: BGE reranking model that performs secondary sorting of retrieval results, placing the most relevant content at the top, effectively improving answer accuracy by 15-30%.
- **gemini_llm.py**: Encapsulates the interaction logic with Google Gemini API, responsible for sending the final context and question to the LLM and returning the generated answer in streaming fashion.

## 3. Application Scenarios

### Personal Knowledge Management and Review
- **Input**: An Obsidian Vault containing long-accumulated, extensive notes, along with a natural language question about the note content (e.g., "What ideas have I recorded about 'decision models'?").
- **Output**: A comprehensive answer based on the user's own notes, potentially revealing hidden connections between notes from different times and topics.
- **Purpose**: Help users combat forgetting, efficiently review, integrate, and reuse their recorded knowledge, truly achieving compound returns on knowledge.

### Creative and Research Assistance
- **Input**: A vague question about a research topic or creative direction (e.g., "What cross-content exists in my notes about 'psychology' and 'marketing'?").
- **Output**: The system identifies all relevant concepts, notes, and connections from the knowledge base, providing a structured overview or summary as a starting point for innovation.
- **Purpose**: Assist in research and creative processes by deeply mining the inherent potential of existing notes to inspire new ideas, rather than starting from scratch.
