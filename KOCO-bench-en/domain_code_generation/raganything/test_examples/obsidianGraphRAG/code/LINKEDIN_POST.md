# LinkedIn Post Template

---

I built a Graph-RAG system that turns your notes into a queryable knowledge base.

If you write a lot, you've forgotten most of what you wrote. This fixes that.

## What it does:

Takes your Obsidian notes and builds a knowledge graph. Then you can ask questions and get answers based on everything you've written.

It finds connections between ideas you didn't know existed. Things you wrote years ago become relevant to what you're thinking about today.

## How it works:

Three steps:

1. Add your Gemini API key (free)
2. Point it to your notes folder
3. Run one command

That's it.

The system chunks your notes, extracts entities and relationships, and builds a graph. When you query it, it traverses the graph to find relevant information.

## Why it matters:

Most people take notes but never look at them again. Search only finds keywords. This understands connections.

Ask "What are my main ideas about X?" and it synthesizes across all your notes. Ask "How does A relate to B?" and it follows the graph.

## Technical details:

Everything runs locally except the LLM. Embeddings use EmbeddingGemma (free). Reranking uses BGE (free). Only Gemini calls cost money, and it's cheap.

Processing 1000 notes costs about $5. After that, queries cost $0.001 each.

There's a web interface. Looks like ChatGPT but talks to your notes instead of the internet.

## Incremental updates:

Once you build the initial database, you don't rebuild everything. The system tracks changes and processes only new or modified notes.

Add five notes, sync in 30 seconds. Compare that to rebuilding 1000 notes, which takes 30 minutes.

## For automated content:

If you want to create content from your knowledge base, you can. The system has an API. Query it programmatically and use the results however you want.

## Setup (actual steps):

```
1. Clone repo
2. Copy .env.example to .env
3. Add API key and vault path
4. pip install -r requirements.txt
5. python run_obsidian_raganything.py
6. python run_ui.py
```

Open http://localhost:8000 and start asking questions.

## Link:

https://github.com/Jinstronda/obsidianGraphRAG

---

Built this because I had 5000 notes and couldn't remember what was in them. Now I can ask questions and actually get answers.

If you write a lot, this is useful.

---

**Alternative shorter version for LinkedIn:**

---

I built a system that turns your Obsidian notes into a queryable knowledge base.

The problem: you write notes but forget what's in them. Search only finds keywords.

The solution: build a knowledge graph. Ask questions. Get answers based on everything you've written.

Setup in 5 minutes:
1. Get free Gemini API key
2. Edit .env file (2 lines)
3. Run installation
4. Start interface

Everything runs locally except LLM calls. Cost: ~$5 for 1000 notes initially, then $0.001 per query.

Web interface included. Incremental sync means you don't rebuild everything when you add notes.

https://github.com/Jinstronda/obsidianGraphRAG

Built this for myself. Sharing because others might find it useful.

---
