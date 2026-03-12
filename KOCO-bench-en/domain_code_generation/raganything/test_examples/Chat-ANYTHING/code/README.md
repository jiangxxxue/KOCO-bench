# 💬 Chat-Anything  
> Chat with your data, the web, and everything in between.

---

### 🚀 Introduction  

**Chat-Anything** is my custom evolution of **RAG-Anything** and **rag-web-ui** — a unified, intelligent chat system built to interact with *any* knowledge source.  

It combines **Retrieval-Augmented Generation (RAG)** with a modern **web interface**, enhanced by **MCP (Model Context Protocol)** to support **live web search** and context injection from real-time data.  

In simple terms:  
> It’s a chat app that can reason, search, and learn — all in one place.  

---

### ✨ Why I Built It  

While experimenting with **RAG-Anything**, I realized how powerful it could be when paired with a smooth web UI and dynamic context updates.  

So I decided to merge:  
- 🧩 **RAG-Anything** → for its modular multimodal RAG pipeline.  
- 🖥️ **rag-web-ui** → for its clean and interactive frontend.  
- 🌐 **MCP (Web Search)** → to let the model pull *fresh* info directly from the internet.  

Now, **Chat-Anything** lets you explore any topic — powered by your own files, your knowledge base, and the entire web.

---

### 🧠 Key Features  

- 🔍 **Integrated Web Search (MCP)**  
  Seamlessly fetch and summarize live data from the internet in real time.  

- 📚 **RAG-Enhanced Answers**  
  Combines local knowledge bases and embeddings for context-aware responses.  

- 💬 **Unified Web Interface**  
  A beautiful, fast chat UI (from rag-web-ui) for an effortless experience.  

- 📁 **Multi-Source Data Loading**  
  Ingest PDFs, text files, URLs, and online results — all unified under one context.  

- ⚙️ **Context Memory Management**  
  Keeps track of your conversation state and source materials intelligently.  

- 🧩 **Open & Extendable**  
  Add your own LLM providers, vector DBs, or tools — fully modular and customizable.

---

### 🧰 Tech Stack  

| Layer | Technologies |
|-------|---------------|
| **Frontend** | Next.js, React, TailwindCSS |
| **Backend** | FastAPI, Python |
| **RAG Engine** | LangChain, LlamaIndex, FAISS / Pinecone |
| **Protocol Layer** | MCP (Model Context Protocol) |
| **Integrations** | OpenAI API, Local file loaders, Web Search tools |
| **Deployment** | Docker, uvicorn, nginx |

---

### 🏗️ Architecture Overview  

    ┌────────────────────┐
    │     Frontend       │  →  Next.js (rag-web-ui)
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │     Backend        │  →  FastAPI + LangChain
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────────────┐
    │        RAG Engine          │
    │ (RAG-Anything + MCP Layer) │
    └────────┬───────────┬───────┘
             │           │
      ┌──────▼──────┐ ┌──▼────────┐
      │ Vector Store│ │ Web Search│
      └─────────────┘ └───────────┘

---

### ⚡ Getting Started  

#### 1. Clone the repository  
```bash
git clone https://github.com/yourusername/chat-anything.git
cd chat-anything
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
# Set environment variables
OPENAI_API_KEY=your_key
SEARCH_API_KEY=your_key
VECTOR_DB_URL=your_url

# Run locally
 # Run backend
uvicorn main:app --reload

# Run frontend
npm run dev
💡 Example Use Cases

Ask for a real-time summary of today’s tech news.

Upload a PDF or documentation and chat about it.

Compare AI models or frameworks using fresh web data.

Mix local knowledge with live internet sources in one conversation.

🗺️ Roadmap

 Multi-user sessions

 Persistent memory across chats

 Enhanced analytics & citation display

 Custom RAG pipelines via UI

 Support for multimodal input (images, audio)

🙏 Acknowledgements

Huge thanks to the creators of:

RAG-Anything

rag-web-ui

Model Context Protocol (MCP)

And to the open-source community that keeps pushing the boundaries of what RAGs can do.

🧾 License

MIT License © 2025 — Developed with ❤️ by Zakaria Eisa

