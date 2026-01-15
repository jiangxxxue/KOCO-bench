"""
FastAPI Web UI for Obsidian RAG System
Provides a ChatGPT-style interface for querying your Obsidian vault
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simple_raganything import SimpleRAGAnything

# Initialize FastAPI app
app = FastAPI(title="Obsidian RAG Chat")

# Global RAG instance
rag_system = None

# Mount static files
app.mount("/static", StaticFiles(directory="ui/static"), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on server startup"""
    global rag_system

    print("=" * 60)
    print("🚀 Starting Obsidian RAG Chat UI")
    print("=" * 60)

    # Configuration from environment or defaults
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro")
    working_dir = os.getenv("RAG_WORKING_DIR", "./rag_storage")

    print(f"📁 Vault: {vault_path}")
    print(f"💾 Storage: {working_dir}")
    print()

    # Initialize RAG system (non-interactive mode for web server)
    try:
        rag_system = SimpleRAGAnything(
            vault_path=vault_path,
            working_dir=working_dir,
            non_interactive=True  # Skip interactive prompts in web server mode
        )

        print("🔄 Initializing RAG system...")
        await rag_system.initialize()

        print()
        print("✅ RAG system ready!")
        print("🌐 Server running at: http://localhost:8000")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("=" * 60)
        raise


@app.get("/")
async def get_chat_page():
    """Serve the main chat interface"""
    with open("ui/static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat with streaming responses

    Protocol:
    - Client sends: {"message": "user question"}
    - Server streams: {"type": "chunk", "content": "..."} for each chunk
    - Server sends: {"type": "done"} when complete
    - Server sends: {"type": "error", "content": "..."} on error
    """
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "").strip()
            query_mode = message_data.get("mode", "hybrid")
            enable_rerank = message_data.get("enable_rerank", True)

            if not user_message:
                await websocket.send_json({
                    "type": "error",
                    "content": "Empty message received"
                })
                continue

            # Log query to terminal (not sent to UI)
            print(f"\n💬 User: {user_message}")
            print(f"🎯 Mode: {query_mode}, Rerank: {enable_rerank}")
            print("🤖 Assistant: ", end="", flush=True)

            try:
                # Query RAG system with settings
                response = await rag_system.query(user_message, mode=query_mode)

                # Log full response to terminal
                print(response)
                print()

                # Stream response to client word by word
                words = response.split()
                for i, word in enumerate(words):
                    # Add space between words (except first word)
                    chunk = word if i == 0 else f" {word}"

                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk
                    })

                    # Small delay for typing effect
                    await asyncio.sleep(0.03)

                # Send completion signal
                await websocket.send_json({"type": "done"})

            except Exception as e:
                error_msg = f"Query failed: {str(e)}"
                print(f"❌ {error_msg}")

                await websocket.send_json({
                    "type": "error",
                    "content": error_msg
                })

    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    import psutil
    import torch
    
    health_data = {
        "status": "healthy",
        "rag_initialized": rag_system is not None,
        "gpu_available": torch.cuda.is_available(),
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
    }
    
    if torch.cuda.is_available():
        health_data["gpu_name"] = torch.cuda.get_device_name(0)
        health_data["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1024 / 1024
        health_data["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(0) / 1024 / 1024
    
    return health_data


@app.get("/api/config")
async def get_config():
    """Get current RAG system configuration"""
    if not rag_system:
        return {"error": "RAG system not initialized"}
    
    return {
        "vault_path": rag_system.vault_path,
        "working_dir": rag_system.working_dir,
        "using_existing_db": rag_system.using_existing_db,
        "non_interactive": rag_system.non_interactive,
        "models": {
            "embedding": "EmbeddingGemma 308M",
            "llm": "Gemini 2.5 Flash",
            "reranker": "BGE Reranker v2-m3"
        }
    }


@app.post("/api/sync")
async def sync_vault():
    """Incrementally sync vault changes"""
    if not rag_system:
        return {"error": "RAG system not initialized"}
    
    try:
        # Import here to avoid circular dependencies
        from src.vault_monitor import VaultMonitor, IncrementalVaultUpdater
        
        # Initialize monitor
        tracking_file = os.path.join(rag_system.working_dir, "vault_tracking.json")
        monitor = VaultMonitor(rag_system.vault_path, tracking_file)
        
        # Scan for changes
        new_files, modified_files, deleted_files = monitor.scan_vault()
        
        # Create updater and sync
        updater = IncrementalVaultUpdater(rag_system, monitor)
        await updater.sync_vault()
        
        return {
            "status": "success",
            "changes": {
                "new": len(new_files),
                "modified": len(modified_files),
                "deleted": len(deleted_files)
            },
            "total_tracked": len(monitor.file_registry)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/api/vault/status")
async def vault_status():
    """Get vault sync status"""
    if not rag_system:
        return {"error": "RAG system not initialized"}
    
    try:
        from src.vault_monitor import VaultMonitor
        
        tracking_file = os.path.join(rag_system.working_dir, "vault_tracking.json")
        monitor = VaultMonitor(rag_system.vault_path, tracking_file)
        
        # Quick scan
        new_files, modified_files, deleted_files = monitor.scan_vault()
        
        return {
            "status": "success",
            "vault_path": rag_system.vault_path,
            "total_tracked": len(monitor.file_registry),
            "pending_changes": {
                "new": len(new_files),
                "modified": len(modified_files),
                "deleted": len(deleted_files)
            },
            "needs_sync": len(new_files) + len(modified_files) + len(deleted_files) > 0
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
