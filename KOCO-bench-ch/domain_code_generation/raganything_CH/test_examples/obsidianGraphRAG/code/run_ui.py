"""
Launcher script for Obsidian RAG Chat UI
Starts the FastAPI server with proper configuration
"""

import os
import sys

if __name__ == "__main__":
    # Set environment variables (optional - can be configured here)
    # os.environ["OBSIDIAN_VAULT_PATH"] = r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro"
    # os.environ["RAG_WORKING_DIR"] = "./rag_storage"

    # Run uvicorn server
    import uvicorn

    print("\n" + "=" * 60)
    print("🚀 Starting Obsidian RAG Chat UI Server")
    print("=" * 60)
    print("📍 URL: http://localhost:8000")
    print("⏹️  Press CTRL+C to stop the server")
    print("=" * 60 + "\n")

    # Check if development mode is enabled
    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
    
    if dev_mode:
        print("⚠️  Development mode: Auto-reload enabled")
        print("   (Set DEV_MODE=false to disable)\n")
    
    uvicorn.run(
        "ui.app:app",
        host="0.0.0.0",
        port=8000,
        reload=dev_mode,  # Only reload in development mode
        log_level="info"
    )
