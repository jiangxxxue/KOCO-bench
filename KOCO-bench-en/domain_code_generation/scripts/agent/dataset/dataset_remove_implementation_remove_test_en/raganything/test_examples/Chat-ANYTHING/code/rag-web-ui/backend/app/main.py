import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.api_v1.api import api_router
from app.core.config import settings
from app.startup.migarate import DatabaseMigrator
import app.core.rag_client as rag_module  # ✅ use module import
from app.services.my_rag import MCPGeminiRAGClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Server starting up...")

    # --- Run database migrations ---
    try:
        migrator = DatabaseMigrator(settings.get_database_url)
        migrator.run_migrations()
        logging.info("Database migrations successful.")
    except Exception as e:
        logging.error(f"Database migration failed: {e}")

    # --- Initialize RAG Client ---
    try:
        mcp_server_script = "server.py"
        rag_module.rag_client = MCPGeminiRAGClient(
            gemini_api_key="AIzaSyBFLZ8o7tOCppO60pLBKX9SsGev0FEsavs"
        )
        await rag_module.rag_client.connect_to_mcp(mcp_server_script)
        logging.info(f"✅ MCPGeminiRAGClient initialized and connected to MCP ({mcp_server_script}).")
        logging.info(f"rag_client instance: {rag_module.rag_client}")
    except Exception as e:
        logging.error(f"❌ FAILED to initialize or connect RAG client: {e}")
        rag_module.rag_client = None

    yield  # <-- Application runs here

    # --- Shutdown logic ---
    logging.info("Server shutting down...")
    if rag_module.rag_client:
        try:
            await rag_module.rag_client.cleanup()
            logging.info("✅ MCP session cleaned up.")
        except Exception as e:
            logging.error(f"MCP cleanup failed: {e}")


# -----------------------------------------------------------------
# FastAPI app setup
# -----------------------------------------------------------------
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"/api/v1/openapi.json",
    lifespan=lifespan
)

# --- CORS setup ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include routers ---
app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {"message": "Welcome to RAG Web UI API"}
