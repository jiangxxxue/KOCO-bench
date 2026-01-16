#!/usr/bin/env python
"""
Example script demonstrating the integration of MinerU parser with RAGAnything

This example shows how to:
1. Process documents with RAGAnything using MinerU parser
2. Perform pure text queries using aquery() method
3. Perform multimodal queries with specific multimodal content using aquery_with_multimodal() method
4. Handle different types of multimodal content (tables, equations) in queries
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


#key
async def process_with_rag(
    file_path: str,
    output_dir: str,
            mode="hybrid",
        )
        logger.info(f"Answer: {multimodal_result}")

        # 3. Another multimodal query with equation content
        logger.info("\n[Multimodal Query]: Mathematical formula analysis")
        equation_result = await rag.aquery_with_multimodal(
            "Explain this formula and relate it to any mathematical concepts in the document",
            multimodal_content=[
                {
                    "type": "equation",
                    "latex": "F1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}",
                    "equation_caption": "F1-score calculation formula",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"Answer: {equation_result}")

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    # parser = argparse.ArgumentParser(description="MinerU RAG Example")
    # parser.add_argument("file_path", help="Path to the document to process")
    # parser.add_argument(
    #     "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    # )
    # parser.add_argument(
    #     "--output", "-o", default="./output", help="Output directory path"
    # )
    # parser.add_argument(
    #     "--api-key",
    #     default=os.getenv("LLM_BINDING_API_KEY"),
    #     help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    # )
    # parser.add_argument(
    #     "--base-url",
    #     default=os.getenv("LLM_BINDING_HOST"),
    #     help="Optional base URL for API",
    # )
    # parser.add_argument(
    #     "--parser",
    #     default=os.getenv("PARSER", "mineru"),
    #     help="Optional base URL for API",
    # )

    # args = parser.parse_args()

    # # Check if API key is provided
    # if not args.api_key:
    #     logger.error("Error: OpenAI API key is required")
    #     logger.error("Set api key environment variable or use --api-key option")
    #     return

    # # Create output directory if specified
    # if args.output:
    #     os.makedirs(args.output, exist_ok=True)

    # Process with RAG
    asyncio.run(
        process_with_rag(
            "./raganything_workspace/docs/SIE.md", # filepath
            "./raganything_workspace/output", # output_dir
            "ollama", # api_key
            "http://brainmachine:11434", # base_url
            "./raganything_workspace/", # working_dir
            "mineru", # parser
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
