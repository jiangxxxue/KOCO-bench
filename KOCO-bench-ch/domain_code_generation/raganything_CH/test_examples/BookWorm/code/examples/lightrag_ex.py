import asyncio
import os
import shutil
from pathlib import Path
import inspect
import logging
import logging.config
from typing import Dict, Any
import tempfile

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

# Configuration constants
WORKING_DIR = "./lightrag_workspace"
DOCUMENT_DIR = "./lightrag_workspace/docs"
DOCUMENT_ARCHIVE_DIR = "./lightrag_workspace/processed_docs"
OUTPUT_DIR = "./lightrag_workspace/output"

# Supported file extensions for LightRAG processing (after PDF conversion)
SUPPORTED_EXTENSIONS = [".txt", ".md", ".docx", ".pptx"]

# Model configuration
# LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:32b")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-coder:30b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
MAX_TOKEN_SIZE = int(os.getenv("MAX_EMBED_TOKENS", "40000"))

# Host configuration
LLM_HOST = os.getenv("LLM_BINDING_HOST", "http://100.95.157.120:11434")
EMBEDDING_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://100.95.157.120:11434")
TIMEOUT = int(os.getenv("TIMEOUT", "3000"))  # 5 minutes in seconds
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "6000"))  # 10 minutes for embeddings in seconds

# PDF Processing configuration
PDF_PROCESSOR = os.getenv("PDF_PROCESSOR", "mineru").lower()  # Options: "mineru", "docling"
SKIP_PDF_CONVERSION = os.getenv("SKIP_PDF_CONVERSION", "false").lower() == "true"


def set_pdf_processor(processor: str) -> bool:
    """
    Set the PDF processor at runtime
    
    Args:
        processor: Either "mineru" or "docling"
        
    Returns:
        True if processor was set successfully, False otherwise
    """
    global PDF_PROCESSOR
    processor = processor.lower()
    
    if processor in ["mineru", "docling"]:
        PDF_PROCESSOR = processor
        print(f"✅ PDF processor set to: {processor.upper()}")
        return True
    else:
        print(f"❌ Invalid PDF processor: {processor}. Valid options: 'mineru', 'docling'")
        return False


def get_pdf_processor() -> str:
    """Get the current PDF processor"""
    return PDF_PROCESSOR


def configure_logging():
    """Configure logging for the application"""
    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_ex.log"))

    print(f"\nLightRAG Enhanced demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

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
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [WORKING_DIR, DOCUMENT_DIR, DOCUMENT_ARCHIVE_DIR, OUTPUT_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/verified directory: {directory}")


async def initialize_rag():
    """Initialize LightRAG instance with configuration"""
    
    # Debug: Print configuration
    print("🔍 Configuration Debug:")
    print(f"  LLM Host: {LLM_HOST}")
    print(f"  Embedding Host: {EMBEDDING_HOST}")
    print(f"  LLM Model: {LLM_MODEL}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Timeout: {TIMEOUT}")
    print(f"  Embedding Dimension: {EMBEDDING_DIM}")
    print(f"  Max Token Size: {MAX_TOKEN_SIZE}")
    
    # # Test Ollama connectivity before initializing RAG
    # print("\n🔍 Testing Ollama connectivity...")
    # try:
    #     # Test embedding service
    #     test_texts = ["test connection"]
    #     test_embedding = await robust_ollama_embed(
    #         test_texts,
    #         embed_model=EMBEDDING_MODEL,
    #         host=EMBEDDING_HOST,
    #     )
    #     if test_embedding is not None:
    #         print(f"✅ Embedding service test successful - Shape: {test_embedding.shape}")
    #     else:
    #         print("❌ Embedding service returned None")
    #         raise Exception("Embedding service returned None")
        
    #     # Test LLM service with direct HTTP request
    #     print("🔍 Testing LLM service...")
    #     import httpx
    #     async with httpx.AsyncClient() as client:
    #         response = await client.post(
    #             f"{LLM_HOST}/api/generate",
    #             json={
    #                 "model": LLM_MODEL,
    #                 "prompt": "Test message",
    #                 "stream": False,
    #                 "options": {"num_ctx": 512}
    #             },
    #             timeout=30.0
    #         )
    #         if response.status_code == 200:
    #             result = response.json()
    #             if result.get("response"):
    #                 print("✅ LLM service test successful")
    #             else:
    #                 print("❌ LLM service returned empty response")
    #                 raise Exception("LLM service returned empty response")
    #         else:
    #             print(f"❌ LLM service failed with status code: {response.status_code}")
    #             raise Exception(f"LLM service failed with status code: {response.status_code}")
        
    # except Exception as e:
    #     print(f"❌ Ollama connectivity test failed: {e}")
    #     print("Please ensure Ollama is running and the models are available")
    #     raise
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_max_token_size=22192,
        llm_model_kwargs={
            "host": LLM_HOST,
            "options": {"num_ctx": 20192},
            "timeout": TIMEOUT,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=MAX_TOKEN_SIZE,
            func=lambda texts: robust_ollama_embed(
                texts,
                embed_model=EMBEDDING_MODEL,
                host=EMBEDDING_HOST
            ),
        ),
        vector_storage="FaissVectorDBStorage",
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": 0.3  # Your desired threshold
        }
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def process_document(file_path: str, rag: LightRAG) -> bool:
    """Process a single document and insert it into the RAG system"""
    try:
        print(f"Processing: {file_path}")
        
        # Read file content based on extension
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            print(f"Unsupported file format: {file_extension}")
            print("Note: PDFs should be converted to markdown before processing")
            return False
            
        # Insert content into RAG system
        await rag.ainsert(content, file_paths=[file_path])
        print(f"Successfully processed: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


async def convert_pdf_to_markdown(pdf_path: str, processor: str | None = None) -> str:
    """Convert PDF file to Markdown format using the specified processor"""
    if processor is None:
        processor = PDF_PROCESSOR
    
    processor = processor.lower()
    
    if processor == "docling":
        return await convert_pdf_with_docling(pdf_path)
    elif processor == "mineru":
        return await convert_pdf_with_mineru(pdf_path)
    else:
        print(f"Unknown PDF processor: {processor}. Using MinerU as fallback.")
        return await convert_pdf_with_mineru(pdf_path)


async def convert_pdf_with_docling(pdf_path: str) -> str:
    """Convert PDF file to Markdown format using Docling"""
    try:
        # Import Docling components locally to avoid global import issues
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
        
        print("Processing with Docling...")
        
        # Configure pipeline options for better performance
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR for scanned documents
        pipeline_options.do_table_structure = True  # Enable table structure recognition
        pipeline_options.table_structure_options.do_cell_matching = True  # Enable cell matching for tables
        
        # Create document converter with pipeline options
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # Convert the document
        result = doc_converter.convert(pdf_path)
        
        # Extract markdown content
        if result and result.document:
            markdown_content = result.document.export_to_markdown()
            return markdown_content
        else:
            print(f"Failed to convert PDF with Docling: {pdf_path}")
            return ""
            
    except ImportError as e:
        print(f"Docling not available: {e}")
        print("Please install docling with: pip install docling")
        print("Falling back to MinerU...")
        return await convert_pdf_with_mineru(pdf_path)
    except Exception as e:
        print(f"Error converting PDF to Markdown using Docling: {e}")
        import traceback
        traceback.print_exc()
        return ""


async def convert_pdf_with_mineru(pdf_path: str) -> str:
    """Convert PDF file to Markdown format using MinerU"""
    try:
        # Import MinerU functions locally to avoid global import issues
        from mineru.cli.common import prepare_env, read_fn
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.utils.enum_class import MakeMode
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
        from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
        
        # Set environment variables to force CPU usage
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare the output directory and file name
            pdf_file_name = Path(pdf_path).stem
            local_image_dir, local_md_dir = prepare_env(temp_dir, pdf_file_name, "auto")
            
            # Read PDF file
            pdf_bytes = read_fn(pdf_path)
            if not pdf_bytes:
                print(f"Failed to read PDF file: {pdf_path}")
                return ""
            
            # Process PDF using MinerU pipeline
            pdf_bytes_list = [pdf_bytes]
            p_lang_list = ["en"]  # Default to English, can be made configurable
            
            # Analyze document with CPU-only mode
            print("Processing with CPU-only mode...")
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                pdf_bytes_list, p_lang_list, parse_method="auto", formula_enable=True, table_enable=True
            )
            
            if not infer_results:
                print(f"Failed to analyze PDF: {pdf_path}")
                return ""
            
            # Process the first (and only) document
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]
            
            # Create data writers
            image_writer = FileBasedDataWriter(local_image_dir)
            
            # Convert to middle JSON format
            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, True
            )
            
            pdf_info = middle_json["pdf_info"]
            
            # Generate markdown content
            image_dir = str(os.path.basename(local_image_dir))
            markdown_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
            
            # Ensure we return a string
            if isinstance(markdown_content, str):
                return markdown_content
            elif isinstance(markdown_content, list):
                return "\n".join(str(item) for item in markdown_content)
            else:
                return str(markdown_content) if markdown_content else ""
            
    except Exception as e:
        print(f"Error converting PDF to Markdown using MinerU: {e}")
        import traceback
        traceback.print_exc()
        return ""


async def process_folder(folder_path: str, rag: LightRAG, recursive: bool = True) -> Dict[str, Any]:
    """Process all documents in a folder"""
    results = {
        "processed": [],
        "failed": [],
        "total": 0,
        "success": 0,
        "failure": 0
    }
    
    folder_path_obj = Path(folder_path)
    
    if not folder_path_obj.exists():
        print(f"Folder does not exist: {folder_path}")
        return results
    
    # Find all supported files
    files_to_process = []
    for ext in SUPPORTED_EXTENSIONS:
        if recursive:
            pattern = f"**/*{ext}"
        else:
            pattern = f"*{ext}"
        files_to_process.extend(folder_path_obj.glob(pattern))
    
    results["total"] = len(files_to_process)
    
    if not files_to_process:
        print(f"No supported files found in {folder_path}")
        return results
    
    print(f"Found {len(files_to_process)} files to process")
    
    # Process each file
    for file_path in files_to_process:
        success = await process_document(str(file_path), rag)
        if success:
            results["processed"].append(str(file_path))
            results["success"] += 1
        else:
            results["failed"].append(str(file_path))
            results["failure"] += 1
    
    return results


def archive_processed_documents():
    """Move processed documents to archive directory"""
    if not os.path.exists(DOCUMENT_DIR):
        print(f"Document directory {DOCUMENT_DIR} does not exist")
        return
        
    processed_files = []
    for ext in SUPPORTED_EXTENSIONS:
        pattern = f"*{ext}"
        for file_path in Path(DOCUMENT_DIR).glob(pattern):
            if file_path.is_file():
                # Move file to archive directory
                archive_path = Path(DOCUMENT_ARCHIVE_DIR) / file_path.name
                shutil.move(str(file_path), str(archive_path))
                processed_files.append(file_path.name)
                print(f"Archived: {file_path.name}")
    
    if processed_files:
        print(f"Successfully archived {len(processed_files)} files")
    else:
        print("No files to archive")


async def print_stream(stream):
    """Print streaming response"""
    async for chunk in stream:
        print(chunk, end="", flush=True)


async def query_with_mode(rag: LightRAG, question: str, mode: str = "hybrid", stream: bool = True):
    """Query the RAG system with specified mode"""
    print(f"\n{'='*60}")
    print(f"🔍 Query mode: {mode}")
    print(f"{'='*60}")
    print(f"Question: {question}")
    print(f"{'-'*60}")
    
    try:
        # Validate mode
        valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
        if mode not in valid_modes:
            print(f"Invalid mode '{mode}'. Using 'hybrid' instead.")
            mode = "hybrid"
        
        resp = await rag.aquery(
            question,
            param=QueryParam(mode=mode, stream=stream),  # type: ignore
        )
        
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)
            
        return resp
        
    except Exception as e:
        print(f"Error during query: {e}")
        return None


async def interactive_query_mode(rag: LightRAG):
    """Interactive query mode with multiple options"""
    print("\n" + "="*60)
    print("🔍 LightRAG Enhanced Query Mode")
    print("="*60)
    print("Available commands:")
    print("1. query - Ask a question")
    print("2. test - Run test queries on all modes")
    print("3. stats - Show system statistics")
    print("4. processor - Switch PDF processor (mineru/docling)")
    print("5. exit - Exit query mode")
    print(f"\nCurrent PDF Processor: {PDF_PROCESSOR.upper()}")
    print("Available modes: hybrid, local, global, naive")
    print("-"*60)

    while True:
        try:
            # Get command
            command = input("\nEnter command (query/test/stats/processor/exit): ").strip().lower()
            
            if command == "exit":
                print("👋 Goodbye!")
                break
            
            elif command == "query":
                # Single query
                question = input("Enter your question: ").strip()
                if not question:
                    continue
                
                mode = input("Enter query mode (hybrid/local/global/naive) [default: hybrid]: ").strip().lower()
                if not mode:
                    mode = "hybrid"
                
                stream = input("Enable streaming (y/n) [default: y]: ").strip().lower()
                stream = stream != "n"
                
                await query_with_mode(rag, question, mode, stream)
                
            elif command == "test":
                # Test all modes
                question = input("Enter test question: ").strip()
                if not question:
                    continue
                
                modes = ["naive", "local", "global", "hybrid"]
                for mode in modes:
                    await query_with_mode(rag, question, mode, stream=False)
                    print("\n" + "="*60 + "\n")
                    
            elif command == "stats":
                # Show system statistics
                print("\n📊 System Statistics:")
                print(f"Working directory: {WORKING_DIR}")
                print(f"Document directory: {DOCUMENT_DIR}")
                print(f"LLM Model: {LLM_MODEL}")
                print(f"Embedding Model: {EMBEDDING_MODEL}")
                print(f"Embedding Dimension: {EMBEDDING_DIM}")
                print(f"PDF Processor: {PDF_PROCESSOR.upper()}")
                
                # Check if storage files exist
                storage_files = [
                    "graph_chunk_entity_relation.graphml",
                    "kv_store_doc_status.json",
                    "kv_store_full_docs.json",
                    "kv_store_text_chunks.json",
                    "vdb_chunks.json",
                    "vdb_entities.json",
                    "vdb_relationships.json",
                ]
                
                print("\n📁 Storage Files:")
                for file in storage_files:
                    file_path = os.path.join(WORKING_DIR, file)
                    exists = "✅" if os.path.exists(file_path) else "❌"
                    size = ""
                    if os.path.exists(file_path):
                        size = f" ({os.path.getsize(file_path)} bytes)"
                    print(f"  {exists} {file}{size}")
                    
            elif command == "processor":
                # Switch PDF processor
                current_processor = get_pdf_processor()
                print(f"\n🔄 Current PDF Processor: {current_processor.upper()}")
                print("Available processors:")
                print("  1. mineru - MinerU (thorough, complex layouts)")
                print("  2. docling - Docling (fast, standard documents)")
                
                new_processor = input("\nEnter new processor (mineru/docling): ").strip().lower()
                if new_processor in ["mineru", "docling"]:
                    if set_pdf_processor(new_processor):
                        print(f"✅ PDF processor switched to: {new_processor.upper()}")
                    else:
                        print("❌ Failed to switch PDF processor")
                else:
                    print("❌ Invalid processor. Please choose 'mineru' or 'docling'")
                    
            else:
                print("Invalid command. Please choose: query, test, stats, processor, or exit")
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


async def check_conversion_consistency(output_dir: str | None = None) -> Dict[str, Any]:
    """
    Check for consistency between markdown files and archived PDFs.
    
    Args:
        output_dir: Directory containing markdown files (if None, uses DOCUMENT_DIR)
        
    Returns:
        Dictionary with consistency check results
    """
    if output_dir is None:
        output_dir = DOCUMENT_DIR
    
    print("🔍 Checking conversion consistency...")
    
    results = {
        "orphaned_markdown": [],      # Markdown files without archived PDFs
        "orphaned_pdfs": [],          # Archived PDFs without markdown files
        "consistent_pairs": []        # Files that have both markdown and archived PDF
    }
    
    # Get all markdown files
    markdown_files = list(Path(output_dir).glob("*.md"))
    
    # Get all archived PDFs
    archived_pdfs = []
    if os.path.exists(DOCUMENT_ARCHIVE_DIR):
        archived_pdfs = list(Path(DOCUMENT_ARCHIVE_DIR).glob("*.pdf"))
    
    # Check for orphaned markdown files (markdown without archived PDF)
    for md_file in markdown_files:
        pdf_name = md_file.stem + ".pdf"
        corresponding_pdf = Path(DOCUMENT_ARCHIVE_DIR) / pdf_name
        
        if corresponding_pdf.exists():
            results["consistent_pairs"].append({
                "markdown": str(md_file),
                "pdf": str(corresponding_pdf)
            })
        else:
            results["orphaned_markdown"].append(str(md_file))
    
    # Check for orphaned archived PDFs (archived PDF without markdown)
    for pdf_file in archived_pdfs:
        md_name = pdf_file.stem + ".md"
        corresponding_md = Path(output_dir) / md_name
        
        if not corresponding_md.exists():
            results["orphaned_pdfs"].append(str(pdf_file))
    
    # Print summary
    if results["consistent_pairs"]:
        print(f"✅ Found {len(results['consistent_pairs'])} consistent markdown-PDF pairs")
    
    if results["orphaned_markdown"]:
        print(f"⚠️  Found {len(results['orphaned_markdown'])} orphaned markdown files (no archived PDF):")
        for md_file in results["orphaned_markdown"]:
            print(f"    - {Path(md_file).name}")
    
    if results["orphaned_pdfs"]:
        print(f"⚠️  Found {len(results['orphaned_pdfs'])} orphaned archived PDFs (no markdown):")
        for pdf_file in results["orphaned_pdfs"]:
            print(f"    - {Path(pdf_file).name}")
    
    return results


async def cleanup_orphaned_files(output_dir: str | None = None, dry_run: bool = True) -> Dict[str, Any]:
    """
    Clean up orphaned files (markdown without archived PDF or vice versa).
    
    Args:
        output_dir: Directory containing markdown files (if None, uses DOCUMENT_DIR)
        dry_run: If True, only report what would be cleaned up without actually doing it
        
    Returns:
        Dictionary with cleanup results
    """
    if output_dir is None:
        output_dir = DOCUMENT_DIR
    
    print(f"🧹 {'Dry run: ' if dry_run else ''}Cleaning up orphaned files...")
    
    # Get consistency check results
    consistency = await check_conversion_consistency(output_dir)
    
    cleanup_results = {
        "orphaned_markdown_handled": [],
        "orphaned_pdfs_handled": [],
        "errors": []
    }
    
    # Handle orphaned markdown files (try to find and archive corresponding PDFs)
    for md_file in consistency["orphaned_markdown"]:
        md_path = Path(md_file)
        pdf_name = md_path.stem + ".pdf"
        
        # Look for the PDF in source directories
        potential_pdf_locations = [
            Path(DOCUMENT_DIR) / pdf_name,
            Path("./pdfs") / pdf_name,
        ]
        
        pdf_found = None
        for pdf_location in potential_pdf_locations:
            if pdf_location.exists():
                pdf_found = pdf_location
                break
        
        if pdf_found:
            archive_path = Path(DOCUMENT_ARCHIVE_DIR) / pdf_name
            if dry_run:
                print(f"Would archive: {pdf_found} -> {archive_path}")
                cleanup_results["orphaned_markdown_handled"].append(str(md_file))
            else:
                try:
                    shutil.move(str(pdf_found), str(archive_path))
                    print(f"📦 Archived: {pdf_found} -> {archive_path}")
                    cleanup_results["orphaned_markdown_handled"].append(str(md_file))
                except Exception as e:
                    error_msg = f"Failed to archive {pdf_found}: {e}"
                    print(f"❌ {error_msg}")
                    cleanup_results["errors"].append(error_msg)
        else:
            print(f"⚠️  No PDF found for orphaned markdown: {md_path.name}")
    
    # Handle orphaned archived PDFs (create placeholder markdown files)
    for pdf_file in consistency["orphaned_pdfs"]:
        pdf_path = Path(pdf_file)
        md_name = pdf_path.stem + ".md"
        md_path = Path(output_dir) / md_name
        
        if dry_run:
            print(f"Would create placeholder markdown: {md_path}")
            cleanup_results["orphaned_pdfs_handled"].append(str(pdf_file))
        else:
            try:
                placeholder_content = f"# {pdf_path.stem}\n\n*This PDF was archived but no markdown conversion was found.*\n\n**Original PDF**: {pdf_path.name}\n**Status**: Needs reconversion\n"
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(placeholder_content)
                print(f"📄 Created placeholder markdown: {md_path}")
                cleanup_results["orphaned_pdfs_handled"].append(str(pdf_file))
            except Exception as e:
                error_msg = f"Failed to create placeholder for {pdf_file}: {e}"
                print(f"❌ {error_msg}")
                cleanup_results["errors"].append(error_msg)
    
    return cleanup_results


async def convert_all_pdfs_to_markdown(source_dirs: list[str] | None = None, output_dir: str | None = None) -> Dict[str, str]:
    """
    Convert all PDF files from source directories to Markdown format.
    
    Args:
        source_dirs: List of directories to search for PDFs (if None, uses DOCUMENT_DIR and pdfs/)
        output_dir: Directory to save the markdown files (if None, uses DOCUMENT_DIR)
        
    Returns:
        Dictionary mapping original PDF paths to converted markdown file paths
    """
    if source_dirs is None:
        source_dirs = [DOCUMENT_DIR, "./pdfs"]
    
    if output_dir is None:
        output_dir = DOCUMENT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check conversion consistency first
    consistency_results = await check_conversion_consistency(output_dir)
    
    print(f"🔍 Scanning for PDF files in directories: {', '.join(source_dirs)}")
    
    # Find all PDF files in source directories
    pdf_paths = []
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if source_path.exists():
            pdf_files = list(source_path.glob("**/*.pdf"))
            pdf_paths.extend([str(f) for f in pdf_files])
            print(f"Found {len(pdf_files)} PDF files in {source_dir}")
        else:
            print(f"Directory does not exist: {source_dir}")
    
    if not pdf_paths:
        print("No PDF files found to convert")
        return {}
    
    print(f"\n📄 Converting {len(pdf_paths)} PDF files to Markdown...")
    processor_name = "MinerU" if PDF_PROCESSOR == "mineru" else "Docling"
    print(f"Using {processor_name} processor")
    
    results = {}
    
    for i, pdf_path in enumerate(pdf_paths, 1):
        try:
            print(f"\n[{i}/{len(pdf_paths)}] Processing: {Path(pdf_path).name}")
            
            # Generate expected markdown filename and path
            pdf_name = Path(pdf_path).stem
            markdown_filename = f"{pdf_name}.md"
            markdown_path = os.path.join(output_dir, markdown_filename)
            
            # Check if PDF has already been processed
            pdf_archive_path = os.path.join(DOCUMENT_ARCHIVE_DIR, Path(pdf_path).name)
            markdown_exists = os.path.exists(markdown_path)
            pdf_already_archived = os.path.exists(pdf_archive_path)
            
            if markdown_exists:
                if pdf_already_archived:
                    print("✅ Already processed: Markdown exists and PDF archived")
                    results[pdf_path] = markdown_path
                    continue
                else:
                    print(f"📄 Markdown exists: {markdown_path}")
                    # Archive the PDF since markdown exists but PDF wasn't archived yet
                    try:
                        shutil.move(pdf_path, pdf_archive_path)
                        print(f"📦 Archived PDF: {pdf_archive_path}")
                        results[pdf_path] = markdown_path
                        continue
                    except Exception as archive_error:
                        print(f"⚠️  Warning: Could not archive PDF: {archive_error}")
                        # Continue with the existing markdown file
                        results[pdf_path] = markdown_path
                        continue
            
            # Convert PDF to Markdown
            print("🔄 Converting PDF to Markdown...")
            markdown_content = await convert_pdf_to_markdown(pdf_path, PDF_PROCESSOR)
            
            if markdown_content:
                # Write markdown content to file
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                results[pdf_path] = markdown_path
                print(f"✅ Converted to: {markdown_path} ({len(markdown_content)} characters)")
                
                # Archive the original PDF (from any source directory)
                try:
                    shutil.move(pdf_path, pdf_archive_path)
                    print(f"📦 Archived PDF: {pdf_archive_path}")
                except Exception as move_error:
                    print(f"⚠️  Warning: Could not archive PDF: {move_error}")
                    # Try to copy instead of move as fallback
                    try:
                        shutil.copy2(pdf_path, pdf_archive_path)
                        os.remove(pdf_path)
                        print(f"📦 Archived PDF (via copy): {pdf_archive_path}")
                    except Exception as copy_error:
                        print(f"❌ Failed to archive PDF: {copy_error}")
            else:
                print(f"❌ Failed to convert: {pdf_path}")
                results[pdf_path] = None
                
        except Exception as e:
            print(f"❌ Error converting {pdf_path}: {e}")
            results[pdf_path] = None
    
    successful_conversions = len([r for r in results.values() if r is not None])
    print(f"\n📊 PDF to Markdown conversion completed: {successful_conversions}/{len(pdf_paths)} successful")
    
    # Run final consistency check
    if successful_conversions > 0:
        print("\n🔍 Final consistency check...")
        final_consistency = await check_conversion_consistency(output_dir)
        if len(final_consistency["consistent_pairs"]) > len(consistency_results["consistent_pairs"]):
            improvement = len(final_consistency["consistent_pairs"]) - len(consistency_results["consistent_pairs"])
            print(f"✅ Improved consistency: +{improvement} consistent pairs")
    
    return results


async def process_pdf_batch(pdf_paths: list[str], output_dir: str | None = None) -> Dict[str, str]:
    """
    Process multiple PDF files in batch using MinerU
    
    Args:
        pdf_paths: List of PDF file paths to process
        output_dir: Optional directory to save markdown files
        
    Returns:
        Dictionary mapping PDF file paths to their markdown content
    """
    results = {}
    
    for pdf_path in pdf_paths:
        print(f"Processing PDF: {pdf_path}")
        try:
            markdown_content = await convert_pdf_to_markdown(pdf_path, PDF_PROCESSOR)
            if markdown_content:
                results[pdf_path] = markdown_content
                
                # Save markdown file if output directory is specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    pdf_name = Path(pdf_path).stem
                    markdown_file = os.path.join(output_dir, f"{pdf_name}.md")
                    with open(markdown_file, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    print(f"Saved markdown: {markdown_file}")
                
                print(f"✅ Successfully processed: {pdf_path} ({len(markdown_content)} chars)")
            else:
                print(f"❌ Failed to process: {pdf_path}")
                results[pdf_path] = ""
                
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            results[pdf_path] = ""
    
    return results


async def process_pdf_directory(pdf_dir: str, rag: LightRAG | None = None, output_dir: str | None = None) -> Dict[str, Any]:
    """
    Process all PDF files in a directory using MinerU and optionally add to RAG
    
    Args:
        pdf_dir: Directory containing PDF files
        rag: Optional LightRAG instance to insert content
        output_dir: Optional directory to save markdown files
        
    Returns:
        Dictionary with processing results
    """
    results = {
        "processed": [],
        "failed": [],
        "total": 0,
        "success": 0,
        "failure": 0,
        "markdown_content": {}
    }
    
    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.exists():
        print(f"Directory does not exist: {pdf_dir}")
        return results
    
    # Find all PDF files
    pdf_files = list(pdf_dir_path.glob("*.pdf"))
    results["total"] = len(pdf_files)
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return results
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        try:
            print(f"\nProcessing: {pdf_file.name}")
            markdown_content = await convert_pdf_to_markdown(str(pdf_file), PDF_PROCESSOR)
            
            if markdown_content:
                results["processed"].append(str(pdf_file))
                results["success"] += 1
                results["markdown_content"][str(pdf_file)] = markdown_content
                
                # Save markdown file if output directory is specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    markdown_file = os.path.join(output_dir, f"{pdf_file.stem}.md")
                    with open(markdown_file, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    print(f"Saved markdown: {markdown_file}")
                
                # Add to RAG if provided
                if rag:
                    await rag.ainsert(markdown_content)
                    print(f"Added to RAG: {pdf_file.name}")
                
                print(f"✅ Successfully processed: {pdf_file.name} ({len(markdown_content)} chars)")
            else:
                results["failed"].append(str(pdf_file))
                results["failure"] += 1
                print(f"❌ Failed to process: {pdf_file.name}")
                
        except Exception as e:
            results["failed"].append(str(pdf_file))
            results["failure"] += 1
            print(f"❌ Error processing {pdf_file.name}: {e}")
    
    return results


async def robust_ollama_embed(texts, embed_model, host, max_retries=3, delay=1):
    """
    Robust wrapper for ollama_embed with retry logic and error handling
    """
    for attempt in range(max_retries):
        try:
            # Use longer timeout for embedding operations
            print(f"Embedding processing attempt {attempt + 1} with model {embed_model} on host {host}")
            return await ollama_embed(
                texts, 
                embed_model=embed_model, 
                host=host,
                timeout=EMBEDDING_TIMEOUT,
                options={"num_threads": 11}
            )
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ Embedding failed after {max_retries} attempts: {e}")
                raise
            
            print(f"⚠️  Embedding attempt {attempt + 1} failed: {e}")
            print(f"🔄 Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff


async def robust_ollama_complete(prompt, model, host, options={}, max_retries=3, delay=1, **kwargs):
    """
    Robust wrapper for ollama_model_complete with retry logic and error handling
    """
    options.update({
        "num_threads": 11
    })
    for attempt in range(max_retries):
        try:
            print(f"LLM completion processing attempt {attempt + 1} with model {model} on host {host}")
            return await ollama_model_complete(prompt, model=model, host=host, options=options, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ LLM completion failed after {max_retries} attempts: {e}")
                raise
            
            print(f"⚠️  LLM completion attempt {attempt + 1} failed: {e}")
            print(f"🔄 Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff


async def main():
    """Main function"""
    rag = None
    try:
        processor_name = "MinerU" if PDF_PROCESSOR == "mineru" else "Docling"
        print(f"🚀 LightRAG Enhanced Demo with {processor_name} PDF Processing")
        print("="*60)
        
        # Create directories
        create_directories()
        
        # First, convert all PDFs to Markdown (unless skipped)
        if not SKIP_PDF_CONVERSION:
            print("\n📄 Step 1: Converting PDFs to Markdown and Archiving...")
            pdf_conversion_results = await convert_all_pdfs_to_markdown()
            
            if pdf_conversion_results:
                successful_conversions = len([r for r in pdf_conversion_results.values() if r is not None])
                total_pdfs = len(pdf_conversion_results)
                print(f"✅ PDF conversion completed: {successful_conversions}/{total_pdfs} PDFs converted to Markdown")
                
                if successful_conversions > 0:
                    print("📦 Original PDFs have been archived to processed_docs directory")
            else:
                print("ℹ️  No PDFs found to convert")
        else:
            print("\n⏭️  Step 1: Skipping PDF conversion (SKIP_PDF_CONVERSION=true)")
        
        # Initialize RAG system
        print("\n📡 Step 2: Initializing RAG system...")
        rag = await initialize_rag()
        
        # # Test embedding function
        # test_text = ["This is a test string for embedding."]
        # if rag.embedding_func:
        #     embedding = await rag.embedding_func(test_text)
        #     embedding_dim = embedding.shape[1]
        #     print("\n✅ Embedding function test successful")
        #     print(f"Test text: {test_text}")
        #     print(f"Detected embedding dimension: {embedding_dim}")
        # else:
        #     print("\n❌ Embedding function is not available")
        
        # Process documents from the document directory (now only markdown and text files)
        print(f"\n📚 Step 3: Processing documents from: {DOCUMENT_DIR}")
        results = await process_folder(DOCUMENT_DIR, rag, recursive=True)
        
        print("\n📊 Processing Results:")
        print(f"Total files: {results['total']}")
        print(f"Successfully processed: {results['success']}")
        print(f"Failed: {results['failure']}")
        
        if results['processed']:
            print("\n✅ Successfully processed files:")
            for file in results['processed']:
                print(f"  - {file}")
        
        if results['failed']:
            print("\n❌ Failed to process files:")
            for file in results['failed']:
                print(f"  - {file}")
        
        # Archive processed documents (optional)
        archive_choice = input("\nArchive processed documents? (y/n) [default: n]: ").strip().lower()
        if archive_choice == "y":
            archive_processed_documents()
        
        # Enter interactive query mode
        print("\n🎯 Entering interactive query mode...")
        await interactive_query_mode(rag)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            try:
                await rag.llm_response_cache.index_done_callback()
                await rag.finalize_storages()
                print("\n✅ RAG system properly finalized")
            except Exception as e:
                print(f"Error finalizing RAG system: {e}")


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    
    print("🔧 Starting LightRAG Enhanced Demo with Configurable PDF Processing")
    print("Configuration loaded from environment variables")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"PDF Processor: {PDF_PROCESSOR.upper()}")
    print(f"Working Directory: {WORKING_DIR}")
    print(f"Document Directory: {DOCUMENT_DIR}")
    
    asyncio.run(main())
    print("\n✅ LightRAG Enhanced Demo completed!")
