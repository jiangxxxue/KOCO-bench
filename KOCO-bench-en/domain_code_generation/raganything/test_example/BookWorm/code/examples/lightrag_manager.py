"""
LightRAG Manager - A Python class wrapper for LightRAG functionality
Provides an easy-to-use interface for document processing and querying
"""

import asyncio
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Union, Literal
import tempfile

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
# from lightrag.rerank import custom_rerank, RerankModel

from dotenv import load_dotenv


class LightRAGManager:
    """
    A comprehensive wrapper class for LightRAG functionality with document processing capabilities
    """
    
    def __init__(
        self,
        working_dir: str = "./lightrag_workspace",
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_host: Optional[str] = None,
        embedding_host: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        max_token_size: Optional[int] = None,
        timeout: Optional[int] = None,
        embedding_timeout: Optional[int] = None,
        load_env: bool = True,
        env_file: str = ".env",
        verbose_debug: bool = False,
        **kwargs
    ):
        """
        Initialize LightRAG Manager
        
        Args:
            working_dir: Directory for LightRAG storage
            llm_model: LLM model name (defaults to env var or qwen2.5-coder:7b)
            embedding_model: Embedding model name (defaults to env var or bge-m3:latest)
            llm_host: LLM host URL (defaults to env var or http://localhost:11434)
            embedding_host: Embedding host URL (defaults to env var or http://localhost:11434)
            embedding_dim: Embedding dimension (defaults to env var or 1024)
            max_token_size: Max token size for embeddings (defaults to env var or 8192)
            timeout: LLM timeout in seconds (defaults to env var or 3000)
            embedding_timeout: Embedding timeout in seconds (defaults to env var or 6000)
            load_env: Whether to load environment variables
            env_file: Path to .env file
            verbose_debug: Enable verbose debug logging
            **kwargs: Additional configuration options
        """
        # Load environment variables if requested
        if load_env:
            load_dotenv(dotenv_path=env_file, override=False)
        
        # Configuration
        self.working_dir = working_dir
        self.document_dir = os.path.join(working_dir, "docs")
        self.document_archive_dir = os.path.join(working_dir, "processed_docs")
        self.output_dir = os.path.join(working_dir, "output")
        
        # Model configuration with fallbacks
        # self.llm_model = llm_model or os.getenv("LLM_MODEL", "deepseek-r1:latest")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "qwen2.5-coder:32b")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
        self.embedding_dim = embedding_dim or int(os.getenv("EMBEDDING_DIM", "1024"))
        self.max_token_size = max_token_size or int(os.getenv("MAX_EMBED_TOKENS", "8192"))
        
        # Host configuration with fallbacks
        self.llm_host = llm_host or os.getenv("LLM_BINDING_HOST", "http://100.95.157.120:11434") #"http://localhost:11434")
        self.embedding_host = embedding_host or os.getenv("EMBEDDING_BINDING_HOST", "http://100.95.157.120:11434")
        self.timeout = timeout or int(os.getenv("TIMEOUT", "3000"))
        self.embedding_timeout = embedding_timeout or int(os.getenv("EMBEDDING_TIMEOUT", "6000"))

        # self.rerank_model = RerankModel(
        #     rerank_func=custom_rerank,
        #     kwargs={
        #         "model": "xitao/bge-reranker-v2-m3:latest",
        #         "base_url": self.llm_host,
        #         "api_key": "ollama",
        #     },
        # )
        
        # Supported file extensions
        self.supported_extensions = [".pdf", ".txt", ".md", ".docx", ".pptx"]
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        if verbose_debug:
            set_verbose_debug(True)
        
        # RAG instance (will be initialized later)
        self.rag: Optional[LightRAG] = None
        self.is_initialized = False
        
        # Store additional kwargs for RAG initialization
        self.additional_kwargs = kwargs
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [self.working_dir, self.document_dir, self.document_archive_dir, self.output_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created/verified directory: {directory}")
    
    async def initialize(self):
        """Initialize the LightRAG system"""
        if self.is_initialized:
            self.logger.info("LightRAG Manager already initialized")
            return
        
        self.logger.info("Initializing LightRAG Manager...")
        
        # Create directories
        self.create_directories()
        
        # Debug configuration
        self.logger.info("Configuration:")
        self.logger.info(f"  Working Directory: {self.working_dir}")
        self.logger.info(f"  LLM Host: {self.llm_host}")
        self.logger.info(f"  Embedding Host: {self.embedding_host}")
        self.logger.info(f"  LLM Model: {self.llm_model}")
        self.logger.info(f"  Embedding Model: {self.embedding_model}")
        self.logger.info(f"  Embedding Dimension: {self.embedding_dim}")
        self.logger.info(f"  Max Token Size: {self.max_token_size}")

        
        
        # Initialize RAG system
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=self.llm_model,
            llm_model_max_token_size=18192,
            llm_model_kwargs={
                "host": self.llm_host,
                "options": {"num_ctx": 18192, "num_threads": 11},
                "timeout": self.timeout,
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=self.max_token_size,
                func=lambda texts: self._robust_ollama_embed(
                    texts,
                    embed_model=self.embedding_model,
                    host=self.embedding_host
                ),
            ),
            # rerank_model_func=rerank_model.rerank,
            vector_storage="FaissVectorDBStorage",
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": 0.3
            },
            **self.additional_kwargs
        )
        
        await self.rag.initialize_storages()
        await initialize_pipeline_status()
        
        self.is_initialized = True
        self.logger.info("LightRAG Manager initialized successfully")
    
    async def _robust_ollama_embed(self, texts, embed_model, host, max_retries=3, delay=1):
        """Robust wrapper for ollama_embed with retry logic and error handling"""
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Embedding processing attempt {attempt + 1} with model {embed_model}")
                return await ollama_embed(
                    texts, 
                    embed_model=embed_model, 
                    host=host,
                    timeout=self.embedding_timeout,
                    options={"num_threads": 11}
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                    raise
                
                self.logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                self.logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
    
    async def add_document(self, content: str, file_path: Optional[str] = None) -> bool:
        """
        Add a document to the RAG system
        
        Args:
            content: Document content as string
            file_path: Optional file path for reference
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            file_paths = [file_path] if file_path else None
            if self.rag:
                await self.rag.ainsert(content, file_paths=file_paths)
                self.logger.info(f"Successfully added document: {file_path or 'content'}")
                return True
            else:
                self.logger.error("RAG system not initialized")
                return False
        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            return False
    
    async def add_documents(self, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Add multiple documents to the RAG system
        
        Args:
            documents: List of dictionaries with 'content' and optional 'file_path' keys
            
        Returns:
            Dict with processing results
        """
        if not self.is_initialized:
            await self.initialize()
        
        results = {"success": 0, "failure": 0, "processed": [], "failed": []}
        
        for doc in documents:
            content = doc.get("content", "")
            file_path = doc.get("file_path")
            
            if not content:
                results["failed"].append(file_path or "empty_content")
                results["failure"] += 1
                continue
            
            success = await self.add_document(content, file_path)
            if success:
                results["processed"].append(file_path or "content")
                results["success"] += 1
            else:
                results["failed"].append(file_path or "content")
                results["failure"] += 1
        
        return results
    
    async def process_file(self, file_path: str) -> bool:
        """
        Process a single file and add it to the RAG system
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            self.logger.info(f"Processing file: {file_path}")
            
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension not in self.supported_extensions:
                self.logger.warning(f"Unsupported file format: {file_extension}")
                return False
            
            # Read file content based on extension
            if file_extension in [".txt", ".md"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            elif file_extension == ".pdf":
                # Convert PDF to Markdown using MinerU
                self.logger.info(f"Converting PDF to Markdown: {file_path}")
                content = await self._convert_pdf_to_markdown(file_path)
                if not content:
                    self.logger.error(f"Failed to convert PDF to Markdown: {file_path}")
                    return False
            else:
                self.logger.warning(f"Unsupported file format: {file_extension}")
                return False
            
            # Add to RAG system
            success = await self.add_document(content, file_path)
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return False
    
    async def process_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Process all supported files in a directory
        
        Args:
            directory_path: Path to the directory to process
            recursive: Whether to process subdirectories recursively
            
        Returns:
            Dict with processing results
        """
        if not self.is_initialized:
            await self.initialize()
        
        results = {"processed": [], "failed": [], "total": 0, "success": 0, "failure": 0}
        
        directory_path_obj = Path(directory_path)
        
        if not directory_path_obj.exists():
            self.logger.error(f"Directory does not exist: {directory_path}")
            return results
        
        # Find all supported files
        files_to_process = []
        for ext in self.supported_extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            files_to_process.extend(directory_path_obj.glob(pattern))
        
        results["total"] = len(files_to_process)
        
        if not files_to_process:
            self.logger.info(f"No supported files found in {directory_path}")
            return results
        
        self.logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process each file
        for file_path in files_to_process:
            success = await self.process_file(str(file_path))
            if success:
                results["processed"].append(str(file_path))
                results["success"] += 1
            else:
                results["failed"].append(str(file_path))
                results["failure"] += 1
        
        return results
    
    async def query(
        self, 
        question: str, 
        mode: str = "hybrid", 
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: Question to ask
            mode: Query mode (local, global, hybrid, naive, mix, bypass)
            stream: Whether to return streaming response
            **kwargs: Additional query parameters
            
        Returns:
            Response from the RAG system
        """
        """Specifies the retrieval mode:
        - "local": Focuses on context-dependent information.
        - "global": Utilizes global knowledge.
        - "hybrid": Combines local and global retrieval methods.
        - "naive": Performs a basic search without advanced techniques.
        - "mix": Integrates knowledge graph and vector retrieval.
        """
        if not self.is_initialized:
            await self.initialize()
        
        valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
        if mode not in valid_modes:
            self.logger.warning(f"Invalid mode '{mode}'. Using 'hybrid' instead.")
            mode = "hybrid"
        
        try:
            self.logger.info(f"Querying with mode: {mode}")
            self.logger.debug(f"Question: {question}")
            
            if self.rag:
                resp = await self.rag.aquery(
                    question,
                    param=QueryParam(mode=mode, stream=stream, **kwargs),  # type: ignore
                )
                return resp
            else:
                self.logger.error("RAG system not initialized")
                return None
            
        except Exception as e:
            self.logger.error(f"Error during query: {e}")
            return None
    
    async def _convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert PDF file to Markdown format using MinerU"""
        try:
            # Import MinerU functions locally
            from mineru.cli.common import prepare_env, read_fn
            from mineru.data.data_reader_writer import FileBasedDataWriter
            from mineru.utils.enum_class import MakeMode
            from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
            from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
            from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
            
            # Force CPU usage
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_file_name = Path(pdf_path).stem
                local_image_dir, local_md_dir = prepare_env(temp_dir, pdf_file_name, "auto")
                
                # Read PDF file
                pdf_bytes = read_fn(pdf_path)
                if not pdf_bytes:
                    self.logger.error(f"Failed to read PDF file: {pdf_path}")
                    return ""
                
                # Process PDF using MinerU pipeline
                pdf_bytes_list = [pdf_bytes]
                p_lang_list = ["en"]
                
                # Analyze document
                infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                    pdf_bytes_list, p_lang_list, parse_method="auto", formula_enable=True, table_enable=True
                )
                
                if not infer_results:
                    self.logger.error(f"Failed to analyze PDF: {pdf_path}")
                    return ""
                
                # Process the document
                model_list = infer_results[0]
                images_list = all_image_lists[0]
                pdf_doc = all_pdf_docs[0]
                _lang = lang_list[0]
                _ocr_enable = ocr_enabled_list[0]
                
                # Create data writer
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
            self.logger.error(f"Error converting PDF to Markdown: {e}")
            return ""
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "working_dir": self.working_dir,
            "document_dir": self.document_dir,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "is_initialized": self.is_initialized,
            "storage_files": {}
        }
        
        # Check storage files
        storage_files = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]
        
        for file in storage_files:
            file_path = os.path.join(self.working_dir, file)
            exists = os.path.exists(file_path)
            size = os.path.getsize(file_path) if exists else 0
            stats["storage_files"][file] = {"exists": exists, "size": size}
        
        return stats
    
    async def cleanup(self):
        """Clean up resources"""
        if self.rag:
            try:
                await self.rag.llm_response_cache.index_done_callback()
                await self.rag.finalize_storages()
                self.logger.info("RAG system properly finalized")
            except Exception as e:
                self.logger.error(f"Error finalizing RAG system: {e}")
        
        self.is_initialized = False
        self.rag = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()


# Example usage and convenience functions
async def create_rag_manager(working_dir: str = "./lightrag_workspace", **kwargs) -> LightRAGManager:
    """
    Convenience function to create and initialize a LightRAG Manager
    
    Args:
        working_dir: Working directory for LightRAG
        **kwargs: Additional configuration options
        
    Returns:
        Initialized LightRAGManager instance
    """
    manager = LightRAGManager(working_dir=working_dir, **kwargs)
    await manager.initialize()
    return manager


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        # Create and initialize manager
        async with LightRAGManager(working_dir="./test_workspace") as manager:
            # Add a document
            # await manager.add_document("This is a test document about Python programming.")
            
            # Process a file
            # await manager.process_file("./example.pdf")
            
            # Process a directory
            # results = await manager.process_directory("./documents")
            
            # Query the system
            # response = await manager.query("What is this document about?")
            # print(f"Response: {response}")
            
            # Get statistics
            stats = await manager.get_stats()
            print(f"Stats: {stats}")
    
    asyncio.run(example_usage())
