"""
Individual document knowledge graph management
Rebuilt from scratch based on LightRAGManager and lightrag_ex.py
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

# LightRAG components
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc, set_verbose_debug
    from lightrag.kg.shared_storage import initialize_pipeline_status
except ImportError:
    LightRAG = None
    QueryParam = None
    ollama_model_complete = None
    ollama_embed = None
    EmbeddingFunc = None
    set_verbose_debug = None
    initialize_pipeline_status = None

from ..library import LibraryManager
from ..utils import BookWormConfig


class DocumentKnowledgeGraph:
    """
    Individual document knowledge graph management
    Based on LightRAGManager with document-specific modifications
    """
    
    def __init__(self, config: BookWormConfig, document_id: str, library_manager: Optional[LibraryManager] = None):
        self.config = config
        self.document_id = document_id
        self.logger = logging.getLogger(f"bookworm.doc_kg.{document_id[:8]}")
        self.rag = None  # Will be LightRAG instance after initialization
        self.library_manager = library_manager or LibraryManager(config)
        self.is_initialized = False
        
        # Create document-specific working directory
        self.doc_working_dir = Path("./workspace/knowledge_graphs") / document_id
        self.doc_working_dir.mkdir(parents=True, exist_ok=True)
        
        if not LightRAG:
            raise ImportError("LightRAG not available. Install lightrag-hku.")

    async def _robust_ollama_embed(self, texts, max_retries=3, delay=1):
        """
        Robust wrapper for ollama_embed with retry logic and error handling
        Based on LightRAGManager implementation
        """
        if not ollama_embed:
            raise ImportError("ollama_embed not available")
            
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Step: Embedding attempt {attempt + 1} with model {self.config.embedding_model} (texts: {len(texts)})")
                result = await ollama_embed(
                    texts,
                    embed_model=self.config.embedding_model,
                    host=self.config.embedding_host,
                    timeout=self.config.embedding_timeout,
                    options={}
                )
                self.logger.debug(f"Step: Embedding successful on attempt {attempt + 1}")
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                    raise
                
                self.logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                self.logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

    async def _robust_ollama_complete(self, prompt: str, **kwargs) -> str:
        """
        Robust wrapper for ollama_model_complete with retry logic and error handling
        Similar to _robust_ollama_embed but for LLM completion
        
        Args:
            prompt (str): The prompt to send to the LLM
            **kwargs: Additional arguments like model, host, etc.
            
        Returns:
            str: The completed text from the LLM
            
        Raises:
            Exception: If all retry attempts fail
        """
        if not ollama_model_complete:
            raise ImportError("ollama_model_complete not available")
            
        max_retries = kwargs.pop('max_retries', 3)
        delay = kwargs.pop('delay', 1.0)
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Step: LLM completion attempt {attempt + 1} with model {kwargs.get('model', 'unknown')}")
                
                # Pass all arguments directly to ollama_model_complete 
                result = await ollama_model_complete(prompt=prompt, **kwargs)
                self.logger.debug(f"Step: LLM completion successful on attempt {attempt + 1}")
                return str(result)
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"LLM completion failed after {max_retries} attempts: {e}")
                    raise
                
                self.logger.warning(f"LLM completion attempt {attempt + 1} failed: {e}")
                self.logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

    async def initialize(self) -> None:
        """
        Initialize document-specific LightRAG system
        Based on LightRAGManager.initialize() with document-specific modifications
        """
        if self.is_initialized:
            self.logger.info(f"Document KG {self.document_id[:8]} already initialized")
            return

        if not LightRAG or not ollama_model_complete or not EmbeddingFunc:
            raise ImportError("Required LightRAG components not available")

        self.logger.info(f"Initializing document knowledge graph for {self.document_id[:8]}...")
        self.logger.debug("Step: Configuration dump")
        self.logger.info("Configuration:")
        self.logger.info(f"  Working Directory: {self.doc_working_dir}")
        self.logger.info(f"  LLM Host: {self.config.llm_host}")
        self.logger.info(f"  Embedding Host: {self.config.embedding_host}")
        self.logger.info(f"  LLM Model: {self.config.llm_model}")
        self.logger.info(f"  Embedding Model: {self.config.embedding_model}")
        self.logger.info(f"  Embedding Dimension: {self.config.embedding_dim}")
        self.logger.info(f"  Max Token Size: {self.config.max_embed_tokens}")

        # Enable verbose debug logging if available
        if set_verbose_debug:
            self.logger.debug("Step: Enabling verbose debug for LightRAG")
            set_verbose_debug(True)

        self.logger.debug("Step: Instantiating LightRAG")
        self.rag = LightRAG(
            working_dir=str(self.doc_working_dir),
            llm_model_func=self._robust_ollama_complete,
            llm_model_name=self.config.llm_model,
            # llm_model_max_token_size=22192,
            llm_model_kwargs={
                "host": self.config.llm_host,
                "options": {"num_ctx": 20768},  # Increased context window
                "timeout": self.config.timeout,
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=self.config.embedding_dim,
                max_token_size=8192, #self.config.max_embed_tokens,
                func=lambda texts: self._robust_ollama_embed(texts),
            ),
            vector_storage="FaissVectorDBStorage",
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": 0.3
            },
        )

        self.logger.debug("Step: Initializing LightRAG storages")
        await self.rag.initialize_storages()
        if initialize_pipeline_status:
            self.logger.debug("Step: Initializing pipeline status")
            # Wrap the call to also log the direct print output
            import sys
            from io import StringIO
            _stderr = sys.stderr
            try:
                sys.stderr = StringIO()
                await initialize_pipeline_status()
                sys.stderr.seek(0)
                msg = sys.stderr.read()
                if msg:
                    for line in msg.strip().splitlines():
                        self.logger.info(line)
            finally:
                sys.stderr = _stderr

        self.is_initialized = True
        self.logger.info(f"✅ Document knowledge graph {self.document_id[:8]} initialized successfully")
    
    async def add_document_content(self, content: str, file_paths: Optional[List[str]] = None) -> None:
        """
        Add document content to this specific knowledge graph
        Based on LightRAGManager.add_document()
        """
        if not self.is_initialized:
            await self.initialize()

        if not content:
            self.logger.warning(f"Document {self.document_id[:8]} has no content, skipping insertion.")
            return

        try:
            self.logger.info(f"Adding document content to KG {self.document_id[:8]} (size: {len(content):,} chars)...")
            
            if self.rag:
                self.logger.debug(f"Step: Inserting content into LightRAG (length: {len(content):,} chars)")
                await self.rag.ainsert(content, file_paths=file_paths)
                self.logger.info(f"✅ Successfully added content to document KG {self.document_id[:8]}")
            else:
                self.logger.error("RAG system not initialized")
                raise RuntimeError("RAG system not initialized")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to add content to document KG {self.document_id[:8]}: {e}")
            raise

    async def query(self, question: str, mode: str = "hybrid", stream: bool = False, **kwargs) -> str:
        """
        Query this document's knowledge graph
        Based on LightRAGManager.query()
        """
        if not self.is_initialized:
            await self.initialize()

        if not QueryParam:
            raise ImportError("QueryParam not available")

        # Valid modes from LightRAGManager
        valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
        if mode not in valid_modes:
            self.logger.warning(f"Invalid mode '{mode}'. Using 'hybrid' instead.")
            mode = "hybrid"

        self.logger.info(f"Querying document KG {self.document_id[:8]} in {mode} mode")
        self.logger.debug(f"Question: {question}")

        if self.rag:
            self.logger.debug(f"Step: Building QueryParam for mode '{mode}'")
            # Create QueryParam with explicit type handling to avoid type checker issues
            query_param = None
            if mode == "local":
                query_param = QueryParam(mode="local", stream=stream, **kwargs)
            elif mode == "global":
                query_param = QueryParam(mode="global", stream=stream, **kwargs)
            elif mode == "naive":
                query_param = QueryParam(mode="naive", stream=stream, **kwargs)
            elif mode == "mix":
                query_param = QueryParam(mode="mix", stream=stream, **kwargs)
            elif mode == "bypass":
                query_param = QueryParam(mode="bypass", stream=stream, **kwargs)
            else:  # default to hybrid
                query_param = QueryParam(mode="hybrid", stream=stream, **kwargs)
            self.logger.debug(f"Step: Executing query on LightRAG")
            resp = await self.rag.aquery(question, param=query_param)
            self.logger.debug(f"Step: Query completed, response length: {len(str(resp))}")
            return str(resp)
        else:
            self.logger.error("RAG system not initialized")
            return ""

    async def generate_description(self) -> str:
        """Generate a description of the document by querying its knowledge graph"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Query the knowledge graph to understand the document's content
            description_query = """What is this document about? Provide a concise summary (2-3 sentences) that includes:
1. The main topic or subject matter
2. The type of content (technical, academic, business, etc.)
3. Key themes or areas covered

Be specific and informative."""

            self.logger.info(f"Generating description for document {self.document_id[:8]} from knowledge graph...")

            # Use hybrid mode for best results combining local and global knowledge
            description = await self.query(description_query, mode="hybrid")

            if description and description.strip():
                # Clean up the description
                description = description.strip()

                # Remove any common prefixes the model might add
                prefixes_to_remove = [
                    "this document is about",
                    "this document discusses", 
                    "the document covers",
                    "this text is about",
                    "the content focuses on",
                    "based on the content",
                    "according to the document"
                ]

                description_lower = description.lower()
                for prefix in prefixes_to_remove:
                    if description_lower.startswith(prefix):
                        description = description[len(prefix):].strip()
                        # Capitalize first letter after removing prefix
                        if description:
                            description = description[0].upper() + description[1:]
                        break

                self.logger.info(f"📝 Generated description from KG for document {self.document_id[:8]}: {description[:80]}...")
                return description
            else:
                # Fallback to a simple description
                fallback = f"Document processed and indexed in knowledge graph {self.document_id[:8]}."
                self.logger.warning(f"Empty description from KG, using fallback: {fallback}")
                return fallback

        except Exception as e:
            self.logger.error(f"Failed to generate description from KG for document {self.document_id[:8]}: {e}")
            # Return a basic fallback description
            return f"Document indexed in knowledge graph {self.document_id[:8]} with content analysis available for querying."

    async def generate_tags(self, max_tags: int = 8) -> List[str]:
        """Generate relevant tags for the document by querying its knowledge graph"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Query the knowledge graph to extract key topics and concepts for tags
            tags_query = f"""Based on the content in this document, extract {max_tags} relevant tags or keywords that best categorize and describe the document.

Focus on:
1. Main topics and subject areas
2. Key concepts and terminology
3. Methodologies or approaches mentioned
4. Application domains or industries
5. Technical terms and processes

Provide ONLY a simple comma-separated list of tags (no explanations, no numbers, no bullets).
Examples: "machine learning, neural networks, data science, artificial intelligence, deep learning"

Make the tags specific and useful for categorization."""

            self.logger.info(f"Generating tags for document {self.document_id[:8]} from knowledge graph...")

            # Use hybrid mode for best results
            tags_response = await self.query(tags_query, mode="hybrid")

            if tags_response and tags_response.strip():
                # Clean up the response and extract tags
                tags_text = tags_response.strip()

                # Remove common prefixes that the AI might add
                prefixes_to_remove = [
                    "based on the document",
                    "the main tags are:",
                    "relevant tags:",
                    "keywords:",
                    "tags:",
                    "here are the tags:",
                    "the following tags"
                ]

                tags_lower = tags_text.lower()
                for prefix in prefixes_to_remove:
                    if tags_lower.startswith(prefix):
                        tags_text = tags_text[len(prefix):].strip()
                        if tags_text.startswith(":"):
                            tags_text = tags_text[1:].strip()
                        break

                # Split by comma and clean up each tag
                raw_tags = [tag.strip() for tag in tags_text.split(',')]

                # Clean and validate tags
                clean_tags = []
                for tag in raw_tags:
                    # Remove quotes, periods, and other unwanted characters
                    tag = tag.strip('"\'.,;:!?()[]{}').strip()

                    # Skip empty tags, very short tags, or tags that are too long
                    if tag and 2 <= len(tag) <= 30:
                        # Convert to lowercase for consistency
                        tag = tag.lower()

                        # Skip duplicates
                        if tag not in clean_tags:
                            clean_tags.append(tag)

                    # Stop if we have enough tags
                    if len(clean_tags) >= max_tags:
                        break

                self.logger.info(f"🏷️  Generated {len(clean_tags)} tags from KG for document {self.document_id[:8]}: {clean_tags[:3]}...")
                return clean_tags
            else:
                # Fallback to basic tags based on document type
                fallback_tags = ["document", "processed"]
                self.logger.warning(f"Empty tags response from KG, using fallback: {fallback_tags}")
                return fallback_tags

        except Exception as e:
            self.logger.error(f"Failed to generate tags from KG for document {self.document_id[:8]}: {e}")
            # Return basic fallback tags
            return ["document", "processed"]

    async def cleanup(self):
        """Clean up resources - based on LightRAGManager.cleanup()"""
        if self.rag:
            try:
                # Safely handle cache cleanup
                if hasattr(self.rag, 'llm_response_cache') and hasattr(self.rag.llm_response_cache, 'index_done_callback'):
                    await self.rag.llm_response_cache.index_done_callback()
                
                # Finalize storages
                if hasattr(self.rag, 'finalize_storages'):
                    await self.rag.finalize_storages()
                    
                self.logger.info(f"RAG system properly finalized for document {self.document_id[:8]}")
            except Exception as e:
                self.logger.error(f"Error finalizing RAG system for document {self.document_id[:8]}: {e}")

        self.is_initialized = False
        self.rag = None
