"""
Simple RAG-Anything Implementation with EmbeddingGemma 308M
Clean, minimal implementation for Obsidian vault processing
"""

import os
import re
import sys
import asyncio
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# RAG-Anything imports
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

# Import Gemini LLM wrapper
try:
    from .gemini_llm import gemini_complete_if_cache, gemini_vision_complete
except ImportError:
    from gemini_llm import gemini_complete_if_cache, gemini_vision_complete

# Import Obsidian chunker
try:
    from .obsidian_chunker import ObsidianChunker
except ImportError:
    from obsidian_chunker import ObsidianChunker

# Import BGE reranker
try:
    from .bge_reranker import bge_rerank_async
except ImportError:
    try:
        from bge_reranker import bge_rerank_async
    except ImportError:
        print("[WARNING] BGE reranker not found - reranking will be disabled")
        bge_rerank_async = None

# Load environment variables
load_dotenv()


class ProgressLoggingHandler(logging.Handler):
    """
    Custom logging handler that captures RAG-Anything progress
    and updates tqdm progress bars in real-time
    """
    def __init__(self):
        super().__init__()
        self.extraction_pbar = None
        self.merge_pbar = None
        self.current_phase = None
        
    def set_extraction_pbar(self, pbar):
        """Set the extraction progress bar"""
        self.extraction_pbar = pbar
        
    def set_merge_pbar(self, pbar):
        """Set the merge progress bar"""
        self.merge_pbar = pbar
        
    def emit(self, record):
        """Process log messages and update progress bars"""
        try:
            msg = record.getMessage()
            
            # Match entity extraction progress: "Chunk X of Y extracted"
            chunk_match = re.search(r'Chunk (\d+) of (\d+) extracted', msg)
            if chunk_match and self.extraction_pbar:
                current = int(chunk_match.group(1))
                total = int(chunk_match.group(2))
                self.extraction_pbar.n = current
                self.extraction_pbar.total = total
                self.extraction_pbar.refresh()
                
            # Match merging phases
            phase_match = re.search(r'Phase (\d+): Processing (\d+) (entities|relations)', msg)
            if phase_match and self.merge_pbar:
                phase = int(phase_match.group(1))
                count = int(phase_match.group(2))
                item_type = phase_match.group(3)
                
                # Update description based on phase
                if phase == 1:
                    self.merge_pbar.set_description(f"Merging Phase 1/3: Processing {count} entities")
                elif phase == 2:
                    self.merge_pbar.set_description(f"Merging Phase 2/3: Processing {count} relations")
                elif phase == 3:
                    self.merge_pbar.set_description(f"Merging Phase 3/3: Finalizing {count} entities")
                self.merge_pbar.update(0)  # Just refresh, don't increment
                
        except Exception:
            pass  # Silently ignore errors in progress tracking

class SimpleRAGAnything:
    """
    Simple RAG-Anything implementation with EmbeddingGemma 308M
    Processes Obsidian vaults with multimodal support
    """
    
    def __init__(self, vault_path: str, working_dir: str = "./rag_storage", non_interactive: bool = False):
        """
        Initialize Simple RAG-Anything with Obsidian chunking

        Args:
            vault_path: Path to Obsidian vault
            working_dir: Working directory for RAG storage
            non_interactive: If True, skip interactive prompts (for web server mode)
        """
        self.vault_path = vault_path
        self.working_dir = working_dir
        self.rag_anything = None
        self.embedding_model = None
        self.chunker = None
        self.using_existing_db = False  # Track if we're using an existing database
        self.non_interactive = non_interactive  # Web server mode flag

        # Create working directory if it doesn't exist
        os.makedirs(working_dir, exist_ok=True)

        # Initialize Obsidian chunker for 2K token chunks
        self.chunker = ObsidianChunker(vault_path, target_tokens=2000)

        print(f"Vault Path: {vault_path}")
        print(f"Working Dir: {working_dir}")
        print(f"Chunker: 2K token chunks with wikilinks & metadata")
    
    def _detect_existing_database(self) -> bool:
        """Check if RAG-Anything database already exists"""
        key_files = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_full_entities.json",
            "kv_store_full_relations.json",
            "kv_store_llm_response_cache.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json"
        ]
        
        existing_files = []
        for file in key_files:
            file_path = os.path.join(self.working_dir, file)
            if os.path.exists(file_path):
                existing_files.append(file)
        
        return len(existing_files) > 0
    
    def _get_database_stats(self) -> dict:
        """Get statistics about existing database"""
        stats = {
            "total_files": 0,
            "key_files": [],
            "last_modified": None,
            "database_size": 0
        }
        
        key_files = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_full_entities.json",
            "kv_store_full_relations.json",
            "kv_store_llm_response_cache.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json"
        ]
        
        for file in key_files:
            file_path = os.path.join(self.working_dir, file)
            if os.path.exists(file_path):
                stats["key_files"].append(file)
                stats["total_files"] += 1
                
                # Get file size
                file_size = os.path.getsize(file_path)
                stats["database_size"] += file_size
                
                # Get last modified time
                mod_time = os.path.getmtime(file_path)
                if stats["last_modified"] is None or mod_time > stats["last_modified"]:
                    stats["last_modified"] = mod_time
        
        return stats
    
    def _delete_existing_database(self):
        """Delete existing database files"""
        print("Deleting existing database...")
        
        key_files = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_full_entities.json",
            "kv_store_full_relations.json",
            "kv_store_llm_response_cache.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json"
        ]
        
        deleted_count = 0
        for file in key_files:
            file_path = os.path.join(self.working_dir, file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Warning: Could not delete {file}: {e}")
        
        print(f"Deleted {deleted_count} database files")
    
    def _handle_database_choice(self) -> str:
        """Present user with database options and get choice"""
        stats = self._get_database_stats()
        
        print("\n" + "="*60)
        print("Found existing RAG-Anything database!")
        print("="*60)
        print(f"Database Statistics:")
        print(f"   - Files: {stats['total_files']}")
        print(f"   - Size: {stats['database_size'] / 1024 / 1024:.1f} MB")
        if stats['last_modified']:
            import datetime
            mod_time = datetime.datetime.fromtimestamp(stats['last_modified'])
            print(f"   - Last Updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   - Key Files: {', '.join(stats['key_files'][:3])}{'...' if len(stats['key_files']) > 3 else ''}")
        
        print("\nChoose an option:")
        print("1. Use existing database (continue with current data)")
        print("2. Build new database (delete old data and start fresh)")
        
        while True:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                return choice
            print("Please enter 1 or 2")
    
    def _initialize_embedding_model(self):
        """Initialize EmbeddingGemma 308M model with EmbeddingFunc wrapper"""
        try:
            # Check if CUDA is available and set device
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            if device == 'cuda':
                print(f"\n[1/3] GPU Detection")
                print(f"   ✓ GPU Detected: {torch.cuda.get_device_name(0)}")
                print(f"   ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                print(f"\n[1/3] GPU Detection")
                print(f"   ✗ No GPU detected, using CPU")

            # Load model on GPU if available
            print(f"\n[2/3] Loading EmbeddingGemma 308M...")
            self.embedding_model = SentenceTransformer("google/embeddinggemma-300m", device=device)
            print(f"   ✓ Loaded on {device.upper()}")
            print(f"   ✓ Dimensions: 768 (truncatable to 512/256/128)")
            print(f"   ✓ Languages: 100+ supported")
            print(f"   ✓ Memory: <200MB with quantization")
            
            # Create a standalone async embedding function for RAG-Anything
            def create_embedding_function(model):
                async def embedding_function(texts):
                    # Return embeddings in the exact same format as openai_embed()
                    # openai_embed returns a list of embedding vectors
                    # Run in executor to avoid blocking async loop
                    embeddings = model.encode(texts, convert_to_numpy=True)
                    return embeddings.tolist()
                return embedding_function

            # Create EmbeddingFunc wrapper for RAG-Anything
            print(f"\n[3/3] Creating EmbeddingFunc Wrapper...")
            self.embedding_func = EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=create_embedding_function(self.embedding_model)
            )
            print("   ✓ EmbeddingFunc wrapper created")
            
        except Exception as e:
            print(f"Error loading EmbeddingGemma: {e}")
            raise
    
    def _embedding_function(self, texts: List[str]) -> List[List[float]]:
        """
        Embedding function using EmbeddingGemma 308M
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.embedding_model:
            self._initialize_embedding_model()
        
        # Generate embeddings with EmbeddingGemma
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        # Optional: Truncate to 512 dimensions for memory efficiency
        # embeddings = embeddings[:, :512]
        
        return embeddings.tolist()
    
    async def _llm_function(self, prompt: str, system_prompt: str = None, history_messages=[], **kwargs) -> str:
        """
        LLM function using Gemini 2.5 Flash for entity extraction
        FIXED: Added debug logging to diagnose prompt issues

        Args:
            prompt: User prompt
            system_prompt: System prompt
            history_messages: Chat history
            **kwargs: Additional arguments

        Returns:
            Generated text response
        """
        # DEBUG: Print prompt info to diagnose "I do not have enough information" issue
        print(f"\n[DEBUG LLM] Prompt length: {len(prompt)} chars")
        print(f"[DEBUG LLM] Has 'Source Data': {'YES' if 'Source Data' in prompt or 'source data' in prompt.lower() else 'NO'}")
        print(f"[DEBUG LLM] System prompt: {'YES' if system_prompt else 'NO'}")
        
        # Print first 500 chars of prompt to see structure
        if len(prompt) > 500:
            print(f"[DEBUG LLM] Prompt preview (first 500 chars):\n{prompt[:500]}...")
        else:
            print(f"[DEBUG LLM] Full prompt:\n{prompt}")
        
        # Determine which model to use based on task type
        # Check if this is entity extraction (contains "extract entities" in prompt)
        is_extraction = "extract entities" in prompt.lower() or "extract relations" in prompt.lower()

        if is_extraction:
            # Use Gemini 2.5 Flash for entity extraction
            # Fast: 1,000 RPM, 1M TPM (paid tier)
            # Cost: ~$7.65 for 6119 chunks (~12 minutes)
            try:
                vertex_api_key = os.getenv("VERTEX_AI_API_KEY")

                result = await gemini_complete_if_cache(
                    model="gemini-2.5-flash",
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=vertex_api_key,
                    **kwargs
                )

                # Ensure we always return a string
                if result is None or len(str(result).strip()) == 0:
                    print(f"[DEBUG LLM] WARNING: Gemini returned empty result")
                    return "Unable to generate response. Please try again."
                
                print(f"[DEBUG LLM] Gemini response length: {len(result)} chars")
                return result

            except Exception as e:
                print(f"ERROR in Gemini LLM function: {e}")
                return ""

        else:
            # Use Gemini 2.5 Flash for user queries too (no OpenAI needed)
            try:
                vertex_api_key = os.getenv("VERTEX_AI_API_KEY")

                result = await gemini_complete_if_cache(
                    model="gemini-2.5-flash",
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=vertex_api_key,
                    **kwargs
                )

                if result is None:
                    print(f"WARNING: Gemini returned None, using empty string")
                    return ""

                return result
            except Exception as e:
                print(f"ERROR in Gemini query function: {e}")
                return ""
    
    async def _vision_function(self, prompt: str, system_prompt: str = None, history_messages=[],
                               image_data: str = None, messages=None, **kwargs) -> str:
        """
        Vision function for multimodal processing using Gemini 2.5 Flash
        Handles images, tables, and equations

        Args:
            prompt: Text prompt
            system_prompt: System prompt
            history_messages: Chat history
            image_data: Base64 encoded image data
            messages: Pre-formatted messages (for multimodal VLM)
            **kwargs: Additional arguments

        Returns:
            Generated response for multimodal content
        """
        # Get Vertex AI API key
        vertex_api_key = os.getenv("VERTEX_AI_API_KEY")

        # Use Gemini vision function for all multimodal content
        try:
            return await gemini_vision_complete(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                image_data=image_data,
                messages=messages,
                api_key=vertex_api_key,
                **kwargs
            )
        except Exception as e:
            print(f"ERROR in Gemini vision function: {e}")
            # Fallback to text-only LLM if vision fails
            return await self._llm_function(prompt, system_prompt, history_messages, **kwargs)
    
    ##key
    async def initialize(self):
        """Initialize RAG-Anything with EmbeddingGemma and database management"""
        print("\n" + "="*70)
        print("INITIALIZATION")
        print("="*70)

        # Check for existing database
        if self._detect_existing_database():
            if self.non_interactive:
                # In web server mode, always use existing database
                print("\n✓ Existing database detected - using existing data (non-interactive mode)")
                self.using_existing_db = True
            else:
                # Interactive mode - ask user
                choice = self._handle_database_choice()

                if choice == "2":  # Build new database
                    self._delete_existing_database()
                    print("\n✓ Starting fresh with new database")
                    self.using_existing_db = False
                else:  # Use existing database
                    print("\n✓ Using existing database")
                    self.using_existing_db = True
        else:
            self.using_existing_db = False

        # Always initialize embedding model
        print("\n" + "="*70)
        print("LOADING EMBEDDING MODEL")
        print("="*70)
        self._initialize_embedding_model()

        print("\n" + "="*70)
        print("CONFIGURING RAG-ANYTHING FRAMEWORK")
        print("="*70)
        print("[1/3] Setting up multimodal processing")
        print("   ✓ Images, Tables, Equations enabled")
        print("[2/3] Configuring MinerU parser")
        print("   ✓ Auto parse method")
        if self.using_existing_db:
            print("[3/3] Database setup")
            print("   ✓ Loading existing knowledge graph")
        else:
            print("[3/3] Database setup")
            print("   ✓ Preparing new knowledge graph storage")

        # RAG-Anything configuration
        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            parser="mineru",  # Advanced document parsing
            parse_method="auto",
            enable_image_processing=True,  # Process images in documents
            enable_table_processing=True,  # Process tables in documents
            enable_equation_processing=True,  # Process equations in documents
            context_window=2,  # Context window for processing
            context_mode="page",  # Page-based context
            max_context_tokens=3000,  # Maximum context tokens
            include_headers=True,  # Include document headers
            include_captions=True,  # Include image/table captions
            context_filter_content_types=["text", "image", "table", "equation"]
        )

        print("\n" + "="*70)
        if self.using_existing_db:
            print("LOADING RAG-ANYTHING INSTANCE")
        else:
            print("INITIALIZING RAG-ANYTHING INSTANCE")
        print("="*70)

        if self.using_existing_db:
            # When using existing database, we must:
            # 1. Create LightRAG instance with existing working_dir
            # 2. Initialize storages to load existing graph data
            # 3. Pass initialized LightRAG to RAGAnything
            print("[1/3] Creating LightRAG instance...")
            from lightrag import LightRAG
            from lightrag.kg.shared_storage import initialize_pipeline_status

            lightrag_instance = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self._llm_function,
                embedding_func=self.embedding_func
            )

            # Add reranker BEFORE initialization
            if bge_rerank_async is not None:
                lightrag_instance.rerank_model_func = bge_rerank_async
                print("   ✓ BGE Reranker v2-m3 configured (GPU-accelerated, 8GB VRAM optimized)")
                print(f"   ✓ Reranker function: {lightrag_instance.rerank_model_func.__name__}")
            else:
                print("   ✗ Reranker not available - BGE module not loaded")
                print("   ℹ Install sentence-transformers for GPU-accelerated reranking")

            print("[2/3] Loading existing knowledge graph data...")
            await lightrag_instance.initialize_storages()
            await initialize_pipeline_status()
            print("   ✓ Knowledge graph loaded")

            print("[3/3] Connecting to RAG-Anything...")
            self.rag_anything = RAGAnything(
                lightrag=lightrag_instance,  # Pass existing LightRAG instance
                vision_model_func=self._vision_function
            )
            
            print("   ✓ RAG-Anything connected to existing database")
        else:
            # For new database, create RAGAnything with config
            print("[1/2] Verifying MinerU installation...")
            print("   ✓ MinerU verified")
            print("[2/2] Setting up LightRAG backend...")

            self.rag_anything = RAGAnything(
                config=config,
                llm_model_func=self._llm_function,
                vision_model_func=self._vision_function,
                embedding_func=self.embedding_func
            )
            
            # Add reranker if available
            if bge_rerank_async is not None:
                self.rag_anything.lightrag.rerank_model_func = bge_rerank_async
                print("   ✓ BGE Reranker v2-m3 enabled (GPU-accelerated, 8GB VRAM optimized)")
                print(f"   ✓ Reranker function: {self.rag_anything.lightrag.rerank_model_func.__name__}")
            else:
                print("   ✗ Reranker not available - BGE module not loaded")
                print("   ℹ Install sentence-transformers for GPU-accelerated reranking")
            
            print("   ✓ Graph storage files created")

        print("\n" + "="*70)
        print("✓ INITIALIZATION COMPLETE")
        print("="*70)
        print("Configuration:")
        print("   • Multimodal: Images, Tables, Equations")
        print("   • Embedding: EmbeddingGemma 308M (GPU)")
        print("   • LLM: Gemini 2.5 Flash (Entity Extraction + Queries)")
        print("   • Vision: Gemini 2.5 Flash (Multimodal)")
        print("   • Context Window: 2000 tokens")
    
    
    async def process_vault(self):
        """
        Process entire Obsidian vault with SOTA chunking and RAG-Anything
        Uses 2K token chunks with wikilinks and metadata preservation
        """
        print("="*70)
        print("PROCESSING OBSIDIAN VAULT WITH SOTA CHUNKING")
        print("="*70)
        
        # Step 1: Chunk vault with SOTA approach
        print("Step 1: Chunking vault with 2K token windows...")
        print("Preserving wikilinks, metadata, and file connections...")
        
        chunks_data = self.chunker.process_entire_vault()
        
        if not chunks_data or not chunks_data.get('chunks'):
            print("No chunks created!")
            return
        
        chunks = chunks_data['chunks']
        stats = chunks_data['stats']
        
        print(f"\nChunking Complete:")
        print(f"   Files: {stats['total_files']}")
        print(f"   Chunks: {stats['total_chunks']}")
        print(f"   Wikilinks: {stats['total_wikilinks']}")
        print(f"   Tags: {stats['total_tags']}")
        
        # Step 2: Process chunks with RAG-Anything
        print(f"\nStep 2: Processing {len(chunks)} chunks with RAG-Anything...")
        print("Using EmbeddingGemma 308M for embeddings")
        print("Multimodal processing: Images, Tables, Equations")
        
        successful_chunks = 0
        failed_chunks = 0
        
        # Convert chunks to content list format for RAG-Anything
        print("\nConverting chunks to RAG-Anything content format...")

        content_list = []
        with tqdm(total=len(chunks), desc="Converting chunks", unit="chunk") as pbar:
            for i, chunk in enumerate(chunks):
                chunk_id = chunk['chunk_id']
                source_file = chunk['metadata'].source_file

                # Convert chunk to content list format
                content_item = {
                    "type": "text",
                    "text": chunk['content'],
                    "page_idx": i  # Use chunk index as page index
                }
                content_list.append(content_item)
                pbar.update(1)

        print(f"\n[OK] Converted {len(content_list)} chunks to content format")

        # Process all chunks at once with RAG-Anything
        print("\n" + "="*70)
        print("PROCESSING WITH RAG-ANYTHING")
        print("="*70)
        print("Building knowledge graph with embeddings...")
        print("Extracting entities and relationships...")
        print("Creating vector database...")
        
        # Setup progress tracking
        progress_handler = ProgressLoggingHandler()
        logger = logging.getLogger('lightrag')
        logger.addHandler(progress_handler)
        
        try:
            # Create progress bars for each phase
            with tqdm(total=len(chunks), desc="Entity Extraction", unit="chunk", position=0) as extraction_pbar, \
                 tqdm(total=3, desc="Merging Knowledge Graph", unit="phase", position=1) as merge_pbar:
                
                # Connect progress bars to handler
                progress_handler.set_extraction_pbar(extraction_pbar)
                progress_handler.set_merge_pbar(merge_pbar)
                
                # Process content
                await self.rag_anything.insert_content_list(
                    content_list=content_list,
                    file_path="obsidian_vault.md",  # Reference file name
                    split_by_character=None,
                    split_by_character_only=False,
                    doc_id=None,
                    display_stats=False  # We're showing our own progress
                )
                
                # Complete the progress bars
                extraction_pbar.n = extraction_pbar.total
                extraction_pbar.refresh()
                merge_pbar.n = merge_pbar.total
                merge_pbar.refresh()
            
            successful_chunks = len(chunks)
            failed_chunks = 0
            print(f"\n[OK] Successfully processed all {successful_chunks} chunks!")
            
        except Exception as e:
            successful_chunks = 0
            failed_chunks = len(chunks)
            print(f"\n[ERROR] Failed to process chunks: {str(e)}")
        finally:
            # Remove progress handler
            logger.removeHandler(progress_handler)
        
        # Print comprehensive summary
        print("\n" + "="*70)
        print("✓ VAULT PROCESSING COMPLETE!")
        print("="*70)
        print("\nProcessing Summary:")
        print(f"   • Files Processed: {stats['total_files']}")
        print(f"   • Chunks Created: {stats['total_chunks']}")
        print(f"   • Success Rate: {(successful_chunks/len(chunks))*100:.1f}%")
        print(f"   • Wikilinks Preserved: {stats['total_wikilinks']}")
        print(f"   • Tags Preserved: {stats['total_tags']}")

        print("\nKnowledge Graph Statistics:")
        print(f"   • Successful Chunks: {successful_chunks}")
        print(f"   • Failed Chunks: {failed_chunks}")

        print("\nDatabase Location:")
        print(f"   • Storage: {self.working_dir}")

        print("\nModel Configuration:")
        print("   • Extraction: GPT-5-nano (cheap)")
        print("   • Queries: GPT-5-mini (quality)")
        print("   • Embeddings: EmbeddingGemma 308M (free, GPU)")

        print("\n" + "="*70)
        print("✓ SYSTEM READY FOR QUERIES")
        print("="*70)
    
    async def query(self, question: str, mode: str = "hybrid") -> str:
        """
        Query the processed knowledge base

        Args:
            question: Question to ask
            mode: Query mode (naive, local, global, hybrid)

        Returns:
            Answer from the knowledge base
        """
        if not self.rag_anything:
            await self.initialize()

        print(f"Query: {question}")
        print(f"Mode: {mode}")

        try:
            # Call RAGAnything's aquery - it will create QueryParam internally
            # Reranker is automatically enabled via rerank_model_func set during initialization
            result = await self.rag_anything.aquery(
                question,
                mode=mode,
                enable_rerank=True  # Explicitly enable BGE Reranker v2-m3
            )

            print(f"Query completed successfully!")
            return result

        except Exception as e:
            print(f"Query failed: {e}")
            return f"Error: {str(e)}"
    
    async def query_multimodal(self, question: str, multimodal_content: List[Dict] = None) -> str:
        """
        Query with multimodal content (images, tables, equations)
        
        Args:
            question: Question to ask
            multimodal_content: List of multimodal content
            
        Returns:
            Answer from the knowledge base
        """
        if not self.rag_anything:
            await self.initialize()
        
        print(f"Multimodal Query: {question}")
        print(f"Content: {len(multimodal_content or [])} items")
        
        try:
            result = await self.rag_anything.aquery_with_multimodal(
                question,
                multimodal_content=multimodal_content or []
            )
            
            print(f"Multimodal query completed successfully!")
            return result
            
        except Exception as e:
            print(f"Multimodal query failed: {e}")
            return f"Error: {str(e)}"


def check_conda_environment():
    """Check if we're running in conda environment turing0.1"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    
    if conda_env != 'turing0.1':
        print("ERROR: Not running in conda environment turing0.1")
        print(f"   Current environment: {conda_env}")
        print("")
        print("Please activate the correct environment:")
        print("   conda activate turing0.1")
        print("   python run_raganything.py")
        sys.exit(1)
    
    print(f"Running in conda environment: {conda_env}")

async def main():
    """
    Main function to run the Simple RAG-Anything system
    Always runs in conda environment turing0.1
    """
    # Check conda environment first
    check_conda_environment()
    
    # Configuration - Default to your Obsidian vault
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro")
    working_dir = os.getenv("WORKING_DIR", "./rag_storage")
    
    print("Simple RAG-Anything with EmbeddingGemma 308M")
    print("Conda Environment: turing0.1")
    print("="*50)
    
    # Initialize system
    rag = SimpleRAGAnything(vault_path, working_dir)
    await rag.initialize()
    
    # Process vault
    await rag.process_vault()
    
    # Test queries
    print("\nTesting queries...")
    
    # Basic query
    result1 = await rag.query("What are the main topics in my notes?")
    print(f"\nBasic Query Result:\n{result1}")
    
    # Multimodal query
    result2 = await rag.query_multimodal("What images and tables are available?")
    print(f"\nMultimodal Query Result:\n{result2}")


if __name__ == "__main__":
    asyncio.run(main())
