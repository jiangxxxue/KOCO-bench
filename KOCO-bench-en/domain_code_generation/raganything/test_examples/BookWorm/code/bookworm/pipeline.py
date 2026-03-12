#!/usr/bin/env python3
"""
BookWorm Document Processing System - Refactored Version

Each document creates its own knowledge graph for better isolation and scalability.
No hardcoded sample data - processes real documents only.
Now supports directory processing for Obsidian vaults and document collections.

Updated to use the new modular architecture:
- processors: DocumentProcessor
- knowledge: KnowledgeGraph  
- generators: MindmapGenerator
"""
import asyncio
from openai import AsyncOpenAI
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# PDF conversion imports will be loaded dynamically as needed

from .utils import BookWormConfig, load_config, setup_logging
from .processors import DocumentProcessor
from .knowledge import KnowledgeGraph
from chartographer import MindmapGenerator
from .library import LibraryManager, DocumentStatus

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookWormPipeline:
    """Main pipeline class to orchestrate document processing"""
    
    def __init__(self, config: BookWormConfig):
        self.config = config
        self.library_manager = LibraryManager(config)
        self.processor = DocumentProcessor(config, self.library_manager)
        # self.knowledge_graph = KnowledgeGraph(config, self.library_manager)
        # self.mindmap_generator = MindmapGenerator(config, self.library_manager)

        # Initialize Ollama client (same pattern as mindmap generator)
        self.ollama_client = None
        try:
            if AsyncOpenAI and self.config.api_provider == "OLLAMA":
                ollama_url = f"{self.config.llm_host}/v1"
                self.ollama_client = AsyncOpenAI(
                    api_key="ollama",  # Ollama doesn't require a real API key
                    base_url=ollama_url
                )
        except ImportError:
            logger.warning("OpenAI client not available for Ollama integration")

    def blank_knowledge(self):
        """Create a blank knowledge graph for testing"""
        self.knowledge_graph = KnowledgeGraph(self.config, self.library_manager)
        self.mindmap_generator = MindmapGenerator(self.config)
        return { "graph": self.knowledge_graph, "mindmap": self.mindmap_generator }
    
    async def convert_and_archive_pdfs(self):
        """Convert and archive PDFs before processing"""
        # Placeholder - actual implementation would go here
        print("\n⏭️  Skipping PDF conversion (not implemented in this version)")
        
    async def scan_workspace(self) -> Tuple[List[Path], List[Path]]:
        """Scan workspace for documents and directories to process"""
        docs_to_process = []
        dirs_to_process = []
        workspace_path = Path(self.config.working_dir)
        docs_dir = workspace_path / "docs"
        
        # Only check the docs directory to avoid duplicates
        if docs_dir.exists():
            for item_path in docs_dir.iterdir():
                if item_path.is_file() and item_path.suffix.lower() in ['.txt', '.md', '.pdf', '.docx']:
                    # Skip if it's in processed directories
                    if 'processed' not in str(item_path) and 'output' not in str(item_path):
                        docs_to_process.append(item_path)
                elif item_path.is_dir() and not item_path.name.startswith('.'):
                    # Check if directory contains supported files
                    has_supported_files = any(
                        f.is_file() and f.suffix.lower() in ['.txt', '.md', '.pdf', '.docx']
                        for f in item_path.rglob("*")
                    )
                    if has_supported_files:
                        dirs_to_process.append(item_path)
        
        return docs_to_process, dirs_to_process
    
    async def process_single_document(self, file_path: Path) -> Optional[Dict]:
        """Process a single document"""
        self.blank_knowledge()

        try:
            # Check if document is already processed in library
            existing_doc = self.library_manager.get_document_by_filename(file_path.name)
            if existing_doc and existing_doc.status == DocumentStatus.PROCESSED:
                print(f"    ⏭️  Already processed, skipping: {file_path.name}")
                return None
            
            # Extract text from document
            processed_doc = await self.processor.process_document(file_path)
            
            if not processed_doc:
                print(f"    ❌ Failed to extract text from: {file_path.name}")
                return None
                
            print(f"    📝 Text extracted: {len(processed_doc.text_content):,} characters")
            
            # Create individual knowledge graph for this document
            print("    🧠 Creating knowledge graph...")
            doc_kg, library_doc_id = await self.knowledge_graph.create_document_graph(processed_doc)
            print(f"    ✅ Knowledge graph created: {processed_doc.id[:8]}")
            
            
            
            # Generate mindmap
            print("    🗺️  Generating mindmap visualization...")
            try:
                mindmap_result = await self.mindmap_generator.generate_mindmap_from_file(file_path, output_dir = f"{self.config.working_dir}/mindmaps/{library_doc_id}", save_files=True)
                print(f"    ✅ Mindmap generated (tokens: {mindmap_result.token_usage.total_cost:,})")
                self.library_manager.add_mindmap(processed_doc.id, mindmap_result, {})
            except Exception as mindmap_error:
                print(f"    ⚠️  Mindmap generation failed: {mindmap_error}")

            # Generate AI description from knowledge graph
            print("    📄 Generating description from knowledge graph...")
            description = await self._generate_description(mindmap_result.markdown)
            
            # Generate tags from knowledge graph
            print("    🏷️  Generating tags from knowledge graph...")
            tags = await self._generate_tags(mindmap_result.markdown)
            
            # Update library with description and tags
            if library_doc_id:
                try:
                    metadata_update = {}
                    if description:
                        metadata_update["description"] = description
                    if tags:
                        metadata_update["tags"] = tags
                    
                    if metadata_update:
                        self.library_manager.update_document_metadata(library_doc_id, metadata_update)
                        print("    📝 Description and tags saved to library")
                except Exception as e:
                    print(f"    ⚠️  Failed to save metadata: {e}")
            
            print(f"    ✅ Completed: {file_path.name}")
            return {
                'document_id': processed_doc.id,
                'knowledge_graph': doc_kg,
                'library_id': library_doc_id,
                'metadata': {
                    'description': description,
                    'tags': tags
                }
            }
                
        except Exception as e:
            print(f"    ❌ Error processing {file_path.name}: {e}")
            return None
    
    async def process_directory_collection(self, directory_path: Path) -> Optional[Dict]:
        """Process an entire directory as a single document"""
        print(f"\ud83d\udcc1 Processing directory collection: {directory_path.name}")
        self.blank_knowledge()
        try:
            # Check if directory is already processed in library
            existing_doc = self.library_manager.get_document_by_filename(directory_path.name)
            if existing_doc and existing_doc.status == DocumentStatus.PROCESSED:
                print(f"    ⏭️  Directory already processed, skipping: {directory_path.name}")
                return None
            
            # Process directory as single document
            processed_doc = await self.processor.process_directory_as_single_document(directory_path)
            
            if not processed_doc:
                print(f"    ❌ Failed to process directory: {directory_path.name}")
                return None
            
            print(f"    📝 Combined text extracted: {len(processed_doc.text_content):,} characters")
            print(f"    📄 Files processed: {processed_doc.metadata.get('file_count', 0)}")
            
            # Create individual knowledge graph for this directory collection
            print("    🧠 Creating knowledge graph...")
            doc_kg, library_doc_id = await self.knowledge_graph.create_document_graph(processed_doc)
            print(f"    ✅ Knowledge graph created: {processed_doc.id[:8]}")
            
            
            
            # Generate mindmap
            print("    🗺️  Generating mindmap visualization...")
            try:
                mindmap_result = await self.mindmap_generator.generate_mindmap_from_text(processed_doc.text_content, output_dir = f"{self.config.working_dir}/mindmaps/{library_doc_id}", save_files=True)
                print(f"    ✅ Mindmap generated (tokens: {mindmap_result.token_usage.get('total_tokens', 0):,})")
                
                self.library_manager.add_mindmap(processed_doc.id, mindmap_result, {})
            except Exception as mindmap_error:
                print(f"    ⚠️  Mindmap generation failed: {mindmap_error}")

            # Generate AI description from knowledge graph
            print("    📄 Generating description from knowledge graph...")
            description = await self._generate_description(mindmap_result.markdown)
            
            # Generate tags from knowledge graph
            print("    🏷️  Generating tags from knowledge graph...")
            tags = await self._generate_tags(mindmap_result.markdown)
            
            # Update library with description and tags
            if library_doc_id:
                try:
                    metadata_update = {}
                    if description:
                        metadata_update["description"] = description
                    if tags:
                        metadata_update["tags"] = tags
                    
                    if metadata_update:
                        self.library_manager.update_document_metadata(library_doc_id, metadata_update)
                        print("    📝 Description and tags saved to library")
                except Exception as e:
                    print(f"    ⚠️  Failed to save metadata: {e}")
            
            print(f"    ✅ Completed directory: {directory_path.name}")
            return {
                'document_id': processed_doc.id,
                'knowledge_graph': doc_kg,
                'library_id': library_doc_id,
                'metadata': {
                    'description': description,
                    'tags': tags
                }
            }
            
        except Exception as e:
            print(f"    ❌ Error processing directory {directory_path.name}: {e}")
            return None
    
    async def _generate_description(self, markdown_mindmap) -> Optional[str]:
        """Generate description from mindmap"""
        try:
            # Prepare the text for description generation
            text_content = markdown_mindmap
            
            # Create prompt for description generation
            prompt = f"""Analyze the following mindmap built from a document and provide a concise, informative description (2-3 sentences) that captures:

1. The main topic or subject matter
2. The type of content (academic, technical, business, etc.)
3. Key themes or focus areas

Text to analyze:
{text_content}

Provide only the description, no additional formatting or labels."""

            # Generate description using Ollama client
            try:
                if self.ollama_client:
                    response = await self.ollama_client.chat.completions.create(
                        model=self.config.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=50000,
                        temperature=0.3
                    )
                    
                    description = response.choices[0].message.content or ""
                    
                    if description:
                        # Clean up the description
                        description = description.strip()
                        # Remove any prefix like "Description:" if the model includes it
                        if description.lower().startswith("description:"):
                            description = description[12:].strip()
                        
                        logger.info(f"📝 Generated description: {description[:100]}...")
                        return description
                    else:
                        msg = "⚠️ Empty description generated"
                        logger.warning(msg)
                        return msg
                        
    
                else:
                    msg = "⚠️ Ollama client not available, using fallback description"
                    logger.warning(msg)
                    return msg

                    
            except Exception as e:
                msg = f"❌ Error calling Ollama for description generation: {e}"
                logger.error(msg)
                return msg
                
        except Exception as e:
            msg = f"❌ Error generating description for document: {e}"
            logger.error(msg)
            return f"❌ Error generating description: {e}"
    
    async def _generate_tags(self, knowledge_graph, is_directory=False):
        """Generate tags from knowledge graph"""
        try:
            # Prepare the text for tag generation
            text_content = knowledge_graph
            
            # Create prompt for tag generation
            prompt = f"""Analyze the following knowledge graph content and extract relevant tags (2-5 comma-separated tags) that capture:

1. The main subject or theme
2. Key concepts or topics
3. Domain or field of study

Text to analyze:
{text_content}

Provide only the tags separated by commas, no additional formatting or labels."""

            # Generate tags using Ollama client
            try:
                if self.ollama_client:
                    response = await self.ollama_client.chat.completions.create(
                        model=self.config.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000,
                        temperature=0.3
                    )
                    
                    tags = response.choices[0].message.content or ""
                    
                    if tags:
                        # Clean up the tags
                        tags = tags.strip()
                        # Split by comma and clean each tag
                        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
                        
                        logger.info(f"🏷️ Generated tags: {tag_list}")
                        return tag_list
                    else:
                        msg = "⚠️ Empty tags generated"
                        logger.warning(msg)
                        return []
                        
                else:
                    msg = "⚠️ Ollama client not available, using fallback tags"
                    logger.warning(msg)
                    return ["untagged"]
                    
            except Exception as e:
                msg = f"❌ Error calling Ollama for tag generation: {e}"
                logger.error(msg)
                return []
                
        except Exception as e:
            msg = f"❌ Error generating tags for document: {e}"
            logger.error(msg)
            return []
        
    async def run(self):
        """Main processing pipeline"""
        print("🐛 BookWorm Document Processing System")
        print("=" * 60)
        
        # Load configuration and set up logging
        setup_logging(self.config)

        # Step 1: Convert and archive PDFs before processing anything else
        if not self.config.skip_pdf_conversion:
            await self.convert_and_archive_pdfs()
        else:
            print("\n⏭️  Skipping PDF conversion (SKIP_PDF_CONVERSION=true)")
        
        print("📚 Library Status:")
        stats = self.library_manager.get_library_stats()
        print(f"  📄 Documents: {stats.total_documents}")
        print(f"  🗺️  Mindmaps: {stats.total_mindmaps}")
        print(f"  ✅ Processed: {stats.processed_documents}")
        print(f"  ⏳ Pending: {stats.pending_documents}")
        print(f"  ❌ Failed: {stats.failed_documents}")
        print(f"  💾 Total size: {stats.total_size_bytes:,} bytes")
        print()
        
        print("📁 Scanning for documents and directories to process...")
        
        # Scan workspace for documents and directories
        docs_to_process, dirs_to_process = await self.scan_workspace()
        
        if not docs_to_process and not dirs_to_process:
            print("📝 No documents or directories found in workspace.")
            print(f"   Add documents/directories to: {Path(self.config.working_dir) / 'docs'}")
            print("   Supported formats: .txt, .md, .pdf, .docx")
            print("   Directories: Obsidian vaults, document collections")
            return
        
        if docs_to_process:
            print(f"📄 Found {len(docs_to_process)} individual documents:")
            for doc_path in docs_to_process:
                print(f"  • {doc_path.name}")
        
        if dirs_to_process:
            print(f"📁 Found {len(dirs_to_process)} directories to process as collections:")
            for dir_path in dirs_to_process:
                # Count files in directory
                file_count = sum(1 for f in dir_path.rglob("*") 
                               if f.is_file() and f.suffix.lower() in ['.txt', '.md', '.pdf', '.docx'])
                print(f"  • {dir_path.name} ({file_count} files)")
        
        print()
        
        # Process documents and directories
        results = []
        
        # Process individual documents first
        for file_path in docs_to_process:
            result = await self.process_single_document(file_path)
            if result:
                results.append(result)
        
        # Process directories as collections
        for dir_path in dirs_to_process:
            result = await self.process_directory_collection(dir_path)
            if result:
                results.append(result)
        
        if not results:
            print("\n❗ No documents were successfully processed.")
            return
        
        print(f"\n🎉 Successfully processed {len(results)} documents")
        print("   Each document has its own isolated knowledge graph")
        
        # Demonstrate querying individual documents
        print("\n🔍 Testing Knowledge Graph Queries...")
        
        # Sample queries to test on each document
        test_queries = [
            "What is the main topic of this document?",
            "What are the key concepts discussed?",
            "Summarize the main points"
        ]
        
        for result in results[:2]:  # Test first 2 documents
            doc_id = result['document_id']
            doc_kg = result['knowledge_graph']
            print(f"\n  📄 Document: {doc_id[:8]}")
            
            for query in test_queries[:1]:  # Test first query only to avoid too much output
                try:
                    result = await doc_kg.query(query, mode="hybrid")
                    print(f"    Q: {query}")
                    print(f"    A: {result[:150]}...")
                    break  # Only show one query per document
                except Exception as e:
                    print(f"    ❌ Query failed: {e}")
        
        # Show cross-document capabilities
        print("\n🔄 Cross-Document Query Example:")
        print(f"   Querying all {len(results)} document graphs...")
        
        try:
            all_document_graphs = [r['knowledge_graph'] for r in results]
            cross_results = await self.knowledge_graph.query_all_documents(
                "What are the main topics across all documents?", 
                mode="global"
            )
            print(f"   ✅ Received results from {len(cross_results)} documents")
            for doc_id, result in list(cross_results.items())[:1]:  # Show one example
                print(f"   {doc_id[:8]}: {result[:100]}...")
        except Exception as e:
            print(f"   ⚠️  Cross-document query failed: {e}")
        
        # Display final library statistics
        print("\n📊 Final Library Statistics:")
        final_stats = self.library_manager.get_library_stats()
        print(f"  📄 Total Documents: {final_stats.total_documents}")
        print(f"  🗺️ Total Mindmaps: {final_stats.total_mindmaps}")
        print(f"  ✅ Successfully Processed: {final_stats.processed_documents}")
        print(f"  💾 Total Content Size: {final_stats.total_size_bytes:,} bytes")
        print(f"  🕒 Last Updated: {final_stats.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show knowledge graph directories
        kg_list = self.knowledge_graph.list_document_graphs()
        print("\n🧠 Knowledge Graph Status:")
        print(f"  📊 Individual graphs created: {len(kg_list)}")
        print("  📁 Graph storage: ./workspace/knowledge_graphs/")
        
        print("\n🎉 Processing completed!")
        print(f"📁 Check outputs in: {self.config.output_dir}")
        print("💡 Architecture:")
        print("  • Each document has its own isolated knowledge graph")
        print("  • All graphs are indexed in the library system")
        print("  • Cross-document queries are supported")
        print("  • Mindmaps are generated per document")


async def main():
    """Main entry point"""
    # Load configuration
    config = load_config()
    
    pipeline = BookWormPipeline(config)
    await pipeline.run()


if __name__ == "__main__":
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main())