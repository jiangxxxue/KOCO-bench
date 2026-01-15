"""
Command Line Interface for BookWorm
"""
import asyncio
import click
import logging
from pathlib import Path
from typing import Optional

from .processors import DocumentProcessor
from .knowledge import KnowledgeGraph
from .generators import MindmapGenerator
from .utils import load_config, setup_logging, ensure_directories


@click.group()
@click.option('--config', '-c', help='Path to .env configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """BookWorm - Advanced Document/Knowledge Ingestion System"""
    # Load configuration
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    
    # Set log level if verbose
    if verbose:
        ctx.obj['config'].log_level = "DEBUG"
    
    # Setup logging
    ctx.obj['logger'] = setup_logging(ctx.obj['config'])
    
    # Ensure directories exist
    ensure_directories(ctx.obj['config'])


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', help='Output directory for processed files')
@click.pass_context
def process(ctx, input_path: str, output_dir: Optional[str]):
    """Process documents and extract text"""
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    if output_dir:
        config.processed_dir = output_dir
        ensure_directories(config)
    
    async def run_process():
        processor = DocumentProcessor(config)
        input_path_obj = Path(input_path)
        
        try:
            if input_path_obj.is_file():
                logger.info(f"Processing single file: {input_path}")
                document = await processor.process_document(input_path_obj)
                click.echo(f"✅ Processed: {document.original_path}")
                click.echo(f"   Output: {document.processed_path}")
                click.echo(f"   Words: {document.metadata.get('word_count', 0)}")
                
            elif input_path_obj.is_dir():
                logger.info(f"Processing directory: {input_path}")
                documents = await processor.process_directory(input_path_obj)
                
                click.echo(f"✅ Processed {len(documents)} documents")
                for doc in documents:
                    click.echo(f"   - {doc.metadata.get('filename', 'Unknown')}: "
                             f"{doc.metadata.get('word_count', 0)} words")
            else:
                click.echo(f"❌ Invalid path: {input_path}")
                return
                
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            click.echo(f"❌ Error: {e}")
            raise click.Abort()
    
    asyncio.run(run_process())


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--mindmap', '-m', is_flag=True, help='Generate mindmaps for processed documents')
@click.pass_context
def ingest(ctx, input_path: str, mindmap: bool):
    """Ingest documents into knowledge graph"""
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    async def run_ingest():
        processor = DocumentProcessor(config)
        knowledge_graph = KnowledgeGraph(config)
        mindmap_generator = MindmapGenerator(config) if mindmap else None
        
        try:
            # Initialize knowledge graph
            logger.info("Initializing knowledge graph...")
            await knowledge_graph.initialize()
            
            # Process documents
            input_path_obj = Path(input_path)
            
            if input_path_obj.is_file():
                logger.info(f"Processing and ingesting file: {input_path}")
                document = await processor.process_document(input_path_obj)
                await knowledge_graph.add_document(document)
                
                click.echo(f"✅ Ingested: {document.metadata.get('filename', 'Unknown')}")
                
                if mindmap:
                    mindmap_result = await mindmap_generator.generate_mindmap(document)
                    click.echo(f"🧠 Mindmap generated for document {document.id}")
                
            elif input_path_obj.is_dir():
                logger.info(f"Processing and ingesting directory: {input_path}")
                documents = await processor.process_directory(input_path_obj)
                await knowledge_graph.batch_add_documents(documents)
                
                click.echo(f"✅ Ingested {len(documents)} documents into knowledge graph")
                
                if mindmap:
                    mindmap_results = await mindmap_generator.batch_generate_mindmaps(documents)
                    click.echo(f"🧠 Generated {len(mindmap_results)} mindmaps")
            
            # Finalize
            await knowledge_graph.finalize()
            logger.info("Knowledge graph operations completed")
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            click.echo(f"❌ Error: {e}")
            raise click.Abort()
    
    asyncio.run(run_ingest())


@cli.command()
@click.argument('query_text')
@click.option('--mode', '-m', default='hybrid', 
              type=click.Choice(['local', 'global', 'hybrid', 'naive', 'mix']),
              help='Query mode for LightRAG')
@click.pass_context
def query(ctx, query_text: str, mode: str):
    """Query the knowledge graph"""
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    async def run_query():
        knowledge_graph = KnowledgeGraph(config)
        
        try:
            # Initialize knowledge graph
            await knowledge_graph.initialize()
            
            # Execute query
            logger.info(f"Executing query: {query_text}")
            result = await knowledge_graph.query(query_text, mode=mode)
            
            click.echo(f"\n📊 Query: {query_text}")
            click.echo(f"🔍 Mode: {mode}")
            click.echo(f"\n📝 Result:\n{result}")
            
            # Finalize
            await knowledge_graph.finalize()
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            click.echo(f"❌ Error: {e}")
            raise click.Abort()
    
    asyncio.run(run_query())


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--provider', '-p', 
              type=click.Choice(['OPENAI', 'CLAUDE', 'DEEPSEEK', 'GEMINI']),
              help='LLM provider for mindmap generation')
@click.pass_context
def mindmap(ctx, input_path: str, provider: Optional[str]):
    """Generate mindmaps from documents"""
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    if provider:
        config.api_provider = provider
    
    async def run_mindmap():
        processor = DocumentProcessor(config)
        mindmap_generator = MindmapGenerator(config)
        
        try:
            input_path_obj = Path(input_path)
            
            if input_path_obj.is_file():
                logger.info(f"Generating mindmap for file: {input_path}")
                document = await processor.process_document(input_path_obj)
                result = await mindmap_generator.generate_mindmap(document)
                
                click.echo(f"🧠 Mindmap generated for: {document.metadata.get('filename', 'Unknown')}")
                click.echo(f"   Provider: {result.provider}")
                click.echo(f"   Generated at: {result.generated_at}")
                
            elif input_path_obj.is_dir():
                logger.info(f"Generating mindmaps for directory: {input_path}")
                documents = await processor.process_directory(input_path_obj)
                results = await mindmap_generator.batch_generate_mindmaps(documents)
                
                click.echo(f"🧠 Generated {len(results)} mindmaps")
                for result in results:
                    doc = next((d for d in documents if d.id == result.document_id), None)
                    filename = doc.metadata.get('filename', 'Unknown') if doc else 'Unknown'
                    click.echo(f"   - {filename}")
            
        except Exception as e:
            logger.error(f"Mindmap generation failed: {e}")
            click.echo(f"❌ Error: {e}")
            raise click.Abort()
    
    asyncio.run(run_mindmap())


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and configuration"""
    config = ctx.obj['config']
    
    click.echo("📋 BookWorm System Status")
    click.echo("=" * 50)
    click.echo(f"Working Directory: {config.working_dir}")
    click.echo(f"Document Directory: {config.document_dir}")
    click.echo(f"Processed Directory: {config.processed_dir}")
    click.echo(f"Output Directory: {config.output_dir}")
    click.echo()
    click.echo(f"LLM Model: {config.llm_model}")
    click.echo(f"Embedding Model: {config.embedding_model}")
    click.echo(f"API Provider: {config.api_provider}")
    click.echo(f"PDF Processor: {config.pdf_processor}")
    click.echo()
    click.echo(f"Max Concurrent Processes: {config.max_concurrent_processes}")
    click.echo(f"Max File Size: {config.max_file_size_mb} MB")
    click.echo(f"Log Level: {config.log_level}")
    
    # Check directory existence
    click.echo("\n📁 Directory Status:")
    directories = [
        ("Working", config.working_dir),
        ("Documents", config.document_dir),
        ("Processed", config.processed_dir),
        ("Output", config.output_dir),
        ("Logs", config.log_dir),
    ]
    
    for name, path in directories:
        exists = Path(path).exists()
        status_icon = "✅" if exists else "❌"
        click.echo(f"   {status_icon} {name}: {path}")


def main():
    """Entry point for the CLI"""
    cli()


if __name__ == '__main__':
    main()
