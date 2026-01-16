#!/usr/bin/env python3
"""
Obsidian RAG-Anything Runner with SOTA Chunking
Always runs in conda environment turing0.1
Processes Obsidian vault with 2K token chunks, wikilinks, and metadata
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_conda_environment():
    """Check if we're running in conda environment turing0.1"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    
    if conda_env != 'turing0.1':
        print("ERROR: Not running in conda environment turing0.1")
        print(f"   Current environment: {conda_env}")
        print("")
        print("Please activate the correct environment:")
        print("   conda activate turing0.1")
        print("   python run_obsidian_raganything.py")
        sys.exit(1)
    
    print(f"Running in conda environment: {conda_env}")

def setup_environment():
    """Setup environment variables and paths"""
    print("Setting up environment...")
    
    # Set default paths
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro")
    working_dir = os.getenv("WORKING_DIR", "./rag_storage")
    
    # Create directories if they don't exist
    os.makedirs(working_dir, exist_ok=True)
    
    print(f"Vault Path: {vault_path}")
    print(f"Working Dir: {working_dir}")
    
    # Check if vault exists
    if not os.path.exists(vault_path):
        print(f"Vault path does not exist: {vault_path}")
        print("Please set the correct vault path:")
        print("   export OBSIDIAN_VAULT_PATH=/path/to/your/vault")
        sys.exit(1)
    
    return vault_path, working_dir

async def main():
    """
    Main function - processes Obsidian vault with SOTA chunking
    Always runs in conda environment turing0.1
    """
    print("Obsidian RAG-Anything with SOTA Chunking")
    print("Conda Environment: turing0.1")
    print("2K Token Chunks with Wikilinks & Metadata")
    print("="*60)
    
    # Check conda environment
    check_conda_environment()
    
    # Setup environment
    vault_path, working_dir = setup_environment()
    
    # Import and run RAG-Anything
    try:
        from src.simple_raganything import SimpleRAGAnything
        
        print("Initializing RAG-Anything system...")
        print("Loading EmbeddingGemma 308M...")
        print("Setting up SOTA chunking...")
        
        # Initialize system
        rag = SimpleRAGAnything(vault_path, working_dir)
        await rag.initialize()

        # Process vault only if NOT using existing database
        if not rag.using_existing_db:
            # Process vault with SOTA chunking
            print("\nProcessing Obsidian vault with SOTA chunking...")
            print("Creating 2K token chunks with wikilinks & metadata...")
            await rag.process_vault()
        else:
            print("\nUsing existing database - skipping vault processing...")
        
        # Interactive query mode
        print("\n" + "="*60)
        print("Interactive Query Mode")
        print("="*60)
        print("Your Obsidian vault has been processed with:")
        print("✅ 2K token chunks with wikilinks preserved")
        print("✅ Metadata and file connections maintained")
        print("✅ EmbeddingGemma 308M embeddings")
        print("✅ RAG-Anything multimodal processing")
        print("")
        print("Type your questions (or 'quit' to exit):")
        
        while True:
            try:
                question = input("\n❓ Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Ask question
                result = await rag.query(question)
                print(f"\n📝 Answer:\n{result}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're in the correct directory and all dependencies are installed")
        print("   Run: bash setup_conda_env.sh")
        sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Always check conda environment first
    check_conda_environment()
    
    # Run the main function
    asyncio.run(main())
