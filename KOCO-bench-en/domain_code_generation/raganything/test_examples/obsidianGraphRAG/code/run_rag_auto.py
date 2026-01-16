#!/usr/bin/env python3
"""
Auto-run RAG system with option to rebuild database automatically
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
        sys.exit(1)

    print(f"Running in conda environment: {conda_env}")

async def main():
    """Main function - processes Obsidian vault with SOTA chunking"""
    print("Obsidian RAG-Anything with SOTA Chunking")
    print("Building NEW database automatically")
    print("="*60)

    # Check conda environment
    check_conda_environment()

    # Setup paths
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro")
    working_dir = os.getenv("WORKING_DIR", "./rag_storage")

    os.makedirs(working_dir, exist_ok=True)

    print(f"Vault Path: {vault_path}")
    print(f"Working Dir: {working_dir}")

    # Check if vault exists
    if not os.path.exists(vault_path):
        print(f"Vault path does not exist: {vault_path}")
        sys.exit(1)

    # Import and run RAG-Anything
    try:
        from src.simple_raganything import SimpleRAGAnything

        print("\nInitializing RAG-Anything system...")

        # Initialize system
        rag = SimpleRAGAnything(vault_path, working_dir)

        # Delete existing database if present
        if rag._detect_existing_database():
            print("\nDeleting existing database to build fresh...")
            rag._delete_existing_database()

        # Initialize
        await rag.initialize()

        # Process vault with SOTA chunking
        print("\nProcessing Obsidian vault with SOTA chunking...")
        await rag.process_vault()

        # Test a query
        print("\n" + "="*60)
        print("Testing Query System")
        print("="*60)

        test_question = "What are the main topics in my notes?"
        print(f"\nQuestion: {test_question}")

        result = await rag.query(test_question)
        print(f"\nAnswer:\n{result}")

        print("\n" + "="*60)
        print("RAG System Ready!")
        print("="*60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    check_conda_environment()
    asyncio.run(main())