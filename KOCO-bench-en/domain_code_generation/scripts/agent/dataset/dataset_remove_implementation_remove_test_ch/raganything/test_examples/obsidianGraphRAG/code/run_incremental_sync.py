#!/usr/bin/env python3
"""
Incremental Vault Sync
Scans your Obsidian vault for changes and incrementally updates the RAG database
Only processes new, modified, or deleted files
"""

import os
import sys
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.simple_raganything import SimpleRAGAnything
from src.vault_monitor import VaultMonitor, IncrementalVaultUpdater


async def main():
    """Run incremental vault sync"""
    print("\n" + "="*70)
    print("OBSIDIAN VAULT - INCREMENTAL SYNC")
    print("="*70)
    print("Only new, modified, or deleted files will be processed")
    print("="*70)
    
    # Configuration
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro")
    working_dir = os.getenv("RAG_WORKING_DIR", "./rag_storage")
    
    print(f"\n📁 Vault: {vault_path}")
    print(f"💾 Storage: {working_dir}")
    
    # Check if vault exists
    if not os.path.exists(vault_path):
        print(f"\n❌ Error: Vault not found at {vault_path}")
        print("Please set OBSIDIAN_VAULT_PATH environment variable")
        return
    
    # Check if RAG database exists
    if not os.path.exists(os.path.join(working_dir, "graph_chunk_entity_relation.graphml")):
        print(f"\n❌ Error: RAG database not found at {working_dir}")
        print("\nYou need to build the initial database first:")
        print("   python run_obsidian_raganything.py")
        print("\nThen you can use incremental sync for future updates")
        return
    
    try:
        # Initialize RAG system (using existing database)
        print("\n" + "="*70)
        print("INITIALIZING RAG SYSTEM")
        print("="*70)
        
        rag = SimpleRAGAnything(
            vault_path=vault_path,
            working_dir=working_dir,
            non_interactive=True  # Always use existing DB
        )
        await rag.initialize()
        
        # Initialize vault monitor
        print("\n" + "="*70)
        print("INITIALIZING VAULT MONITOR")
        print("="*70)
        
        monitor = VaultMonitor(
            vault_path=vault_path,
            tracking_file=os.path.join(working_dir, "vault_tracking.json")
        )
        
        # Initialize incremental updater
        updater = IncrementalVaultUpdater(rag, monitor)
        
        # Perform sync
        await updater.sync_vault()
        
        print("\n" + "="*70)
        print("✅ SYNC COMPLETE - DATABASE UP TO DATE")
        print("="*70)
        
        # Optional: Test query
        user_choice = input("\nWould you like to test a query? (y/n): ").strip().lower()
        if user_choice == 'y':
            question = input("Enter your question: ").strip()
            if question:
                print(f"\n🔍 Querying: {question}")
                result = await rag.query(question, mode="hybrid")
                print(f"\n📝 Answer:\n{result}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Sync interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    asyncio.run(main())
