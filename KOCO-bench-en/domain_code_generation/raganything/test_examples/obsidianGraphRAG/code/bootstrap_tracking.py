#!/usr/bin/env python3
"""
Bootstrap Tracking File
Marks all existing vault files as already processed, so incremental sync only handles NEW changes
"""

import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.vault_monitor import VaultMonitor


def bootstrap_tracking(vault_path: str, working_dir: str, cutoff_days: int = 1):
    """
    Mark all files modified before X days ago as already processed
    
    Args:
        vault_path: Path to Obsidian vault
        working_dir: RAG storage directory
        cutoff_days: Mark files modified before X days ago (default: 1 = yesterday)
    """
    print("\n" + "="*70)
    print("BOOTSTRAP TRACKING FILE")
    print("="*70)
    print(f"This will mark existing files as already synced")
    print(f"Only files modified in the last {cutoff_days} day(s) will be synced")
    print("="*70)
    
    # Initialize monitor
    tracking_file = os.path.join(working_dir, "vault_tracking.json")
    monitor = VaultMonitor(vault_path, tracking_file)
    
    # Get cutoff timestamp
    cutoff_time = datetime.now() - timedelta(days=cutoff_days)
    cutoff_timestamp = cutoff_time.timestamp()
    
    print(f"\nCutoff date: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Files modified BEFORE this date will be marked as synced")
    print(f"Files modified AFTER this date will need syncing\n")
    
    # Scan vault
    marked_count = 0
    recent_count = 0
    
    print("Scanning vault files...")
    
    for root, dirs, files in os.walk(vault_path):
        # Skip .obsidian directory
        if '.obsidian' in root:
            continue
        
        for file in files:
            if file.endswith('.md'):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, vault_path)
                
                # Check file modification time
                try:
                    stat = os.stat(abs_path)
                    file_mtime = stat.st_mtime
                    
                    if file_mtime < cutoff_timestamp:
                        # Mark as already processed
                        doc_id = monitor.generate_doc_id(rel_path)
                        monitor.mark_file_processed(rel_path, doc_id)
                        marked_count += 1
                        
                        if marked_count % 50 == 0:
                            print(f"   Marked {marked_count} files...")
                    else:
                        recent_count += 1
                        
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not process {rel_path}: {e}")
    
    # Save tracking data
    monitor._save_tracking_data()
    
    print("\n" + "="*70)
    print("✅ BOOTSTRAP COMPLETE")
    print("="*70)
    print(f"   • Files marked as synced: {marked_count}")
    print(f"   • Recent files (need sync): {recent_count}")
    print(f"   • Total tracked: {len(monitor.file_registry)}")
    print(f"\nTracking file saved to: {tracking_file}")
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run incremental sync:")
    print("   python run_incremental_sync.py")
    print("\n2. Or use the web UI:")
    print("   python run_ui.py")
    print("   (Open settings → Sync Vault)")
    print("\n3. Only files modified in the last day will be processed!")
    print("="*70)


def main():
    """Main function"""
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro")
    working_dir = os.getenv("RAG_WORKING_DIR", "./rag_storage")
    
    print("\n" + "="*70)
    print("BOOTSTRAP TRACKING SETUP")
    print("="*70)
    print(f"📁 Vault: {vault_path}")
    print(f"💾 Storage: {working_dir}")
    
    # Check if vault exists
    if not os.path.exists(vault_path):
        print(f"\n❌ Error: Vault not found at {vault_path}")
        return
    
    # Check if RAG database exists
    if not os.path.exists(os.path.join(working_dir, "graph_chunk_entity_relation.graphml")):
        print(f"\n❌ Error: RAG database not found at {working_dir}")
        print("Please build the database first: python run_obsidian_raganything.py")
        return
    
    # Check if tracking file already exists
    tracking_file = os.path.join(working_dir, "vault_tracking.json")
    if os.path.exists(tracking_file):
        print(f"\n⚠️  Warning: Tracking file already exists at {tracking_file}")
        response = input("Do you want to OVERWRITE it? (yes/no): ").strip().lower()
        if response != 'yes':
            print("❌ Cancelled")
            return
        print("\n✓ Overwriting existing tracking file...")
    
    # Ask for cutoff days
    print("\n" + "="*70)
    print("CUTOFF DATE SELECTION")
    print("="*70)
    print("Files modified BEFORE the cutoff will be marked as already synced")
    print("Files modified AFTER the cutoff will need syncing")
    print("\nOptions:")
    print("  1 = Yesterday (files from today will sync)")
    print("  2 = 2 days ago (files from last 2 days will sync)")
    print("  7 = Last week (files from last week will sync)")
    print("  30 = Last month (files from last month will sync)")
    
    try:
        days_input = input("\nEnter number of days (default: 1): ").strip()
        cutoff_days = int(days_input) if days_input else 1
        
        if cutoff_days < 0:
            print("❌ Error: Days must be positive")
            return
            
    except ValueError:
        print("❌ Error: Invalid number")
        return
    
    # Confirm
    cutoff_time = datetime.now() - timedelta(days=cutoff_days)
    print(f"\n✓ Cutoff: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Files modified BEFORE this date → Already synced")
    print(f"   Files modified AFTER this date → Will need syncing")
    
    response = input("\nProceed? (yes/no): ").strip().lower()
    if response != 'yes':
        print("❌ Cancelled")
        return
    
    # Bootstrap
    try:
        bootstrap_tracking(vault_path, working_dir, cutoff_days)
        print("\n✅ Success! You can now run incremental sync.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
