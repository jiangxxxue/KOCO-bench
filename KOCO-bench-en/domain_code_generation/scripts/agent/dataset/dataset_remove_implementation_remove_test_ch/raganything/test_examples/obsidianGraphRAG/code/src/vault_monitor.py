"""
Obsidian Vault Monitor - Incremental Updates
Tracks changes in your vault and incrementally updates the RAG database
"""

import os
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime


class VaultMonitor:
    """
    Monitors Obsidian vault for changes and manages incremental updates
    """
    
    def __init__(self, vault_path: str, tracking_file: str = "./rag_storage/vault_tracking.json"):
        """
        Initialize vault monitor
        
        Args:
            vault_path: Path to Obsidian vault
            tracking_file: Path to file tracking database state
        """
        self.vault_path = vault_path
        self.tracking_file = tracking_file
        self.file_registry: Dict[str, dict] = {}
        
        # Load existing tracking data
        self._load_tracking_data()
    
    def _load_tracking_data(self):
        """Load existing file tracking data"""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    self.file_registry = json.load(f)
                print(f"✓ Loaded tracking data: {len(self.file_registry)} files tracked")
            except Exception as e:
                print(f"⚠️  Warning: Could not load tracking data: {e}")
                self.file_registry = {}
        else:
            print("ℹ️  No existing tracking data found - will create new")
            self.file_registry = {}
    
    def _save_tracking_data(self):
        """Save file tracking data"""
        try:
            os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.file_registry, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved tracking data: {len(self.file_registry)} files")
        except Exception as e:
            print(f"❌ Error saving tracking data: {e}")
    
    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute MD5 hash of file content
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash string
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"⚠️  Warning: Could not hash {file_path}: {e}")
            return ""
    
    def _get_file_info(self, file_path: str) -> dict:
        """
        Get file metadata
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file info
        """
        try:
            stat = os.stat(file_path)
            return {
                "hash": self._compute_file_hash(file_path),
                "modified": stat.st_mtime,
                "size": stat.st_size,
                "last_processed": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"⚠️  Warning: Could not get info for {file_path}: {e}")
            return {}
    
    def scan_vault(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Scan vault and detect changes
        
        Returns:
            Tuple of (new_files, modified_files, deleted_files)
        """
        print("\n" + "="*70)
        print("SCANNING VAULT FOR CHANGES")
        print("="*70)
        
        # Find all markdown files in vault
        current_files = set()
        for root, dirs, files in os.walk(self.vault_path):
            # Skip .obsidian directory
            if '.obsidian' in root:
                continue
                
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    # Store relative path for portability
                    rel_path = os.path.relpath(file_path, self.vault_path)
                    current_files.add(rel_path)
        
        # Compare with tracked files
        tracked_files = set(self.file_registry.keys())
        
        # Detect new files
        new_files = []
        for file in current_files - tracked_files:
            new_files.append(file)
        
        # Detect modified files
        modified_files = []
        for file in current_files & tracked_files:
            abs_path = os.path.join(self.vault_path, file)
            current_hash = self._compute_file_hash(abs_path)
            tracked_hash = self.file_registry[file].get('hash', '')
            
            if current_hash != tracked_hash:
                modified_files.append(file)
        
        # Detect deleted files
        deleted_files = list(tracked_files - current_files)
        
        # Print summary
        print(f"\n📊 Scan Results:")
        print(f"   • Total files in vault: {len(current_files)}")
        print(f"   • Tracked files: {len(tracked_files)}")
        print(f"   • New files: {len(new_files)}")
        print(f"   • Modified files: {len(modified_files)}")
        print(f"   • Deleted files: {len(deleted_files)}")
        
        return new_files, modified_files, deleted_files
    
    def mark_file_processed(self, file_path: str, doc_id: str):
        """
        Mark a file as processed
        
        Args:
            file_path: Relative path to file
            doc_id: Document ID assigned in RAG system
        """
        abs_path = os.path.join(self.vault_path, file_path)
        file_info = self._get_file_info(abs_path)
        file_info['doc_id'] = doc_id
        self.file_registry[file_path] = file_info
    
    def mark_files_processed(self, file_paths: List[str], doc_ids: List[str]):
        """
        Mark multiple files as processed
        
        Args:
            file_paths: List of relative paths
            doc_ids: List of document IDs
        """
        for file_path, doc_id in zip(file_paths, doc_ids):
            self.mark_file_processed(file_path, doc_id)
        
        # Save after batch update
        self._save_tracking_data()
    
    def remove_deleted_files(self, deleted_files: List[str]) -> List[str]:
        """
        Remove deleted files from tracking
        
        Args:
            deleted_files: List of deleted file paths
            
        Returns:
            List of doc_ids that need to be removed from RAG
        """
        doc_ids_to_remove = []
        
        for file_path in deleted_files:
            if file_path in self.file_registry:
                doc_id = self.file_registry[file_path].get('doc_id')
                if doc_id:
                    doc_ids_to_remove.append(doc_id)
                del self.file_registry[file_path]
        
        if deleted_files:
            self._save_tracking_data()
        
        return doc_ids_to_remove
    
    def get_doc_id_for_file(self, file_path: str) -> str:
        """
        Get document ID for a file
        
        Args:
            file_path: Relative path to file
            
        Returns:
            Document ID or None
        """
        return self.file_registry.get(file_path, {}).get('doc_id')
    
    def generate_doc_id(self, file_path: str) -> str:
        """
        Generate unique document ID for a file
        
        Args:
            file_path: Relative path to file
            
        Returns:
            Document ID
        """
        # Use file path hash for consistent IDs
        path_hash = hashlib.md5(file_path.encode('utf-8')).hexdigest()[:12]
        return f"doc_{path_hash}"


class IncrementalVaultUpdater:
    """
    Manages incremental updates to RAG database when vault changes
    """
    
    def __init__(self, rag_system, vault_monitor: VaultMonitor):
        """
        Initialize updater
        
        Args:
            rag_system: SimpleRAGAnything instance
            vault_monitor: VaultMonitor instance
        """
        self.rag = rag_system
        self.monitor = vault_monitor
        self.chunker = rag_system.chunker
    
    async def process_new_files(self, new_files: List[str]):
        """
        Process and add new files to RAG database
        
        Args:
            new_files: List of new file paths (relative)
        """
        if not new_files:
            print("\nℹ️  No new files to process")
            return
        
        print(f"\n{'='*70}")
        print(f"PROCESSING {len(new_files)} NEW FILES")
        print("="*70)
        
        for i, file_path in enumerate(new_files, 1):
            abs_path = os.path.join(self.monitor.vault_path, file_path)
            print(f"\n[{i}/{len(new_files)}] Processing: {file_path}")
            
            try:
                # Chunk the file
                chunks = self.chunker.process_markdown_file(Path(abs_path))
                
                if not chunks:
                    print(f"   ⚠️  No chunks generated")
                    continue
                
                print(f"   ✓ Generated {len(chunks)} chunks")
                
                # Convert to content list format
                content_list = []
                for idx, chunk in enumerate(chunks):
                    content_item = {
                        "type": "text",
                        "text": chunk['content'],
                        "page_idx": idx  # Use enumerate index
                    }
                    content_list.append(content_item)
                
                # Generate doc ID
                doc_id = self.monitor.generate_doc_id(file_path)
                
                # Insert into RAG
                await self.rag.rag_anything.insert_content_list(
                    content_list=content_list,
                    file_path=file_path,  # For citations
                    doc_id=doc_id,
                    display_stats=False
                )
                
                # Mark as processed
                self.monitor.mark_file_processed(file_path, doc_id)
                
                print(f"   ✓ Added to RAG database (doc_id: {doc_id})")
                
            except Exception as e:
                print(f"   ❌ Error processing {file_path}: {e}")
        
        # Save tracking data
        self.monitor._save_tracking_data()
        
        print(f"\n✓ Completed processing new files")
    
    async def process_modified_files(self, modified_files: List[str]):
        """
        Process and update modified files in RAG database
        
        Args:
            modified_files: List of modified file paths (relative)
        """
        if not modified_files:
            print("\nℹ️  No modified files to process")
            return
        
        print(f"\n{'='*70}")
        print(f"PROCESSING {len(modified_files)} MODIFIED FILES")
        print("="*70)
        
        for i, file_path in enumerate(modified_files, 1):
            abs_path = os.path.join(self.monitor.vault_path, file_path)
            print(f"\n[{i}/{len(modified_files)}] Updating: {file_path}")
            
            try:
                # Get existing doc ID
                doc_id = self.monitor.get_doc_id_for_file(file_path)
                
                if doc_id:
                    # Delete old version
                    print(f"   🗑️  Removing old version (doc_id: {doc_id})")
                    await self.rag.rag_anything.lightrag.adelete_by_doc_id(doc_id)
                else:
                    # Generate new doc ID if not tracked
                    doc_id = self.monitor.generate_doc_id(file_path)
                
                # Chunk the file
                chunks = self.chunker.process_markdown_file(Path(abs_path))
                
                if not chunks:
                    print(f"   ⚠️  No chunks generated")
                    continue
                
                print(f"   ✓ Generated {len(chunks)} chunks")
                
                # Convert to content list format
                content_list = []
                for idx, chunk in enumerate(chunks):
                    content_item = {
                        "type": "text",
                        "text": chunk['content'],
                        "page_idx": idx  # Use enumerate index
                    }
                    content_list.append(content_item)
                
                # Insert updated version
                await self.rag.rag_anything.insert_content_list(
                    content_list=content_list,
                    file_path=file_path,
                    doc_id=doc_id,
                    display_stats=False
                )
                
                # Update tracking
                self.monitor.mark_file_processed(file_path, doc_id)
                
                print(f"   ✓ Updated in RAG database")
                
            except Exception as e:
                print(f"   ❌ Error updating {file_path}: {e}")
        
        # Save tracking data
        self.monitor._save_tracking_data()
        
        print(f"\n✓ Completed processing modified files")
    
    async def process_deleted_files(self, deleted_files: List[str]):
        """
        Remove deleted files from RAG database
        
        Args:
            deleted_files: List of deleted file paths (relative)
        """
        if not deleted_files:
            print("\nℹ️  No deleted files to process")
            return
        
        print(f"\n{'='*70}")
        print(f"PROCESSING {len(deleted_files)} DELETED FILES")
        print("="*70)
        
        # Get doc IDs to remove
        doc_ids = self.monitor.remove_deleted_files(deleted_files)
        
        if not doc_ids:
            print("\nℹ️  No documents to remove from RAG")
            return
        
        # Remove from RAG
        for i, doc_id in enumerate(doc_ids, 1):
            print(f"\n[{i}/{len(doc_ids)}] Removing: {doc_id}")
            try:
                await self.rag.rag_anything.lightrag.adelete_by_doc_id(doc_id)
                print(f"   ✓ Removed from RAG database")
            except Exception as e:
                print(f"   ❌ Error removing {doc_id}: {e}")
        
        print(f"\n✓ Completed processing deleted files")
    
    async def sync_vault(self):
        """
        Scan vault and synchronize all changes with RAG database
        """
        print("\n" + "="*70)
        print("INCREMENTAL VAULT SYNC")
        print("="*70)
        
        # Scan for changes
        new_files, modified_files, deleted_files = self.monitor.scan_vault()
        
        total_changes = len(new_files) + len(modified_files) + len(deleted_files)
        
        if total_changes == 0:
            print("\n✅ Vault is up to date - no changes detected")
            return
        
        print(f"\n🔄 Synchronizing {total_changes} changes...")
        
        # Process changes
        await self.process_deleted_files(deleted_files)
        await self.process_modified_files(modified_files)
        await self.process_new_files(new_files)
        
        print("\n" + "="*70)
        print("✅ VAULT SYNC COMPLETE")
        print("="*70)
        print(f"   • New files added: {len(new_files)}")
        print(f"   • Files updated: {len(modified_files)}")
        print(f"   • Files removed: {len(deleted_files)}")
        print(f"   • Total files tracked: {len(self.monitor.file_registry)}")
