"""
Obsidian Chunker - SOTA Implementation
Chunks Obsidian vault files into 2K token windows while preserving:
- Wikilinks and connections
- Metadata (frontmatter, tags, file info)
- File relationships and chunk connections
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import frontmatter
from dataclasses import dataclass

@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_id: str
    chunk_index: int
    source_file: str
    source_path: str
    file_name: str
    file_created: str
    file_modified: str
    wikilinks: List[str]
    tags: List[str]
    frontmatter: Dict[str, Any]
    word_count: int
    token_estimate: int
    has_wikilinks: bool
    wikilink_count: int
    tag_count: int
    is_first_chunk: bool
    is_last_chunk: bool
    total_chunks_in_file: int
    previous_chunk_id: str = None
    next_chunk_id: str = None

class ObsidianChunker:
    """
    SOTA Obsidian chunker that preserves wikilinks and metadata
    Chunks files into 2K token windows with full context preservation
    """
    
    def __init__(self, vault_path: str, target_tokens: int = 2000):
        """
        Initialize Obsidian chunker
        
        Args:
            vault_path: Path to Obsidian vault
            target_tokens: Target tokens per chunk (default: 2000)
        """
        self.vault_path = Path(vault_path)
        self.target_tokens = target_tokens
        self.tokens_per_word = 1.3  # Approximate tokens per word
        
        # Regex patterns for Obsidian features
        self.wikilink_pattern = re.compile(r'\[\[([^\]]+)\]\]')
        self.embed_pattern = re.compile(r'!\[\[([^\]]+)\]\]')
        self.tag_pattern = re.compile(r'#([a-zA-Z0-9_/\-]+)')
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
        print(f"Vault Path: {self.vault_path}")
        print(f"Target Tokens: {self.target_tokens}")
        print(f"Tokens per Word: {self.tokens_per_word}")
    
    def _extract_wikilinks(self, text: str) -> List[str]:
        """Extract all wikilinks from text"""
        wikilinks = []
        
        # Find all wikilinks (excluding embeds)
        for match in self.wikilink_pattern.finditer(text):
            link = match.group(1)
            # Handle aliases [[Link|Alias]] -> Link
            if '|' in link:
                link = link.split('|')[0]
            # Handle section links [[Link#Section]] -> Link
            if '#' in link:
                link = link.split('#')[0]
            wikilinks.append(link.strip())
        
        return list(set(wikilinks))  # Remove duplicates
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        tags = []
        for match in self.tag_pattern.finditer(text):
            tags.append(match.group(1))
        return list(set(tags))
    
    def _extract_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter from content"""
        try:
            post = frontmatter.loads(content)
            return post.metadata, post.content
        except Exception:
            return {}, content
    
    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get file system metadata"""
        file_stat = file_path.stat()
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_stem': file_path.stem,
            'file_size': file_stat.st_size,
            'created_time': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text"""
        words = len(text.split())
        return int(words * self.tokens_per_word)
    
    def _create_chunk_id(self, file_name: str, chunk_index: int) -> str:
        """Create unique chunk ID"""
        return f"{file_name}_{chunk_index:03d}"
    
    def _enhance_chunk_with_metadata(self, chunk_text: str, metadata: ChunkMetadata) -> str:
        """
        Enhance chunk with metadata and connections
        This is the SOTA approach for preserving context
        """
        enhanced_chunk = ""
        
        # Add file metadata header
        enhanced_chunk += f"# Document: {metadata.file_name}\n"
        enhanced_chunk += f"**File Path**: {metadata.source_path}\n"
        enhanced_chunk += f"**Chunk**: {metadata.chunk_index + 1}/{metadata.total_chunks_in_file}\n"
        enhanced_chunk += f"**Created**: {metadata.file_created}\n"
        enhanced_chunk += f"**Modified**: {metadata.file_modified}\n\n"
        
        # Add frontmatter if exists
        if metadata.frontmatter:
            enhanced_chunk += "## Metadata\n"
            for key, value in metadata.frontmatter.items():
                enhanced_chunk += f"- **{key}**: {value}\n"
            enhanced_chunk += "\n"
        
        # Add wikilinks context (CRITICAL for RAG connections)
        if metadata.wikilinks:
            enhanced_chunk += "## Connected Notes\n"
            enhanced_chunk += f"**Wikilinks**: {', '.join([f'[[{link}]]' for link in metadata.wikilinks])}\n"
            enhanced_chunk += f"**Connections**: This note connects to {len(metadata.wikilinks)} other notes\n\n"
        
        # Add tags context
        if metadata.tags:
            enhanced_chunk += "## Topics\n"
            enhanced_chunk += f"**Tags**: {', '.join([f'#{tag}' for tag in metadata.tags])}\n\n"
        
        # Add chunk connection info
        if metadata.total_chunks_in_file > 1:
            enhanced_chunk += "## Document Structure\n"
            if metadata.is_first_chunk:
                enhanced_chunk += f"**Position**: First chunk of {metadata.total_chunks_in_file} chunks\n"
                if metadata.next_chunk_id:
                    enhanced_chunk += f"**Next**: {metadata.next_chunk_id}\n"
            elif metadata.is_last_chunk:
                enhanced_chunk += f"**Position**: Last chunk of {metadata.total_chunks_in_file} chunks\n"
                if metadata.previous_chunk_id:
                    enhanced_chunk += f"**Previous**: {metadata.previous_chunk_id}\n"
            else:
                enhanced_chunk += f"**Position**: Chunk {metadata.chunk_index + 1} of {metadata.total_chunks_in_file}\n"
                if metadata.previous_chunk_id:
                    enhanced_chunk += f"**Previous**: {metadata.previous_chunk_id}\n"
                if metadata.next_chunk_id:
                    enhanced_chunk += f"**Next**: {metadata.next_chunk_id}\n"
            enhanced_chunk += "\n"
        
        # Add original content
        enhanced_chunk += "## Content\n"
        enhanced_chunk += chunk_text
        
        return enhanced_chunk
    
    def _chunk_text_with_context(self, text: str, file_metadata: Dict, frontmatter: Dict) -> List[Dict[str, Any]]:
        """
        Chunk text into 2K token windows while preserving context
        SOTA approach for maintaining connections
        """
        # Estimate total tokens
        total_tokens = self._estimate_tokens(text)
        chunks_needed = max(1, (total_tokens + self.target_tokens - 1) // self.target_tokens)
        
        # Split text into words for precise chunking
        words = text.split()
        words_per_chunk = len(words) // chunks_needed
        
        chunks = []
        
        for i in range(chunks_needed):
            start_idx = i * words_per_chunk
            end_idx = start_idx + words_per_chunk if i < chunks_needed - 1 else len(words)
            
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Extract wikilinks and tags from this specific chunk
            chunk_wikilinks = self._extract_wikilinks(chunk_text)
            chunk_tags = self._extract_tags(chunk_text)
            
            # Create chunk metadata
            chunk_id = self._create_chunk_id(file_metadata['file_stem'], i)
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                chunk_index=i,
                source_file=file_metadata['file_name'],
                source_path=file_metadata['file_path'],
                file_name=file_metadata['file_name'],
                file_created=file_metadata['created_time'],
                file_modified=file_metadata['modified_time'],
                wikilinks=chunk_wikilinks,
                tags=chunk_tags,
                frontmatter=frontmatter,
                word_count=len(chunk_words),
                token_estimate=self._estimate_tokens(chunk_text),
                has_wikilinks=len(chunk_wikilinks) > 0,
                wikilink_count=len(chunk_wikilinks),
                tag_count=len(chunk_tags),
                is_first_chunk=(i == 0),
                is_last_chunk=(i == chunks_needed - 1),
                total_chunks_in_file=chunks_needed
            )
            
            # Set chunk connections
            if i > 0:
                metadata.previous_chunk_id = self._create_chunk_id(file_metadata['file_stem'], i - 1)
            if i < chunks_needed - 1:
                metadata.next_chunk_id = self._create_chunk_id(file_metadata['file_stem'], i + 1)
            
            # Enhance chunk with metadata
            enhanced_chunk = self._enhance_chunk_with_metadata(chunk_text, metadata)
            
            chunks.append({
                'chunk_id': chunk_id,
                'content': enhanced_chunk,
                'original_content': chunk_text,
                'metadata': metadata,
                'enhanced_metadata': {
                    'wikilinks': chunk_wikilinks,
                    'tags': chunk_tags,
                    'connections': {
                        'previous': metadata.previous_chunk_id,
                        'next': metadata.next_chunk_id,
                        'file_chunks': chunks_needed
                    }
                }
            })
        
        return chunks
    
    def process_markdown_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a single markdown file into chunks with full context
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            List of chunk dictionaries with metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            # Extract frontmatter and clean content
            frontmatter_data, clean_content = self._extract_frontmatter(raw_content)
            
            # Get file metadata
            file_metadata = self._get_file_metadata(file_path)
            
            # Chunk the content
            chunks = self._chunk_text_with_context(clean_content, file_metadata, frontmatter_data)
            
            print(f"Processed: {file_path.name} -> {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            return []
    
    def discover_markdown_files(self) -> List[Path]:
        """Discover all .md files in the vault"""
        md_files = []
        
        if not self.vault_path.exists():
            print(f"Vault path does not exist: {self.vault_path}")
            return md_files
        
        print(f"Discovering markdown files in: {self.vault_path}")
        
        for root, dirs, files in os.walk(self.vault_path):
            # Skip .obsidian folder
            if '.obsidian' in root:
                continue
                
            for file in files:
                if file.endswith('.md'):
                    file_path = Path(root) / file
                    md_files.append(file_path)
        
        print(f"Found {len(md_files)} markdown files")
        return md_files
    
    def process_entire_vault(self) -> Dict[str, Any]:
        """
        Process entire Obsidian vault into chunks with full context
        
        Returns:
            Dictionary with all chunks and metadata
        """
        print("="*70)
        print("PROCESSING OBSIDIAN VAULT WITH SOTA CHUNKING")
        print("="*70)
        
        # Discover markdown files
        md_files = self.discover_markdown_files()
        
        if not md_files:
            print("No markdown files found!")
            return {}
        
        all_chunks = []
        processing_stats = {
            'total_files': len(md_files),
            'total_chunks': 0,
            'total_wikilinks': 0,
            'total_tags': 0,
            'files_with_wikilinks': 0,
            'files_with_tags': 0,
            'processing_errors': 0
        }
        
        # Process each file
        for i, file_path in enumerate(md_files, 1):
            print(f"\n[{i}/{len(md_files)}] Processing: {file_path.name}")
            
            file_chunks = self.process_markdown_file(file_path)
            
            if file_chunks:
                all_chunks.extend(file_chunks)
                processing_stats['total_chunks'] += len(file_chunks)
                
                # Update statistics
                for chunk in file_chunks:
                    processing_stats['total_wikilinks'] += chunk['metadata'].wikilink_count
                    processing_stats['total_tags'] += chunk['metadata'].tag_count
                
                if any(chunk['metadata'].has_wikilinks for chunk in file_chunks):
                    processing_stats['files_with_wikilinks'] += 1
                
                if any(chunk['metadata'].tag_count > 0 for chunk in file_chunks):
                    processing_stats['files_with_tags'] += 1
            else:
                processing_stats['processing_errors'] += 1
        
        # Print processing summary
        print("\n" + "="*70)
        print("PROCESSING COMPLETE!")
        print("="*70)
        print(f"Files Processed: {processing_stats['total_files']}")
        print(f"Total Chunks: {processing_stats['total_chunks']}")
        print(f"Total Wikilinks: {processing_stats['total_wikilinks']}")
        print(f"Total Tags: {processing_stats['total_tags']}")
        print(f"Files with Wikilinks: {processing_stats['files_with_wikilinks']}")
        print(f"Files with Tags: {processing_stats['files_with_tags']}")
        print(f"Processing Errors: {processing_stats['processing_errors']}")
        print(f"Avg Chunks per File: {processing_stats['total_chunks']/processing_stats['total_files']:.1f}")
        print("="*70)
        
        return {
            'chunks': all_chunks,
            'stats': processing_stats,
            'vault_path': str(self.vault_path),
            'processing_time': datetime.now().isoformat()
        }
    
    def save_chunks_to_json(self, chunks_data: Dict[str, Any], output_file: str = "obsidian_chunks.json"):
        """Save processed chunks to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            print(f"Chunks saved to: {output_file}")
        except Exception as e:
            print(f"Error saving chunks: {e}")


def main():
    """Main function to run Obsidian chunking"""
    vault_path = r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro"
    
    print("Obsidian Chunker - SOTA Implementation")
    print("="*50)
    
    # Initialize chunker
    chunker = ObsidianChunker(vault_path, target_tokens=2000)
    
    # Process entire vault
    chunks_data = chunker.process_entire_vault()
    
    # Save to JSON
    if chunks_data:
        chunker.save_chunks_to_json(chunks_data)
        
        # Show sample chunk
        if chunks_data['chunks']:
            sample_chunk = chunks_data['chunks'][0]
            print(f"\nSample Chunk Preview:")
            print(f"Chunk ID: {sample_chunk['chunk_id']}")
            print(f"Source: {sample_chunk['metadata'].source_file}")
            print(f"Wikilinks: {sample_chunk['metadata'].wikilinks}")
            print(f"Tags: {sample_chunk['metadata'].tags}")
            print(f"Content Preview: {sample_chunk['content'][:200]}...")


if __name__ == "__main__":
    main()
