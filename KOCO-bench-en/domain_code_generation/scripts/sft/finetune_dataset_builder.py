#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Builder for Code Repository
Convert valuable files from a code repository into LLM training dataset
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime
from transformers import AutoTokenizer

class DatasetBuilder:
    """Code repository dataset builder"""
    
    # Supported file types
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.md': 'markdown', 
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.sh': 'shell',
        '.json': 'json',
        '.txt': 'text',
        '.rst': 'rst',
        '.cfg': 'config',
        '.ini': 'config',
        '.dockerfile': 'dockerfile',
        '.gitignore': 'gitignore',
        '.requirements': 'requirements'
    }
    
    # Directories to exclude
    EXCLUDED_DIRS = {
        '__pycache__', '.git', '.pytest_cache', 'node_modules', 
        '.vscode', '.idea', 'build', 'dist', '.tox', 'venv', 
        'env', '.env', 'logs', 'tmp', '.DS_Store'
    }
    
    # Files to exclude
    EXCLUDED_FILES = {
        '.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll',
        '.exe', '.bin', '.log', '.tmp', '.swp', '~'
    }
    
    def __init__(self, source_dir: str, output_file: str, max_file_size: int = 1024*1024, 
                 tokenizer_path: Optional[str] = None):
        """
        Initialize dataset builder
        
        Args:
            source_dir: Source code directory path
            output_file: Output file path
            max_file_size: Maximum file size limit (bytes)
            tokenizer_path: Tokenizer model path for token counting
        """
        self.source_dir = Path(source_dir)
        self.output_file = Path(output_file)
        self.max_file_size = max_file_size
        self.processed_files = 0
        self.skipped_files = 0
        
        # Initialize tokenizer
        self.tokenizer = None
        if tokenizer_path:
            try:
                print(f"Loading tokenizer from: {tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                print("✓ Tokenizer loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load tokenizer: {e}")
                print("Token counting will be skipped.")
        
    def should_process_file(self, file_path: Path) -> bool:
        """Determine if the file should be processed"""
        
        # Check file extension
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            # Check special file names
            if file_path.name.lower() not in ['dockerfile', 'makefile', 'readme', 'license', 'requirements.txt']:
                return False
        
        # Check file size
        try:
            if file_path.stat().st_size > self.max_file_size:
                return False
        except OSError:
            return False
            
        # Check if file is excluded
        if any(excluded in file_path.name.lower() for excluded in self.EXCLUDED_FILES):
            return False
            
        return True
    
    def should_process_dir(self, dir_path: Path) -> bool:
        """Determine if the directory should be processed"""
        return dir_path.name not in self.EXCLUDED_DIRS
    
    def get_file_type(self, file_path: Path) -> str:
        """Get file type"""
        ext = file_path.suffix.lower()
        if ext in self.SUPPORTED_EXTENSIONS:
            return self.SUPPORTED_EXTENSIONS[ext]
        
        # Handle special file names
        name = file_path.name.lower()
        if name == 'dockerfile':
            return 'dockerfile'
        elif name in ['makefile']:
            return 'makefile'
        elif name in ['readme', 'readme.txt']:
            return 'markdown'
        elif name in ['license', 'license.txt']:
            return 'text'
        elif name == 'requirements.txt':
            return 'requirements'
        else:
            return 'unknown'
    
    def read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    # Basic content validation
                    if len(content.strip()) == 0:
                        return None
                    return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return None
        
        print(f"Could not decode file: {file_path}")
        return None
    
    def create_training_sample(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Create training sample"""
        relative_path = file_path.relative_to(self.source_dir)
        file_type = self.get_file_type(file_path)
        
        # Generate file content hash
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Calculate token count
        token_count = 0
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(content, add_special_tokens=False)
                token_count = len(tokens)
            except Exception as e:
                print(f"Warning: Failed to tokenize {file_path}: {e}")
                token_count = 0
        
        sample = {
            'file_path': str(relative_path),
            'file_name': file_path.name,
            'file_type': file_type,
            'content': content,
            'size': len(content),
            'lines': len(content.splitlines()),
            'tokens': token_count,
            'content_hash': content_hash,
            'created_at': datetime.now().isoformat()
        }
        
        # Add file type specific metadata
        if file_type == 'python':
            sample['language'] = 'python'
            sample['has_docstring'] = '"""' in content or "'''" in content
            sample['has_imports'] = 'import ' in content or 'from ' in content
        elif file_type == 'markdown':
            sample['language'] = 'markdown'
            sample['has_code_blocks'] = '```' in content
        elif file_type in ['yaml', 'json', 'toml']:
            sample['language'] = file_type
            sample['is_config'] = True
            
        return sample
    
    def scan_directory(self) -> List[Dict[str, Any]]:
        """Scan directory and generate training samples"""
        samples = []
        
        print(f"Scanning directory: {self.source_dir}")
        
        for root, dirs, files in os.walk(self.source_dir):
            root_path = Path(root)
            
            # Filter directories
            dirs[:] = [d for d in dirs if self.should_process_dir(root_path / d)]
            
            for file in files:
                file_path = root_path / file
                
                if not self.should_process_file(file_path):
                    self.skipped_files += 1
                    continue
                
                # Read file content
                content = self.read_file_content(file_path)
                if content is None:
                    self.skipped_files += 1
                    continue
                
                # Create training sample
                try:
                    sample = self.create_training_sample(file_path, content)
                    samples.append(sample)
                    self.processed_files += 1
                    
                    if self.processed_files % 50 == 0:
                        print(f"Processed {self.processed_files} files...")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    self.skipped_files += 1
                    continue
        
        return samples
    
    def save_dataset(self, samples: List[Dict[str, Any]], format: str = 'jsonl') -> None:
        """Save dataset"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'jsonl':
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        elif format.lower() == 'json':
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def generate_statistics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate dataset statistics"""
        file_types = {}
        total_size = 0
        total_lines = 0
        total_tokens = 0
        
        for sample in samples:
            file_type = sample['file_type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_size += sample['size']
            total_lines += sample['lines']
            total_tokens += sample.get('tokens', 0)
        
        stats = {
            'total_files': len(samples),
            'processed_files': self.processed_files,
            'skipped_files': self.skipped_files,
            'file_types': file_types,
            'total_size_chars': total_size,
            'total_lines': total_lines,
            'total_tokens': total_tokens,
            'average_file_size': total_size / len(samples) if samples else 0,
            'average_lines_per_file': total_lines / len(samples) if samples else 0,
            'average_tokens_per_file': total_tokens / len(samples) if samples else 0,
            'source_directory': str(self.source_dir),
            'output_file': str(self.output_file),
            'generated_at': datetime.now().isoformat()
        }
        
        return stats
    
    def build_dataset(self, output_format: str = 'jsonl') -> Dict[str, Any]:
        """Build complete dataset"""
        print(f"Building dataset from: {self.source_dir}")
        print(f"Output file: {self.output_file}")
        
        # Scan directory and generate samples
        samples = self.scan_directory()
        
        if not samples:
            print("No valid files found!")
            return {}
        
        # Save dataset
        self.save_dataset(samples, output_format)
        
        # Generate statistics
        stats = self.generate_statistics(samples)
        
        # Save statistics
        stats_file = self.output_file.with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\nDataset build completed!")
        print(f"Total files processed: {stats['processed_files']}")
        print(f"Files skipped: {stats['skipped_files']}")
        print(f"File types: {stats['file_types']}")
        print(f"Total size: {stats['total_size_chars']:,} characters")
        print(f"Total lines: {stats['total_lines']:,}")
        if stats['total_tokens'] > 0:
            print(f"Total tokens: {stats['total_tokens']:,}")
        print(f"Dataset saved to: {self.output_file}")
        print(f"Statistics saved to: {stats_file}")
        
        return stats

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Build training dataset from codebase')
    parser.add_argument('--source-dir', '-s', required=True,
                       help='Source directory containing VERL codebase')
    parser.add_argument('--output-file', '-o', required=True,
                       help='Output file path for the dataset')
    parser.add_argument('--format', '-f', choices=['json', 'jsonl'], default='jsonl',
                       help='Output format (default: jsonl)')
    parser.add_argument('--max-file-size', '-m', type=int, default=1024*1024,
                       help='Maximum file size in bytes (default: 1MB)')
    parser.add_argument('--tokenizer-path', '-t', type=str, default=None,
                       help='Path to tokenizer model for token counting')
    
    args = parser.parse_args()
    
    # Validate input directory
    source_path = Path(args.source_dir)
    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: Source directory does not exist: {source_path}")
        return 1
    
    # Create dataset builder
    builder = DatasetBuilder(
        source_dir=args.source_dir,
        output_file=args.output_file,
        max_file_size=args.max_file_size,
        tokenizer_path=args.tokenizer_path
    )
    
    # Build dataset
    try:
        stats = builder.build_dataset(args.format)
        if stats:
            print(f"\nSuccess! Generated {stats['total_files']} training samples.")
            return 0
        else:
            print("Failed to build dataset.")
            return 1
    except Exception as e:
        print(f"Error building dataset: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
