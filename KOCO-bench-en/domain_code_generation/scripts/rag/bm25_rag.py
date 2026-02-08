#!/usr/bin/env python3
"""
BM25-based Retrieval Augmented Generation (RAG) for Code

Usage:
    python bm25_rag.py \\
        --input data/algorithm_methods.jsonl \\
        --metadata metadata.json \\
        --knowledge-base /path/to/code/repo1 \\
        --knowledge-base /path/to/code/repo2 \\
        --framework framework_name \\
        --output output.jsonl
"""

import json
import os
import re
import ast
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
import tokenize
import io

from rank_bm25 import BM25Okapi

class CodeChunk:
    """
    Represents a code chunk extracted from source files.
    
    Attributes:
        code: The source code text
        file_path: Path to the source file
        start_line: Starting line number in the file
        end_line: Ending line number in the file
        chunk_type: Type of chunk (function, class, or module)
    """
    
    def __init__(self, code: str, file_path: str, start_line: int, end_line: int, chunk_type: str = "function"):
        self.code = code
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.chunk_type = chunk_type
    
    def __repr__(self):
        return f"CodeChunk({self.chunk_type}, {self.file_path}:{self.start_line}-{self.end_line})"


class CodeTokenizer:
    """
    Tokenizer for extracting relevant keywords from code.
    
    This tokenizer extracts identifiers, keywords, and string literals
    from Python code for BM25 indexing and retrieval.
    """
    
    @staticmethod
    def tokenize_code(code: str) -> List[str]:
        """
        Tokenize code by extracting identifiers, keywords, and string literals.
        
        Args:
            code: Source code string
            
        Returns:
            List of lowercase tokens
        """
        tokens = []
        try:
            # Use Python's tokenize module
            token_gen = tokenize.generate_tokens(io.StringIO(code).readline)
            for tok in token_gen:
                token_type = tok.type
                token_str = tok.string
                
                # Extract identifiers, keywords, and string literals
                if token_type in (tokenize.NAME, tokenize.STRING):
                    # For strings, extract content
                    if token_type == tokenize.STRING:
                        # Remove quotes and extract content
                        content = token_str.strip('"\'').strip('"""').strip("'''")
                        # Split into words
                        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content)
                        tokens.extend([w.lower() for w in words])
                    else:
                        tokens.append(token_str.lower())
        except:
            # If tokenization fails, use simple regex
            tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
            tokens = [t.lower() for t in tokens]
        
        return tokens
    
    @staticmethod
    def extract_function_signature_tokens(signature: str) -> List[str]:
        """
        Extract tokens from function signature for retrieval.
        
        Args:
            signature: Function signature string
            
        Returns:
            List of lowercase tokens from function and parameter names
        """
        tokens = []
        # Extract function name
        func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', signature)
        if func_match:
            func_name = func_match.group(1)
            # Split camelCase and snake_case
            tokens.extend(re.split(r'[_\.]', func_name))
        
        # Extract parameter names
        param_match = re.search(r'\(([^)]*)\)', signature)
        if param_match:
            params_str = param_match.group(1)
            for param in params_str.split(','):
                param = param.strip()
                # Remove type annotations and default values
                param_name = param.split(':')[0].split('=')[0].strip()
                if param_name:
                    tokens.extend(re.split(r'[_\.]', param_name))
        
        return [t.lower() for t in tokens if t]


class CodeChunker:
    """
    Splits Python source files into function/class-level code chunks.
    
    Uses AST parsing to extract meaningful code segments for retrieval.
    """
    
    @staticmethod
    def chunk_file(file_path: str) -> List[CodeChunk]:
        """
        Split a Python file into code chunks.
        
        Args:
            file_path: Path to Python source file
            
        Returns:
            List of CodeChunk objects
        """
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            return chunks
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # If parsing fails, return entire file as one chunk
            return [CodeChunk(content, file_path, 1, len(lines), "module")]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Extract function code
                start_line = node.lineno
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                
                # Get function code
                func_lines = lines[start_line-1:end_line]
                func_code = '\n'.join(func_lines)
                
                chunks.append(CodeChunk(
                    func_code,
                    file_path,
                    start_line,
                    end_line,
                    "function"
                ))
            
            elif isinstance(node, ast.ClassDef):
                # Extract class code
                start_line = node.lineno
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                
                class_lines = lines[start_line-1:end_line]
                class_code = '\n'.join(class_lines)
                
                chunks.append(CodeChunk(
                    class_code,
                    file_path,
                    start_line,
                    end_line,
                    "class"
                ))
        
        # If no functions or classes found, return entire file
        if not chunks:
            chunks.append(CodeChunk(content, file_path, 1, len(lines), "module"))
        
        return chunks


class BM25RAG:
    """
    BM25-based Retrieval Augmented Generation system for code.
    
    This class builds a BM25 index over code chunks from knowledge bases
    and retrieves relevant code examples for query functions.
    """
    
    def __init__(self, knowledge_base_paths: List[str], exclude_paths: Set[str] = None):
        """
        Initialize BM25 RAG system.
        
        Args:
            knowledge_base_paths: List of paths to code repositories (knowledge bases)
            exclude_paths: Set of file paths to exclude (e.g., ground-truth files)
        """
        # Support single path (backward compatibility) or multiple paths
        if isinstance(knowledge_base_paths, str):
            knowledge_base_paths = [knowledge_base_paths]
        self.knowledge_base_paths = [Path(p) for p in knowledge_base_paths]
        self.exclude_paths = exclude_paths or set()
        self.chunks: List[CodeChunk] = []
        self.tokenizer = CodeTokenizer()
        self.bm25 = None
        
    def build_index(self):
        """
        Build BM25 index from code chunks in knowledge bases.
        
        Scans all Python files in knowledge base paths, chunks them,
        and builds a BM25 index for retrieval.
        """
        print("Scanning codebase and building chunks...")
        print(f"Knowledge base paths: {len(self.knowledge_base_paths)}")
        for kb_path in self.knowledge_base_paths:
            print(f"  - {kb_path}")
        
        total_chunks = 0
        excluded_files = 0
        all_python_files = []
        
        # Scan all Python files in knowledge base paths
        for kb_path in self.knowledge_base_paths:
            if not kb_path.exists():
                print(f"Warning: Knowledge base path does not exist: {kb_path}")
                continue
            python_files = list(kb_path.rglob("*.py"))
            all_python_files.extend([(py_file, kb_path) for py_file in python_files])
        
        print(f"Found {len(all_python_files)} Python files in total")
        
        for py_file, kb_path in all_python_files:
            # Convert to relative path for matching
            rel_path = py_file.relative_to(kb_path)
            rel_path_str = str(rel_path).replace('\\', '/')
            
            # Check if should exclude
            should_exclude = False
            for exclude_path in self.exclude_paths:
                # Exact match
                if rel_path_str == exclude_path:
                    should_exclude = True
                    break
                # Path suffix match
                if rel_path_str.endswith('/' + exclude_path) or rel_path_str.endswith(exclude_path):
                    should_exclude = True
                    break
            
            if should_exclude:
                excluded_files += 1
                continue
            
            # Chunk the file
            chunks = CodeChunker.chunk_file(str(py_file))
            self.chunks.extend(chunks)
            total_chunks += len(chunks)
        
        print(f"Excluded {excluded_files} files (ground-truth)")
        print(f"Created {total_chunks} code chunks")
        
        if not self.chunks:
            print("Warning: No code chunks found!")
            return
        
        # Build BM25 index
        print("Building BM25 index...")
        tokenized_corpus = []
        for chunk in self.chunks:
            tokens = self.tokenizer.tokenize_code(chunk.code)
            tokenized_corpus.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built successfully!")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """
        Retrieve similar code chunks for a query.
        
        Args:
            query: Query string (typically a function signature)
            top_k: Number of top results to return
            
        Returns:
            List of (CodeChunk, score) tuples sorted by relevance
        """
        if not self.bm25:
            return []
        
        # Tokenize query
        query_tokens = self.tokenizer.extract_function_signature_tokens(query)
        if not query_tokens:
            # If no tokens from signature, use entire query
            query_tokens = self.tokenizer.tokenize_code(query)
        
        # BM25 retrieval
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return results with positive scores
                results.append((self.chunks[idx], scores[idx]))
        
        return results


def parse_groundtruth_paths(jsonl_path: str, knowledge_base_path: str) -> Set[str]:
    """
    Parse ground-truth file paths from JSONL data to exclude from retrieval.
    
    This prevents the RAG system from retrieving the target implementation
    (ground-truth) itself, which would constitute data leakage.
    
    Args:
        jsonl_path: Path to JSONL data file
        knowledge_base_path: Path to knowledge base directory
    
    Returns:
        Set of relative file paths to exclude
    """
    exclude_paths = set()
    knowledge_base = Path(knowledge_base_path)
    
    print(f"Knowledge base path (retrieval scope): {knowledge_base}")
    
    if not knowledge_base.exists():
        print(f"Warning: Knowledge base directory does not exist: {knowledge_base}")
        return exclude_paths
    
    # Read JSONL file and parse implementation_location fields
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            impl_location = data.get('implementation_location', '')
            
            if not impl_location:
                continue
            
            # Parse format: "code/path/to/file.py:line X-Y"
            match = re.search(r'(.+\.py):', impl_location)
            if match:
                file_path = match.group(1)
                # Remove "code/" prefix if present
                if file_path.startswith('code/'):
                    file_path = file_path[5:]
                
                # Full path to ground-truth file
                gt_file = knowledge_base / file_path
                
                # Check if ground-truth file exists
                if not gt_file.exists():
                    print(f"Warning: Ground-truth file does not exist: {gt_file}")
                    continue
                
                # Add to exclusion list (relative to knowledge_base)
                rel_path = gt_file.relative_to(knowledge_base)
                exclude_paths.add(str(rel_path).replace('\\', '/'))
                print(f"  Will exclude ground-truth: {rel_path}")
    
    return exclude_paths


def format_retrieved_code(chunks_with_scores: List[Tuple[CodeChunk, float]], knowledge_base_path: str = None) -> str:
    """
    Format retrieved code chunks for inclusion in prompts.
    
    Args:
        chunks_with_scores: List of (CodeChunk, score) tuples
        knowledge_base_path: Base path for computing relative paths
        
    Returns:
        Formatted string with retrieved code examples
    """
    if not chunks_with_scores:
        return ""
    
    formatted = "\n\n**Retrieved Similar Code Examples:**\n\n"
    
    knowledge_base = Path(knowledge_base_path) if knowledge_base_path else None
    
    for i, (chunk, score) in enumerate(chunks_with_scores, 1):
        # Get relative path for display
        if knowledge_base and Path(chunk.file_path).is_absolute():
            try:
                rel_path = Path(chunk.file_path).relative_to(knowledge_base)
                rel_path_str = str(rel_path).replace('\\', '/')
            except ValueError:
                # If relative path cannot be computed, use filename
                rel_path_str = Path(chunk.file_path).name
        else:
            # If already relative or no knowledge_base, use as-is
            rel_path_str = chunk.file_path
        
        formatted += f"Example {i} (relevance score: {score:.2f}) from `{rel_path_str}`:\n"
        formatted += "```python\n"
        formatted += chunk.code
        formatted += "\n```\n\n"
    
    return formatted


def create_rag_user_prompt(function_name, overview, function_signature, input_parameters, 
                          detailed_description, output, retrieved_code: str = ""):
    """
    Create a user prompt with RAG-enhanced context.
    
    Args:
        function_name: Name of the function to implement
        overview: High-level function overview
        function_signature: Function signature
        input_parameters: Description of input parameters
        detailed_description: Detailed implementation requirements
        output: Expected output description
        retrieved_code: Formatted retrieved code examples (optional)
        
    Returns:
        Complete user prompt string
    """
    
    user_prompt = f"""Please implement a function named `{function_name}`.

**Function Overview:**
{overview}

**Function Signature:**
```python
{function_signature}
```

**Input Parameters:**
{input_parameters}

**Detailed Description:**
{detailed_description}

**Expected Output:**
{output}"""

    # Add retrieved code if available
    if retrieved_code:
        user_prompt += retrieved_code
    
    user_prompt += """

Please implement this function based on the above information, ensuring that:
1. The function signature matches the one provided above
2. The implementation follows the logic described in the detailed description
3. All input parameters are properly handled
4. The return value matches the expected output format
5. The code style is clear with necessary comments
6. You may refer to the retrieved similar code examples for implementation 

Please provide the Python function implementation code directly."""

    return user_prompt


def add_rag_prompts_to_data(input_file_path: str, output_file_path: str, metadata_path: str,
                            knowledge_base_paths: List[str], framework_name: str, top_k: int = 5):
    """
    Add RAG-enhanced prompts to JSONL data.
    
    Processes each entry in the input JSONL file, retrieves relevant code examples,
    and creates enhanced prompts with retrieved context.
    
    Args:
        input_file_path: Input JSONL file path
        output_file_path: Output JSONL file path
        metadata_path: Framework metadata JSON file path
        knowledge_base_paths: List of knowledge base repository paths
        framework_name: Name of the framework
        top_k: Number of top-k code chunks to retrieve
        
    Returns:
        Number of processed records
    """
    
    # Load framework metadata
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        framework_name_from_meta = metadata.get("framework_name", framework_name)
        framework_description = metadata.get("framework_description", "")
    except Exception as e:
        print(f"Warning: Could not load metadata from {metadata_path}: {e}")
        framework_name_from_meta = framework_name
        framework_description = f"{framework_name} is a framework for domain-specific code generation"
    
    system_prompt = f"You are a Python programmer developing with the {framework_name_from_meta} framework. {framework_description}"
    
    # Parse ground-truth paths to exclude
    print("Parsing ground-truth paths to exclude...")
    primary_kb_path = knowledge_base_paths[0] if knowledge_base_paths else ""
    exclude_paths = parse_groundtruth_paths(input_file_path, primary_kb_path)
    print(f"Will exclude {len(exclude_paths)} ground-truth files")
    if exclude_paths:
        print(f"Excluded paths: {list(exclude_paths)}")
    
    # Initialize BM25 RAG system
    rag = BM25RAG(knowledge_base_paths, exclude_paths)
    rag.build_index()
    
    # Read input file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    updated_data = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        
        data = json.loads(line.strip())
        print(f"Processing {i}/{len(lines)}: {data['function_name']}")
        
        # Retrieve using function signature
        function_signature = data.get('function_signature', '')
        retrieved_chunks = rag.retrieve(function_signature, top_k=top_k)
        
        # Format retrieval results
        retrieved_code = format_retrieved_code(retrieved_chunks, primary_kb_path)
        
        # Create user prompt
        user_prompt = create_rag_user_prompt(
            data['function_name'],
            data['overview'],
            function_signature,
            data['input_parameters'],
            data['detailed_description'],
            data['output'],
            retrieved_code
        )
        
        # Create prompt structure
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Add prompt to data
        data['prompt'] = prompt
        updated_data.append(data)
    
    # Write output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for data in updated_data:
            json_line = json.dumps(data, ensure_ascii=False, indent=None)
            f.write(json_line + '\n')
    
    print(f"\nSuccessfully added RAG-enhanced prompts to {len(updated_data)} records")
    return len(updated_data)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Add RAG-enhanced prompt field to algorithm methods data")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file path")
    parser.add_argument("--metadata", "-m", required=True, help="Metadata JSON file path")
    parser.add_argument("--output", "-o", help="Output JSONL file path (default: overwrite input file)")
    parser.add_argument("--knowledge-base", "-k", required=True, action='append', 
                        help="Knowledge base code directory path (can specify multiple times, e.g., -k path1 -k path2)")
    parser.add_argument("--framework", "-f", required=True, help="Framework name (e.g., verl, smolagents)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved code chunks (default: 5)")
    
    args = parser.parse_args()
    
    # Set default output file
    output_file = args.output if args.output else args.input
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist!")
        return 1
    
    # Validate metadata file
    if not Path(args.metadata).exists():
        print(f"Error: Metadata file {args.metadata} does not exist!")
        return 1
    
    # Validate knowledge base paths
    knowledge_base_paths = args.knowledge_base
    for kb_path in knowledge_base_paths:
        if not Path(kb_path).exists():
            print(f"Error: Knowledge base path {kb_path} does not exist!")
            return 1
    
    print(f"Input file: {args.input}")
    print(f"Metadata file: {args.metadata}")
    print(f"Knowledge bases ({len(knowledge_base_paths)}):")
    for kb_path in knowledge_base_paths:
        print(f"  - {kb_path}")
    print(f"Framework: {args.framework}")
    print(f"Output file: {output_file}")
    print(f"Top-K: {args.top_k}")
    print()
    
    # Add RAG-enhanced prompts
    try:
        num_records = add_rag_prompts_to_data(
            args.input,
            output_file,
            args.metadata,
            knowledge_base_paths,
            args.framework,
            args.top_k
        )
    except Exception as e:
        print(f"Error processing files: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Verify results
    print(f"\nVerifying the updated file...")
    with open(output_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        first_record = json.loads(first_line.strip())
        
        print(f"Fields in updated record: {list(first_record.keys())}")
        print(f"Prompt structure: {len(first_record['prompt'])} messages")
        print(f"System prompt length: {len(first_record['prompt'][0]['content'])}")
        print(f"User prompt length: {len(first_record['prompt'][1]['content'])}")
        
        # Check if retrieved code is included
        user_content = first_record['prompt'][1]['content']
        if "Retrieved Similar Code Examples" in user_content:
            print("✓ RAG retrieval results found in prompt")
        else:
            print("⚠ Warning: No RAG retrieval results found")
    
    return 0


if __name__ == "__main__":
    exit(main())
