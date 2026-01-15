#!/usr/bin/env python3
"""
Script to parse the algorithm and core methods documentation and extract function information in the required format.
"""

import os
import re
import json
import argparse
from typing import Dict, List, Any
from pathlib import Path


def parse_markdown_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse the markdown file and extract function information."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content by function sections
    function_sections = re.split(r'\n# FUNCTION:', content)[1:]  # Skip the first part (title)
    
    functions_data = []
    
    for section in function_sections:
        func_data = {}
        
        # Extract function name
        lines = section.strip().split('\n')
        func_name = lines[0].strip()
        func_data['function_name'] = func_name
        
        # Initialize fields
        func_data['overview'] = ""
        func_data['function_signature'] = ""
        func_data['input_parameters'] = ""
        func_data['detailed_description'] = ""
        func_data['output'] = ""
        func_data['func_implementation'] = ""
        func_data['test_code_path'] = ""
        
        current_section = None
        current_content = []
        implementation_location = ""
        test_code_location = ""
        
        for line in lines[1:]:
            line_stripped = line.strip()
            
            # Process previous section before starting new one
            # Support both Chinese and English headers
            if line_stripped.startswith('## 功能概述') or line_stripped.startswith('## Function Overview'):
                current_section = 'overview'
                current_content = []
            elif line_stripped.startswith('## 函数签名') or line_stripped.startswith('## Function Signature'):
                if current_section == 'overview':
                    func_data['overview'] = '\n'.join(current_content)
                current_section = 'function_signature'
                current_content = []
            elif line_stripped.startswith('## 输入参数') or line_stripped.startswith('## Input Parameters'):
                if current_section == 'overview':
                    func_data['overview'] = '\n'.join(current_content)
                elif current_section == 'function_signature':
                    func_data['function_signature'] = '\n'.join(current_content)
                current_section = 'input_parameters'
                current_content = []
            elif line_stripped.startswith('## 详细描述') or line_stripped.startswith('## Detailed Description'):
                if current_section == 'input_parameters':
                    func_data['input_parameters'] = '\n'.join(current_content)
                current_section = 'detailed_description'
                current_content = []
            elif line_stripped.startswith('## 输出') or line_stripped.startswith('## Output'):
                if current_section == 'detailed_description':
                    func_data['detailed_description'] = '\n'.join(current_content)
                current_section = 'output'
                current_content = []
            elif line_stripped.startswith('## 函数实现') or line_stripped.startswith('## Function Implementation'):
                if current_section == 'output':
                    func_data['output'] = '\n'.join(current_content)
                current_section = 'implementation'
                current_content = []
            elif line_stripped.startswith('## 测试代码') or line_stripped.startswith('## Test Code'):
                if current_section == 'implementation':
                    implementation_location = '\n'.join(current_content)
                current_section = 'test_code'
                current_content = []
            elif line_stripped.startswith('---'):
                # Process the last section before ending
                if current_section == 'test_code':
                    test_code_location = '\n'.join(current_content)
                break
            elif line_stripped and not line_stripped.startswith('#'):
                current_content.append(line_stripped)
        
        # Handle the last section if it doesn't end with ---
        if current_content:
            section_text = '\n'.join(current_content)
            if current_section == 'overview':
                func_data['overview'] = section_text
            elif current_section == 'function_signature':
                func_data['function_signature'] = section_text
            elif current_section == 'input_parameters':
                func_data['input_parameters'] = section_text
            elif current_section == 'detailed_description':
                func_data['detailed_description'] = section_text
            elif current_section == 'output':
                func_data['output'] = section_text
            elif current_section == 'implementation':
                implementation_location = section_text
            elif current_section == 'test_code':
                test_code_location = section_text
        
        functions_data.append((func_data, implementation_location, test_code_location))
    
    return functions_data


def extract_function_implementation(implementation_location: str, function_name: str, code_base_path: str) -> str:
    """Extract the specific function implementation from the source file."""
    if not implementation_location:
        return ""
    
    # Parse implementation location (e.g., "code/recipe/prime/prime_core_algos.py:line 21-79")
    match = re.search(r'(.+\.py):line (\d+)-(\d+)', implementation_location)
    if not match:
        return ""
    
    file_path = match.group(1)
    start_line = int(match.group(2))
    end_line = int(match.group(3))
    
    # Construct full path
    base_path = Path(code_base_path)
    source_file_path = base_path / file_path
    
    if not source_file_path.exists():
        return ""
    
    try:
        with open(source_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract the specified lines (convert to 0-based indexing)
        if start_line <= len(lines) and end_line <= len(lines):
            function_lines = lines[start_line-1:end_line]
            return ''.join(function_lines).strip()
        else:
            return ""
    
    except Exception as e:
        print(f"Error reading file {source_file_path}: {e}")
        return ""


def process_functions(functions_data: List, code_base_path: str, test_base_path: str) -> List[Dict[str, Any]]:
    """Process the parsed function data to extract implementations and test paths."""
    
    processed_functions = []
    
    for func_data, implementation_location, test_code_location in functions_data:
        # Extract function implementation from source code
        func_data['func_implementation'] = extract_function_implementation(
            implementation_location, func_data['function_name'], code_base_path
        )
        
        # Add implementation location from markdown file (now contains full path)
        func_data['implementation_location'] = implementation_location
        
        # Set test code path (now contains full path from markdown)
        func_data['test_code_path'] = test_code_location
        
        processed_functions.append(func_data)
    
    return processed_functions


def main():
    """Main function to parse markdown and generate jsonl output."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parse algorithm methods documentation and extract function information")
    parser.add_argument("--input", "-i", required=True, help="Input markdown file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    parser.add_argument("--code-base", "-c", required=True, help="Code base directory path for searching source files")
    parser.add_argument("--test-base", "-t", help="Test base path prefix (default: verl/test_examples/prime/code/tests)")
    
    args = parser.parse_args()
    
    # Set default test base path if not provided
    test_base_path = args.test_base if args.test_base else "verl/test_examples/prime/code/tests"
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input markdown file {args.input} does not exist!")
        return 1
    
    # Validate code base directory exists
    if not Path(args.code_base).exists():
        print(f"Error: Code base directory {args.code_base} does not exist!")
        return 1
    
    print(f"Input markdown file: {args.input}")
    print(f"Output JSONL file: {args.output}")
    print(f"Code base directory: {args.code_base}")
    print(f"Test base path: {test_base_path}")
    print()
    
    # Parse markdown file
    print("Parsing markdown file...")
    try:
        functions_data = parse_markdown_file(args.input)
    except Exception as e:
        print(f"Error parsing markdown file: {e}")
        return 1
    
    # Process functions to extract implementations and set test paths
    print("Processing function implementations...")
    processed_functions = process_functions(functions_data, args.code_base, test_base_path)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to jsonl file
    print(f"Saving data to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for func_data in processed_functions:
            json_line = json.dumps(func_data, ensure_ascii=False, indent=None)
            f.write(json_line + '\n')
    
    print(f"Successfully processed {len(processed_functions)} functions and saved to {args.output}")
    print()
    
    # Print summary
    for i, func_data in enumerate(processed_functions, 1):
        print(f"{i}. {func_data['function_name']}")
        print(f"   Implementation length: {len(func_data['func_implementation'])} characters")
        print(f"   Test path: {func_data['test_code_path']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
