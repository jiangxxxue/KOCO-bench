#!/usr/bin/env python3
"""
Script to add prompt and keyword fields to algorithm_methods_data.jsonl
This script generates complete prompt structures with repository information for Claude Code usage.
"""

import json
import os
import argparse
from pathlib import Path


def create_user_prompt(function_name, overview, function_signature, input_parameters, detailed_description, output):
    """Create a user prompt based on function information."""

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
{output}

Please implement this function based on the above information, ensuring that:
1. The function signature matches the one provided above
2. The implementation follows the logic described in the detailed description
3. All input parameters are properly handled
4. The return value matches the expected output format
5. The code style is clear with necessary comments

Please provide the Python function implementation code directly."""

    return user_prompt


def create_system_prompt(framework_name, framework_description, knowledge_corpus_path, test_example_path):
    """
    Create an enhanced system prompt with repository information.

    Args:
        framework_name: Name of the framework (e.g., "RAGAnything", "verl")
        framework_description: Description of the framework
        knowledge_corpus_path: Relative path to knowledge corpus (e.g., "./raganything/knowledge_corpus")
        test_example_path: Relative path to test example code (e.g., "./raganything/test_examples/rag4chat")

    Returns:
        Complete system prompt string
    """
    system_prompt = f"""You are a Python programmer developing with the {framework_name} framework. {framework_description}

The framework knowledge base is located at: {knowledge_corpus_path}. This contains the core {framework_name} framework implementation and documentation that provides the foundational knowledge and patterns for this domain.

The development repository for this specific project is located at: {test_example_path}. This contains the project-specific code, requirements, and implementation that builds upon the framework knowledge base.

This dual-repository setup means you have access to both the foundational framework knowledge and the specific project implementation context when developing functions for this domain."""

    return system_prompt


def build_keyword_paths(base_dir, framework, test_example):
    """
    Build absolute paths for the keyword field.

    Args:
        base_dir: Base directory path (absolute)
        framework: Framework name (e.g., "raganything", "verl")
        test_example: Test example name (e.g., "rag4chat", "ARES")

    Returns:
        List of absolute paths [knowledge_corpus_path, test_example_code_path]
    """
    base_path = Path(base_dir).resolve()

    # Build knowledge corpus path
    knowledge_corpus_path = str(base_path / framework / "knowledge_corpus")

    # Build test example code path
    test_example_code_path = str(base_path / framework / "test_examples" / test_example / "code")

    return [knowledge_corpus_path, test_example_code_path]


def build_relative_paths(framework, test_example):
    """
    Build relative paths for system prompt.

    Args:
        framework: Framework name (e.g., "raganything", "verl")
        test_example: Test example name (e.g., "rag4chat", "ARES")

    Returns:
        Tuple of (knowledge_corpus_relative_path, test_example_relative_path)
    """
    knowledge_corpus_path = f"./{framework}/knowledge_corpus"
    test_example_path = f"./{framework}/test_examples/{test_example}"

    return knowledge_corpus_path, test_example_path


def add_prompts_and_keywords(input_file_path, output_file_path, metadata_path, base_dir, framework, test_example):
    """
    Add prompt and keyword fields to each record in the JSONL file.

    Args:
        input_file_path: Input JSONL file path
        output_file_path: Output JSONL file path
        metadata_path: Metadata JSON file path
        base_dir: Base directory for building absolute paths
        framework: Framework name
        test_example: Test example name
    """

    # Load framework metadata
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        framework_name = metadata.get("framework_name", framework)
        framework_description = metadata.get("framework_description",
            f"{framework} is a flexible and efficient framework for domain-specific development")
    except Exception as e:
        print(f"Warning: Could not load metadata from {metadata_path}: {e}")
        # Fallback values
        framework_name = framework
        framework_description = f"{framework} is a flexible and efficient framework for domain-specific development"

    # Build paths
    keyword_paths = build_keyword_paths(base_dir, framework, test_example)
    knowledge_corpus_rel, test_example_rel = build_relative_paths(framework, test_example)

    # Create enhanced system prompt
    system_prompt = create_system_prompt(
        framework_name,
        framework_description,
        knowledge_corpus_rel,
        test_example_rel
    )

    # Read the original file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process each line and add prompt and keyword
    updated_data = []
    for line in lines:
        if not line.strip():
            continue

        data = json.loads(line.strip())

        # Create user prompt
        user_prompt = create_user_prompt(
            data['function_name'],
            data['overview'],
            data.get('function_signature', ''),
            data['input_parameters'],
            data['detailed_description'],
            data['output']
        )

        # Create the prompt structure
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Add prompt and keyword to data
        data['prompt'] = prompt
        data['keyword'] = keyword_paths

        updated_data.append(data)

    # Write to output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for data in updated_data:
            json_line = json.dumps(data, ensure_ascii=False, indent=None)
            f.write(json_line + '\n')

    print(f"Successfully added prompts and keywords to {len(updated_data)} records")
    return len(updated_data)


def main():
    """Main function."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Add prompt and keyword fields to algorithm methods data"
    )
    parser.add_argument("--input", "-i", required=True,
                       help="Input JSONL file path")
    parser.add_argument("--metadata", "-m", required=True,
                       help="Metadata JSON file path")
    parser.add_argument("--output", "-o",
                       help="Output JSONL file path (default: {input}_with_prompts_v2.jsonl)")
    parser.add_argument("--base-dir", "-b", required=True,
                       help="Base directory for building absolute paths (e.g., /path/to/domain_knowledge_dataset)")
    parser.add_argument("--framework", "-f", required=True,
                       help="Framework name (e.g., raganything, verl)")
    parser.add_argument("--test-example", "-t", required=True,
                       help="Test example name (e.g., rag4chat, ARES)")

    args = parser.parse_args()

    # Set default output file if not specified
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input)
        output_file = str(input_path.parent / f"{input_path.stem}_with_prompts_v2.jsonl")

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist!")
        return 1

    # Validate metadata file exists
    if not Path(args.metadata).exists():
        print(f"Error: Metadata file {args.metadata} does not exist!")
        return 1

    # Validate base directory exists
    if not Path(args.base_dir).exists():
        print(f"Error: Base directory {args.base_dir} does not exist!")
        return 1

    print("=" * 60)
    print("Prompt and Keyword Generation")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Metadata file: {args.metadata}")
    print(f"Output file: {output_file}")
    print(f"Base directory: {args.base_dir}")
    print(f"Framework: {args.framework}")
    print(f"Test example: {args.test_example}")
    print()

    # Build and display paths
    keyword_paths = build_keyword_paths(args.base_dir, args.framework, args.test_example)
    print("Keyword paths (absolute):")
    for i, path in enumerate(keyword_paths, 1):
        exists = "✓" if Path(path).exists() else "✗"
        print(f"  {i}. [{exists}] {path}")
    print()

    # Add prompts and keywords to data
    try:
        num_records = add_prompts_and_keywords(
            args.input,
            output_file,
            args.metadata,
            args.base_dir,
            args.framework,
            args.test_example
        )
    except Exception as e:
        print(f"Error processing files: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Verify the result
    print(f"\nVerifying the updated file...")
    with open(output_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        first_record = json.loads(first_line.strip())

        print(f"Fields in updated record: {list(first_record.keys())}")
        print(f"Prompt structure: {len(first_record['prompt'])} messages")
        print(f"System prompt length: {len(first_record['prompt'][0]['content'])} characters")
        print(f"User prompt length: {len(first_record['prompt'][1]['content'])} characters")
        print(f"Keyword paths: {len(first_record['keyword'])} paths")

        # Show a sample of the system prompt
        print("\nSample system prompt:")
        print("-" * 60)
        sample_system_prompt = first_record['prompt'][0]['content'][:400] + "..."
        print(sample_system_prompt)
        print("-" * 60)

        # Show keyword paths
        print("\nKeyword paths:")
        for i, path in enumerate(first_record['keyword'], 1):
            print(f"  {i}. {path}")

    print("\n" + "=" * 60)
    print("✓ Successfully generated prompts with keywords!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
