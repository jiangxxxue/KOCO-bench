#!/usr/bin/env python3
"""
Script to add prompt field to algorithm_methods_data.jsonl
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


def add_prompts_to_data(input_file_path, output_file_path, metadata_path):
    """Add prompt field to each record in the JSONL file."""
    
    # Load framework metadata
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        framework_name = metadata.get("framework_name", "verl")
        framework_description = metadata.get("framework_description", "verl is a flexible and efficient RL training framework for large language models")
    except Exception as e:
        print(f"Warning: Could not load metadata from {metadata_path}: {e}")
        # Fallback values
        framework_name = "verl"
        framework_description = "verl is a flexible and efficient RL training framework for large language models"
    
    system_prompt = f"You are a Python programmer developing with the {framework_name} framework. {framework_description}"
    
    # Read the original file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line and add prompt
    updated_data = []
    for line in lines:
        data = json.loads(line.strip())
        
        # Create user prompt
        user_prompt = create_user_prompt(
            data['function_name'],
            data['overview'],
            data.get('function_signature', ''),  # Use get() for backward compatibility
            data['input_parameters'],
            data['detailed_description'],
            data['output']
        )
        
        # Create the prompt structure
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Add prompt to data
        data['prompt'] = prompt
        
        updated_data.append(data)
    
    # Write to output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for data in updated_data:
            json_line = json.dumps(data, ensure_ascii=False, indent=None)
            f.write(json_line + '\n')
    
    print(f"Successfully added prompts to {len(updated_data)} records")
    return len(updated_data)


def main():
    """Main function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Add prompt field to algorithm methods data")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file path")
    parser.add_argument("--metadata", "-m", required=True, help="Metadata JSON file path")
    parser.add_argument("--output", "-o", help="Output JSONL file path (default: overwrite input file)")
    
    args = parser.parse_args()
    
    # Set default output file to input file if not specified
    output_file = args.output if args.output else args.input
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist!")
        return 1
    
    # Validate metadata file exists
    if not Path(args.metadata).exists():
        print(f"Error: Metadata file {args.metadata} does not exist!")
        return 1
    
    print(f"Input file: {args.input}")
    print(f"Metadata file: {args.metadata}")
    print(f"Output file: {output_file}")
    print()
    
    # Add prompts to data
    try:
        num_records = add_prompts_to_data(args.input, output_file, args.metadata)
    except Exception as e:
        print(f"Error processing files: {e}")
        return 1
    
    # Verify the result
    print(f"\nVerifying the updated file...")
    with open(output_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        first_record = json.loads(first_line.strip())
        
        print(f"Fields in updated record: {list(first_record.keys())}")
        print(f"Prompt structure: {len(first_record['prompt'])} messages")
        print(f"System prompt length: {len(first_record['prompt'][0]['content'])}")
        print(f"User prompt length: {len(first_record['prompt'][1]['content'])}")
        
        # Show a sample of the user prompt
        sample_user_prompt = first_record['prompt'][1]['content'][:200] + "..."
        print(f"Sample user prompt: {sample_user_prompt}")
    
    return 0


if __name__ == "__main__":
    exit(main())
