#!/usr/bin/env python3
"""
Create Claude Code v2 output files with updated implementations.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any

def load_implementations_summary(summary_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load all implementations from summary file and organize by framework and function name.
    
    Returns:
        Dict with structure: {framework: {function_name: implementation}}
    """
    with open(summary_path, 'r', encoding='utf-8') as f:
        implementations = json.load(f)
    
    # Organize by framework and function name
    organized = {}
    for impl in implementations:
        framework = impl['framework']
        function_name = impl['function_name']
        
        if framework not in organized:
            organized[framework] = {}
        
        # Only include successful implementations
        if impl['status'] == 'success' and impl['implementation']:
            organized[framework][function_name] = impl['implementation']
    
    return organized

def create_framework_structure_v2():
    """
    Create Claude Code v2 framework-based directory structure.
    """
    
    # Framework to category mapping
    framework_mapping = {
        # open-r1 category
        "AlphaDrive": "open-r1",
        "CANOE": "open-r1", 
        "VLM-R1": "open-r1",
        "VisualQuality-R1": "open-r1",
        "ml-diffucoder": "open-r1",
        
        # raganything category
        "BookWorm": "raganything",
        "Chat-ANYTHING": "raganything",
        "law-unleashed-rag": "raganything", 
        "obsidianGraphRAG": "raganything",
        "rag4chat": "raganything",
        
        # smolagents category
        "DeepSearchAgents": "smolagents",
        "examples": "smolagents",
        "smolcc": "smolagents",
        "ToolBrain": "smolagents",


        # tensorrt_model_optimizer category
        "FlagScale": "tensorrt_model_optimizer",
        "Nemo": "tensorrt_model_optimizer",
        "TensorRT-Incubator": "tensorrt_model_optimizer", 
        "byteperf": "tensorrt_model_optimizer",
        
        # verl category
        "ARES": "verl",
        "DAPO": "verl",
        "LUFFY": "verl",
        "PACS": "verl", 
        "PURE": "verl",
        "critic-rl": "verl",
        "prime": "verl"
    }
    
    base_dir = Path("/home/gudako/repo/try/daily_try/jiangxue_domaincode/domain_knowledge_dataset")
    output_dir = base_dir / "output"
    
    # Load new implementations summary
    summary_file = output_dir / "generated_results" / "all_implementations_summary.json"
    implementations = load_implementations_summary(str(summary_file))
    
    print(f"Loaded implementations for {len(implementations)} frameworks")
    
    # Create new data_v2 directory structure
    data_v2_dir = output_dir / "data_v2"
    data_v2_dir.mkdir(exist_ok=True)
    
    # Create category directories and claude_code subdirectories
    for category in set(framework_mapping.values()):
        category_dir = data_v2_dir / category
        category_dir.mkdir(exist_ok=True)
        
        claude_code_dir = category_dir / "claude_code"
        claude_code_dir.mkdir(exist_ok=True)
    
    # Copy original data files to new structure
    original_data_dir = output_dir / "data"
    if original_data_dir.exists():
        for category in set(framework_mapping.values()):
            original_category_dir = original_data_dir / category
            new_category_dir = data_v2_dir / category
            
            if original_category_dir.exists():
                # Copy original data files (without completions)
                for orig_file in original_category_dir.glob("algorithm_methods_data_*.jsonl"):
                    if not orig_file.name.endswith("_output.jsonl"):
                        shutil.copy2(orig_file, new_category_dir / orig_file.name)
                        print(f"Copied original: {category}/{orig_file.name}")
    
    # Generate output files with updated implementations
    processed_count = 0
    
    for framework, category in framework_mapping.items():
        if framework in implementations:
            # Generate output filename
            output_filename = f"algorithm_methods_data_{framework}_output.jsonl"
            output_path = data_v2_dir / category / "claude_code" / output_filename
            
            # Find original file to base on
            original_file = None
            for original_dir in [output_dir / "data" / category, output_dir]:
                potential_file = original_dir / f"algorithm_methods_data_{framework}.jsonl"
                if not potential_file.exists():
                    potential_file = original_dir / f"{framework}_algorithm_methods.jsonl"
                
                if potential_file.exists():
                    original_file = potential_file
                    break
            
            if original_file:
                # Process the file with new implementations
                process_framework_file(
                    str(original_file), 
                    str(output_path), 
                    implementations[framework]
                )
                print(f"Generated v2: {category}/claude_code/{output_filename}")
                processed_count += 1
            else:
                print(f"Warning: No original file found for framework '{framework}'")
    
    return data_v2_dir, processed_count

def process_framework_file(input_file: str, output_file: str, implementations: Dict[str, str]):
    """
    Process a single framework file, adding completions field with updated implementations.
    
    Args:
        input_file: Path to original algorithm methods file
        output_file: Path to output file with completions
        implementations: Dict of {function_name: implementation} for this framework
    """
    output_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse JSON line
            data = json.loads(line)
            function_name = data.get('function_name', '')
            
            # Add completions field if implementation exists
            if function_name in implementations:
                # Format completion as a list (similar to open-r1 format)
                data['completions'] = [implementations[function_name]]
                data['num_completions'] = 1
                data['server'] = "claude_code_v2"  # Indicate updated source
                data['version'] = "v2"  # Mark as version 2
            else:
                # No implementation found
                data['completions'] = []
                data['num_completions'] = 0
                data['server'] = "claude_code_v2"
                data['version'] = "v2"
            
            # Write updated line
            output_lines.append(json.dumps(data, ensure_ascii=False))
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')

def main():
    data_v2_dir, processed_count = create_framework_structure_v2()
    
    print(f"\nSuccessfully created Claude Code v2 with {processed_count} framework files")
    print(f"New structure created at: {data_v2_dir}")
    
    # Print the final structure
    print("\n" + "="*60)
    print("Claude Code v2 Directory Structure:")
    print("="*60)
    
    for category_dir in sorted(data_v2_dir.glob("*")):
        if category_dir.is_dir():
            print(f"📁 {category_dir.name}/")
            
            # List original files
            for orig_file in sorted(category_dir.glob("algorithm_methods_data_*.jsonl")):
                if not orig_file.name.endswith("_output.jsonl"):
                    print(f"   📄 {orig_file.name}")
            
            # List claude_code files
            claude_code_dir = category_dir / "claude_code"
            if claude_code_dir.exists():
                print(f"   📁 claude_code/")
                for cc_file in sorted(claude_code_dir.glob("*.jsonl")):
                    print(f"      📄 {cc_file.name}")
            print()

if __name__ == "__main__":
    main()