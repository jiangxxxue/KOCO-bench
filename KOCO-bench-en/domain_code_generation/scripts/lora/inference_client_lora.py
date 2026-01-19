#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_client_lora.py — LoRA inference client

Call LoRA inference server via HTTP requests for code generation
Similar to inference_client.py, but specifically for LoRA server
"""

import json
import argparse
import requests
import time
from typing import List, Dict, Any
from pathlib import Path


class LoRAInferenceClient:
    """LoRA inference client"""
    
    def __init__(self, server_url: str):
        """
        Initialize client
        
        Args:
            server_url: Server address, e.g., http://localhost:8000
        """
        self.server_url = server_url.rstrip('/')
        self.health_url = f"{self.server_url}/health"
        self.generate_url = f"{self.server_url}/generate"
    
    def check_health(self) -> Dict[str, Any]:
        """Check server health status"""
        try:
            response = requests.get(self.health_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Server health check failed: {e}")
    
    def generate(
        self,
        prompts: List[str],
        num_completions: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> List[List[str]]:
        """
        Generate code completions
        
        Args:
            prompts: Prompt list
            num_completions: Number of completions per prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter
            top_p: Top-p sampling
            do_sample: Whether to sample
        
        Returns:
            Completion list, outer list corresponds to each prompt, inner list corresponds to multiple completions per prompt
        """
        payload = {
            "prompts": prompts,
            "num_completions": num_completions,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["completions"]
        
        except requests.exceptions.Timeout:
            raise RuntimeError("Request timeout")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl_data(data: List[Dict[str, Any]], file_path: str):
    """Save JSONL data"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def format_prompt(prompt_data):
    """
    Format prompt data to string
    
    Args:
        prompt_data: Can be a string or conversation list
    
    Returns:
        Formatted string
    """
    if isinstance(prompt_data, str):
        return prompt_data
    elif isinstance(prompt_data, list):
        # Conversation list format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        formatted_parts = []
        for message in prompt_data:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        return "\n\n".join(formatted_parts)
    else:
        return str(prompt_data)


def process_dataset(
    client: LoRAInferenceClient,
    input_file: str,
    output_file: str,
    model_name: str,
    num_completions: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 1,
):
    """
    Process dataset
    
    Args:
        client: Inference client
        input_file: Input file path
        output_file: Output file path
        model_name: Model name (for output directory)
        num_completions: Number of completions per task
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature parameter
        top_p: Top-p sampling
        batch_size: Batch size
    """
    # Load data
    print(f"📂 Loading data: {input_file}")
    data = load_jsonl_data(input_file)
    print(f"  Found {len(data)} tasks")
    
    # Build output path
    output_path = Path(output_file)
    if not output_path.is_absolute():
        # If relative path, base on input file directory
        input_path = Path(input_file)
        base_dir = input_path.parent
        
        # Create model output directory
        model_dir = base_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_filename = input_path.stem + "_output" + input_path.suffix
        output_path = model_dir / output_filename
    
    print(f"📂 Output file: {output_path}")
    print()
    
    # Batch processing
    total = len(data)
    processed = 0
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        # Directly use prompt (can be string or conversation list, server will handle formatting)
        batch_prompts = [item["prompt"] for item in batch]
        
        print(f"🔄 Processing batch {i // batch_size + 1} / {(total + batch_size - 1) // batch_size}")
        print(f"   Tasks {i + 1}-{min(i + len(batch), total)} / {total}")
        
        try:
            # Generate
            start_time = time.time()
            completions = client.generate(
                prompts=batch_prompts,
                num_completions=num_completions,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            elapsed = time.time() - start_time
            
            # Save results
            for j, item in enumerate(batch):
                item["completions"] = completions[j]
            
            processed += len(batch)
            print(f"   ✅ Completed ({elapsed:.2f}s) - Progress: {processed}/{total}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            # Save empty results
            for item in batch:
                item["completions"] = [""] * num_completions
        
        print()
    
    # Save output
    print(f"💾 Saving results: {output_path}")
    save_jsonl_data(data, str(output_path))
    print()
    print("✅ Processing completed!")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA code completion inference client - Call inference server via HTTP requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic usage
  python inference_client_lora.py \\
    --server_url http://localhost:8000 \\
    --input_file ../data/algorithm_methods_data.jsonl \\
    --output_file ../data/algorithm_methods_output.jsonl

  # Generate multiple completions
  python inference_client_lora.py \\
    --server_url http://localhost:8000 \\
    --input_file ../data/algorithm_methods_data.jsonl \\
    --output_file ../data/algorithm_methods_output.jsonl \\
    --num_completions 4 \\
    --temperature 0.2

  # Use batch processing
  python inference_client_lora.py \\
    --server_url http://localhost:8000 \\
    --input_file ../data/algorithm_methods_data.jsonl \\
    --output_file ../data/algorithm_methods_output.jsonl \\
    --batch_size 4
        """
    )
    
    # Server configuration
    parser.add_argument(
        "--server_url",
        type=str,
        required=True,
        help="Inference server address, e.g., http://localhost:8000"
    )
    
    # File paths
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSONL file path (default: auto-generated)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="lora-model",
        help="Model name (for output directory)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_completions",
        type=int,
        default=1,
        help="Number of completions per task (default: 1)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature parameter (default: 0.7)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling (default: 0.95)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Set output file
    if args.output_file is None:
        input_path = Path(args.input_file)
        args.output_file = str(input_path.parent / f"{input_path.stem}_output{input_path.suffix}")
    
    print("=" * 60)
    print("🚀 LoRA Inference Client")
    print("=" * 60)
    print(f"Server: {args.server_url}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model name: {args.model_name}")
    print(f"Completions: {args.num_completions}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    print()
    
    # Create client
    client = LoRAInferenceClient(args.server_url)
    
    # Health check
    print("🔍 Checking server health status...")
    try:
        health = client.check_health()
        print(f"✅ Server is healthy")
        print(f"  - Status: {health['status']}")
        print(f"  - Base model: {health['base_model']}")
        print(f"  - LoRA adapter: {health['lora_adapter']}")
        print(f"  - Device: {health['device']}")
        print()
    except Exception as e:
        print(f"❌ Server unavailable: {e}")
        return 1
    
    # Process dataset
    try:
        process_dataset(
            client=client,
            input_file=args.input_file,
            output_file=args.output_file,
            model_name=args.model_name,
            num_completions=args.num_completions,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
        )
        return 0
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

