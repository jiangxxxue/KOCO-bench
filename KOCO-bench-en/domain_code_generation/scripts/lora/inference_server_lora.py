#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_server_lora.py — LoRA model inference server

Provides code generation service based on FastAPI, supports LoRA adapter loading
"""

import os
import sys
import argparse
import torch
import uvicorn
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


# ========================================
# Global Variables
# ========================================
model = None
tokenizer = None
generation_config = None
base_model_path = None
lora_adapter_path = None


# ========================================
# Request/Response Models
# ========================================

class GenerationRequest(BaseModel):
    """Generation request"""
    prompts: List[Any] = Field(..., description="Input prompt list (can be strings or conversation lists)")
    num_completions: int = Field(1, ge=1, le=10, description="Number of completions per prompt")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature parameter")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p sampling")
    do_sample: bool = Field(True, description="Whether to use sampling")


class GenerationResponse(BaseModel):
    """Generation response"""
    completions: List[List[str]] = Field(..., description="Generation results, outer list corresponds to input prompts, inner list corresponds to multiple completions per prompt")
    model: str = Field(..., description="Model used")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field("healthy", description="Service status")
    model: str = Field(..., description="Loaded model")
    base_model: str = Field(..., description="Base model path")
    lora_adapter: str = Field(..., description="LoRA adapter path")
    device: str = Field(..., description="Device information")


# ========================================
# Model Loading
# ========================================

def load_lora_model(
    base_model: str,
    lora_adapter: str,
    device: str = "auto",
    torch_dtype: str = "bfloat16",
    max_context_len: int = 4096,
):
    """
    Load base model + LoRA adapter
    
    Args:
        base_model: Base model path
        lora_adapter: LoRA adapter path
        device: Device, "auto" or "cuda:0" etc.
        torch_dtype: Data type
        max_context_len: Maximum context length
    
    Returns:
        model, tokenizer, generation_config
    """
    print(f"📦 Loading base model: {base_model}")
    
    # Process torch_dtype
    if torch_dtype == "auto":
        dtype = "auto"
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",  # Avoid flash-attention issues
    )
    
    print(f"📦 Loading LoRA adapter: {lora_adapter}")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base,
        lora_adapter,
        torch_dtype=dtype,
    )
    
    # Merge LoRA weights to improve inference speed (optional)
    # model = model.merge_and_unload()
    
    print(f"📦 Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="left",  # Left padding needed for batch generation
    )
    
    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Generation configuration
    gen_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    print("✅ Model loading completed")
    print(f"  - Base model: {base_model}")
    print(f"  - LoRA adapter: {lora_adapter}")
    print(f"  - Device: {device}")
    print(f"  - Dtype: {dtype}")
    
    return model, tokenizer, gen_config


# ========================================
# Generation Functions
# ========================================

def format_prompt(prompt_data: Any) -> str:
    """
    Format prompt to string
    
    Args:
        prompt_data: Can be a string or conversation list
    
    Returns:
        Formatted string
    """
    global tokenizer
    
    if isinstance(prompt_data, str):
        return prompt_data
    elif isinstance(prompt_data, list):
        # Conversation list format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        # Use tokenizer's apply_chat_template method if available
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                return tokenizer.apply_chat_template(
                    prompt_data,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback to simple formatting if failed
                pass
        
        # Simple formatting
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
        return "\n\n".join(formatted_parts) + "\n\nAssistant: "
    else:
        return str(prompt_data)


def generate_completions(
    prompts: List[Any],
    num_completions: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> List[List[str]]:
    """
    Batch generate code completions
    
    Args:
        prompts: Prompt list (can be strings or conversation lists)
        num_completions: Number of completions per prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature parameter
        top_p: Top-p sampling
        do_sample: Whether to sample
    
    Returns:
        Completion list, outer list corresponds to each prompt, inner list corresponds to multiple completions per prompt
    """
    global model, tokenizer, generation_config
    
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")
    
    results = []
    
    for prompt_data in prompts:
        prompt_completions = []
        
        # Format prompt
        prompt = format_prompt(prompt_data)
        
        # Generate separately for each completion
        for _ in range(num_completions):
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            prompt_completions.append(completion)
        
        results.append(prompt_completions)
    
    return results


# ========================================
# FastAPI Application
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Load model on startup
    global model, tokenizer, generation_config, base_model_path, lora_adapter_path
    
    print("=" * 60)
    print("🚀 Starting LoRA inference server")
    print("=" * 60)
    
    model, tokenizer, generation_config = load_lora_model(
        base_model=base_model_path,
        lora_adapter=lora_adapter_path,
        device=app.state.device,
        torch_dtype=app.state.torch_dtype,
        max_context_len=app.state.max_context_len,
    )
    
    print("=" * 60)
    print("✅ Server startup completed")
    print("=" * 60)
    
    yield
    
    # Cleanup on shutdown
    print("🛑 Shutting down server...")


app = FastAPI(
    title="LoRA Code Generation Inference Server",
    description="Code completion inference service supporting LoRA adapter",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    device_name = str(next(model.parameters()).device)
    
    return HealthResponse(
        status="healthy",
        model=f"{base_model_path} + {lora_adapter_path}",
        base_model=base_model_path,
        lora_adapter=lora_adapter_path,
        device=device_name,
    )


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Code generation endpoint"""
    try:
        completions = generate_completions(
            prompts=request.prompts,
            num_completions=request.num_completions,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
        )
        
        return GenerationResponse(
            completions=completions,
            model=f"{base_model_path} + {lora_adapter_path}",
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ========================================
# Main Function
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="LoRA code completion inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Start server (default port 8000)
  python inference_server_lora.py \\
    --base_model /path/to/base/model \\
    --lora_adapter ../models/your_framework-lora

  # Specify port
  python inference_server_lora.py \\
    --base_model /path/to/base/model \\
    --lora_adapter ../models/your_framework-lora \\
    --port 8001

  # Test health check
  curl http://localhost:8000/health

  # Test generation
  curl -X POST http://localhost:8000/generate \\
    -H "Content-Type: application/json" \\
    -d '{"prompts": ["def hello():\\n    "], "num_completions": 1}'
"""
    )
    
    # Model parameters
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model path"
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        required=True,
        help="LoRA adapter path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (default: auto)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model data type (default: bfloat16)"
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=4096,
        help="Maximum context length (default: 4096)"
    )
    
    # Server parameters
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Set global variables
    global base_model_path, lora_adapter_path
    base_model_path = args.base_model
    lora_adapter_path = args.lora_adapter
    
    # Save parameters to app.state
    app.state.device = args.device
    app.state.torch_dtype = args.torch_dtype
    app.state.max_context_len = args.max_context_len
    
    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

