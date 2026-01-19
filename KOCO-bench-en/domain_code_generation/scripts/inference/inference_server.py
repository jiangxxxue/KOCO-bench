#!/usr/bin/env python3
"""
Code completion inference server
Provides HTTP API using FastAPI, loads model only once, continuously provides inference service
"""

import json
import torch
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from transformers import AutoTokenizer, AutoModelForCausalLM

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# Request/Response Models
# ========================================

class GenerationRequest(BaseModel):
    """Generation request"""
    prompts: List[Any] = Field(..., description="List of prompts to generate (supports string or conversation format)")
    num_completions: int = Field(1, description="Number of completions per prompt")
    max_tokens: int = Field(512, description="Maximum number of tokens per completion")
    temperature: float = Field(0.2, description="Sampling temperature")
    top_p: float = Field(0.95, description="Top-p sampling parameter")

class GenerationResponse(BaseModel):
    """Generation response"""
    completions: List[List[str]] = Field(..., description="Generated completions list")
    model: str = Field(..., description="Model used")
    status: str = Field("success", description="Status")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    model: str = ""
    device: str = ""

# ========================================
# Global Variables (model and tokenizer)
# ========================================

model = None
tokenizer = None
model_name = ""
device = ""

# ========================================
# FastAPI Application
# ========================================

app = FastAPI(
    title="Code Completion Inference Service",
    description="HTTP API service for code completion inference",
    version="1.0.0"
)

@app.get("/", response_model=HealthResponse)
async def root():
    """Root path - health check"""
    return HealthResponse(
        status="healthy",
        model=model_name,
        device=device
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=model_name,
        device=device
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Generate code completions
    
    Request example:
    {
        "prompts": ["def hello():\\n    "],  # String format
        or
        "prompts": [[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]],  # Conversation format
        "num_completions": 2,
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.95
    }
    """
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        logger.info(f"Received generation request: {len(request.prompts)} prompts")
        
        all_completions = []
        
        for i, prompt in enumerate(request.prompts):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing {i+1}/{len(request.prompts)}...")
            
            # Process prompt format
            if isinstance(prompt, list):
                # Conversation format: use chat template
                if hasattr(tokenizer, 'apply_chat_template'):
                    try:
                        formatted_prompt = tokenizer.apply_chat_template(
                            prompt,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    except Exception as e:
                        logger.warning(f"Sample {i} failed to use chat template: {e}, falling back to message concatenation")
                        # Fallback: simple message content concatenation
                        formatted_prompt = "\n\n".join(msg.get("content", "") for msg in prompt if msg.get("content"))
                else:
                    # No chat template, simple concatenation
                    formatted_prompt = "\n\n".join(msg.get("content", "") for msg in prompt if msg.get("content"))
            else:
                # String format: use directly
                formatted_prompt = prompt
            
            # Tokenize input
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,  # Use fixed maximum length
            ).to(model.device)
            
            input_len = inputs.input_ids.shape[1]
            
            # Generate multiple completions
            completions = []
            for _ in range(request.num_completions):
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature if request.temperature > 0 else 1.0,
                        top_p=request.top_p,
                        do_sample=request.temperature > 0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.0,
                    )
                
                # Extract only newly generated part
                generated_tokens = outputs[0][input_len:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                completions.append(generated_text)
            
            all_completions.append(completions)
        
        logger.info(f"✓ Generation completed")
        
        return GenerationResponse(
            completions=all_completions,
            model=model_name,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# Model Loading
# ========================================

def load_model(model_path: str, max_context_len: int = 4096):
    """Load model and tokenizer"""
    global model, tokenizer, model_name, device
    
    logger.info("="*60)
    logger.info("Starting inference server...")
    logger.info("="*60)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Max context length: {max_context_len}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("✓ Tokenizer loaded")
    
    # Load model
    logger.info("Loading model...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using precision: {torch_dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    device = str(model.device)
    model_name = Path(model_path).name
    
    logger.info(f"✓ Model loaded")
    logger.info(f"Device: {device}")
    logger.info(f"Model name: {model_name}")
    logger.info("="*60)
    logger.info("✅ Server ready!")
    logger.info("="*60)

# ========================================
# Main Function
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="Code completion inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Start server (default port 8000)
  python inference_server.py --model_path ../models/your_model

  # Specify port
  python inference_server.py \\
    --model_path ../models/your_model \\
    --port 8001

  # Test health check
  curl http://localhost:8000/health

  # Test generation
  curl -X POST http://localhost:8000/generate \\
    -H "Content-Type: application/json" \\
    -d '{"prompts": ["def hello():\\n    "], "num_completions": 1}'
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local model path"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=4096,
        help="Maximum context length (default: 4096)"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"❌ Model path does not exist: {model_path}")
        return
    
    # Load model
    load_model(str(model_path), args.max_context_len)
    
    # Start server
    logger.info(f"Starting server: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

