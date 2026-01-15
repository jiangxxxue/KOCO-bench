#!/usr/bin/env python3
"""
Knowledge Understanding Inference Server
Provides HTTP API for multiple-choice question answering with local LLM
"""

import json
import torch
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
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

class EvaluationRequest(BaseModel):
    """Evaluation request for multiple-choice questions"""
    prompts: List[str] = Field(..., description="List of prompts (question with instruction)")
    temperature: float = Field(0.0, description="Sampling temperature")
    max_tokens: int = Field(4096, description="Maximum tokens to generate")
    top_p: float = Field(1.0, description="Top-p sampling parameter")

class EvaluationResponse(BaseModel):
    """Evaluation response"""
    responses: List[str] = Field(..., description="Generated responses")
    model: str = Field(..., description="Model used")
    status: str = Field("success", description="Status")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    model: str = ""
    device: str = ""

# ========================================
# Global variables (model and tokenizer)
# ========================================

model = None
tokenizer = None
model_name = ""
device = ""

# ========================================
# FastAPI Application
# ========================================

app = FastAPI(
    title="Knowledge Understanding Inference Server",
    description="HTTP API for LLM-based multiple-choice question evaluation",
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

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """
    Evaluate multiple-choice questions
    
    Request example:
    {
        "prompts": ["Question with options..."],
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 1.0
    }
    """
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        logger.info(f"Received evaluation request: {len(request.prompts)} prompts")
        
        all_responses = []
        
        for i, prompt in enumerate(request.prompts):
            if (i + 1) % 5 == 0:
                logger.info(f"Processing {i+1}/{len(request.prompts)}...")
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=8192,  # Large context for reading code
            ).to(model.device)
            
            input_len = inputs.input_ids.shape[1]
            
            # Generate response
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
            all_responses.append(generated_text)
        
        logger.info(f"✓ Generation complete")
        
        return EvaluationResponse(
            responses=all_responses,
            model=model_name,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# Model Loading
# ========================================

def load_model(model_path: str, max_context_len: int = 8192):
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
        description="Knowledge Understanding Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Start server (default port 8000)
  python inference_server.py --model_path /path/to/model

  # Specify port
  python inference_server.py \\
    --model_path /path/to/model \\
    --port 8001

  # Test health check
  curl http://localhost:8000/health

  # Test evaluation
  curl -X POST http://localhost:8000/evaluate \\
    -H "Content-Type: application/json" \\
    -d '{"prompts": ["Question..."], "temperature": 0.0}'
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
        default=8192,
        help="Maximum context length (default: 8192)"
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

