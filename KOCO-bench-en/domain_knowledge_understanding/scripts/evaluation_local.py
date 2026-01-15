#!/usr/bin/env python3
"""
Evaluate multiple-choice questions using local LLM inference server.

Features:
- Parse JSON files containing problem definitions with stem and ground-truth answers
- Send questions to local inference server
- Parse model output and compare with ground-truth
- Report accuracy and per-question results

Usage:
  python evaluation_local.py \
    --server_url "http://localhost:8000" \
    --input problems_xxx_EN.json \
    --output results_xxx.json
"""

import re
import argparse
import json
import os
import logging
import time
import requests
from typing import List, Dict, Set, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    server_url: str = "http://localhost:8000"
    model_name: str = ""
    
    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    delay: float = 0.5  # Delay between API calls
    
    # Input/Output
    input_file: str = ""
    output_file: str = ""


class IOParser:
    @staticmethod
    def parse_json(json_text: str) -> List[Dict]:
        """Parse JSON text and extract problems.

        Expected JSON structure:
        {
            "meta": {...},
            "problems": [
                {
                    "id": int,
                    "dataset": str,
                    "instruction": str (e.g., "choose ONE option" or "choose MULTIPLE options"),
                    "stem": str (question with options),
                    "explanation": str (optional),
                    "gt": [list of correct answer letters]
                },
                ...
            ]
        }

        Returns list of dicts: {id: int, dataset: str, instruction: str, stem: str, explanation: str, correct: Set[str]}
        """
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return []

        problems = data.get('problems', [])
        results: List[Dict] = []

        for item in problems:
            pid = item.get('id', 0)
            dataset = item.get('dataset', '')
            instruction = item.get('instruction', '')
            stem = item.get('stem', '')
            explanation = item.get('explanation', '')

            # Convert gt list to set of uppercase letters
            gt_list = item.get('gt', [])
            correct_letters: Set[str] = set()
            for letter in gt_list:
                if isinstance(letter, str):
                    correct_letters.add(letter.upper())

            results.append({
                'id': pid,
                'dataset': dataset,
                'instruction': instruction,
                'stem': stem,
                'explanation': explanation,
                'correct': correct_letters
            })

        logger.info(f"Parsed {len(results)} problems from JSON")
        return results

    
    @staticmethod
    def prepare_prompt(stem: str, instruction: str) -> str:
        """Prepare an instruction prompt that asks the model to show step-by-step reasoning
        and then provide the final answer wrapped in \\boxed{...} so it can be unambiguously parsed.
        Example final line: "The final answer is \\boxed{C}" or "Final: \\boxed{A,B}".
        
        Args:
            stem: The question with options
            instruction: Instruction (e.g., "choose ONE option" or "choose MULTIPLE options")
        """
        prompt = f"""You are given a multiple-choice question with options. First, give a brief step-by-step reasoning
explaining why you choose the answer. After your reasoning, provide the final answer ONLY in the
following exact format on its own line: \\boxed{{LETTER}} for single-choice or \\boxed{{A,B}} for multiple-choice.
Do not add other text on the final answer line. The \\boxed notion should only appear once in all your output.

Instruction: {instruction}

{stem}

Now provide reasoning, then the final answer in the boxed format:
"""
        return prompt.strip()

    @staticmethod
    def extract_letters(text: str) -> Set[str]:
        """Extract letter choices (A-Z) from model output text."""
        if not text:
            return set()
        
        # Prefer boxed answer \boxed{...}
        box_match = re.search(r"\\boxed\{([^}]+)\}", text)
        if box_match:
            content = box_match.group(1)
            letters = re.findall(r"[A-Za-z]", content)
            return set([c.upper() for c in letters])

        # Fallback: look for 'Final: A' or 'Answer: A' patterns
        simple_match = re.search(r"(?:Final|Answer)[:\s]*([A-Za-z,\s]+)", text, flags=re.I)
        if simple_match:
            content = simple_match.group(1)
            letters = re.findall(r"[A-Za-z]", content)
            if letters:
                return set([c.upper() for c in letters])

        # Last resort: collect any standalone letters A-Z in the text
        letters = re.findall(r"\b([A-Za-z])\b", text)
        return set([c.upper() for c in letters])


class LocalInferenceEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # Check server health
        logger.info(f"Checking connection to inference server: {config.server_url}")
        if not self.check_server_health():
            raise ConnectionError(f"Cannot connect to inference server at {config.server_url}")
        
        logger.info("✓ Server connection established")
    
    def check_server_health(self, max_retries: int = 3) -> bool:
        """Check if the inference server is healthy"""
        health_url = f"{self.config.server_url}/health"
        
        for i in range(max_retries):
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    self.config.model_name = data.get('model', 'local-model')
                    logger.info(f"  Model: {self.config.model_name}")
                    logger.info(f"  Device: {data.get('device', 'N/A')}")
                    return True
            except requests.exceptions.RequestException:
                if i < max_retries - 1:
                    logger.warning(f"Connection attempt {i+1}/{max_retries} failed, retrying...")
                    time.sleep(2)
        
        return False
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response from the local inference server."""
        try:
            evaluate_url = f"{self.config.server_url}/evaluate"
            
            request_data = {
                "prompts": [prompt],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p
            }
            
            response = requests.post(
                evaluate_url,
                json=request_data,
                timeout=1800  # 30 minutes timeout for long generation
            )
            
            if response.status_code == 200:
                result = response.json()
                responses = result.get("responses", [])
                if responses:
                    return responses[0].strip()
                else:
                    logger.warning("Empty response from server")
                    return ""
            else:
                logger.error(f"Server returned error: {response.status_code}")
                logger.error(f"Error message: {response.text}")
                return ""
            
        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            return ""
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return ""

    def evaluate(self, input_file: str, output_file: str = None):
        """Evaluate the input JSON file using local inference server and save results."""
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        problems = IOParser.parse_json(content)
        logger.info(f"Loaded {len(problems)} problems from {input_file}")

        results = []
        correct_count = 0

        logger.info(f"🚀 Starting evaluation with local model: {self.config.model_name}")
        logger.info("")

        for idx, prob in enumerate(problems, 1):
            logger.info(f"[{idx}/{len(problems)}] Processing problem ID {prob['id']}")
            
            # Prepare prompt
            prompt = IOParser.prepare_prompt(prob['stem'], prob['instruction'])
            
            # Generate response
            raw_output = self.generate_response(prompt)
            
            if not raw_output:
                logger.warning(f"  ⚠️  Empty response for problem {prob['id']}")
            
            # Extract predicted letters
            pred_letters = IOParser.extract_letters(raw_output)

            # Check correctness
            is_correct = False
            if prob['correct']:
                if pred_letters == prob['correct']:
                    is_correct = True

            if is_correct:
                correct_count += 1

            results.append({
                'id': prob['id'],
                'dataset': prob['dataset'],
                'instruction': prob['instruction'],
                'stem': prob['stem'],
                'explanation': prob['explanation'],
                'pred_raw': raw_output,
                'pred_letters': sorted(list(pred_letters)),
                'gt_letters': sorted(list(prob['correct'])),
                'is_correct': is_correct
            })
            
            status_icon = "✅" if is_correct else "❌"
            logger.info(f"  {status_icon} Predicted: {sorted(list(pred_letters))} | Ground Truth: {sorted(list(prob['correct']))}")
            logger.info("")
            
            # Delay to avoid overloading
            if idx < len(problems):
                time.sleep(self.config.delay)

        total = len(results)
        accuracy = correct_count / total * 100 if total > 0 else 0.0

        summary = {
            'model': self.config.model_name,
            'server_url': self.config.server_url,
            'total': total,
            'correct': correct_count,
            'incorrect': total - correct_count,
            'accuracy_percent': accuracy
        }

        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 Evaluation Summary")
        logger.info("=" * 80)
        logger.info(f"Model: {summary['model']}")
        logger.info(f"Server: {summary['server_url']}")
        logger.info(f"Total problems: {summary['total']}")
        logger.info(f"✅ Correct: {summary['correct']}")
        logger.info(f"❌ Incorrect: {summary['incorrect']}")
        logger.info(f"🎯 Accuracy: {summary['accuracy_percent']:.2f}%")
        logger.info("=" * 80)

        # Save results
        out_path = output_file or self.config.output_file
        if out_path:
            with open(out_path, 'w', encoding='utf-8') as out_f:
                json.dump({'summary': summary, 'results': results}, out_f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Detailed results saved to {out_path}")

        return summary, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple-choice questions using local inference server")
    
    # Server configuration
    parser.add_argument("--server_url", type=str, default="http://localhost:8000",
                        help="Local inference server URL (default: http://localhost:8000)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for deterministic)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum tokens to generate")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling parameter")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between requests in seconds")
    
    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON file with problems")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for results")

    args = parser.parse_args()

    # Create configuration
    config = EvaluationConfig(
        server_url=args.server_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        delay=args.delay,
        input_file=args.input,
        output_file=args.output
    )

    logger.info("=" * 80)
    logger.info("🤖 KOCO-BENCH KNOWLEDGE UNDERSTANDING EVALUATION (LOCAL)")
    logger.info("=" * 80)
    logger.info(f"Server: {config.server_url}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Temperature: {config.temperature}")
    logger.info(f"Max tokens: {config.max_tokens}")
    logger.info("=" * 80)
    logger.info("")

    # Run evaluation
    try:
        evaluator = LocalInferenceEvaluator(config)
        evaluator.evaluate(args.input, args.output)
    except ConnectionError as e:
        logger.error(f"❌ {e}")
        logger.error("")
        logger.error("Please start the inference server first:")
        logger.error("  bash start_inference_server.sh")
        return 1
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

