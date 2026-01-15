#!/usr/bin/env python3
"""
Evaluate multiple-choice questions using OpenRouter API.

Features:
- Parse JSON files containing problem definitions with stem and ground-truth answers
- Send questions to various models via OpenRouter API
- Parse model output and compare with ground-truth
- Report accuracy and per-question results

Usage:
  python evaluation_openrouter.py \
    --model "qwen/qwen2.5-coder-7b-instruct" \
    --input problems_xxx_EN.json \
    --output results_xxx.json
"""

import re
import argparse
import json
import os
import logging
import time
from typing import List, Dict, Set, Any
from dataclasses import dataclass
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    model_name: str = "qwen/qwen2.5-coder-7b-instruct"
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    
    # API parameters
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    delay: float = 0.5  # Delay between API calls to avoid rate limiting
    
    # Input/Output
    input_file: str = ""
    output_file: str = ""
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")


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


class OpenRouterEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # Validate API key
        if not self.config.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set. Please set it via environment variable or config.")
        
        # Initialize OpenAI client for OpenRouter
        logger.info(f"Initializing OpenRouter client for model: {config.model_name}")
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        logger.info("OpenRouter client initialized successfully")
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response from the model via OpenRouter API."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            content = response.choices[0].message.content
            if content is None:
                logger.warning(f"Response returned None")
                return ""
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return ""

    def evaluate(self, input_file: str, output_file: str = None):
        """Evaluate the input JSON file using OpenRouter API and save results."""
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        problems = IOParser.parse_json(content)
        logger.info(f"Loaded {len(problems)} problems from {input_file}")

        results = []
        correct_count = 0

        logger.info(f"🚀 Starting evaluation with model: {self.config.model_name}")
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
            
            # Delay to avoid rate limiting
            if idx < len(problems):
                time.sleep(self.config.delay)

        total = len(results)
        accuracy = correct_count / total * 100 if total > 0 else 0.0

        summary = {
            'model': self.config.model_name,
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
    parser = argparse.ArgumentParser(description="Evaluate multiple-choice questions using OpenRouter API")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="qwen/qwen2.5-coder-7b-instruct",
                        help="Model name on OpenRouter (e.g., qwen/qwen2.5-coder-7b-instruct)")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENROUTER_API_KEY"),
                        help="OpenRouter API Key (default: from OPENROUTER_API_KEY env var)")
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1",
                        help="OpenRouter API base URL")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for deterministic)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum tokens to generate")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling parameter")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls in seconds")
    
    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON file with problems")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for results")

    args = parser.parse_args()

    # Create configuration
    config = EvaluationConfig(
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        delay=args.delay,
        input_file=args.input,
        output_file=args.output
    )

    logger.info("=" * 80)
    logger.info("🤖 KOCO-BENCH KNOWLEDGE UNDERSTANDING EVALUATION")
    logger.info("=" * 80)
    logger.info(f"API: OpenRouter")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Temperature: {config.temperature}")
    logger.info(f"Max tokens: {config.max_tokens}")
    logger.info("=" * 80)
    logger.info("")

    # Run evaluation
    evaluator = OpenRouterEvaluator(config)
    evaluator.evaluate(args.input, args.output)


if __name__ == '__main__':
    main()

