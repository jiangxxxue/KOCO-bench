#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RepoLevelCodeRunner:
    def __init__(self, data_dir: str, output_dir: str = None, model: str = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir.parent / "generated_results"
        self.output_dir.mkdir(exist_ok=True)
        
        self.model = model  # "claude-sonnet-4-5-20250929"
        
        self.temp_base = Path("/tmp/claude_code_gen")
        self.temp_base.mkdir(exist_ok=True)
    
    def copy_repositories(self, keyword_paths: List[str], work_dir: Path) -> Dict[str, Path]:
        repo_mappings = {}
        
        for i, repo_path in enumerate(keyword_paths):
            if not os.path.exists(repo_path):
                logger.warning(f"Repository path does not exist: {repo_path}")
                continue
                
            repo_name = f"repo_{i}_{Path(repo_path).name}"
            target_path = work_dir / repo_name
            
            logger.info(f"Copying {repo_path} -> {target_path}")
            shutil.copytree(repo_path, target_path, symlinks=True)
            repo_mappings[repo_path] = target_path
            
        return repo_mappings
    
    def enhance_prompt(self, original_prompt: List[Dict[str, str]], repo_mappings: Dict[str, Path], task_data: Dict[str, Any]) -> str:
        
        user_prompt = ""
        for message in original_prompt:
            if message["role"] == "system":
                user_prompt += message["content"]
            if message["role"] == "user":
                user_prompt += message["content"]
                break
        
        enhanced_prompt = user_prompt + "\n\n"
        enhanced_prompt += "You can freely explore the following known repositories to obtain the required information: for example, domain knowledge and callable functions in the framework knowledge base, as well as utility functions in the development repository, but please do not over-explore.\n"

        for original_path, copied_path in repo_mappings.items():
            if "knowledge_corpus" in original_path:
                enhanced_prompt += f"- Framework Knowledge Base: {copied_path}\n"
            else:
                enhanced_prompt += f"- Development Repository: {copied_path}\n"

        enhanced_prompt += "\nPlease use the code in these repositories to implement the required functionality."
        enhanced_prompt += "\n\n**Important: After completing the implementation, you must create a file named 'implementation_result.json' in the working directory with the following format:**"
        enhanced_prompt += "\n```json"
        enhanced_prompt += "\n{"
        enhanced_prompt += f'\n  "function_name": "{task_data["function_name"]}",'
        enhanced_prompt += '\n  "implementation": "Your complete function implementation code (including the def line)"'
        enhanced_prompt += "\n}"
        enhanced_prompt += "\n```"
        enhanced_prompt += "\n\nThis JSON file is required, and I will extract your implementation result from it. Please ensure the implementation field contains complete executable function code."
        
        logger.info(f"Enhanced prompt length: {len(enhanced_prompt)} characters")
        logger.debug(f"Enhanced prompt preview: {enhanced_prompt[:]}...")

        return enhanced_prompt
    
    def execute_claude_command(self, prompt: str, work_dir: Path, max_turns: int = 50) -> Tuple[bool, str, str]:
        try:
            cmd = [
                "claude",
                "-p", prompt,
                "--allowedTools", "Read,Write,Edit,Grep,Glob,Bash",
                "--max-turns", str(max_turns)
            ]
            
            if self.model:
                cmd.extend(["--model", self.model])
            
            logger.info(f"Executing Claude command in {work_dir}")
            logger.info(f"Command: {' '.join(cmd[:2])} [prompt] {' '.join(cmd[3:])}")
            
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            success = result.returncode == 0
            logger.info(f"Claude command completed. Return code: {result.returncode}")
            
            if result.stdout:
                logger.info(f"Claude stdout length: {len(result.stdout)} characters")
                logger.info(f"Claude stdout content: {result.stdout}")
            else:
                logger.warning("Claude stdout is empty")
                
            if result.stderr:
                logger.warning(f"Claude stderr length: {len(result.stderr)} characters")
                logger.warning(f"Claude stderr content: {result.stderr}")
            else:
                logger.info("No stderr output")
                
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            logger.error("Claude command timed out")
            return False, "", "Command timed out"
        except Exception as e:
            logger.error(f"Error executing Claude command: {e}")
            return False, "", str(e)
    
    def extract_implementation_from_result_file(self, work_dir: Path, function_name: str) -> str:
        result_file = work_dir / "implementation_result.json"
        
        logger.info(f"Looking for implementation result file: {result_file}")
        
        if not result_file.exists():
            logger.warning("No implementation result file found")
            return ""
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            if result_data.get("function_name") == function_name:
                implementation = result_data.get("implementation", "")
                logger.info(f"Successfully extracted {len(implementation)} characters from result file")
                return implementation
            else:
                logger.warning(f"Function name mismatch in result file: expected {function_name}, got {result_data.get('function_name')}")
                return ""
                
        except Exception as e:
            logger.error(f"Error reading result file: {e}")
            return ""
    
    def process_single_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        function_name = task_data["function_name"]
        
        with tempfile.TemporaryDirectory(dir=self.temp_base, prefix=f"{function_name}_") as temp_dir:
            work_dir = Path(temp_dir)
            logger.info(f"Processing {function_name} in {work_dir}")
            
            try:
                keyword_paths = task_data["keyword"]
                repo_mappings = self.copy_repositories(keyword_paths, work_dir)
                
                if not repo_mappings:
                    logger.error(f"No valid repositories found for {function_name}")
                    return {
                        "function_name": function_name,
                        "status": "failed",
                        "error": "No valid repositories",
                        "implementation": ""
                    }
                
                enhanced_prompt = self.enhance_prompt(task_data["prompt"], repo_mappings, task_data)
                
                success, stdout, stderr = self.execute_claude_command(enhanced_prompt, work_dir)
                
                if not success:
                    logger.error(f"Claude command failed for {function_name}: {stderr}")
                    return {
                        "function_name": function_name,
                        "status": "failed", 
                        "error": stderr,
                        "implementation": ""
                    }
                
                implementation = self.extract_implementation_from_result_file(work_dir, function_name)
                
                return {
                    "function_name": function_name,
                    "status": "success",
                    "implementation": implementation,
                    "claude_output": stdout
                }
                
            except Exception as e:
                logger.error(f"Error processing {function_name}: {e}")
                return {
                    "function_name": function_name,
                    "status": "failed",
                    "error": str(e),
                    "implementation": ""
                }
    
    def process_framework_file(self, framework_file: Path) -> None:
        framework_name = framework_file.stem.replace("_with_prompts_v2", "")
        output_file = self.output_dir / f"{framework_name}_results.json"
        
        logger.info(f"Processing framework: {framework_name}")
        
        results = []
        
        try:
            with open(framework_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        task_data = json.loads(line.strip())
                        logger.info(f"Processing task {line_num}: {task_data.get('function_name', 'unknown')}")
                        
                        result = self.process_single_task(task_data)
                        results.append(result)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading framework file {framework_file}: {e}")
            return
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Results saved to {output_file}")
            
            successful = sum(1 for r in results if r["status"] == "success")
            total = len(results)
            logger.info(f"Framework {framework_name}: {successful}/{total} tasks successful")
            
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
    
    def run_all_frameworks(self):
        jsonl_files = list(self.data_dir.glob("*_with_prompts_v2.jsonl"))
        
        if not jsonl_files:
            logger.error(f"No framework files found in {self.data_dir}")
            return
        
        logger.info(f"Found {len(jsonl_files)} framework files to process")
        
        for framework_file in jsonl_files:
            try:
                self.process_framework_file(framework_file)
            except KeyboardInterrupt:
                logger.info("Process interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing {framework_file}: {e}")
                continue
        
        logger.info("All frameworks processing completed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Repo-level Code Generation Runner")
    parser.add_argument(
        "--data-dir", 
        required=True,
        help="Directory containing *_with_prompts_v2.jsonl files"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (default: data_dir/../generated_results)"
    )
    parser.add_argument(
        "--framework",
        help="Process only specific framework (e.g., 'ARES')"
    )
    parser.add_argument(
        "--model",
        help="Claude model to use (e.g., 'claude-sonnet-4-5-20250929')"
    )
    
    args = parser.parse_args()
    
    runner = RepoLevelCodeRunner(args.data_dir, args.output_dir, args.model)
    
    if args.framework:
        framework_file = Path(args.data_dir) / f"{args.framework}_with_prompts_v2.jsonl"
        if framework_file.exists():
            runner.process_framework_file(framework_file)
        else:
            logger.error(f"Framework file not found: {framework_file}")
    else:
        runner.run_all_frameworks()


if __name__ == "__main__":
    main()