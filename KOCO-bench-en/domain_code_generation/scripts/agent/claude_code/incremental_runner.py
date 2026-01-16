#!/usr/bin/env python3
"""
Incremental processing and aggregation functionality for repo-level code generator
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Set, Any
from repo_level_code_runner import RepoLevelCodeRunner
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IncrementalRepoLevelRunner:
    def __init__(self, data_dir: str, output_dir: str = None, model: str = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir.parent / "generated_results"
        self.output_dir.mkdir(exist_ok=True)
        
        self.summary_file = self.output_dir / "all_implementations_summary_new.json"
        
        self.base_runner = RepoLevelCodeRunner(str(self.data_dir), str(self.output_dir), model)
        
        self.completed_tasks: Set[str] = set()
        self.load_completed_tasks()
    
    def get_task_key(self, framework: str, function_name: str) -> str:
        return f"{framework}::{function_name}"
    
    def load_completed_tasks(self):
        if self.summary_file.exists():
            try:
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                for item in summary_data:
                    framework = item.get("framework", "")
                    function_name = item.get("function_name", "")
                    if framework and function_name:
                        task_key = self.get_task_key(framework, function_name)
                        self.completed_tasks.add(task_key)
                
                logger.info(f"Loaded {len(self.completed_tasks)} completed tasks")
            except Exception as e:
                logger.error(f"Failed to load summary file: {e}")
                self.completed_tasks = set()
        else:
            logger.info("Summary file does not exist, will create a new summary")
    
    def save_to_summary(self, framework: str, result: Dict[str, Any]):
        summary_item = {
            "framework": framework,
            "function_name": result["function_name"],
            "status": result["status"],
            "implementation": result.get("implementation", ""),
            "error": result.get("error", ""),
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        existing_data = []
        if self.summary_file.exists():
            try:
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read summary file: {e}")
                existing_data = []
        
        existing_data.append(summary_item)
        
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Added {framework}::{result['function_name']} to summary file")
        except Exception as e:
            logger.error(f"Failed to save summary file: {e}")
    
    def process_framework_incremental(self, framework_file: Path) -> None:
        framework_name = framework_file.stem.replace("_with_prompts_v2", "")
        logger.info(f"Starting incremental processing for framework: {framework_name}")
        
        all_tasks = []
        pending_tasks = []
        
        try:
            with open(framework_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        task_data = json.loads(line.strip())
                        function_name = task_data.get("function_name", "")
                        
                        if not function_name:
                            continue
                        
                        all_tasks.append((task_data, function_name))
                        task_key = self.get_task_key(framework_name, function_name)
                        
                        if task_key not in self.completed_tasks:
                            pending_tasks.append((task_data, function_name, task_key))
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to read framework file {framework_file}: {e}")
            return
        
        total_tasks = len(all_tasks)
        skipped_tasks = total_tasks - len(pending_tasks)
        
        logger.info(f"Framework {framework_name}: Total tasks {total_tasks}, Completed {skipped_tasks}, Pending {len(pending_tasks)}")
        
        if not pending_tasks:
            logger.info("All tasks completed, no processing needed")
            return
        
        successful_tasks = 0
        failed_tasks = 0
        
        progress_bar = tqdm(
            pending_tasks, 
            desc=f"Processing {framework_name}",
            unit="task",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Success: {postfix}"
        )
        
        for task_data, function_name, task_key in progress_bar:
            try:
                progress_bar.set_description(f"{framework_name}::{function_name[:20]}...")
                
                result = self.base_runner.process_single_task(task_data)
                
                self.save_to_summary(framework_name, result)
                
                if result["status"] == "success":
                    self.completed_tasks.add(task_key)
                    successful_tasks += 1
                else:
                    failed_tasks += 1
                
                progress_bar.set_postfix_str(f"✅{successful_tasks} ❌{failed_tasks}")
                
            except Exception as e:
                failed_tasks += 1
                progress_bar.set_postfix_str(f"✅{successful_tasks} ❌{failed_tasks}")
                logger.error(f"Failed to process task {task_key}: {e}")
                continue
        
        progress_bar.close()
        
        logger.info(f"Framework {framework_name} processing completed:")
        logger.info(f"  Total tasks: {total_tasks}")
        logger.info(f"  Skipped: {skipped_tasks}")
        logger.info(f"  Processed this time: {len(pending_tasks)}")
        logger.info(f"  Successful: {successful_tasks}")
        logger.info(f"  Failed: {failed_tasks}")
        logger.info(f"  Success rate: {successful_tasks/(len(pending_tasks)) * 100:.1f}%" if pending_tasks else "N/A")
    
    def process_all_frameworks_incremental(self):
        jsonl_files = list(self.data_dir.glob("*_with_prompts_v2.jsonl"))
        
        if not jsonl_files:
            logger.error(f"No framework files found in {self.data_dir}")
            return
        
        logger.info(f"Found {len(jsonl_files)} framework files for incremental processing")
        logger.info(f"Summary file: {self.summary_file}")
        
        framework_progress = tqdm(
            sorted(jsonl_files),
            desc="Processing Frameworks",
            unit="framework",
            position=0
        )
        
        for framework_file in framework_progress:
            framework_name = framework_file.stem.replace("_with_prompts_v2", "")
            framework_progress.set_description(f"Framework: {framework_name}")
            
            try:
                self.process_framework_incremental(framework_file)
            except KeyboardInterrupt:
                logger.info("User interrupted processing")
                break
            except Exception as e:
                logger.error(f"Failed to process framework file {framework_file}: {e}")
                continue
        
        framework_progress.close()
        
        self.print_final_summary()
    
    def process_specific_framework(self, framework_name: str):
        framework_file = self.data_dir / f"{framework_name}_with_prompts_v2.jsonl"
        
        if not framework_file.exists():
            logger.error(f"Framework file does not exist: {framework_file}")
            return
        
        logger.info(f"Starting to process specific framework: {framework_name}")
        self.process_framework_incremental(framework_file)
        self.print_final_summary()
    
    def print_final_summary(self):
        if not self.summary_file.exists():
            logger.warning("Summary file does not exist")
            return
        
        try:
            with open(self.summary_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            total = len(summary_data)
            successful = sum(1 for item in summary_data if item.get("status") == "success")
            failed = total - successful
            
            framework_stats = {}
            for item in summary_data:
                fw = item.get("framework", "unknown")
                if fw not in framework_stats:
                    framework_stats[fw] = {"total": 0, "success": 0}
                framework_stats[fw]["total"] += 1
                if item.get("status") == "success":
                    framework_stats[fw]["success"] += 1
            
            logger.info("=== Final Summary Statistics ===")
            logger.info(f"Total tasks: {total}")
            logger.info(f"Successful: {successful} ({successful/total*100:.1f}%)")
            logger.info(f"Failed: {failed} ({failed/total*100:.1f}%)")

            logger.info("\n=== Framework Statistics ===")
            for fw, stats in sorted(framework_stats.items()):
                success_rate = stats["success"] / stats["total"] * 100
                logger.info(f"{fw}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

        except Exception as e:
            logger.error(f"Failed to read summary statistics: {e}")
    
    def resume_failed_tasks(self):
        logger.info("Starting to reprocess failed tasks...")
        
        if not self.summary_file.exists():
            logger.warning("Summary file does not exist, cannot resume failed tasks")
            return
        
        failed_tasks = []
        try:
            with open(self.summary_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            for item in summary_data:
                if item.get("status") != "success":
                    failed_tasks.append({
                        "framework": item.get("framework"),
                        "function_name": item.get("function_name")
                    })
            
            logger.info(f"Found {len(failed_tasks)} failed tasks")
            
            failed_progress = tqdm(failed_tasks, desc="Retrying Failed Tasks", unit="task")
            
            for task_info in failed_progress:
                framework = task_info["framework"]
                function_name = task_info["function_name"]
                task_key = self.get_task_key(framework, function_name)
                
                failed_progress.set_description(f"Retry: {framework}::{function_name[:20]}...")
                
                self.completed_tasks.discard(task_key)
                
                framework_file = self.data_dir / f"{framework}_with_prompts_v2.jsonl"
                if framework_file.exists():
                    self.process_framework_incremental(framework_file)
            
            failed_progress.close()
        
        except Exception as e:
            logger.error(f"Error while resuming failed tasks: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Incremental processing for Repo-level code generation")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing *_with_prompts_v2.jsonl files"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: data_dir/../generated_results)"
    )
    parser.add_argument(
        "--framework",
        help="Process only specific framework (e.g., 'ARES')"
    )
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="Reprocess failed tasks"
    )
    parser.add_argument(
        "--model",
        help="Claude model (e.g., 'claude-sonnet-4-5-20250929')"
    )
    
    args = parser.parse_args()
    
    runner = IncrementalRepoLevelRunner(args.data_dir, args.output_dir, args.model)
    
    if args.resume_failed:
        runner.resume_failed_tasks()
    elif args.framework:
        runner.process_specific_framework(args.framework)
    else:
        runner.process_all_frameworks_incremental()


if __name__ == "__main__":
    main()