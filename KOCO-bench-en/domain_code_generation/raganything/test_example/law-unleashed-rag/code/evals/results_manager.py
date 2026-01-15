"""
Results manager for organizing evaluation outputs
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class EvaluationResultsManager:
    """Manages evaluation results storage and organization"""
    
    def __init__(self, results_dir: str = "evals/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_evaluation_run(self, evaluation_run_id: str, results: Dict[str, Any]) -> str:
        """Save evaluation run results to organized directory structure"""
        
        # Create directory structure: results/YYYY-MM-DD/evaluation_run_id/
        date_str = datetime.now().strftime("%Y-%m-%d")
        run_dir = self.results_dir / date_str / evaluation_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = run_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary = self._extract_summary(results)
        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save individual evaluation case results
        evaluation_cases_dir = run_dir / "evaluation_cases"
        evaluation_cases_dir.mkdir(exist_ok=True)
        
        for result in results.get('detailed_results', []):
            evaluation_case_file = evaluation_cases_dir / f"{result['evaluation_case_id']}_{result['rag_approach']}.json"
            with open(evaluation_case_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        # Create README for the run
        readme_file = run_dir / "README.md"
        self._create_run_readme(readme_file, results)
        
        return str(run_dir)
    
    def _extract_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary information from evaluation results"""
        
        evaluation_run = results.get('evaluation_run', {})
        comparison_summary = results.get('comparison_summary', {})
        
        return {
            "evaluation_run_id": evaluation_run.get('id'),
            "name": evaluation_run.get('name'),
            "description": evaluation_run.get('description'),
            "evaluation_suite_id": evaluation_run.get('evaluation_suite_id'),
            "rag_approaches": evaluation_run.get('rag_approaches', []),
            "status": evaluation_run.get('status'),
            "started_at": evaluation_run.get('started_at'),
            "completed_at": evaluation_run.get('completed_at'),
            "best_approach": comparison_summary.get('best_approach'),
            "average_scores": comparison_summary.get('average_scores', {}),
            "success_rate": comparison_summary.get('success_rate'),
            "total_evaluation_cases": evaluation_run.get('total_evaluation_cases', 0),
            "completed_evaluation_cases": evaluation_run.get('completed_evaluation_cases', 0),
            "failed_evaluation_cases": evaluation_run.get('failed_evaluation_cases', 0),
            "recommendations": results.get('recommendations', [])
        }
    
    def _create_run_readme(self, readme_file: Path, results: Dict[str, Any]):
        """Create a README file for the evaluation run"""
        
        evaluation_run = results.get('evaluation_run', {})
        comparison_summary = results.get('comparison_summary', {})
        
        readme_content = f"""# Evaluation Run: {evaluation_run.get('name', 'Unknown')}

## Overview
- **Evaluation ID**: {evaluation_run.get('id', 'Unknown')}
- **Evaluation Suite**: {evaluation_run.get('evaluation_suite_id', 'Unknown')}
- **Status**: {evaluation_run.get('status', 'Unknown')}
- **Started**: {evaluation_run.get('started_at', 'Unknown')}
- **Completed**: {evaluation_run.get('completed_at', 'Unknown')}

## RAG Approaches Evaluated
{', '.join(evaluation_run.get('rag_approaches', []))}

## Results Summary
- **Best Approach**: {comparison_summary.get('best_approach', 'Unknown')}
- **Success Rate**: {comparison_summary.get('success_rate', 0):.1%}
- **Total Evaluation Cases**: {evaluation_run.get('total_evaluation_cases', 0)}
- **Completed**: {evaluation_run.get('completed_evaluation_cases', 0)}
- **Failed**: {evaluation_run.get('failed_evaluation_cases', 0)}

## Average Scores by Approach
"""
        
        for approach, score in comparison_summary.get('average_scores', {}).items():
            readme_content += f"- **{approach}**: {score:.3f}\n"
        
        readme_content += "\n## Recommendations\n"
        for rec in results.get('recommendations', []):
            readme_content += f"- {rec}\n"
        
        readme_content += f"""
## Files in this Directory
- `evaluation_results.json` - Complete evaluation results
- `summary.json` - Summary information
- `evaluation_cases/` - Individual evaluation case results
- `README.md` - This file

## Evaluation Case Results
"""
        
        for result in results.get('detailed_results', []):
            readme_content += f"- `{result['evaluation_case_id']}_{result['rag_approach']}.json` - {result['evaluation_case_id']} ({result['rag_approach']})\n"
        
        with open(readme_file, 'w') as f:
            f.write(readme_content)
    
    def list_evaluation_runs(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all evaluation runs, optionally filtered by date"""
        
        runs = []
        
        if date:
            # List runs for specific date
            date_dir = self.results_dir / date
            if date_dir.exists():
                for run_dir in date_dir.iterdir():
                    if run_dir.is_dir():
                        runs.append(self._get_run_info(run_dir))
        else:
            # List all runs
            for date_dir in self.results_dir.iterdir():
                if date_dir.is_dir():
                    for run_dir in date_dir.iterdir():
                        if run_dir.is_dir():
                            runs.append(self._get_run_info(run_dir))
        
        return sorted(runs, key=lambda x: x.get('completed_at', ''), reverse=True)
    
    def _get_run_info(self, run_dir: Path) -> Dict[str, Any]:
        """Get information about a specific evaluation run"""
        
        summary_file = run_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        
        return {
            "evaluation_run_id": run_dir.name,
            "name": "Unknown",
            "status": "Unknown",
            "completed_at": None
        }
    
    def get_evaluation_run(self, evaluation_run_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific evaluation run by ID"""
        
        for date_dir in self.results_dir.iterdir():
            if date_dir.is_dir():
                run_dir = date_dir / evaluation_run_id
                if run_dir.exists():
                    results_file = run_dir / "evaluation_results.json"
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            return json.load(f)
        
        return None
    
    def create_comparison_report(self, evaluation_run_ids: List[str]) -> str:
        """Create a comparison report across multiple evaluation runs"""
        
        runs = []
        for run_id in evaluation_run_ids:
            run_data = self.get_evaluation_run(run_id)
            if run_data:
                runs.append(run_data)
        
        if not runs:
            return "No evaluation runs found for comparison"
        
        # Create comparison report
        report = "# RAG Approach Comparison Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Evaluation Runs Compared\n"
        for run in runs:
            eval_run = run.get('evaluation_run', {})
            report += f"- **{eval_run.get('name', 'Unknown')}** ({eval_run.get('id', 'Unknown')})\n"
            report += f"  - Test Suite: {eval_run.get('test_suite_id', 'Unknown')}\n"
            report += f"  - Completed: {eval_run.get('completed_at', 'Unknown')}\n"
        
        report += "\n## Performance Comparison\n\n"
        
        # Compare average scores
        all_approaches = set()
        for run in runs:
            comparison = run.get('comparison_summary', {})
            all_approaches.update(comparison.get('average_scores', {}).keys())
        
        if all_approaches:
            report += "| RAG Approach | "
            for run in runs:
                eval_run = run.get('evaluation_run', {})
                report += f"{eval_run.get('name', 'Unknown')} | "
            report += "\n|--------------|"
            for _ in runs:
                report += "----------------|"
            report += "\n"
            
            for approach in sorted(all_approaches):
                report += f"| **{approach}** | "
                for run in runs:
                    comparison = run.get('comparison_summary', {})
                    score = comparison.get('average_scores', {}).get(approach, 0)
                    report += f"{score:.3f} | "
                report += "\n"
        
        # Save comparison report
        date_str = datetime.now().strftime("%Y-%m-%d")
        comparison_dir = self.results_dir / date_str / "comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = comparison_dir / f"comparison_{datetime.now().strftime('%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        return str(report_file)
