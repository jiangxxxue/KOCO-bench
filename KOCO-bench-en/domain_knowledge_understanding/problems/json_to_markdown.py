#!/usr/bin/env python3
"""
Convert JSON problem files to Markdown format.
Reads JSON files with structure like problems_*_EN.json and outputs formatted markdown.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def format_problem_type(instruction: str, is_chinese: bool = False) -> str:
    """Convert instruction to problem type label."""
    if "ONE" in instruction.upper() or "一个" in instruction:
        return "单选题" if is_chinese else "Single Choice"
    elif "MULTIPLE" in instruction.upper() or "多个" in instruction:
        return "多选题" if is_chinese else "Multiple Choice"
    return "未知" if is_chinese else "Unknown"


def format_correct_answer(gt: List[str]) -> str:
    """Format the ground truth answer(s)."""
    return ", ".join(sorted(gt))


def format_problem(problem: Dict[str, Any], index: int, is_chinese: bool = False) -> str:
    """Format a single problem as markdown."""
    repos = problem["dataset"]
    problem_type = format_problem_type(problem["instruction"], is_chinese)
    stem = problem["stem"]
    explanation = problem["explanation"]
    correct_answer = format_correct_answer(problem["gt"])
    
    if is_chinese:
        md = f"## {repos} - 问题 {index} ({problem_type})\n"
        md += f"{stem}\n\n"
        md += f"**正确答案：{correct_answer}**\n\n"
        md += f"**解析：** {explanation}\n\n"
    else:
        md = f"## {repos} - Problem {index} ({problem_type})\n"
        md += f"{stem}\n\n"
        md += f"**Correct Answer: {correct_answer}**\n\n"
        md += f"**Explanation:** {explanation}\n\n"
    
    md += "---\n\n"
    
    return md


def convert_json_to_markdown(json_path: Path, output_path: Path = None) -> str:
    """
    Convert a JSON problem file to markdown format.
    
    Args:
        json_path: Path to the input JSON file
        output_path: Optional path for output file. If None, returns string only.
    
    Returns:
        The markdown content as a string
    """
    # Detect if this is a Chinese file
    is_chinese = "ZH_CN" in json_path.name or "zh_cn" in json_path.name.lower()
    
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problems = data.get("problems", [])
    
    if not problems:
        raise ValueError(f"No problems found in {json_path}")
    
    # Extract dataset name from first problem
    dataset_name = problems[0].get("dataset", "Unknown")
    
    if is_chinese:
        title = f"{dataset_name.title()} 选择题"
    else:
        title = f"{dataset_name.title()} Multiple Choice Problems"
    
    # Build markdown content
    md_content = f"# {title}\n\n"
    
    for idx, problem in enumerate(problems, start=1):
        md_content += format_problem(problem, idx, is_chinese)
    
    # Write to file if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"✓ Converted {json_path.name} -> {output_path.name}")
    
    return md_content


def main():
    """Main function to process JSON files."""
    parser = argparse.ArgumentParser(
        description="Convert JSON problem files to Markdown format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Convert all problems_*_EN.json and problems_*_ZH_CN.json files
  %(prog)s file1.json file2.json        # Convert specific files
  %(prog)s -o output_dir file.json      # Specify output directory
  %(prog)s -p "problems_*.json"         # Use custom pattern
        """
    )
    
    parser.add_argument(
        'files',
        nargs='*',
        type=Path,
        help='JSON files to convert (if not specified, auto-discovers files)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=None,
        help='Output directory for markdown files (default: same as input file)'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default=None,
        help='Glob pattern to find JSON files (e.g., "problems_*.json")'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Determine which files to process
    if args.files:
        json_files = args.files
    elif args.pattern:
        json_files = list(Path('.').glob(args.pattern))
    else:
        # Default: find all *_EN.json and *_ZH_CN.json files
        json_files = list(Path('.').glob('problems_*_EN.json'))
        json_files.extend(Path('.').glob('problems_*_ZH_CN.json'))
    
    if not json_files:
        print("No JSON files found.")
        if args.pattern:
            print(f"Pattern used: {args.pattern}")
        else:
            print("Default patterns: 'problems_*_EN.json', 'problems_*_ZH_CN.json'")
        parser.print_help()
        return 1
    
    # Create output directory if specified
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.verbose:
            print(f"Output directory: {args.output_dir}\n")
    
    print(f"Found {len(json_files)} JSON file(s) to convert\n")
    
    success_count = 0
    error_count = 0
    
    for json_path in json_files:
        if not json_path.exists():
            print(f"✗ File not found: {json_path}")
            error_count += 1
            continue
        
        # Generate output filename
        if args.output_dir:
            output_path = args.output_dir / json_path.with_suffix('.md').name
        else:
            output_path = json_path.with_suffix('.md')
        
        try:
            convert_json_to_markdown(json_path, output_path)
            success_count += 1
        except Exception as e:
            print(f"✗ Error converting {json_path.name}: {e}")
            error_count += 1
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    print(f"\n✓ Conversion complete! ({success_count} successful, {error_count} failed)")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
