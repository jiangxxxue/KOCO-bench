#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_aggregate_metrics.py - Batch aggregate evaluation metrics for multiple models

Process multiple model directories at once and generate summary table
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

# Import aggregate_metrics module
from aggregate_metrics import aggregate_metrics, discover_test_examples


def batch_aggregate(
    base_dir: str,
    model_names: List[str],
    test_examples: List[str] = None,
    framework: str = None,
    output_csv: str = None,
) -> List[Dict[str, Any]]:
    """
    Batch aggregate metrics for multiple models
    
    Args:
        base_dir: Base directory path
        model_names: List of model names
        test_examples: List of test instances (optional, auto-discover instances under each model directory if not specified)
        framework: Framework name
        output_csv: Output CSV file path (optional)
    
    Returns:
        List of aggregated results for all models
    """
    base_path = Path(base_dir)
    all_results = []
    
    # List of test instances for CSV output (dynamically collected in auto-discover mode)
    all_test_examples = set()
    
    print("=" * 80)
    print("📊 Batch Aggregate Evaluation Metrics")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Number of models: {len(model_names)}")
    if test_examples:
        print(f"Test instances: {', '.join(test_examples)}")
    else:
        print("Test instances: (auto-discover)")
    print("=" * 80)
    print()
    
    for i, model_name in enumerate(model_names, 1):
        print(f"\n[{i}/{len(model_names)}] Processing model: {model_name}")
        print("-" * 80)
        
        model_dir = base_path / model_name
        
        if not model_dir.exists():
            print(f"⚠️  Warning: Model directory does not exist: {model_dir}")
            continue
        
        try:
            result = aggregate_metrics(
                model_dir=str(model_dir),
                test_examples=test_examples,
                framework=framework,
            )
            
            # Add model name
            result['model_name'] = model_name
            all_results.append(result)
            
            # Collect all test instances (for CSV output)
            for item in result['individual_metrics']:
                all_test_examples.add(item['example'])
            
            # Print brief results
            agg = result['aggregate_metrics']
            print(f"✓ pass@1: {agg['pass@1']:.4f}, avg_pass_ratio: {agg['avg_pass_ratio']:.4f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if not all_results:
        raise ValueError("No models were successfully processed")
    
    # Generate summary table
    print("\n" + "=" * 80)
    print("📋 Summary Table")
    print("=" * 80)
    print()
    
    # Table header
    header = f"{'Model Name':<30} {'Total Functions':>15} {'Passed':>10} {'pass@1':>12} {'avg_pass_ratio':>15}"
    print(header)
    print("-" * 80)
    
    # Data rows
    for result in all_results:
        model_name = result['model_name']
        agg = result['aggregate_metrics']
        
        row = (f"{model_name:<30} "
               f"{agg['total_functions']:>15} "
               f"{agg['total_passed']:>10} "
               f"{agg['pass@1']:>12.4f} "
               f"{agg['avg_pass_ratio']:>15.4f}")
        print(row)
    
    print("=" * 80)
    
    # Save CSV
    if output_csv:
        # Use specified test_examples or all collected test_examples
        csv_examples = test_examples if test_examples else sorted(all_test_examples)
        save_csv(all_results, output_csv, csv_examples)
        print(f"\n💾 CSV saved to: {output_csv}")
    
    return all_results


def save_csv(results: List[Dict[str, Any]], output_path: str, test_examples: List[str]):
    """Save results as CSV file"""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['model_name', 'total_functions', 'total_passed', 'pass@1', 'avg_pass_ratio']
        
        # Add columns for each test instance
        for example in test_examples:
            header.extend([
                f'{example}_functions',
                f'{example}_passed',
                f'{example}_pass@1',
                f'{example}_avg_pass_ratio'
            ])
        
        writer.writerow(header)
        
        # Write data
        for result in results:
            agg = result['aggregate_metrics']
            row = [
                result['model_name'],
                agg['total_functions'],
                agg['total_passed'],
                f"{agg['pass@1']:.4f}",
                f"{agg['avg_pass_ratio']:.4f}",
            ]
            
            # Add detailed data for each instance
            individual_dict = {item['example']: item['metrics'] 
                             for item in result['individual_metrics']}
            
            for example in test_examples:
                if example in individual_dict:
                    m = individual_dict[example]
                    row.extend([
                        m.get('total_functions', 0),
                        m.get('total_passed', 0),
                        f"{m.get('pass_at_k', {}).get('pass@1', 0.0):.4f}",
                        f"{m.get('avg_pass_ratio', 0.0):.4f}",
                    ])
                else:
                    row.extend(['', '', '', ''])
            
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Batch aggregate evaluation metrics for multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Auto-discover all test instances (recommended)
  python batch_aggregate_metrics.py \\
    --base_dir scripts/data/your_framework \\
    --model_names model1 model2 model3

  # Specify specific test instances
  python batch_aggregate_metrics.py \\
    --base_dir scripts/data/your_framework \\
    --model_names model1 model2 model3 \\
    --test_examples example1 example2 example3

  # Save as CSV
  python batch_aggregate_metrics.py \\
    --base_dir scripts/data/your_framework \\
    --model_names model1 model2 \\
    --output_csv comparison.csv

  # Use wildcards (requires shell support)
  python batch_aggregate_metrics.py \\
    --base_dir scripts/data/your_framework \\
    --model_names model-*
        """
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Base directory path'
    )
    parser.add_argument(
        '--model_names',
        type=str,
        nargs='+',
        required=True,
        help='List of model names (space-separated)'
    )
    parser.add_argument(
        '--test_examples',
        type=str,
        nargs='+',
        default=None,
        help='List of test instance names (space-separated). If not specified, auto-discover instances under each model directory'
    )
    parser.add_argument(
        '--framework',
        type=str,
        default=None,
        help='Framework name (optional)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Output CSV file path (optional)'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help='Output JSON file path (optional)'
    )
    
    args = parser.parse_args()
    
    try:
        # Batch aggregate
        results = batch_aggregate(
            base_dir=args.base_dir,
            model_names=args.model_names,
            test_examples=args.test_examples,
            framework=args.framework,
            output_csv=args.output_csv,
        )
        
        # Save JSON
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 JSON saved to: {args.output_json}")
        
        print("\n✅ Batch aggregation completed!")
        return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

