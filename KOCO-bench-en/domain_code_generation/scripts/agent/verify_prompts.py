#!/usr/bin/env python3
"""
Verify generated prompt files in source/output directory
"""

import json
import os
from pathlib import Path

def verify_prompt_file(file_path):
    """Verify a single prompt file"""
    results = {
        'file': os.path.basename(file_path),
        'exists': False,
        'records': 0,
        'has_prompt': False,
        'has_keyword': False,
        'keyword_paths_exist': [],
        'errors': []
    }

    if not os.path.exists(file_path):
        results['errors'].append('File does not exist')
        return results

    results['exists'] = True

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            results['records'] = len(lines)

            if lines:
                # Check first record
                first_record = json.loads(lines[0].strip())

                # Check prompt field
                if 'prompt' in first_record:
                    results['has_prompt'] = True
                    if len(first_record['prompt']) != 2:
                        results['errors'].append(f"Expected 2 prompt messages, got {len(first_record['prompt'])}")
                else:
                    results['errors'].append('Missing prompt field')

                # Check keyword field
                if 'keyword' in first_record:
                    results['has_keyword'] = True
                    for path in first_record['keyword']:
                        results['keyword_paths_exist'].append(os.path.exists(path))
                else:
                    results['errors'].append('Missing keyword field')

    except json.JSONDecodeError as e:
        results['errors'].append(f'JSON decode error: {e}')
    except Exception as e:
        results['errors'].append(f'Error: {e}')

    return results

def main():
    output_dir = Path('/home/gudako/repo/try/claude-code-domain-dataset/source/output')

    print("=" * 60)
    print("Prompt Files Verification")
    print("=" * 60)
    print()
    print(f"Output directory: {output_dir}")
    print()

    # Find all jsonl files
    jsonl_files = list(output_dir.glob('*_with_prompts_v2.jsonl'))

    if not jsonl_files:
        print("❌ No prompt files found!")
        return 1

    print(f"Found {len(jsonl_files)} file(s)")
    print()

    # Verify each file
    all_passed = True
    for file_path in sorted(jsonl_files):
        results = verify_prompt_file(file_path)

        # Print results
        status = "✓" if not results['errors'] else "✗"
        print(f"{status} {results['file']}")
        print(f"   Records: {results['records']}")
        print(f"   Has prompt: {results['has_prompt']}")
        print(f"   Has keyword: {results['has_keyword']}")

        if results['keyword_paths_exist']:
            all_exist = all(results['keyword_paths_exist'])
            paths_status = "✓" if all_exist else "✗"
            print(f"   Keyword paths exist: {paths_status} ({sum(results['keyword_paths_exist'])}/{len(results['keyword_paths_exist'])})")

        if results['errors']:
            all_passed = False
            for error in results['errors']:
                print(f"   ⚠️  {error}")

        print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("✓ All files passed verification!")
    else:
        print("✗ Some files have issues")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
