#!/usr/bin/env python3
"""
KOCO-bench CLI — Cross-platform entry point.

Usage (works on Windows / Linux / macOS):
    # steps 1-3: parse algorithm methods, construct prompts, generate code via API
    python cli.py generate  --framework verl --model deepseek/deepseek-v3.2

    # steps 4-5: run execution evaluation + aggregate metrics
    python cli.py score     --framework verl --model deepseek/deepseek-v3.2

    # step 4 only: run execution evaluation inside Docker
    python cli.py evaluate  --framework verl --model deepseek/deepseek-v3.2

    # step 5 only: aggregate evaluation metrics
    python cli.py aggregate --framework verl --model deepseek/deepseek-v3.2

    # full pipeline (steps 1-5): generate + score
    python cli.py run       --framework verl --model deepseek/deepseek-v3.2

    # validate Docker image: run ground truth tests inside Docker
    python cli.py validate-image --framework verl
    python cli.py validate-image --framework verl --test-example ARES
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# Ensure the scripts/ directory is on sys.path so we can import sibling modules.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import (
    SCRIPTS_DIR,
    PROJECT_ROOT,
    get_test_examples,
    get_metadata_path,
    get_docker_image,
)


# Step 1 — Parse algorithm methods
def step1_parse(framework: str, examples: list) -> dict:
    """Parse algorithm methods from markdown into JSONL.

    Replaces: run_parse_algorithm_methods.sh (165 lines)
    """
    from parse_algorithm_methods import parse_markdown_file, process_functions

    success, skip, fail = 0, 0, 0

    for example in examples:
        input_file = (
            PROJECT_ROOT / framework / "test_examples" / example
            / "requirements" / "03_algorithm_and_core_methods.md"
        )
        code_base = PROJECT_ROOT / framework / "test_examples" / example / "code"
        test_base = PROJECT_ROOT / framework / "test_examples" / example / "code" / "tests"
        output_dir = SCRIPTS_DIR / "data" / framework
        output_file = output_dir / f"algorithm_methods_data_{example}.jsonl"

        print(f"\n--- {example} ---")

        if not input_file.exists():
            print(f"  Skipping: input file not found ({input_file.name})")
            skip += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            functions_data = parse_markdown_file(str(input_file))
            processed = process_functions(functions_data, str(code_base), str(test_base))

            with open(output_file, "w", encoding="utf-8") as fh:
                for func in processed:
                    fh.write(json.dumps(func, ensure_ascii=False) + "\n")

            print(f"  Parsed {len(processed)} functions -> {output_file.name}")
            success += 1
        except Exception as exc:
            print(f"  FAILED: {exc}")
            fail += 1

    return {"success": success, "skip": skip, "fail": fail}


# Step 2 — Construct prompts
def step2_prompts(framework: str, examples: list) -> dict:
    """Add prompt field to each JSONL record.

    Replaces: run_prompts_construction.sh (161 lines)
    """
    from prompts_construction import add_prompts_to_data

    metadata_path = get_metadata_path(framework)
    success, skip, fail = 0, 0, 0

    for example in examples:
        data_file = SCRIPTS_DIR / "data" / framework / f"algorithm_methods_data_{example}.jsonl"

        print(f"\n--- {example} ---")

        if not data_file.exists():
            print(f"  Skipping: data file not found")
            skip += 1
            continue

        try:
            add_prompts_to_data(str(data_file), str(data_file), str(metadata_path))
            success += 1
        except Exception as exc:
            print(f"  FAILED: {exc}")
            fail += 1

    return {"success": success, "skip": skip, "fail": fail}


# Step 3 — Generate code via OpenRouter API
def step3_generate(
    framework: str,
    model: str,
    examples: list,
    num_completions: int = 1,
    timeout: float = 300,
    connect_timeout: float = 30,
    max_retries: int = 5,
) -> dict:
    """Call OpenRouter API to generate code completions.

    Replaces: apicall/run_openrouter.sh (161 lines)
    """
    # Make the apicall subpackage importable
    apicall_dir = str(SCRIPTS_DIR / "apicall")
    if apicall_dir not in sys.path:
        sys.path.insert(0, apicall_dir)
    from generate_completions_openrouter import main as gen_main

    # Check API key early
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("\nERROR: OPENROUTER_API_KEY is not set.")
        print("Set it in scripts/.env or export it as an environment variable.")
        return {"success": 0, "skip": 0, "fail": len(examples)}

    model_dir_name = model.split("/")[-1]
    output_dir = SCRIPTS_DIR / "data" / framework / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    success, skip, fail = 0, 0, 0

    for example in examples:
        input_file = SCRIPTS_DIR / "data" / framework / f"algorithm_methods_data_{example}.jsonl"
        output_file = output_dir / f"algorithm_methods_data_{example}_output.jsonl"

        print(f"\n--- {example} ---")

        if not input_file.exists():
            print(f"  Skipping: input file not found")
            skip += 1
            continue

        # Inject sys.argv so the existing argparse-based main() can parse them.
        saved_argv = sys.argv
        sys.argv = [
            "generate_completions_openrouter.py",
            "--model", model,
            "--input_file", str(input_file),
            "--output_file", str(output_file),
            "--num_completions", str(num_completions),
            "--max_tokens", "30000",
            "--temperature", "0.0",
            "--top_p", "1.0",
            "--delay", "0.5",
            "--timeout", str(timeout),
            "--connect_timeout", str(connect_timeout),
            "--max_retries", str(max_retries),
            "--debug",
        ]
        try:
            rc = gen_main()
            if rc is None or rc == 0:
                success += 1
            else:
                fail += 1
        except SystemExit as exc:
            # argparse may call sys.exit; treat 0 as success
            if exc.code == 0:
                success += 1
            else:
                fail += 1
        except Exception as exc:
            print(f"  FAILED: {exc}")
            fail += 1
        finally:
            sys.argv = saved_argv

    return {"success": success, "skip": skip, "fail": fail}



# Step 4 — Docker execution evaluation
def cmd_evaluate(framework: str, model: str, test_example: str = None) -> int:
    """Run execution evaluation inside Docker.

    Replaces: run_execution_evaluation_pure.sh (118 lines)
              run_batch_execution_evaluation_pure.sh (241 lines)

    Uses pathlib for native OS paths, avoiding MSYS path mangling on Windows.
    """
    image = get_docker_image(framework)
    container_mnt = "/workspace/project"
    host_root = str(PROJECT_ROOT)  # Native path on every OS

    # Verify Docker is available
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Docker is not running or not installed.")
        print("  - Windows/macOS: Start Docker Desktop")
        print("  - Linux: sudo systemctl start docker")
        return 1

    # Verify image exists
    result = subprocess.run(
        ["docker", "images", "-q", image],
        capture_output=True, text=True,
    )
    if not result.stdout.strip():
        print(f"ERROR: Docker image '{image}' not found.")
        print(f"Build it first:  cd Build-Env/Docker && make FRAMEWORK={framework}")
        return 1

    examples = [test_example] if test_example else get_test_examples(framework)
    model_dir_name = model.split("/")[-1]
    total, passed, failed = 0, 0, 0

    for example in examples:
        total += 1
        input_name = f"algorithm_methods_data_{example}_output.jsonl"
        output_name = f"algorithm_methods_data_{example}_result.jsonl"

        data_subdir = f"scripts/data/{framework}/{model_dir_name}"
        source_subdir = f"{framework}/test_examples/{example}/code"

        # Check that input file exists on host
        host_input = PROJECT_ROOT / data_subdir / input_name
        if not host_input.exists():
            print(f"  {example}: SKIP (input file not found)")
            continue

        docker_cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{host_root}:{container_mnt}",
            image,
            "python3", f"{container_mnt}/scripts/execution_evaluation_pure.py",
            "--source_dir", f"{container_mnt}/{source_subdir}",
            "--input_file", f"{container_mnt}/{data_subdir}/{input_name}",
            "--output_file", f"{container_mnt}/{data_subdir}/{output_name}",
        ]

        if platform.system() == "Linux":
            uid, gid = os.getuid(), os.getgid()
            docker_cmd[3:3] = [
                "--user", f"{uid}:{gid}",
                "-e", "HOME=/tmp",
                "-e", "USER=benchuser",
            ]

        print(f"\n--- {example} ---")
        print(f"  Docker image: {image}")

        rc = subprocess.run(docker_cmd).returncode
        if rc == 0:
            # Try to show test summary from metrics file
            metrics_file = PROJECT_ROOT / data_subdir / output_name.replace('_result.jsonl', '_result.metrics.json')
            if metrics_file.exists():
                with open(metrics_file) as f:
                    m = json.load(f)
                tp = m.get('total_passed', 0)
                tt = m.get('total_tests', 0)
                p1 = m.get('pass_at_k', {}).get('pass@1', 0)
                print(f"  COMPLETED ({tp}/{tt} passed, pass@1={p1:.4f})")
            else:
                print(f"  COMPLETED")
            passed += 1
        else:
            print(f"  FAIL (exit code {rc})")
            failed += 1

    print(f"\nEvaluation summary: {passed}/{total} passed, {failed} failed")
    return 1 if failed else 0


# Validate Docker image — run ground truth tests inside Docker
def cmd_validate_image(framework: str, test_example: str = None) -> int:
    """Run ground truth tests inside Docker to validate images.

    For each test example, runs run_tests_with_stats.py inside the
    framework's Docker container. If all tests pass at 100%, the image
    is considered valid.

    Usage:
        python cli.py validate-image --framework verl
        python cli.py validate-image --framework verl --test-example ARES
    """
    image = get_docker_image(framework)
    container_mnt = "/workspace/project"
    host_root = str(PROJECT_ROOT)

    # Verify Docker is available
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Docker is not running or not installed.")
        print("  - Windows/macOS: Start Docker Desktop")
        print("  - Linux: sudo systemctl start docker")
        return 1

    # Verify image exists
    result = subprocess.run(
        ["docker", "images", "-q", image],
        capture_output=True, text=True,
    )
    if not result.stdout.strip():
        print(f"ERROR: Docker image '{image}' not found.")
        print(f"Build it first:  cd Build-Env/Docker && make FRAMEWORK={framework}")
        return 1

    examples = [test_example] if test_example else get_test_examples(framework)
    total, passed, failed, skipped = 0, 0, 0, 0

    for example in examples:
        total += 1
        test_code_dir = (
            PROJECT_ROOT / framework / "test_examples" / example
            / "code" / "test_code"
        )
        runner_file = test_code_dir / "run_tests_with_stats.py"

        if not runner_file.exists():
            print(f"\n--- {example} ---")
            print(f"  SKIP: run_tests_with_stats.py not found")
            skipped += 1
            continue

        # Paths inside container
        container_test_code = (
            f"{container_mnt}/{framework}/test_examples/{example}/code/test_code"
        )
        container_runner = f"{container_test_code}/run_tests_with_stats.py"

        docker_cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{host_root}:{container_mnt}",
            "-w", container_test_code,
            image,
            "python3", container_runner,
        ]

        if platform.system() == "Linux":
            uid, gid = os.getuid(), os.getgid()
            docker_cmd[3:3] = [
                "--user", f"{uid}:{gid}",
                "-e", "HOME=/tmp",
                "-e", "USER=benchuser",
            ]

        print(f"\n--- {example} ---")
        print(f"  Docker image : {image}")
        print(f"  Test runner  : {framework}/test_examples/{example}/code/test_code/run_tests_with_stats.py")

        proc = subprocess.run(docker_cmd)
        if proc.returncode == 0:
            print(f"  Result: PASS (100% ground truth tests passed)")
            passed += 1
        else:
            print(f"  Result: FAIL (exit code {proc.returncode})")
            failed += 1

    # Summary
    print()
    print("=" * 60)
    print("  Docker Image Validation Summary")
    print("=" * 60)
    print(f"  Framework    : {framework}")
    print(f"  Docker image : {image}")
    print(f"  Total        : {total}")
    print(f"  Passed (100%): {passed}")
    print(f"  Failed       : {failed}")
    print(f"  Skipped      : {skipped}")

    if failed == 0 and skipped == 0:
        print(f"\n  ALL {passed} test examples passed — image is VALID")
    elif failed == 0:
        print(f"\n  {passed} passed, {skipped} skipped — image looks OK (some runners missing)")
    else:
        print(f"\n  {failed} FAILED — image may have issues")

    print("=" * 60)
    return 1 if failed else 0


# Step 5 — Aggregate metrics
def cmd_aggregate(framework: str, model: str) -> int:
    """Aggregate evaluation metrics.

    Replaces: run_aggregate_metrics.sh (193 lines)
    """
    from aggregate_metrics import main as agg_main

    model_dir_name = model.split("/")[-1]
    model_dir = SCRIPTS_DIR / "data" / framework / model_dir_name

    if not model_dir.is_dir():
        print(f"ERROR: Model output directory not found: {model_dir}")
        return 1

    saved_argv = sys.argv
    sys.argv = [
        "aggregate_metrics.py",
        "--model_dir", str(model_dir),
        "--framework", framework,
    ]
    try:
        rc = agg_main()
        return rc if rc is not None else 0
    except SystemExit as exc:
        return exc.code if exc.code else 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        sys.argv = saved_argv


# CLI entry point
def _print_banner(title: str, **kwargs):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    for k, v in kwargs.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print()


def _print_step_result(step_name: str, result: dict):
    total = result["success"] + result["skip"] + result["fail"]
    print(f"\n{step_name}: {result['success']}/{total} succeeded", end="")
    if result["skip"]:
        print(f", {result['skip']} skipped", end="")
    if result["fail"]:
        print(f", {result['fail']} FAILED", end="")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="KOCO-bench CLI — cross-platform benchmark runner",
    )
    subparsers = parser.add_subparsers(dest="command")

    # generate
    gen = subparsers.add_parser(
        "generate",
        help="Steps 1-3: parse algorithm methods, construct prompts, generate code via API",
    )
    gen.add_argument("--framework", required=True, help="Framework name (e.g. verl)")
    gen.add_argument("--model", required=True, help="Model name (e.g. deepseek/deepseek-v3.2)")
    gen.add_argument("--test-example", default=None, help="Single test example (default: all)")
    gen.add_argument("--num-completions", type=int, default=1, help="Completions per sample")
    gen.add_argument("--timeout", type=float, default=300, help="API read timeout in seconds (default: 300)")
    gen.add_argument("--connect-timeout", type=float, default=30, help="API connect timeout in seconds (default: 30)")
    gen.add_argument("--max-retries", type=int, default=5, help="Max API retries (default: 5)")

    # evaluate
    ev = subparsers.add_parser(
        "evaluate",
        help="Step 4: run execution evaluation inside Docker",
    )
    ev.add_argument("--framework", required=True)
    ev.add_argument("--model", required=True)
    ev.add_argument("--test-example", default=None)

    # aggregate
    ag = subparsers.add_parser(
        "aggregate",
        help="Step 5: aggregate evaluation metrics",
    )
    ag.add_argument("--framework", required=True)
    ag.add_argument("--model", required=True)

    # score (steps 4-5)
    sc = subparsers.add_parser(
        "score",
        help="Steps 4-5: run Docker execution evaluation + aggregate metrics",
    )
    sc.add_argument("--framework", required=True, help="Framework name (e.g. verl)")
    sc.add_argument("--model", required=True, help="Model name (e.g. deepseek/deepseek-v3.2)")
    sc.add_argument("--test-example", default=None, help="Single test example (default: all)")

    # validate-image
    vi = subparsers.add_parser(
        "validate-image",
        help="Validate Docker image by running ground truth tests to 100%% pass",
    )
    vi.add_argument("--framework", required=True, help="Framework name (e.g. verl)")
    vi.add_argument("--test-example", default=None, help="Single test example (default: all)")

    # run (full pipeline)
    rn = subparsers.add_parser(
        "run",
        help="Full pipeline: generate + evaluate + aggregate",
    )
    rn.add_argument("--framework", required=True)
    rn.add_argument("--model", required=True)
    rn.add_argument("--test-example", default=None)
    rn.add_argument("--num-completions", type=int, default=1)
    rn.add_argument("--timeout", type=float, default=300)
    rn.add_argument("--connect-timeout", type=float, default=30)
    rn.add_argument("--max-retries", type=int, default=5)

    # parse
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Resolve test examples
    test_example = getattr(args, "test_example", None)

    # generate
    if args.command == "generate":
        _print_banner(
            "KOCO-bench: Code Generation (Steps 1-3)",
            Framework=args.framework,
            Model=args.model,
            TestExample=test_example or "all",
            Completions=args.num_completions,
        )

        examples = [test_example] if test_example else get_test_examples(args.framework)

        # Step 1
        print("\n>>> Step 1/3: Parse algorithm methods")
        r1 = step1_parse(args.framework, examples)
        _print_step_result("Step 1", r1)
        if r1["fail"] and not r1["success"]:
            print("\nStep 1 failed completely. Aborting.")
            return 1

        # Step 2
        print("\n>>> Step 2/3: Construct prompts")
        r2 = step2_prompts(args.framework, examples)
        _print_step_result("Step 2", r2)
        if r2["fail"] and not r2["success"]:
            print("\nStep 2 failed completely. Aborting.")
            return 1

        # Step 3
        print("\n>>> Step 3/3: Generate code via OpenRouter API")
        r3 = step3_generate(
            args.framework, args.model, examples, args.num_completions,
            args.timeout, args.connect_timeout, args.max_retries,
        )
        _print_step_result("Step 3", r3)

        # Summary
        model_dir_name = args.model.split("/")[-1]
        print("\n" + "=" * 60)
        print("  All 3 steps completed!")
        print("=" * 60)
        print(f"  1. Parse algorithm methods    {'PASS' if not r1['fail'] else 'PARTIAL'}")
        print(f"  2. Construct prompts          {'PASS' if not r2['fail'] else 'PARTIAL'}")
        print(f"  3. Generate code (OpenRouter) {'PASS' if not r3['fail'] else 'PARTIAL'}")
        print(f"\n  Output: scripts/data/{args.framework}/{model_dir_name}/")
        print("=" * 60)

        return 1 if (r1["fail"] or r2["fail"] or r3["fail"]) else 0

    # validate-image
    elif args.command == "validate-image":
        _print_banner(
            "KOCO-bench: Validate Docker Image (Ground Truth Tests)",
            Framework=args.framework,
            TestExample=test_example or "all",
        )
        return cmd_validate_image(args.framework, test_example)

    # evaluate
    elif args.command == "evaluate":
        _print_banner(
            "KOCO-bench: Docker Execution Evaluation (Step 4)",
            Framework=args.framework,
            Model=args.model,
            TestExample=test_example or "all",
        )
        return cmd_evaluate(args.framework, args.model, test_example)

    # aggregate
    elif args.command == "aggregate":
        _print_banner(
            "KOCO-bench: Aggregate Metrics (Step 5)",
            Framework=args.framework,
            Model=args.model,
        )
        return cmd_aggregate(args.framework, args.model)

    # score (steps 4-5)
    elif args.command == "score":
        _print_banner(
            "KOCO-bench: Evaluation & Aggregation (Steps 4-5)",
            Framework=args.framework,
            Model=args.model,
            TestExample=test_example or "all",
        )

        # Step 4
        print("\n>>> Step 1/2: Docker execution evaluation")
        rc4 = cmd_evaluate(args.framework, args.model, test_example)

        # Step 5 — run even if step 4 had partial failures
        print("\n>>> Step 2/2: Aggregate metrics")
        rc5 = cmd_aggregate(args.framework, args.model)

        # Summary
        model_dir_name = args.model.split("/")[-1]
        print("\n" + "=" * 60)
        print("  Steps 4-5 completed!")
        print("=" * 60)
        print(f"  4. Docker evaluation  {'PASS' if rc4 == 0 else 'FAIL'}")
        print(f"  5. Aggregate metrics  {'PASS' if rc5 == 0 else 'FAIL'}")
        print(f"\n  Results: scripts/data/{args.framework}/{model_dir_name}/")
        print("=" * 60)

        return 1 if (rc4 or rc5) else 0

    # run (full pipeline)
    elif args.command == "run":
        _print_banner(
            "KOCO-bench: Full Pipeline (Steps 1-5)",
            Framework=args.framework,
            Model=args.model,
            TestExample=test_example or "all",
            Completions=args.num_completions,
        )

        examples = [test_example] if test_example else get_test_examples(args.framework)

        # Steps 1-3
        print("\n>>> Step 1/5: Parse algorithm methods")
        r1 = step1_parse(args.framework, examples)
        _print_step_result("Step 1", r1)
        if r1["fail"] and not r1["success"]:
            return 1

        print("\n>>> Step 2/5: Construct prompts")
        r2 = step2_prompts(args.framework, examples)
        _print_step_result("Step 2", r2)
        if r2["fail"] and not r2["success"]:
            return 1

        print("\n>>> Step 3/5: Generate code via OpenRouter API")
        r3 = step3_generate(
            args.framework, args.model, examples, args.num_completions,
            args.timeout, args.connect_timeout, args.max_retries,
        )
        _print_step_result("Step 3", r3)
        if r3["fail"] and not r3["success"]:
            return 1

        # Step 4
        print("\n>>> Step 4/5: Docker execution evaluation")
        rc4 = cmd_evaluate(args.framework, args.model, test_example)
        if rc4 != 0:
            print("\nStep 4 failed. Aborting.")
            return 1

        # Step 5
        print("\n>>> Step 5/5: Aggregate metrics")
        rc5 = cmd_aggregate(args.framework, args.model)

        print("\n" + "=" * 60)
        print("  Full pipeline completed!")
        print("=" * 60)
        return rc5

    return 0


if __name__ == "__main__":
    sys.exit(main())
