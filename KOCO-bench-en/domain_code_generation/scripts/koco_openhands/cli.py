#!/usr/bin/env python3
"""
KOCO-bench OpenHands CLI — agent-based code generation.

Uses OpenHands agent (via OpenRouter API) to explore repositories and
implement functions, then evaluates via the existing Docker pipeline.

Usage (run from the scripts/ directory):
    # agent inference: explore repo, read code, generate implementation
    python koco_openhands/cli.py infer --framework verl --model deepseek/deepseek-v3.2

    # agent inference: single test example only
    python koco_openhands/cli.py infer --framework verl --model deepseek/deepseek-v3.2 --test-example ARES

    # agent inference: specific functions only
    python koco_openhands/cli.py infer --framework verl --model deepseek/deepseek-v3.2 --instance-ids compute_score

    # agent inference: discard previous results and re-run from scratch
    python koco_openhands/cli.py infer --framework verl --model deepseek/deepseek-v3.2 --force

    # evaluation: run execution evaluation inside Docker + aggregate metrics (reuses cli.py steps 4-5)
    python koco_openhands/cli.py eval  --framework verl --model deepseek/deepseek-v3.2

    # full pipeline: infer + eval (resumes by default, skipping completed instances)
    python koco_openhands/cli.py run   --framework verl --model deepseek/deepseek-v3.2

    # full pipeline: discard previous infer results and re-run everything from scratch
    python koco_openhands/cli.py run   --framework verl --model deepseek/deepseek-v3.2 --force

Prerequisites:
    pip install openhands-ai numpy
    export OPENROUTER_API_KEY="sk-or-..."

Output:
    data/{framework}/openhands/{model}/algorithm_methods_data_{example}_output.jsonl
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add the parent scripts/ directory to sys.path so we can import
# config.py, aggregate_metrics.py, etc.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Also add our own directory so "import runner" works
_OPENHANDS_DIR = Path(__file__).resolve().parent
if str(_OPENHANDS_DIR) not in sys.path:
    sys.path.insert(0, str(_OPENHANDS_DIR))

from config import (
    SCRIPTS_DIR,
    PROJECT_ROOT,
    get_test_examples,
    get_docker_image,
    get_metadata_path,
)


# ── helpers ───────────────────────────────────────────────────────────────

def _ensure_input_data(framework: str, example: str) -> bool:
    """Run Step 1 (parse) + Step 2 (prompts) for a single test example.

    Returns True if the input JSONL file exists afterwards.
    """
    from parse_algorithm_methods import parse_markdown_file, process_functions
    from prompts_construction import add_prompts_to_data

    # Step 1: parse algorithm methods from markdown → JSONL
    input_md = (
        PROJECT_ROOT / framework / "test_examples" / example
        / "requirements" / "03_algorithm_and_core_methods.md"
    )
    code_base = PROJECT_ROOT / framework / "test_examples" / example / "code"
    test_base = code_base / "tests"
    output_dir = SCRIPTS_DIR / "data" / framework
    output_file = output_dir / f"algorithm_methods_data_{example}.jsonl"

    if not input_md.exists():
        print(f"    Step 1: markdown not found ({input_md.name})")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        functions_data = parse_markdown_file(str(input_md))
        processed = process_functions(functions_data, str(code_base), str(test_base))
        with open(output_file, "w", encoding="utf-8") as fh:
            for func in processed:
                fh.write(json.dumps(func, ensure_ascii=False) + "\n")
        print(f"    Step 1: parsed {len(processed)} functions")
    except Exception as exc:
        print(f"    Step 1 failed: {exc}")
        return False

    # Step 2: add prompts to each record
    metadata_path = get_metadata_path(framework)
    try:
        add_prompts_to_data(str(output_file), str(output_file), str(metadata_path))
        print(f"    Step 2: prompts added")
    except Exception as exc:
        print(f"    Step 2 failed: {exc}")
        return False

    return output_file.exists()


# ── infer ─────────────────────────────────────────────────────────────────

def cmd_infer(
    framework: str,
    model: str,
    test_example: str = None,
    api_key: str = "",
    base_url: str = "https://openrouter.ai/api/v1",
    max_iterations: int = 50,
    instance_ids: list = None,
    concurrency: int = 3,
    force: bool = False,
    debug: bool = False,
) -> int:
    """Run OpenHands agent inference for code generation.

    Args:
        force: If True, delete existing progress and output files before
               running, so that all instances are re-inferred from scratch.
               By default (False), completed instances are skipped (resume).
        debug: If True, preserve workspace artifacts (logs, code snapshot,
               agent events) for every run, not just failures.
    """
    from runner import (
        run_single_instance,
        load_jsonl,
        save_jsonl,
        load_completed_ids,
        save_completed_ids,
    )

    # Resolve API key: arg > env > .env file
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        env_file = SCRIPTS_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        print("\nERROR: OPENROUTER_API_KEY is not set.")
        print("Set it via --api-key, export OPENROUTER_API_KEY=..., or in scripts/.env")
        return 1

    model_dir_name = model.split("/")[-1]
    examples = [test_example] if test_example else get_test_examples(framework)

    success, fail = 0, 0

    for example in examples:
        print(f"\n--- {example} ---")

        # Input data file (shared with cli.py) — auto-generate if missing
        data_file = SCRIPTS_DIR / "data" / framework / f"algorithm_methods_data_{example}.jsonl"
        if not data_file.exists():
            print(f"  Input data not found, running Step 1-2 (parse + prompts)...")
            if not _ensure_input_data(framework, example):
                print(f"  Skipping: failed to generate input data")
                fail += 1
                continue

        # Output directory — under openhands/ to avoid collision with cli.py output
        output_dir = SCRIPTS_DIR / "data" / framework / "openhands" / model_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"algorithm_methods_data_{example}_output.jsonl"
        progress_file = output_dir / f".{example}_progress.json"

        # --force: wipe previous infer + eval results so everything re-runs
        result_file = output_dir / f"algorithm_methods_data_{example}_result.jsonl"
        metrics_file = output_dir / f"algorithm_methods_data_{example}_result.metrics.json"
        if force:
            for f in (output_file, progress_file, result_file, metrics_file):
                if f.exists():
                    f.unlink()
                    print(f"  Removed (--force): {f.name}")

        # Workspace (the repo code the agent explores)
        workspace_root = str(
            PROJECT_ROOT / framework / "test_examples" / example / "code"
        )
        if not os.path.isdir(workspace_root):
            print(f"  Skipping: workspace not found ({workspace_root})")
            fail += 1
            continue

        # Knowledge corpus (framework docs the agent can reference)
        knowledge_corpus_root = str(
            PROJECT_ROOT / framework / "knowledge_corpus"
        )
        if not os.path.isdir(knowledge_corpus_root):
            print(f"  Skipping: knowledge_corpus not found ({knowledge_corpus_root})")
            fail += 1
            continue

        # Load task data
        all_records = load_jsonl(str(data_file))
        print(f"  Loaded {len(all_records)} functions")

        # Filter by specific function names
        records = all_records
        if instance_ids:
            records = [r for r in records if r["function_name"] in instance_ids]
            print(f"  Filtered to {len(records)} by --instance-ids")

        # Resume support
        completed = load_completed_ids(str(progress_file))
        if completed:
            print(f"  Resuming: {len(completed)} already completed")

        # Load existing output (for incremental append)
        existing = {}
        if output_file.exists():
            for r in load_jsonl(str(output_file)):
                existing[r["function_name"]] = r

        pending = [r for r in records if r["function_name"] not in completed]

        if not pending:
            print(f"  All {len(records)} instances already completed")
            success += 1
            continue

        workers = min(concurrency, len(pending))
        print(f"  Processing {len(pending)} pending instances ({workers} concurrent)")

        # Lock protects shared state: existing, completed, file writes
        lock = threading.Lock()

        def _process_one(record):
            result = run_single_instance(
                record=record,
                framework=framework,
                example=example,
                workspace_root=workspace_root,
                knowledge_corpus_root=knowledge_corpus_root,
                model=model,
                api_key=api_key,
                base_url=base_url,
                max_iterations=max_iterations,
                debug=debug,
            )
            with lock:
                existing[result["function_name"]] = result
                completed.add(result["function_name"])
                save_completed_ids(completed, str(progress_file))
                save_jsonl(list(existing.values()), str(output_file))
            return result

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, r): r for r in pending}
            for future in as_completed(futures):
                r = future.result()
                print(f"  [{r['function_name']}] done -> {r['status']}")

        print(f"  Saved {len(existing)} results -> {output_file.name}")
        success += 1

    print(f"\nInference summary: {success} example(s) processed, {fail} failed")
    return 0


# ── eval (reuses cli.py steps 4-5) ───────────────────────────────────────

def cmd_eval(framework: str, model: str, test_example: str = None) -> int:
    """Run Docker evaluation + aggregate metrics.

    Reads inference output from the openhands/ subdirectory, runs
    execution_evaluation_pure.py inside Docker, then aggregates metrics.

    Runs Docker eval and aggregation directly (instead of delegating to
    the parent cli.py) so that the data path correctly points to
    ``scripts/data/{framework}/openhands/{model_dir_name}/``.
    """
    from aggregate_metrics import main as agg_main

    model_dir_name = model.split("/")[-1]
    # Correct path: scripts/data/{framework}/openhands/{model_dir_name}
    data_subdir = f"scripts/data/{framework}/openhands/{model_dir_name}"

    image = get_docker_image(framework)
    container_mnt = "/workspace/project"
    host_root = str(PROJECT_ROOT)

    # Verify Docker is available
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
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

    # --- Step 1: Docker execution evaluation ---
    print("\n>>> Step 1/2: Docker execution evaluation")
    total, passed, failed = 0, 0, 0

    for example in examples:
        total += 1
        input_name = f"algorithm_methods_data_{example}_output.jsonl"
        output_name = f"algorithm_methods_data_{example}_result.jsonl"
        source_subdir = f"{framework}/test_examples/{example}/code"

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
            metrics_file = PROJECT_ROOT / data_subdir / output_name.replace(
                '_result.jsonl', '_result.metrics.json')
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
    eval_rc = 1 if failed else 0

    # --- Step 2: Aggregate metrics ---
    print("\n>>> Step 2/2: Aggregate metrics")
    model_dir = SCRIPTS_DIR / "data" / framework / "openhands" / model_dir_name

    if not model_dir.is_dir():
        print(f"ERROR: Model output directory not found: {model_dir}")
        agg_rc = 1
    else:
        saved_argv = sys.argv
        sys.argv = [
            "aggregate_metrics.py",
            "--model_dir", str(model_dir),
            "--framework", framework,
        ]
        try:
            rc = agg_main()
            agg_rc = rc if rc is not None else 0
        except SystemExit as exc:
            agg_rc = exc.code if exc.code else 0
        except Exception as exc:
            print(f"ERROR: {exc}")
            agg_rc = 1
        finally:
            sys.argv = saved_argv

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Evaluation completed!")
    print("=" * 60)
    print(f"  Docker evaluation:  {'PASS' if eval_rc == 0 else 'FAIL'}")
    print(f"  Aggregate metrics:  {'PASS' if agg_rc == 0 else 'FAIL'}")
    print(f"\n  Results: data/{framework}/openhands/{model_dir_name}/")
    print("=" * 60)

    return 1 if (eval_rc or agg_rc) else 0


# ── run (infer + eval) ───────────────────────────────────────────────────

def cmd_run(
    framework: str,
    model: str,
    test_example: str = None,
    **infer_kwargs,
) -> int:
    """Full pipeline: agent inference + Docker evaluation."""
    print("\n>>> Phase 1: Agent inference")
    rc1 = cmd_infer(framework, model, test_example, **infer_kwargs)
    if rc1 != 0:
        print("\nInference failed. Aborting.")
        return 1

    print("\n>>> Phase 2: Evaluation")
    rc2 = cmd_eval(framework, model, test_example)
    return rc2


# ── CLI ───────────────────────────────────────────────────────────────────

def _print_banner(title: str, **kwargs):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    for k, v in kwargs.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="KOCO-bench OpenHands CLI — agent-based code generation",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Shared arguments
    def add_common(sp):
        sp.add_argument("--framework", required=True, help="Framework name (e.g. verl)")
        sp.add_argument("--model", default="deepseek/deepseek-v3.2", help="OpenRouter model (default: deepseek/deepseek-v3.2)")
        sp.add_argument("--test-example", default=None, help="Single test example (default: all)")

    def add_infer_args(sp):
        sp.add_argument("--api-key", default="", help="OpenRouter API key (default: OPENROUTER_API_KEY env)")
        sp.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="API base URL")
        sp.add_argument("--max-iterations", type=int, default=50, help="Max agent turns (default: 50)")
        sp.add_argument("--instance-ids", nargs="+", help="Specific function names to process")
        sp.add_argument("--concurrency", "-j", type=int, default=10, help="Concurrent agents per framework (default: 10)")
        sp.add_argument("--force", action="store_true", help="Discard previous infer results and re-run from scratch")
        sp.add_argument("--debug", action="store_true", help="Preserve workspace artifacts (logs, code snapshot, agent events) for post-mortem analysis")

    # infer
    inf = subparsers.add_parser("infer", help="Run OpenHands agent inference")
    add_common(inf)
    add_infer_args(inf)

    # eval
    ev = subparsers.add_parser("eval", help="Run Docker evaluation + aggregate metrics")
    add_common(ev)

    # run
    rn = subparsers.add_parser("run", help="Full pipeline: infer + eval")
    add_common(rn)
    add_infer_args(rn)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    test_example = getattr(args, "test_example", None)

    if args.command == "infer":
        _print_banner(
            "KOCO-bench OpenHands: Agent Inference",
            Framework=args.framework,
            Model=args.model,
            TestExample=test_example or "all",
            Concurrency=args.concurrency,
            Debug=args.debug,
        )
        return cmd_infer(
            args.framework, args.model, test_example,
            api_key=args.api_key,
            base_url=args.base_url,
            max_iterations=args.max_iterations,
            instance_ids=args.instance_ids,
            concurrency=args.concurrency,
            force=args.force,
            debug=args.debug,
        )

    elif args.command == "eval":
        _print_banner(
            "KOCO-bench OpenHands: Evaluation",
            Framework=args.framework,
            Model=args.model,
            TestExample=test_example or "all",
        )
        return cmd_eval(args.framework, args.model, test_example)

    elif args.command == "run":
        _print_banner(
            "KOCO-bench OpenHands: Full Pipeline",
            Framework=args.framework,
            Model=args.model,
            TestExample=test_example or "all",
            Concurrency=args.concurrency,
            Debug=args.debug,
        )
        return cmd_run(
            args.framework, args.model, test_example,
            api_key=args.api_key,
            base_url=args.base_url,
            max_iterations=args.max_iterations,
            instance_ids=args.instance_ids,
            concurrency=args.concurrency,
            force=args.force,
            debug=args.debug,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
