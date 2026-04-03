# koco_openhands

OpenHands SDK agent evaluation pipeline for KOCO-bench domain code generation. Runs an AI agent that explores a framework repository, reads existing code, and implements target functions — then evaluates results via Docker execution.

## Prerequisites

- Python ≥ 3.12, < 3.14
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for evaluation step)

## Setup

```bash
cd KOCO-bench-en/domain_code_generation/scripts/koco_openhands

# install dependencies (auto-creates venv)
uv sync
```

Set API keys in `scripts/.env` (see `scripts/.env.example`):

```
OPENROUTER_API_KEY=sk-or-v1-...     # required — LLM access via OpenRouter
OPENAI_API_KEY=sk-...               # optional — enables semantic embeddings for knowledge_search
```

## Usage

All commands run from this directory (`koco_openhands/`):

```bash
# agent inference: explore repo and generate implementations
python cli.py infer --framework verl --model deepseek/deepseek-v3.2

# single test example only
python cli.py infer --framework verl --model deepseek/deepseek-v3.2 --test-example ARES

# specific functions only
python cli.py infer --framework verl --model deepseek/deepseek-v3.2 --instance-ids compute_score

# discard previous results and re-run from scratch
python cli.py infer --framework verl --model deepseek/deepseek-v3.2 --force

# preserve workspace artifacts (logs, code snapshot, agent events) for debugging
python cli.py infer --framework verl --model deepseek/deepseek-v3.2 --debug

# evaluation only (Docker execution + metrics aggregation)
python cli.py eval --framework verl --model deepseek/deepseek-v3.2

# full pipeline: infer + eval (resumes by default)
python cli.py run --framework verl --model deepseek/deepseek-v3.2
```

### Key CLI Options

| Option | Default | Description |
|---|---|---|
| `--framework` | *(required)* | Framework name (e.g., `verl`) |
| `--model` | `deepseek/deepseek-v3.2` | OpenRouter model identifier |
| `--test-example` | all | Single test example name (e.g., `ARES`) |
| `--instance-ids` | all | Comma-separated function names |
| `--concurrency` | `10` | Number of concurrent agents |
| `--max-iterations` | `50` | Max agent turns per function |
| `--force` | `false` | Discard previous results and re-run |
| `--debug` | `false` | Preserve workspace artifacts on completion |

### Pipeline Flow

1. **Parse** — Extract function specs from algorithm methods markdown → JSONL
2. **Prompt** — Build system context + user task prompts
3. **Infer** — For each target function:
   - Copy workspace to isolated temp dir and stub ground-truth function bodies
   - Run SDK agent with `terminal`, `file_editor`, and optionally `knowledge_search` tools
   - Extract implementation from modified source file (fallback: scan agent events)
4. **Eval** — Run Docker execution evaluation and aggregate metrics

### Knowledge Search Tool

Hybrid BM25 + semantic search over framework source code and documentation. Allows the agent to efficiently query the knowledge base instead of manually exploring files.

- **BM25** (weight 0.3): SQLite FTS5 keyword search
- **Semantic** (weight 0.7): Cosine similarity via `text-embedding-3-small` (requires `OPENAI_API_KEY`)
- Falls back to BM25-only when no embedding API key is available

## Output

Results are written to `scripts/data/{framework}/openhands/{model}/`:

```
algorithm_methods_data_{example}_output.jsonl   # inference results
algorithm_methods_data_{example}_result.jsonl    # evaluation results
algorithm_methods_data_{example}_result.metrics.json  # aggregated metrics
```

Debug artifacts (on failure or `--debug`): `scripts/data/{framework}/openhands/debug/{example}/{function_name}/`

## Tests

```bash
cd KOCO-bench-en/domain_code_generation/scripts/koco_openhands
uv run pytest test_cli.py test_runner.py tools/knowledge_search/test_knowledge_search.py -v
```
