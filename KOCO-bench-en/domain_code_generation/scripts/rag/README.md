# RAG (Retrieval-Augmented Generation)

## Usage

```bash
python bm25_rag.py \
    --input data/input.jsonl \
    --metadata metadata.json \
    --knowledge-base /path/to/code/repo1 \
    --knowledge-base /path/to/code/repo2 \
    --framework framework_name \
    --output data/output.jsonl \
    --top-k 5
```
