# Simple RAG Project Management System

## Overview

A lightweight system to track and manage RAG projects without complex Firebase integration. Uses local JSON files and simple storage organization.

## Core Concept

Each RAG project is a single RAG database setup with a unique storage key: `{project_id}_{rag_approach}_{iteration}`

## Storage Structure

```
projects/rag_anything/
├── projectA_raganything_1/    # One RAG database for projectA
│   ├── storage/              # Processed documents
│   └── temp/                 # Temporary files
├── projectA_raganything_2/    # Another RAG database for projectA (different config)
│   ├── storage/
│   └── temp/
└── projectB_raganything_1/    # One RAG database for projectB
    ├── storage/
    └── temp/
```

## Project Registry

### File: `rag_projects.json`

```json
{
  "rag_databases": {
    "NbbabGQy3gkCJIDzkSoE_raganything_1": {
      "name": "Thomaston Chiropractic Clinic Records - RAGAnything v1",
      "project_id": "NbbabGQy3gkCJIDzkSoE",
      "rag_approach": "raganything",
      "iteration": 1,
      "user_id": "7CtdhckRcxOIjU3Dh7Ao3jvigg13",
      "workspace_id": "RahGZZ7sQXX6De61oAl4",
      "gcs_path": "lawunleashed-dev.firebasestorage.app/workspaces/RahGZZ7sQXX6De61oAl4/projects/NbbabGQy3gkCJIDzkSoE/case_files",
      "status": "loaded",
      "parser": "unstructured",
      "model": "gpt-4o-mini",
      "created_at": "2025-01-20T10:05:00Z",
      "document_count": 0
    },
    "NbbabGQy3gkCJIDzkSoE_raganything_2": {
      "name": "Thomaston Chiropractic Clinic Records - RAGAnything v2",
      "project_id": "NbbabGQy3gkCJIDzkSoE",
      "rag_approach": "raganything",
      "iteration": 2,
      "user_id": "7CtdhckRcxOIjU3Dh7Ao3jvigg13",
      "workspace_id": "RahGZZ7sQXX6De61oAl4",
      "gcs_path": "lawunleashed-dev.firebasestorage.app/workspaces/RahGZZ7sQXX6De61oAl4/projects/NbbabGQy3gkCJIDzkSoE/case_files",
      "status": "loaded",
      "parser": "unstructured",
      "model": "gpt-4o-mini",
      "created_at": "2025-01-20T10:10:00Z",
      "document_count": 0
    }
  }
}
```

## Simple Management Script

### RAG Manager Script
A simple Python script that:
1. Takes a project ID as input
2. Looks up the configuration from `rag_projects.json`
3. Calls the existing `/process-folder` endpoint

### Usage
```bash
python rag_manager.py NbbabGQy3gkCJIDzkSoE_raganything_1
```

### Script Flow
1. **Load Configuration**: Read `rag_projects.json` and find the project config
2. **Call Existing API**: Use the existing `/process-folder` endpoint with the config
3. **Track Progress**: Monitor the job status using existing `/processing-status/{job_id}` endpoint
4. **Query Results**: Use existing `/query` endpoint to test the RAG database

### No New Endpoints Needed
- Uses existing `/process-folder` endpoint
- Uses existing `/processing-status/{job_id}` endpoint  
- Uses existing `/query` endpoint
- Just adds a simple management layer on top

## Implementation

### RAG Manager Script
```python
#!/usr/bin/env python3
"""
Simple RAG Manager - manages RAG databases using existing API endpoints
"""

import json
import asyncio
import aiohttp
import sys
from typing import Dict, Any

class RAGManager:
    def __init__(self, registry_file="rag_projects.json", api_base="http://localhost:8000"):
        self.registry_file = registry_file
        self.api_base = api_base
        self.rag_databases = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load RAG database registry from JSON file"""
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"rag_databases": {}}
    
    def _save_registry(self):
        """Save RAG database registry to JSON file"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.rag_databases, f, indent=2)
    
    def add_rag_database(self, storage_key: str, config: Dict[str, Any]):
        """Add a new RAG database to the registry"""
        if "rag_databases" not in self.rag_databases:
            self.rag_databases["rag_databases"] = {}
        
        self.rag_databases["rag_databases"][storage_key] = config
        self._save_registry()
    
    def get_rag_database(self, storage_key: str) -> Dict[str, Any]:
        """Get RAG database configuration"""
        return self.rag_databases["rag_databases"].get(storage_key)
    
    def list_rag_databases(self) -> Dict[str, Any]:
        """List all RAG databases"""
        return self.rag_databases["rag_databases"]
    
    async def process_documents(self, storage_key: str) -> str:
        """Process documents for a RAG database using existing API"""
        config = self.get_rag_database(storage_key)
        if not config:
            raise ValueError(f"RAG database {storage_key} not found")
        
        # Call existing /process-folder endpoint
        async with aiohttp.ClientSession() as session:
            payload = {
                "user_id": config["user_id"],
                "project_id": storage_key,  # Use storage_key as project_id for isolation
                "workspace_id": config["workspace_id"],
                "gcs_folder_path": config["gcs_path"],
                "rag_approach": config["rag_approach"],
                "parser": config["parser"],
                "parse_method": "auto",
                "model": config["model"],
                "file_extensions": [".pdf", ".docx", ".txt", ".md"]
            }
            
            async with session.post(f"{self.api_base}/process-folder", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["job_id"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to start processing: {error_text}")
    
    async def query_database(self, storage_key: str, query: str) -> Dict[str, Any]:
        """Query a RAG database using existing API"""
        config = self.get_rag_database(storage_key)
        if not config:
            raise ValueError(f"RAG database {storage_key} not found")
        
        # Call existing /query endpoint
        async with aiohttp.ClientSession() as session:
            payload = {
                "user_id": config["user_id"],
                "project_id": storage_key,  # Use storage_key as project_id for isolation
                "rag_approach": config["rag_approach"],
                "query": query,
                "model": config["model"]
            }
            
            async with session.post(f"{self.api_base}/query", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Query failed: {error_text}")

async def main():
    if len(sys.argv) != 2:
        print("Usage: python rag_manager.py <storage_key>")
        print("Example: python rag_manager.py NbbabGQy3gkCJIDzkSoE_raganything_1")
        sys.exit(1)
    
    storage_key = sys.argv[1]
    manager = RAGManager()
    
    try:
        # Process documents
        print(f"🚀 Processing documents for {storage_key}...")
        job_id = await manager.process_documents(storage_key)
        print(f"✅ Processing started with job ID: {job_id}")
        
        # TODO: Add job status monitoring and query testing
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

## Benefits

1. **Simple**: No complex Firebase setup
2. **Local**: Everything stored locally in JSON
3. **Flexible**: Easy to add new projects and configurations
4. **Isolated**: Each configuration has its own storage
5. **Trackable**: Clear history of what's been loaded
6. **Portable**: Easy to backup and move between environments

## Usage Examples

### 1. Add a RAG database to the registry
```python
from rag_manager import RAGManager

manager = RAGManager()

# Add a new RAG database configuration
config = {
    "name": "Thomaston Chiropractic Clinic Records - RAGAnything v1",
    "project_id": "NbbabGQy3gkCJIDzkSoE",
    "rag_approach": "raganything",
    "iteration": 1,
    "user_id": "7CtdhckRcxOIjU3Dh7Ao3jvigg13",
    "workspace_id": "RahGZZ7sQXX6De61oAl4",
    "gcs_path": "lawunleashed-dev.firebasestorage.app/workspaces/RahGZZ7sQXX6De61oAl4/projects/NbbabGQy3gkCJIDzkSoE/case_files",
    "parser": "unstructured",
    "model": "gpt-4o-mini"
}

manager.add_rag_database("NbbabGQy3gkCJIDzkSoE_raganything_1", config)
```

### 2. Process documents using the script
```bash
python rag_manager.py NbbabGQy3gkCJIDzkSoE_raganything_1
```

### 3. Query the RAG database
```python
import asyncio
from rag_manager import RAGManager

async def query_example():
    manager = RAGManager()
    result = await manager.query_database(
        "NbbabGQy3gkCJIDzkSoE_raganything_1", 
        "What are the main topics in this case?"
    )
    print(result["answer"])

asyncio.run(query_example())
```

## File Structure

```
rag_projects.json          # RAG database registry
projects/rag_anything/         # All RAG storage
├── NbbabGQy3gkCJIDzkSoE_raganything_1/    # One RAG database
├── NbbabGQy3gkCJIDzkSoE_raganything_2/    # Another RAG database (different config)
└── OtherProject_raganything_1/            # Another RAG database
```

## Key Concepts

- **One RAG Database = One Entry**: Each `{project_id}_{rag_approach}_{iteration}` is a complete RAG database
- **Storage Key = Database ID**: The storage key uniquely identifies each RAG database
- **Simple Registry**: Just a JSON file tracking all your RAG databases
- **Isolated Storage**: Each RAG database has its own storage directory
- **Easy Management**: Create, list, query, and manage RAG databases independently

This system is much simpler, easier to understand, and doesn't require any external dependencies beyond what you already have!
