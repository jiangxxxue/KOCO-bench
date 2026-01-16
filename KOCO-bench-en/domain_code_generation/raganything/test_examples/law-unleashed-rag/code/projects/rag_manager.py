#!/usr/bin/env python3
"""
Simple RAG Manager - manages RAG databases using existing API endpoints
"""

import json
import asyncio
import aiohttp
import sys
from typing import Dict, Any, Optional

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
    
    def _update_database_status(self, storage_key: str, status: str):
        """Update the status of a RAG database"""
        if storage_key in self.rag_databases["rag_databases"]:
            self.rag_databases["rag_databases"][storage_key]["status"] = status
            self._save_registry()
    
    def _update_database_job_id(self, storage_key: str, job_id: str):
        """Update the job_id of a RAG database"""
        if storage_key in self.rag_databases["rag_databases"]:
            self.rag_databases["rag_databases"][storage_key]["last_job_id"] = job_id
            self._save_registry()
    
    def _update_database_corpus_info(self, storage_key: str, corpus_info: Dict[str, Any]):
        """Update the corpus information of a RAG database"""
        if storage_key in self.rag_databases["rag_databases"]:
            self.rag_databases["rag_databases"][storage_key]["corpus_info"] = corpus_info
            self._save_registry()
    
    def _find_storage_key_by_job_id(self, job_id: str) -> Optional[str]:
        """Find storage key by job ID"""
        for storage_key, config in self.rag_databases["rag_databases"].items():
            if config.get("last_job_id") == job_id:
                return storage_key
        return None
    
    async def process_documents(self, storage_key: str) -> str:
        """Process documents for a RAG database using existing API"""
        config = self.get_rag_database(storage_key)
        if not config:
            raise ValueError(f"RAG database {storage_key} not found")
        
        # Update status to processing
        self._update_database_status(storage_key, "processing")
        
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
                    job_id = result["job_id"]
                    # Store job_id for tracking
                    self._update_database_job_id(storage_key, job_id)
                    return job_id
                else:
                    # Update status to failed if request fails
                    self._update_database_status(storage_key, "failed")
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
                "project_id": storage_key,  # Use storage_key as project_id for RAG service identification
                "rag_approach": config["rag_approach"],
                "query": query,
                "model": config["model"],
                "corpus_info": config.get("corpus_info")  # Pass corpus info if available
            }
            
            async with session.post(f"{self.api_base}/query", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Query failed: {error_text}")
    
    async def check_processing_status(self, job_id: str, user_id: str) -> Dict[str, Any]:
        """Check the status of a processing job"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base}/processing-status/{job_id}?user_id={user_id}") as response:
                if response.status == 200:
                    status_data = await response.json()
                    
                    # If job is completed and has corpus info, store it
                    if (status_data.get('status') == 'completed' and 
                        'corpus_info' in status_data and 
                        status_data['corpus_info']):
                        
                        storage_key = self._find_storage_key_by_job_id(job_id)
                        if storage_key:
                            self._update_database_corpus_info(storage_key, status_data['corpus_info'])
                            print(f"✅ Stored corpus info for {storage_key}: {status_data['corpus_info']['corpus_name']}")
                    
                    return status_data
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to check status: {error_text}")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_manager.py <command> [args]")
        print("Commands:")
        print("  process <storage_key>     - Process documents for a RAG database")
        print("  query <storage_key> <query> - Query a RAG database")
        print("  list                     - List all RAG databases")
        print("  status <job_id> <user_id> - Check processing status")
        print("  check-db <storage_key>   - Check database status and update if needed")
        print("Examples:")
        print("  python rag_manager.py process NbbabGQy3gkCJIDzkSoE_raganything_1")
        print("  python rag_manager.py process NbbabGQy3gkCJIDzkSoE_vertex_rag_1")
        print("  python rag_manager.py query NbbabGQy3gkCJIDzkSoE_raganything_1 'What are the main topics?'")
        print("  python rag_manager.py query NbbabGQy3gkCJIDzkSoE_vertex_rag_1 'What are the main topics?'")
        print("  python rag_manager.py list")
        print("  python rag_manager.py status job_123")
        sys.exit(1)
    
    command = sys.argv[1]
    manager = RAGManager()
    
    try:
        if command == "process":
            if len(sys.argv) != 3:
                print("Usage: python rag_manager.py process <storage_key>")
                sys.exit(1)
            
            storage_key = sys.argv[2]
            print(f"🚀 Processing documents for {storage_key}...")
            job_id = await manager.process_documents(storage_key)
            print(f"✅ Processing started with job ID: {job_id}")
            print(f"💡 Use 'python rag_manager.py status {job_id}' to check progress")
            
        elif command == "query":
            if len(sys.argv) < 4:
                print("Usage: python rag_manager.py query <storage_key> <query> [--debug]")
                sys.exit(1)
            
            # Check for debug flag
            debug_mode = "--debug" in sys.argv
            
            storage_key = sys.argv[2]
            query = sys.argv[3]
            print(f"🔍 Querying {storage_key}...")
            result = await manager.query_database(storage_key, query)
            
            print(f"📝 Answer: {result.get('answer', 'No answer found')}")
            
            # Show sources if available
            sources = result.get('sources', [])
            if sources:
                print(f"\n📚 Sources ({len(sources)} found):")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source.get('content', 'No content')[:200]}...")
                    if 'metadata' in source:
                        print(f"     Metadata: {source['metadata']}")
                    if 'score' in source:
                        print(f"     Score: {source['score']}")
            
            # Show metadata if available
            metadata = result.get('metadata', {})
            if metadata:
                print(f"\n📊 Metadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            
            # Show processing time if available
            processing_time = result.get('processing_time')
            if processing_time:
                print(f"\n⏱️  Processing time: {processing_time:.2f} seconds")
            
            # Show raw response in debug mode
            if debug_mode:
                print(f"\n🔍 Debug - Raw Response:")
                import json
                print(json.dumps(result, indent=2, default=str))
            
        elif command == "list":
            databases = manager.list_rag_databases()
            if not databases:
                print("📋 No RAG databases found in registry")
            else:
                print("📋 RAG Databases:")
                for storage_key, config in databases.items():
                    print(f"  • {storage_key}")
                    print(f"    Name: {config.get('name', 'N/A')}")
                    print(f"    Status: {config.get('status', 'N/A')}")
                    print(f"    Approach: {config.get('rag_approach', 'N/A')}")
                    print(f"    Model: {config.get('model', 'N/A')}")
                    if 'last_job_id' in config:
                        print(f"    Last Job: {config.get('last_job_id', 'N/A')}")
                    print(f"    Created: {config.get('created_at', 'N/A')}")
                    print()
                    
        elif command == "status":
            if len(sys.argv) != 4:
                print("Usage: python rag_manager.py status <job_id> <user_id>")
                sys.exit(1)
            
            job_id = sys.argv[2]
            user_id = sys.argv[3]
            print(f"📊 Checking status for job {job_id}...")
            status = await manager.check_processing_status(job_id, user_id)
            print(f"Status: {status.get('status', 'Unknown')}")
            if 'progress' in status:
                print(f"Progress: {status['progress']}")
            if 'message' in status:
                print(f"Message: {status['message']}")
                
        elif command == "check-db":
            if len(sys.argv) != 3:
                print("Usage: python rag_manager.py check-db <storage_key>")
                sys.exit(1)
            
            storage_key = sys.argv[2]
            config = manager.get_rag_database(storage_key)
            if not config:
                print(f"❌ RAG database {storage_key} not found")
                sys.exit(1)
            
            print(f"📊 Checking database status for {storage_key}...")
            print(f"Current status: {config.get('status', 'Unknown')}")
            
            # If there's a last job ID, check its status
            if 'last_job_id' in config:
                job_id = config['last_job_id']
                print(f"Last job ID: {job_id}")
                try:
                    job_status = await manager.check_processing_status(job_id, config['user_id'])
                    print(f"Job status: {job_status.get('status', 'Unknown')}")
                    
                    # Update database status based on job status
                    if job_status.get('status') == 'completed':
                        manager._update_database_status(storage_key, "completed")
                        print("✅ Database status updated to 'completed'")
                    elif job_status.get('status') == 'failed':
                        manager._update_database_status(storage_key, "failed")
                        print("❌ Database status updated to 'failed'")
                    elif job_status.get('status') == 'processing':
                        print("⏳ Job is still processing...")
                        
                except Exception as e:
                    print(f"⚠️  Could not check job status: {e}")
            else:
                print("ℹ️  No job ID found - database has not been processed yet")
                
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python rag_manager.py' without arguments to see usage")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
