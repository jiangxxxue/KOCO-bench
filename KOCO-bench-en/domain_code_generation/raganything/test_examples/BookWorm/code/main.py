#!/usr/bin/env python3
"""
Simple BookWorm Main Runner

This script processes all documents in a specified folder, 
only processing those that haven't been processed yet.
"""
import asyncio
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from bookworm import BookWormPipeline
from bookworm.utils import load_config

async def main():
    """Main function to process documents in a folder"""
    
    # Load configuration from the default location
    config = load_config()
    
    # Create pipeline instance
    pipeline = BookWormPipeline(config)
    
    print("Starting BookWorm processing...")
    print(f"Processing documents in: {Path(config.working_dir) / 'docs'}")
    
    try:
        # Run the complete pipeline
        await pipeline.run()
        
        print("\n✅ Processing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Processing failed with error: {e}")
        raise

if __name__ == "__main__":
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main())