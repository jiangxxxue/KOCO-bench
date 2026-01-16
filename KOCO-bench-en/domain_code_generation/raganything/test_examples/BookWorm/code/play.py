#!/usr/bin/env python3
"""
Example usage of BookWorm library system
This demonstrates how to:
1. Load the library
2. List documents in the library
3. Access document metadata
4. Query knowledge graphs for specific documents
5. Retrieve and display mindmap content
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bookworm.library import LibraryManager
from bookworm.utils import load_config

def demonstrate_library_usage():
    """Demonstrate how to use the BookWorm library system"""
    
    print("📚 BookWorm Library Usage Example")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    print(f"🔧 Configuration loaded from: {config.config_file}")
    print(f"📁 Working directory: {config.working_dir}")
    print()
    
    # Create library manager instance
    print("📖 Loading library...")
    library = LibraryManager(config)
    
    # Get library statistics
    stats = library.get_library_stats()
    print(f"📊 Library Statistics:")
    print(f"   Total documents: {stats.total_documents}")
    print(f"   Processed: {stats.processed_documents}")
    print(f"   Pending: {stats.pending_documents}")
    print(f"   Failed: {stats.failed_documents}")
    print(f"   Mindmaps: {stats.total_mindmaps}")
    print()
    
    # List documents in the library
    if not library.documents:
        print("⚠️  No documents found in library")
        return
    
    print("📄 Documents in Library:")
    print("-" * 30)
    
    # Sort documents by status and then by filename
    sorted_docs = sorted(library.documents.values(), 
                        key=lambda x: (x.status.value, x.filename.lower()))
    
    for doc_id, document in list(library.documents.items())[:10]:  # Show first 10 documents
        print(f"ID: {doc_id[:8]}...")
        print(f"   File: {document.filename}")
        print(f"   Status: {document.status.value}")
        print(f"   Size: {document.file_size:,} bytes")
        if document.is_directory:
            print(f"   Type: Directory ({document.file_count or 0} files)")
        else:
            print(f"   Type: File")
        
        # Show metadata if available
        if document.title:
            print(f"   Title: {document.title}")
        if document.author:
            print(f"   Author: {document.author}")
        if document.tags:
            print(f"   Tags: {', '.join(list(document.tags)[:3])}{'...' if len(document.tags) > 3 else ''}")
        
        # Show processing info
        if document.processed_at:
            print(f"   Processed: {document.processed_at.strftime('%Y-%m-%d %H:%M')}")
        
        print()
    
    # Demonstrate accessing specific documents
    print("🔍 Accessing Document by ID:")
    print("-" * 30)
    
    # Get first processed document
    processed_docs = [doc for doc in library.documents.values() 
                     if doc.status.value == "processed"]
    
    if processed_docs:
        sample_doc = processed_docs[0]
        doc_id = sample_doc.id
        print(f"Sample document ID: {doc_id[:12]}...")
        print(f"Filename: {sample_doc.filename}")
        print(f"Filepath: {sample_doc.filepath}")
        print(f"Status: {sample_doc.status.value}")
        
        # Show additional information if available
        if sample_doc.knowledge_graph_path:
            print(f"Knowledge Graph: {Path(sample_doc.knowledge_graph_path).name}")
        if sample_doc.mindmap_id:
            print(f"Mindmap ID: {sample_doc.mindmap_id[:12]}...")
        
        # Show tags and description
        if sample_doc.tags:
            print(f"Tags: {', '.join(list(sample_doc.tags))}")
        if sample_doc.description:
            print(f"Description: {sample_doc.description[:100]}...")
        
        print()
    else:
        print("No processed documents found in library")
        return
    
    # Query the sample document (this would require importing KnowledgeGraph)
    print("🧠 Querying Knowledge Graph:")
    print("-" * 30)
    
    try:   
        # Access mindmap information if available
        mindmap = library.get_mindmap(sample_doc.mindmap_id)
        if mindmap:
            print(f"📝 Mindmap Details:")
            print(f"   Filename: {mindmap.filename}")
            print(f"   Mermaid file: {Path(mindmap.mermaid_file).name}")
            print(f"   Markdown file: {Path(mindmap.markdown_file).name}")
            print(f"   Topics: {mindmap.topic_count}")
            print(f"   Subtopics: {mindmap.subtopic_count}")
            print(f"   Details: {mindmap.detail_count}")
            
            # Show the content by reading files
            try:
                if Path(mindmap.markdown_file).exists():
                    with open(mindmap.markdown_file, 'r') as f:
                        content = f.read()[:300] + "..." if len(content) > 300 else content
                        print(f"   Markdown preview: {content}")
            except Exception as e:
                print(f"   Failed to read markdown file: {e}")
            
        else:
            print("⚠️ No mindmap found for this document")
            
        print()
        
        # Demonstrate filtering documents by tags
        print("🏷️  Filtering Documents by Tags:")
        print("-" * 30)
        
        all_tags = set()
        for doc in library.documents.values():
            all_tags.update(doc.tags)
        
        print(f"Available tags: {sorted(list(all_tags))}")
        
        if all_tags:
            # Test with first tag
            sample_tag = list(all_tags)[0]
            tagged_docs = library.find_documents(tags=[sample_tag])
            if tagged_docs:
                print(f"Documents tagged '{sample_tag}':")
                for doc in tagged_docs[:3]:  # Show first 3
                    print(f"   - {doc.filename}")
        
        print()
        
        # Show directory processing info
        print("📁 Directory Collections:")
        print("-" * 30)
        
        directories = [doc for doc in library.documents.values() 
                       if doc.is_directory and doc.status.value == "processed"]
        
        for dir_doc in directories[:3]:  # Show first 3 directories
            print(f"Collection: {dir_doc.filename}")
            print(f"   Files processed: {dir_doc.file_count}")
            print(f"   Size: {dir_doc.file_size:,} bytes")
            if dir_doc.description:
                print(f"   Description: {dir_doc.description[:100]}...")
        
        print()
        
        # Show search functionality
        print("🔍 Document Search:")
        print("-" * 30)
        
        filename_search = "test"  # Simple search term
        found_docs = library.find_documents(filename=filename_search)
        
        if found_docs:
            print(f"Documents containing '{filename_search}':")
            for doc in found_docs[:3]:  # Show first 3
                print(f"   - {doc.filename} ({doc.status.value})")
        else:
            print(f"No documents found containing '{filename_search}'")
        
    except Exception as e:
        print(f"Error during library demonstration: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_query_functionality():
    """Demonstrate how to query knowledge graphs"""
    
    print("\n" + "="*50)
    print("🧠 Knowledge Graph Query Demonstration")
    print("="*50)
    
    try:
        # Note: This would require actual KnowledgeGraph implementation
        # In a real environment, this is how you'd structure the queries
        
        config = load_config()
        library = LibraryManager(config)
        
        processed_docs = [doc for doc in library.documents.values() 
                         if doc.status.value == "processed" and doc.knowledge_graph_path]
        
        if processed_docs:
            print("Querying sample documents...")
            
            sample_doc = processed_docs[0] 
            print(f"Sample document: {sample_doc.filename}")
            print(f"Knowledge graph path: {sample_doc.knowledge_graph_path}")
            print("Note: Actual querying would require KnowledgeGraph implementation")
            
        else:
            print("No processed documents with knowledge graphs found")
        
    except Exception as e:
        print(f"Error in query demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 BookWorm Library Usage Examples")
    
    # Run the main demonstrations
    demonstrate_library_usage()
    demonstrate_query_functionality()
    
    print("\n" + "="*50)
    print("📖 Usage Guide Summary:")
    print("="*50)
    print("""
1. Loading the library:
   from bookworm.library import LibraryManager
   from bookworm.utils import load_config
   config = load_config()
   library = LibraryManager(config)

2. Listing documents:
   docs = list(library.documents.values())  # All documents
   processed = [doc for doc in docs if doc.status.value == "processed"]

3. Accessing document metadata:
   filename = document.filename
   status = document.status.value
   size = document.file_size

4. Getting knowledge graphs:
   # This requires KnowledgeGraph implementation
   kg = KnowledgeGraph(config, library)
   results = await kg.query("what is the main topic?") 

5. Accessing mindmaps:
   mindmap = library.get_mindmap(document.mindmap_id)
   if mindmap:
       with open(mindmap.markdown_file, 'r') as f:
           content = f.read()
""")