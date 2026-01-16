# BookWorm: RAG-Based Book Q&A System

## 1. System Overview

BookWorm is an interactive book question-answering system based on Retrieval-Augmented Generation (RAG) technology. It utilizes Streamlit to build the user interface and employs the RAG-Anything library as its core backend, enabling deep parsing and intelligent Q&A of book content.

When a user selects a book and asks a question, the system first uses the RAG-Anything engine to parse the book file, perform chunking and vectorization, and build an index. Then, based on the user's question, the system retrieves the most relevant content segments from the index and passes them along with the question to a Large Language Model (LLM), ultimately generating an accurate answer that is faithful to the original text.

## 2. Workflow

```
Algorithm: BookWorm RAG Pipeline
Input: User-selected book, User query
Output: Answer to the query

1: Initialize Streamlit UI
2: book_path ← get_selected_book_from_user()
3: query ← get_query_from_user()
4: 
5: // Get or Create Assistant for the book
6: assistant ← get_or_create_assistant(book_path)
7: 
8: // Query the assistant
9: response ← assistant.chat(query)
10: 
11: // Display response
12: display_response_in_ui(response)
13: return response

Function get_or_create_assistant(book_path):
    // Check if an assistant for this book already exists
    if assistant_exists(book_path) then
        return load_assistant(book_path)
    else
        // Create a new assistant
        rag_anything ← initialize_RAGAnything()
        // Parse the book file and insert its content into the vector store
        rag_anything.parse_file_and_insert(book_path)
        save_assistant(rag_anything, book_path)
        return rag_anything
    end if
```

Key functions in the pipeline:
- `initialize_RAGAnything()`: Initializes the RAG-Anything engine for document parsing and retrieval.
- `parse_file_and_insert(book_path)`: Parses the specified book file and stores its processed content into the vector database.
- `get_or_create_assistant(book_path)`: Retrieves or creates a Q&A "assistant" bound to a specific book.
- `assistant.chat(query)`: Receives user questions, executes the RAG workflow (retrieval + generation), and returns answers.
- `Streamlit UI`: Provides user interaction interface, including book selection, question input, and answer display.


## 3. Application Scenarios

### Interactive Book Content Q&A
- **Input**: User-specified book files (such as PDF, TXT, etc.) and natural language questions about the book content.
- **Output**: Natural language answers based on the book content.
- **Validation Method**: Users can verify the accuracy of answers by comparing them with the original text in the book.
- **Data Source**: Local book files uploaded by users.

