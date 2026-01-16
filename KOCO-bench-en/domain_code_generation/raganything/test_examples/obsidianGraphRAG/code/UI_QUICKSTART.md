# 🚀 Obsidian RAG UI - Quick Start Guide

## What Was Fixed

### 🐛 Critical Bug #1: Server Hanging on Startup
**Problem**: The UI server would hang indefinitely when starting because the RAG system was waiting for user input (interactive prompt asking whether to use existing database or create new one).

**Solution**: 
- Added `non_interactive` parameter to `SimpleRAGAnything` class
- When set to `True`, automatically uses existing database without prompting
- Updated UI backend to use `non_interactive=True` mode

**Files Modified**:
- `src/simple_raganything.py`: Added non-interactive mode support
- `ui/app.py`: Pass `non_interactive=True` flag during initialization

### 🐛 Critical Bug #2: Infinite Reload Loop
**Problem**: The server would continuously restart in an infinite loop. During RAG initialization, file changes (cache, model loading) triggered uvicorn's auto-reload feature, causing the server to restart, which triggered initialization again.

**Solution**:
- Disabled auto-reload by default in production mode
- Added `DEV_MODE` environment variable to enable reload for development
- Server now runs in stable production mode unless explicitly enabled

**Files Modified**:
- `run_ui.py`: Disabled auto-reload by default, added dev mode flag

### ✨ SOTA Features Added

1. **Chat History Persistence**
   - Conversations automatically saved to browser localStorage
   - Survives browser refresh
   - Clear chat with confirmation dialog

2. **Settings Panel**
   - Query mode selection (Hybrid, Naive, Local, Global, Mix)
   - Toggle BGE Reranker on/off
   - View system info (GPU status, memory usage)
   - Export chat functionality

3. **Enhanced Markdown Rendering**
   - Syntax highlighting for code blocks (Highlight.js)
   - Better styling for tables, lists, blockquotes
   - Improved contrast and readability

4. **Copy to Clipboard**
   - Hover over AI responses to reveal copy button
   - Toast notification on successful copy

5. **Suggested Prompts**
   - Quick-start buttons for common queries
   - Click to populate input field

6. **System Monitoring**
   - Real-time connection status with pulse animation
   - GPU and memory usage tracking
   - RAG initialization status

7. **Modern UI Design**
   - Gradient effects and smooth animations
   - Better color scheme with improved contrast
   - Responsive design for mobile

8. **Enhanced API Endpoints**
   - `/health` - System health with GPU metrics
   - `/api/config` - Current RAG configuration

## 🏃 Getting Started

### Step 1: Ensure You Have a RAG Database

Before starting the UI, you should have already processed your Obsidian vault. If not:

```bash
conda activate turing0.1

# Set your API key
$env:VERTEX_AI_API_KEY="your-api-key-here"

# Process your vault (this creates the database)
python run_obsidian_raganything.py
```

This will create the RAG database in `./rag_storage/`. The UI will use this existing database.

### Step 2: Start the UI Server

```bash
conda activate turing0.1
python run_ui.py
```

You should see:
```
🚀 Starting Obsidian RAG Chat UI Server
📍 URL: http://localhost:8000
⏹️  Press CTRL+C to stop the server
```

The server will:
1. Initialize RAG system in non-interactive mode (10-30 seconds)
2. Load embedding model (EmbeddingGemma 308M)
3. Start FastAPI server on port 8000

### Step 3: Open Your Browser

Navigate to: **http://localhost:8000**

You should see the modern chat interface with:
- Welcome message with AI branding
- Suggested prompt buttons
- Connection status indicator (green = connected)
- Settings button in header

### Step 4: Start Chatting!

Try these example queries:

**Basic Questions:**
- "What are the main topics in my notes?"
- "Summarize my notes about [topic]"

**Connections:**
- "Show me related concepts to [topic]"
- "Find connections between [topic A] and [topic B]"

**Search:**
- "Find notes about [keyword]"
- "What did I write about [topic] last month?"

**Analysis:**
- "What are the most referenced concepts?"
- "Generate a reading list for [topic]"

## 🎯 Using the Settings Panel

### Opening Settings
Click the ⚙️ icon in the header (top-right)

### Query Mode Selection
- **Hybrid** (Default): Best for most queries - combines local and global knowledge
- **Naive**: Simple vector similarity search
- **Local**: Context-dependent search within related documents
- **Global**: Searches across all documents using global knowledge graph
- **Mix**: Combines knowledge graph and vector search

### Reranker Toggle
- **Enabled** (Default): Uses BGE Reranker v2-m3 for +15-30% better relevance
- **Disabled**: Skips reranking (faster but less accurate)

### Export Chat
Click "Export Chat" to download your conversation as a text file:
- Format: `obsidian-rag-chat-YYYY-MM-DD.txt`
- Contains all messages with timestamps
- Useful for documentation or review

### System Info
View real-time system metrics:
- Status: healthy / error
- RAG: Ready / Initializing
- GPU status (if available)
- Memory usage

## 🔧 Configuration

### Changing Vault Path

Edit `run_ui.py` (lines 11-12):

```python
os.environ["OBSIDIAN_VAULT_PATH"] = r"C:\path\to\your\vault"
os.environ["RAG_WORKING_DIR"] = "./rag_storage"
```

Restart the server for changes to take effect.

### Changing Port

Edit `run_ui.py` (line 24):

```python
uvicorn.run(
    "ui.app:app",
    host="0.0.0.0",
    port=8001,  # Change this
    reload=True,
    log_level="info"
)
```

## 🎨 UI Features Guide

### Message Actions
Hover over AI responses to reveal action buttons:
- 📋 Copy to clipboard

### Keyboard Shortcuts
- **Enter**: Send message
- **Shift+Enter**: New line in message
- **ESC**: Close settings panel (when open)

### Clear Chat
Click the 🗑️ icon in header to clear all messages:
- Confirmation dialog appears
- Clears localStorage
- Shows welcome message again

### Connection Status
Watch the status dot in header:
- 🟢 Green (pulsing): Connected
- 🔴 Red: Disconnected
- Automatically reconnects every 3 seconds

## 🐛 Troubleshooting

### Server Hangs on Startup

**Old Issue** (Fixed in this version):
- Previously: Server would wait for user input
- Now: Automatically uses existing database

If you still experience hanging:
1. Check terminal for error messages
2. Verify `./rag_storage/` directory exists
3. Ensure database files are present (*.json, *.graphml)

### Infinite Reload Loop

**Symptoms**: Server keeps restarting with messages like:
```
WARNING: StatReload detected changes in 'src\simple_raganything.py'. Reloading...
INFO: Shutting down
INFO: Started server process [xxxxx]
```

**Cause**: Auto-reload was enabled, triggering restarts during initialization.

**Solution** (Fixed in this version):
- Auto-reload is now disabled by default
- Server runs in stable production mode

**If you need auto-reload for development**:
```bash
# Windows PowerShell
$env:DEV_MODE="true"
python run_ui.py

# Linux/Mac
export DEV_MODE=true
python run_ui.py
```

**Quick fix if still occurring**:
1. Stop the server (Ctrl+C)
2. Kill any remaining processes:
   ```bash
   # Windows PowerShell
   Get-Process python | Stop-Process
   
   # Linux/Mac
   pkill -f "python.*run_ui"
   ```
3. Restart: `python run_ui.py`

### "Disconnected" Status

**Causes**:
1. Server is still initializing (wait 10-30 seconds)
2. RAG system failed to initialize (check terminal)
3. Network issue (check firewall)

**Solutions**:
```bash
# Check if server is running
curl http://localhost:8000/health

# Check logs in terminal
# Look for "✅ RAG system ready!" message
```

### First Query Takes Long Time

**Expected Behavior**:
- First query: 20-30 seconds (loading BGE Reranker)
- Subsequent queries: 1-3 seconds

This is normal! The reranker model (~1GB) is loaded on first use.

### Chat History Not Saving

**Causes**:
1. Browser localStorage disabled
2. Private/Incognito mode
3. Browser restrictions

**Solutions**:
- Enable localStorage in browser settings
- Use normal browsing mode (not incognito)
- Try a different browser (Chrome, Firefox, Edge)

### API Key Errors

**Error**: "VERTEX_AI_API_KEY not found"

**Solution**:
```bash
# Windows PowerShell
$env:VERTEX_AI_API_KEY="your-api-key-here"

# Linux/Mac
export VERTEX_AI_API_KEY="your-api-key-here"

# Restart the server
python run_ui.py
```

## 📊 Performance Tips

### For Best Performance
1. **Use GPU**: Ensure CUDA is available for faster embeddings and reranking
2. **Enable Reranker**: Better results, only slight speed decrease
3. **Use Hybrid Mode**: Best balance of speed and quality
4. **Clear Old Chats**: Export and clear long conversations periodically

### Monitor System Resources
Open Settings → System Info to check:
- GPU memory usage
- RAM usage
- RAG initialization status

### If Running Slow
1. Check GPU memory isn't full
2. Close other GPU-intensive applications
3. Restart the server to clear caches
4. Consider using CPU mode (slower but works without GPU)

## 🎓 Advanced Usage

### Custom Query Examples

**Complex Queries:**
```
"Compare and contrast the concepts of [A] and [B] based on my notes,
and suggest how they might be related to [C]"
```

**Multi-part Questions:**
```
"First, summarize my notes about machine learning.
Then, find any notes that mention both ML and ethics.
Finally, suggest connections I might have missed."
```

**Citation Requests:**
```
"What are my notes about quantum computing?
Include references to specific note titles where possible."
```

### Using Different Query Modes

**When to use each mode:**

1. **Hybrid** - General questions, best all-around
   - "What are the main themes in my vault?"
   - "Explain [concept] from my notes"

2. **Local** - Context-specific queries
   - "In the context of [topic], what did I note about [subtopic]?"
   - "Within my project notes, what are the key milestones?"

3. **Global** - Broad overviews
   - "Give me a bird's eye view of all my research"
   - "What are all the topics I've written about?"

4. **Mix** - Comprehensive results
   - "Find everything related to [topic]"
   - "Show me all connections to [concept]"

5. **Naive** - Simple keyword search
   - "Find notes containing [exact phrase]"
   - "Quick search for [keyword]"

### Export and Analysis

1. **Export Chat**: Get your conversation history
2. **Analyze Patterns**: Review what topics you query most
3. **Document Knowledge**: Save important Q&A sessions
4. **Share Insights**: Export for team collaboration

## 🔮 What's Next?

### Planned Features (Future)
- Dark/Light theme toggle
- Voice input support
- Image upload for multimodal queries
- Advanced filtering (by date, note type, tags)
- Collaborative features (share queries)
- Custom prompt templates
- Query history search
- Keyboard shortcuts customization

### Contributing
Found a bug or have a feature request? The codebase is well-documented:
- `ui/app.py` - Backend server
- `ui/static/index.html` - Frontend interface
- `src/simple_raganything.py` - RAG system

## 📚 Additional Resources

- **Main README**: Overview of the RAG system
- **UI README**: Detailed technical documentation
- **SOLUTION_SUMMARY.md**: Complete fix documentation
- **QUICK_FIX_GUIDE.md**: Common issues and solutions

## 💡 Tips for Best Results

1. **Be Specific**: "Explain the concept of [X] as it relates to [Y]" works better than "Tell me about X"

2. **Use Context**: "In my notes about [topic], what are the main arguments?" is more targeted

3. **Iterate**: Start broad ("What are my main research topics?") then drill down ("Tell me more about [specific topic]")

4. **Leverage History**: The UI saves your chat, so you can refer back to previous answers

5. **Export Important Sessions**: Don't lose valuable insights - export conversations you want to keep

6. **Experiment with Modes**: Try different query modes to see which gives best results for your use case

7. **Monitor Performance**: Keep an eye on system info to ensure optimal performance

---

## 🎉 You're All Set!

Your Obsidian RAG UI is now powered by SOTA techniques:
- ✅ Non-blocking initialization
- ✅ Chat history persistence
- ✅ Real-time settings adjustment
- ✅ Modern, responsive design
- ✅ Enhanced error handling
- ✅ System monitoring
- ✅ Export functionality

**Enjoy exploring your knowledge base with AI! 🚀**

For issues or questions, check the troubleshooting section above or the detailed README in the `ui/` directory.

