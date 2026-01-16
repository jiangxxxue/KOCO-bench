# 🐛 BookWorm - Advanced Document/Knowledge Ingestion System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

An advanced document and knowledge ingestion system that combines **LightRAG** for knowledge graph construction with **intelligent mindmap generation** for comprehensive document analysis and visualization.

## 🌟 Features

### 📚 Document Processing
- **Multi-format Support**: PDF, DOCX, TXT, Markdown, and more
- **Intelligent Text Extraction**: Optimized processors for different file types
- **Batch Processing**: Handle entire directories with concurrent processing
- **Metadata Extraction**: Automatic extraction of document metadata and statistics

### 🧠 Knowledge Graph Integration
- **LightRAG Integration**: Build sophisticated knowledge graphs from document content
- **Multi-modal Queries**: Support for local, global, hybrid, and mixed query modes
- **Semantic Understanding**: Extract entities, relationships, and complex patterns
- **Persistent Storage**: Maintain knowledge across sessions

### 🗺️ Advanced Mindmap Generation
- **Hierarchical Analysis**: Extract topics, subtopics, and detailed insights
- **Document Type Detection**: Adaptive processing for technical, scientific, business, and general documents
- **Multi-provider LLM Support**: OpenAI, Anthropic Claude, DeepSeek, and Google Gemini
- **Multiple Output Formats**: Mermaid syntax, interactive HTML, and Markdown outlines
- **Cost Optimization**: Efficient token usage and provider switching

### 🔧 System Features
- **Flexible Configuration**: Environment-based configuration management
- **Comprehensive Logging**: Detailed logging with rotation and levels
- **CLI Interface**: Full command-line interface for all operations
- **Error Handling**: Robust error handling and recovery mechanisms
- **Extensible Architecture**: Modular design for easy extension

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- uv package manager (recommended) or pip
- API keys for your chosen LLM providers

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd BookWorm
```

2. **Install dependencies using uv:**
```bash
uv sync
```

Or with pip:
```bash
pip install -e .
```

3. **Set up configuration:**
```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

### Basic Usage

#### Process Documents
```bash
# Process a single document
uv run bookworm process document.pdf

# Process an entire directory
uv run bookworm process ./documents/

# Process with custom output directory
uv run bookworm process ./documents/ -o ./processed/
```

#### Build Knowledge Graph
```bash
# Ingest documents into knowledge graph
uv run bookworm ingest ./documents/

# Ingest with mindmap generation
uv run bookworm ingest ./documents/ --mindmap
```

#### Query Knowledge
```bash
# Query the knowledge graph
uv run bookworm query "What are the main concepts in the documents?"

# Use different query modes
uv run bookworm query "Explain the technical architecture" --mode hybrid
```

#### Generate Mindmaps
```bash
# Generate mindmaps for documents
uv run bookworm mindmap ./documents/

# Use specific LLM provider
uv run bookworm mindmap ./documents/ --provider CLAUDE
```

#### Check System Status
```bash
uv run bookworm status
```

## 📖 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# LLM API Keys
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
DEEPSEEK_API_KEY="your-deepseek-api-key"
GEMINI_API_KEY="your-gemini-api-key"

# Primary LLM Provider
API_PROVIDER="OPENAI"  # Options: OPENAI, CLAUDE, DEEPSEEK, GEMINI

# LightRAG Configuration
LLM_MODEL="gpt-4o-mini"
EMBEDDING_MODEL="text-embedding-3-small"
WORKING_DIR="./workspace"

# Processing Settings
MAX_CONCURRENT_PROCESSES="4"
MAX_FILE_SIZE_MB="100"
PDF_PROCESSOR="pymupdf"  # Options: pymupdf, pdfplumber
```

### Directory Structure

BookWorm creates the following directory structure:

```
workspace/
├── docs/           # Input documents
├── processed/      # Processed text files
├── output/         # Generated mindmaps
└── rag_storage/    # LightRAG knowledge graph data
```

## 🏗️ Architecture

### Core Components

1. **DocumentProcessor**: Handles file ingestion and text extraction
2. **KnowledgeGraph**: Manages LightRAG operations and queries
3. **MindmapGenerator**: Creates hierarchical mindmaps from documents
4. **BookWormMindmapGenerator**: Advanced mindmap generation with LLM integration

### Processing Pipeline

```
Documents → Text Extraction → Knowledge Graph → Queries/Analysis
    ↓
Mindmap Generation → Mermaid/HTML/Markdown Outputs
```

### Supported File Types

- **PDFs**: `.pdf`
- **Text Files**: `.txt`, `.md`, `.markdown`
- **Documents**: `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`
- **Code Files**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.h`, `.cs`
- **Data Files**: `.json`, `.yaml`, `.yml`, `.xml`, `.csv`

## 🎯 Use Cases

### Academic Research
- **Literature Review**: Extract key concepts from research papers
- **Thesis Organization**: Structure complex research findings
- **Knowledge Mapping**: Visualize relationships across multiple sources

### Business Intelligence
- **Document Analysis**: Extract insights from reports and presentations
- **Strategic Planning**: Organize strategic documents into actionable frameworks
- **Competitive Research**: Structure competitor analysis and market research

### Technical Documentation
- **System Architecture**: Map complex technical systems
- **API Documentation**: Organize endpoint functionality
- **Code Analysis**: Understand large codebases through documentation

### Legal and Compliance
- **Document Review**: Extract key provisions from legal documents
- **Compliance Mapping**: Structure regulatory requirements
- **Case Analysis**: Organize legal precedents and arguments

## 🔧 Development

### Setting up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd BookWorm

# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Code formatting
uv run black .
uv run isort .

# Type checking
uv run mypy .
```

### Project Structure

```
BookWorm/
├── bookworm/                 # Main package
│   ├── __init__.py
│   ├── core.py              # Core components
│   ├── utils.py             # Utilities and configuration
│   ├── cli.py               # Command-line interface
│   └── mindmap_generator.py # Advanced mindmap generation
├── tests/                   # Test suite
├── examples/               # Example scripts and documents
├── docs/                   # Documentation
├── main.py                 # Demo script
├── pyproject.toml          # Project configuration
├── .env.example            # Environment template
└── README.md
```

## 📊 Performance and Costs

### Token Usage Optimization
- **Intelligent Chunking**: Efficient content segmentation
- **Provider Switching**: Cost-optimal provider selection
- **Caching**: Reduce redundant API calls
- **Early Stopping**: Halt processing when sufficient quality is achieved

### Supported LLM Providers
- **OpenAI**: GPT-4o-mini (cost-effective, high quality)
- **Anthropic**: Claude 3.5 Haiku (fast, efficient)
- **DeepSeek**: DeepSeek-Chat/Reasoner (competitive pricing)
- **Google**: Gemini 2.0 Flash (emerging option)

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:
- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LightRAG Team**: For the excellent knowledge graph framework
- **Dicklesworthstone**: For the original mindmap generator inspiration
- **Open Source Community**: For the amazing tools and libraries

## 🔗 Related Projects

- [LightRAG](https://github.com/HKUDS/LightRAG) - Simple and Fast Retrieval-Augmented Generation
- [Mindmap Generator](https://github.com/Dicklesworthstone/mindmap-generator) - Original mindmap generation system
- [RAG-Anything](https://github.com/HKUDS/RAG-Anything) - Multimodal RAG system

## 📞 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join discussions in GitHub Discussions
- **Documentation**: Check our [documentation](./docs/) for detailed guides

---

**BookWorm** - Transform your documents into structured knowledge and beautiful visualizations! 🐛📚🧠
