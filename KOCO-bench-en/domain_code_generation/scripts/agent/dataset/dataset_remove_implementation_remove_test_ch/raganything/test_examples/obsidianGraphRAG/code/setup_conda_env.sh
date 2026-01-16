#!/bin/bash

# Clean Setup Script for RAG-Anything with EmbeddingGemma 308M
# Always uses conda environment turing0.1

echo "🚀 Setting up RAG-Anything with EmbeddingGemma 308M"
echo "📦 Using conda environment: turing0.1"
echo "="*50

# Activate conda environment
echo "🔄 Activating conda environment turing0.1..."
conda activate turing0.1

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment turing0.1"
    echo "💡 Please create the environment first:"
    echo "   conda create -n turing0.1 python=3.9"
    echo "   conda activate turing0.1"
    exit 1
fi

echo "✅ Conda environment turing0.1 activated"

# Install from requirements.txt
echo "📦 Installing packages from requirements.txt..."
pip install -r requirements.txt

# Install additional dependencies for RAG-Anything
echo "🔄 Installing additional RAG-Anything dependencies..."
pip install frontmatter
pip install dataclasses

echo "✅ All packages installed successfully!"
echo ""
echo "🎯 Ready to run RAG-Anything with EmbeddingGemma 308M"
echo "📁 Default vault path: C:\\Users\\joaop\\Documents\\Obsidian Vault\\Segundo Cerebro"
echo ""
echo "🚀 Run with: python run_obsidian_raganything.py"
