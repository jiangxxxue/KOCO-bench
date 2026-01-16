# from huggingface_hub import login

# login()
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever


from smolagents import Tool, CodeAgent, InferenceClientModel


if __name__ == "__main__":
    # Example query
    query = "For a transformers model training, which is slower, the forward or the backward pass?"
    rag_with_lexical_search(query)
