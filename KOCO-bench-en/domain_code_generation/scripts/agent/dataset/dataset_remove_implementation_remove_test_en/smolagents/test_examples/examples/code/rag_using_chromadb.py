import os

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from transformers import AutoTokenizer

# from langchain_openai import OpenAIEmbeddings
from smolagents import LiteLLMModel, Tool
from smolagents.agents import CodeAgent


# from smolagents.agents import ToolCallingAgent


