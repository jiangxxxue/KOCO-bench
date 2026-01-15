import unittest
from unittest.mock import patch, MagicMock, ANY
import sys
import os

# Add the parent directory to the sys.path to allow for correct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to be tested
import rag_using_chromadb

class TestRAGWithVectorDatabase(unittest.TestCase):
    """Tests the RAG with vector database functionality."""

    @patch.dict(os.environ, {'GROQ_API_KEY': 'test_api_key'})
    @patch('rag_using_chromadb.datasets.load_dataset')
    @patch('rag_using_chromadb.RecursiveCharacterTextSplitter.from_huggingface_tokenizer')
    @patch('rag_using_chromadb.AutoTokenizer.from_pretrained')
    @patch('rag_using_chromadb.HuggingFaceEmbeddings')
    @patch('rag_using_chromadb.Chroma')
    @patch('rag_using_chromadb.CodeAgent')
    @patch('rag_using_chromadb.LiteLLMModel')
    def test_rag_with_vector_database_flow(
        self, MockLiteLLMModel, MockCodeAgent, MockChroma, MockHuggingFaceEmbeddings,
        mock_from_pretrained, mock_from_huggingface_tokenizer, mock_load_dataset
    ):
        """
        Test the entire rag_with_vector_database function flow.
        """
        # --- Mock External Dependencies ---

        # 1. Mock dataset loading
        mock_dataset = [
            {"text": "To push a model, use `model.push_to_hub()`.", "source": "huggingface/transformers/hub"}
        ]
        mock_load_dataset.return_value = mock_dataset

        # 2. Mock tokenizer and text splitter
        mock_splitter_instance = MagicMock()
        mock_processed_docs = [MagicMock(page_content="Processed doc")]
        mock_splitter_instance.split_documents.return_value = mock_processed_docs
        mock_from_huggingface_tokenizer.return_value = mock_splitter_instance

        # 3. Mock embeddings
        mock_embeddings_instance = MagicMock()
        MockHuggingFaceEmbeddings.return_value = mock_embeddings_instance

        # 4. Mock ChromaDB
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.similarity_search.return_value = [MagicMock(page_content="Retrieved doc")]
        MockChroma.from_documents.return_value = mock_vector_store_instance

        # 5. Mock model
        mock_model_instance = MagicMock()
        MockLiteLLMModel.return_value = mock_model_instance

        # 6. Mock agent
        mock_agent_instance = MagicMock()
        expected_result = "You can push a model to the Hub using `model.push_to_hub()`."
        mock_agent_instance.run.return_value = expected_result
        MockCodeAgent.return_value = mock_agent_instance

        # --- Execute the function under test ---
        query = "How can I push a model to the Hub?"
        result = rag_using_chromadb.rag_with_vector_database(query)

        # --- Assertions ---

        # Assert dataset loading
        mock_load_dataset.assert_called_once_with("m-ric/huggingface_doc", split="train")

        # Assert text splitter was used
        mock_from_huggingface_tokenizer.assert_called_once()
        mock_splitter_instance.split_documents.assert_called()

        # Assert embeddings and vector store were created
        MockHuggingFaceEmbeddings.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")
        MockChroma.from_documents.assert_called_once_with(
            ANY, mock_embeddings_instance, persist_directory="./chroma_db"
        )

        # Assert model was instantiated
        MockLiteLLMModel.assert_called_once_with(
            model_id="groq/openai/gpt-oss-120b",
            api_key="test_api_key",
        )

        # Assert agent was created with the retriever tool
        MockCodeAgent.assert_called_once()
        args, kwargs = MockCodeAgent.call_args
        self.assertEqual(len(kwargs['tools']), 1)
        self.assertEqual(kwargs['tools'][0].name, 'retriever')
        self.assertEqual(kwargs['model'], mock_model_instance)

        # Assert agent's run method was called
        mock_agent_instance.run.assert_called_once_with(query)

        # Assert the final result
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()