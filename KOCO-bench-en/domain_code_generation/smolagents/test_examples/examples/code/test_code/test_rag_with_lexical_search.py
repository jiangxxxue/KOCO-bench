import unittest
from unittest.mock import patch, MagicMock, ANY
import sys
import os

# Add the parent directory to the sys.path to allow for correct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to be tested
import rag

class TestRAGWithLexicalSearch(unittest.TestCase):
    """Tests the RAG with lexical search functionality."""

    @patch('rag.datasets.load_dataset')
    @patch('rag.RecursiveCharacterTextSplitter')
    @patch('rag.BM25Retriever')
    @patch('rag.InferenceClientModel')
    @patch('rag.CodeAgent')
    def test_rag_with_lexical_search_flow(
        self, MockCodeAgent, MockInferenceClientModel, MockBM25Retriever, 
        MockTextSplitter, mock_load_dataset
    ):
        """
        Test the entire rag_with_lexical_search function flow.
        """
        # --- Mock External Dependencies ---

        # 1. Mock dataset loading
        mock_dataset = MagicMock()
        mock_dataset.filter.return_value = [
            {"text": "The forward pass is faster.", "source": "huggingface/transformers/perf"},
            {"text": "The backward pass is slower.", "source": "huggingface/transformers/perf"}
        ]
        mock_load_dataset.return_value = mock_dataset

        # 2. Mock text splitter
        mock_splitter_instance = MagicMock()
        mock_processed_docs = [MagicMock(page_content="Processed doc")]
        mock_splitter_instance.split_documents.return_value = mock_processed_docs
        MockTextSplitter.return_value = mock_splitter_instance

        # 3. Mock retriever
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.invoke.return_value = [MagicMock(page_content="Retrieved doc")]
        MockBM25Retriever.from_documents.return_value = mock_retriever_instance

        # 4. Mock model
        mock_model_instance = MagicMock()
        MockInferenceClientModel.return_value = mock_model_instance

        # 5. Mock agent
        mock_agent_instance = MagicMock()
        expected_result = "The backward pass is generally slower."
        mock_agent_instance.run.return_value = expected_result
        MockCodeAgent.return_value = mock_agent_instance

        # --- Execute the function under test ---
        query = "Which is slower, forward or backward pass?"
        result = rag.rag_with_lexical_search(query)

        # --- Assertions ---

        # Assert dataset loading and filtering
        mock_load_dataset.assert_called_once_with("m-ric/huggingface_doc", split="train")
        mock_dataset.filter.assert_called_once()

        # Assert text splitter was used
        MockTextSplitter.assert_called_once_with(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True,
            strip_whitespace=True,
            separators=ANY
        )
        mock_splitter_instance.split_documents.assert_called_once()

        # Assert retriever was created and used
        MockBM25Retriever.from_documents.assert_called_once_with(mock_processed_docs, k=10)
        
        # Assert model was instantiated
        MockInferenceClientModel.assert_called_once_with(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking")

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