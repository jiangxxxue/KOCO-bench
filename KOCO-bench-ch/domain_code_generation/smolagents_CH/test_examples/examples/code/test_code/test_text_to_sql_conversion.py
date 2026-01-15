import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the sys.path to allow for correct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module and functions to be tested
import text_to_sql

class TestTextToSQLConversion(unittest.TestCase):
    """Tests the text-to-SQL conversion functionality."""

    def test_sql_engine_tool(self):
        """
        Tests the sql_engine tool function with a mocked database engine.
        """
        with patch('text_to_sql.engine') as mock_engine:
            # Mock the connection and execution
            mock_connection = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_connection
            
            # 1. Test successful query
            mock_connection.execute.return_value = [("Woodrow Wilson",), ("Margaret James",)]
            result = text_to_sql.sql_engine("SELECT customer_name FROM receipts")
            self.assertIn("Woodrow Wilson", result)
            self.assertIn("Margaret James", result)

            # 2. Test query with no results
            mock_connection.execute.return_value = []
            result = text_to_sql.sql_engine("SELECT customer_name FROM receipts WHERE price > 100")
            self.assertEqual(result, "")

            # 3. Test for SQL error
            mock_connection.execute.side_effect = Exception("Invalid SQL")
            with self.assertRaises(Exception) as context:
                text_to_sql.sql_engine("SELECT invalid_column FROM receipts")
            self.assertIn("Invalid SQL", str(context.exception))

    @patch('text_to_sql.CodeAgent')
    @patch('text_to_sql.InferenceClientModel')
    def test_text_to_sql_conversion_flow(self, MockInferenceClientModel, MockCodeAgent):
        """
        Tests the full text_to_sql_conversion function flow.
        """
        # --- Mock Initialization ---
        mock_model_instance = MagicMock()
        MockInferenceClientModel.return_value = mock_model_instance

        mock_agent_instance = MagicMock()
        expected_result = "Woodrow Wilson"
        mock_agent_instance.run.return_value = expected_result
        MockCodeAgent.return_value = mock_agent_instance

        # --- Execute the function under test ---
        query = "Can you give me the name of the client who got the most expensive receipt?"
        result = text_to_sql.text_to_sql_conversion(query)

        # --- Assertions ---
        # 1. Assert model was instantiated
        MockInferenceClientModel.assert_called_once_with(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")

        # 2. Assert agent was created with the sql_engine tool
        MockCodeAgent.assert_called_once_with(
            tools=[text_to_sql.sql_engine],
            model=mock_model_instance
        )

        # 3. Assert agent's run method was called
        mock_agent_instance.run.assert_called_once_with(query)

        # 4. Assert the final result
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()