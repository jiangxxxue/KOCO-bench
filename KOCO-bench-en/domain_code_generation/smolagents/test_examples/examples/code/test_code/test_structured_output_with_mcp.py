import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the sys.path to allow for correct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to be tested
import structured_output_tool

class TestStructuredOutputWithMCP(unittest.TestCase):
    """Tests the structured output with MCP functionality."""

    @patch('structured_output_tool.CodeAgent')
    @patch('structured_output_tool.LiteLLMModel')
    @patch('structured_output_tool.MCPClient')
    @patch('structured_output_tool.StdioServerParameters')
    @patch('structured_output_tool.weather_server_script', return_value="mock_script")
    def test_structured_output_flow(
        self, mock_weather_script, MockStdioServer, MockMCPClient,
        MockLiteLLMModel, MockCodeAgent
    ):
        """
        Test the entire structured_output_with_mcp function flow.
        """
        # --- Mock Initialization ---
        
        # Mock model
        mock_model_instance = MagicMock()
        MockLiteLLMModel.return_value = mock_model_instance

        # Mock server parameters
        mock_server_params_instance = MagicMock()
        MockStdioServer.return_value = mock_server_params_instance

        # Mock MCPClient context manager
        mock_mcp_tools = [MagicMock()]
        mock_mcp_client_context = MagicMock()
        mock_mcp_client_context.__enter__.return_value = mock_mcp_tools
        MockMCPClient.return_value = mock_mcp_client_context

        # Mock agent
        mock_agent_instance = MagicMock()
        expected_result = "The temperature is 72.5°F."
        mock_agent_instance.run.return_value = expected_result
        MockCodeAgent.return_value = mock_agent_instance

        # --- Execute the function under test ---
        query = "What is the temperature in Tokyo in Fahrenheit?"
        result = structured_output_tool.structured_output_with_mcp(query)

        # --- Assertions ---

        # 1. Assert that the weather server script function was called
        mock_weather_script.assert_called_once()

        # 2. Assert that the server parameters were configured correctly
        MockStdioServer.assert_called_once_with(command="python", args=["-c", "mock_script"])

        # 3. Assert that the MCPClient was instantiated correctly
        MockMCPClient.assert_called_once_with(mock_server_params_instance, structured_output=True)

        # 4. Assert that the model was instantiated
        MockLiteLLMModel.assert_called_once_with(model_id="mistral/mistral-small-latest")

        # 5. Assert that the agent was created with the tools from MCP
        MockCodeAgent.assert_called_once_with(tools=mock_mcp_tools, model=mock_model_instance)

        # 6. Assert that the agent's run method was called
        mock_agent_instance.run.assert_called_once_with(query)

        # 7. Assert the final result
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()