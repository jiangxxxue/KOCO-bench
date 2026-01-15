import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the sys.path to allow for correct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to be tested
import multi_llm_agent

class TestMultiLLMLoadBalancing(unittest.TestCase):
    """Tests the multi-LLM load balancing functionality."""

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key',
        'AWS_ACCESS_KEY_ID': 'test_access_key',
        'AWS_SECRET_ACCESS_KEY': 'test_secret_key',
        'AWS_REGION': 'test_region'
    })
    @patch('multi_llm_agent.CodeAgent')
    @patch('multi_llm_agent.LiteLLMRouterModel')
    @patch('multi_llm_agent.WebSearchTool')
    def test_load_balancing_agent_execution(self, MockWebSearchTool, MockLiteLLMRouterModel, MockCodeAgent):
        """
        Test that the multi_llm_load_balancing function correctly configures and runs the agent.
        """
        # --- Mock Initialization ---
        mock_router_instance = MagicMock()
        MockLiteLLMRouterModel.return_value = mock_router_instance

        mock_agent_instance = MagicMock()
        expected_result = "The agent ran successfully."
        mock_agent_instance.run.return_value = expected_result
        MockCodeAgent.return_value = mock_agent_instance

        mock_search_tool_instance = MagicMock()
        MockWebSearchTool.return_value = mock_search_tool_instance

        # --- Get model list and execute the function ---
        model_list = multi_llm_agent.get_model_list()
        routing_strategy = "simple-shuffle"
        
        result = multi_llm_agent.multi_llm_load_balancing(model_list, routing_strategy)

        # --- Assertions ---
        # 1. Assert LiteLLMRouterModel was called correctly
        MockLiteLLMRouterModel.assert_called_once_with(
            model_id="model-group-1",
            model_list=model_list,
            client_kwargs={"routing_strategy": routing_strategy},
        )

        # 2. Assert WebSearchTool was instantiated
        MockWebSearchTool.assert_called_once()

        # 3. Assert CodeAgent was called correctly
        MockCodeAgent.assert_called_once_with(
            tools=[mock_search_tool_instance],
            model=mock_router_instance,
            stream_outputs=True,
            return_full_result=True
        )

        # 4. Assert the agent's run method was called
        mock_agent_instance.run.assert_called_once_with(
            "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"
        )

        # 5. Assert the function returns the expected result
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()