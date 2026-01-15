import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the sys.path to allow for correct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to be tested
import inspect_multiagent_run

class TestMultiAgentCoordination(unittest.TestCase):
    """Tests the multi-agent coordination functionality."""

    @patch('inspect_multiagent_run.CodeAgent')
    @patch('inspect_multiagent_run.ToolCallingAgent')
    @patch('inspect_multiagent_run.InferenceClientModel')
    @patch('inspect_multiagent_run.WebSearchTool')
    @patch('inspect_multiagent_run.VisitWebpageTool')
    def test_multi_agent_coordination_flow(
        self, MockVisitWebpageTool, MockWebSearchTool, MockInferenceClientModel,
        MockToolCallingAgent, MockCodeAgent
    ):
        """
        Test the entire multi_agent_coordination function flow.
        """
        # --- Mock Initialization ---

        # Mock model
        mock_model_instance = MagicMock()
        MockInferenceClientModel.return_value = mock_model_instance

        # Mock tools
        mock_search_tool_instance = MagicMock()
        MockWebSearchTool.return_value = mock_search_tool_instance
        mock_visit_tool_instance = MagicMock()
        MockVisitWebpageTool.return_value = mock_visit_tool_instance

        # Mock search agent
        mock_search_agent_instance = MagicMock()
        MockToolCallingAgent.return_value = mock_search_agent_instance

        # Mock manager agent
        mock_manager_agent_instance = MagicMock()
        mock_run_result = MagicMock()
        mock_run_result.token_usage = {"total_tokens": 100}
        mock_run_result.timing = {"total_time": 5.0}
        mock_manager_agent_instance.run.return_value = mock_run_result
        MockCodeAgent.return_value = mock_manager_agent_instance

        # --- Execute the function under test ---
        task = "If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?"
        result = inspect_multiagent_run.multi_agent_coordination(task)

        # --- Assertions ---

        # 1. Assert model was instantiated
        MockInferenceClientModel.assert_called_once_with(provider="nebius")

        # 2. Assert tools were instantiated
        MockWebSearchTool.assert_called_once()
        MockVisitWebpageTool.assert_called_once()

        # 3. Assert search agent was created correctly
        MockToolCallingAgent.assert_called_once_with(
            tools=[mock_search_tool_instance, mock_visit_tool_instance],
            model=mock_model_instance,
            name="search_agent",
            description="This is an agent that can do web search.",
            return_full_result=True,
        )

        # 4. Assert manager agent was created correctly with the search agent
        MockCodeAgent.assert_called_once_with(
            tools=[],
            model=mock_model_instance,
            managed_agents=[mock_search_agent_instance],
            return_full_result=True,
        )

        # 5. Assert the manager agent's run method was called
        mock_manager_agent_instance.run.assert_called_once_with(task)

        # 6. Assert the final result is the one returned by the manager agent
        self.assertEqual(result, mock_run_result)
        self.assertEqual(result.token_usage, {"total_tokens": 100})
        self.assertEqual(result.timing, {"total_time": 5.0})

if __name__ == '__main__':
    unittest.main()