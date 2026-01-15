import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from starlette.testclient import TestClient

# Add the parent directory to the sys.path to allow for correct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'async_agent')))

# Import the app from the main module
from main import app

class TestAsyncAgentExecution(unittest.TestCase):
    """Tests the asynchronous agent execution via Starlette app."""

    def setUp(self):
        """Set up the test client."""
        self.client = TestClient(app)

    @patch('main.anyio.to_thread.run_sync')
    @patch('main.get_agent')
    def test_run_agent_endpoint_success(self, mock_get_agent, mock_run_sync):
        """Test the /run-agent endpoint for a successful execution."""
        # --- Mock Initialization ---
        mock_agent_instance = MagicMock()
        mock_get_agent.return_value = mock_agent_instance
        
        expected_result = "The agent ran successfully."
        mock_run_sync.return_value = expected_result

        # --- Make the request ---
        task = "This is a test task."
        response = self.client.post("/run-agent", json={"task": task})

        # --- Assertions ---
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"result": expected_result})
        mock_get_agent.assert_called_once()
        mock_run_sync.assert_called_once_with(mock_agent_instance.run, task)

    def test_run_agent_endpoint_missing_task(self):
        """Test the /run-agent endpoint when the 'task' parameter is missing."""
        response = self.client.post("/run-agent", json={"not_a_task": "some_value"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": 'Missing "task" in request body.'})

    @patch('main.anyio.to_thread.run_sync')
    @patch('main.get_agent')
    def test_run_agent_endpoint_agent_error(self, mock_get_agent, mock_run_sync):
        """Test the /run-agent endpoint when the agent's run method raises an exception."""
        # --- Mock Initialization ---
        mock_get_agent.return_value = MagicMock()
        
        error_message = "Something went wrong in the agent."
        mock_run_sync.side_effect = Exception(error_message)

        # --- Make the request ---
        response = self.client.post("/run-agent", json={"task": "a task that will fail"})

        # --- Assertions ---
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": error_message})

if __name__ == '__main__':
    unittest.main()