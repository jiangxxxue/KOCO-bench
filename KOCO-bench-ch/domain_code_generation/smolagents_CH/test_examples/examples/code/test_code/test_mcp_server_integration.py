import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from starlette.testclient import TestClient


class TestMCPServerIntegration(unittest.TestCase):
    """Tests the MCP server integration functionality."""

    @patch('server.main.MCPClient')
    @patch('server.main.InferenceClientModel')
    def setUp(self, MockInferenceClientModel, MockMCPClient):
        """Set up the test client for the Starlette app."""
        # Mock the MCPClient to return a mock with a get_tools method
        self.mock_mcp_client_instance = MagicMock()
        self.mock_mcp_client_instance.get_tools.return_value = [MagicMock()]
        MockMCPClient.return_value = self.mock_mcp_client_instance

        # Mock the InferenceClientModel
        self.mock_model_instance = MagicMock()
        MockInferenceClientModel.return_value = self.mock_model_instance
        
        # Now that mocks are in place, we can import the app
        from server.main import app
        self.client = TestClient(app)

    def test_homepage_endpoint(self):
        """Test that the homepage endpoint returns a 200 OK response."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<h1>🤖 Smolagents Demo</h1>", response.text)

    @patch('server.main.to_thread.run_sync')
    def test_chat_endpoint_success(self, mock_run_sync):
        """Test the /chat endpoint for a successful agent run."""
        # --- Mock Initialization ---
        expected_result = "The agent ran successfully."
        mock_run_sync.return_value = expected_result

        # --- Make the request ---
        message = "This is a test message."
        response = self.client.post("/chat", json={"message": message})

        # --- Assertions ---
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"reply": expected_result})
        
        # Assert that the agent's run method was called via the thread
        mock_run_sync.assert_called_once()
        # The first argument to run_sync is the function to run (agent.run)
        # The second is the argument to that function (the message)
        self.assertEqual(mock_run_sync.call_args[0][1], message)

    def tearDown(self):
        """Disconnect the client after tests."""
        # The shutdown event handler in the app will call this.
        # We can also call it explicitly if needed, but TestClient handles shutdown.
        pass

# This allows the test to be run from the command line
if __name__ == '__main__':
    unittest.main()