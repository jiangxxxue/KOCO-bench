import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from starlette.testclient import TestClient


class TestMCPServerIntegration(unittest.TestCase):
    """Tests the MCP server integration functionality."""

    @classmethod
    def setUpClass(cls):
        """Patch smolagents classes before server.main is imported so
        module-level instantiation uses mocks instead of real clients."""
        cls.mock_mcp_client_instance = MagicMock()
        cls.mock_mcp_client_instance.get_tools.return_value = [MagicMock()]

        cls.mock_model_instance = MagicMock()

        cls._patcher_mcp = patch('smolagents.MCPClient', return_value=cls.mock_mcp_client_instance)
        cls._patcher_model = patch('smolagents.InferenceClientModel', return_value=cls.mock_model_instance)
        cls._patcher_agent = patch('smolagents.CodeAgent')

        cls._patcher_mcp.start()
        cls._patcher_model.start()
        cls._patcher_agent.start()

        # Force a fresh import so module-level code runs with mocked deps
        sys.modules.pop('server.main', None)
        sys.modules.pop('server', None)

        from server.main import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls._patcher_mcp.stop()
        cls._patcher_model.stop()
        cls._patcher_agent.stop()
        sys.modules.pop('server.main', None)
        sys.modules.pop('server', None)

    def setUp(self):
        self.client = TestClient(self.app)

    def test_homepage_endpoint(self):
        """Test that the homepage endpoint returns a 200 OK response."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<h1>🤖 Smolagents Demo</h1>", response.text)

    @patch('server.main.to_thread.run_sync')
    def test_chat_endpoint_success(self, mock_run_sync):
        """Test the /chat endpoint for a successful agent run."""
        expected_result = "The agent ran successfully."
        mock_run_sync.return_value = expected_result

        message = "This is a test message."
        response = self.client.post("/chat", json={"message": message})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"reply": expected_result})

        mock_run_sync.assert_called_once()
        self.assertEqual(mock_run_sync.call_args[0][1], message)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
