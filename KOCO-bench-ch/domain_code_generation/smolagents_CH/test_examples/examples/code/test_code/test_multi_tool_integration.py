import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import requests

# Add the parent directory to the sys.path to allow for correct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module and functions to be tested
import multiple_tools

class TestMultiToolIntegration(unittest.TestCase):
    """Tests the multi-tool integration functionality."""

    def test_get_weather_tool(self):
        """Test the get_weather tool function."""
        with patch('multiple_tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "current": {"weather_descriptions": ["Sunny"], "temperature": 72}
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = multiple_tools.get_weather("Test City")
            self.assertEqual(result, "The current weather in Test City is Sunny with a temperature of 72 °F.")

            # Test API error
            mock_response.json.return_value = {"error": {"info": "API error"}}
            result = multiple_tools.get_weather("Test City")
            self.assertIn("Error: API error", result)

            # Test request exception
            mock_get.side_effect = requests.exceptions.RequestException("Connection error")
            result = multiple_tools.get_weather("Test City")
            self.assertIn("Error fetching weather data: Connection error", result)

    def test_convert_currency_tool(self):
        """Test the convert_currency tool function."""
        with patch('multiple_tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"conversion_rates": {"EUR": 0.9}}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = multiple_tools.convert_currency(100, "USD", "EUR")
            self.assertEqual(result, "100 USD is equal to 90.0 EUR.")
            
            # Test invalid currency
            mock_response.json.return_value = {"conversion_rates": {}}
            result = multiple_tools.convert_currency(100, "USD", "XYZ")
            self.assertIn("Unable to find exchange rate", result)

    def test_get_news_headlines_tool(self):
        """Test the get_news_headlines tool function."""
        with patch('multiple_tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "articles": [{"title": "Test Headline", "source": {"name": "Test Source"}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = multiple_tools.get_news_headlines()
            self.assertEqual(result, "Test Headline - Test Source")

    def test_get_joke_tool(self):
        """Test the get_joke tool function."""
        with patch('multiple_tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"joke": "This is a test joke."}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = multiple_tools.get_joke()
            self.assertEqual(result, "This is a test joke.")

    def test_get_time_in_timezone_tool(self):
        """Test the get_time_in_timezone tool function."""
        with patch('multiple_tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"datetime": "2025-01-01T12:00:00"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = multiple_tools.get_time_in_timezone("Europe/London")
            self.assertEqual(result, "The current time in Europe/London is 2025-01-01T12:00:00.")

    def test_get_random_fact_tool(self):
        """Test the get_random_fact tool function."""
        with patch('multiple_tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"text": "This is a random fact."}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = multiple_tools.get_random_fact()
            self.assertEqual(result, "Random Fact: This is a random fact.")

    def test_search_wikipedia_tool(self):
        """Test the search_wikipedia tool function."""
        with patch('multiple_tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"title": "Test Page", "extract": "This is a test summary."}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = multiple_tools.search_wikipedia("Test Query")
            self.assertEqual(result, "Summary for Test Page: This is a test summary.")

    @patch('multiple_tools.CodeAgent')
    @patch('multiple_tools.InferenceClientModel')
    def test_agent_integration(self, MockInferenceClientModel, MockCodeAgent):
        """Test the integration of all tools with the CodeAgent."""
        mock_model_instance = MagicMock()
        MockInferenceClientModel.return_value = mock_model_instance

        mock_agent_instance = MagicMock()
        expected_response = "Agent executed successfully."
        mock_agent_instance.run.return_value = expected_response
        MockCodeAgent.return_value = mock_agent_instance

        query = "Test query"
        result = multiple_tools.multi_tool_integration(query)

        MockInferenceClientModel.assert_called_once()
        MockCodeAgent.assert_called_once_with(
            tools=multiple_tools.get_all_tools(),
            model=mock_model_instance,
            stream_outputs=True,
        )
        mock_agent_instance.run.assert_called_once_with(query)
        self.assertEqual(result, expected_response)

if __name__ == '__main__':
    unittest.main()