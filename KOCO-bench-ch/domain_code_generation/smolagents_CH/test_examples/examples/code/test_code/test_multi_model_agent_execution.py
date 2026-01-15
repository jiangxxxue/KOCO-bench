import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the sys.path to allow for correct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module itself to be able to patch its contents
import agent_from_any_llm

# The patch target must be where the object is looked up.
# In this case, it's in the 'agent_from_any_llm' module's namespace.
MODEL_CLASSES = {
    "inference_client": "agent_from_any_llm.InferenceClientModel",
    "transformers": "agent_from_any_llm.TransformersModel",
    "ollama": "agent_from_any_llm.LiteLLMModel",
    "litellm": "agent_from_any_llm.LiteLLMModel",
    "openai": "agent_from_any_llm.OpenAIModel",
}

class TestMultiModelAgentExecution(unittest.TestCase):
    
    def test_all_inference_types(self):
        """
        Test that multi_model_agent_execution function can correctly initialize
        different model types and that the created agents can execute a run.
        This test iterates through all available inference types.
        """
        for chosen_inference in agent_from_any_llm.available_inferences:
            with self.subTest(chosen_inference=chosen_inference):
                expected_tool_response = "The weather is UNGODLY with torrential rains and temperatures below -10°C"
                model_class_to_mock = MODEL_CLASSES[chosen_inference]

                with patch(model_class_to_mock) as MockModel, \
                     patch('agent_from_any_llm.ToolCallingAgent') as MockToolCallingAgent, \
                     patch('agent_from_any_llm.CodeAgent') as MockCodeAgent:

                    # Mock Model Initialization
                    mock_model_instance = MagicMock()
                    MockModel.return_value = mock_model_instance

                    # Mock Agent Initialization and Run
                    mock_tool_calling_agent_instance = MagicMock()
                    mock_tool_calling_agent_instance.run.return_value = expected_tool_response
                    MockToolCallingAgent.return_value = mock_tool_calling_agent_instance

                    mock_code_agent_instance = MagicMock()
                    mock_code_agent_instance.run.return_value = expected_tool_response
                    MockCodeAgent.return_value = mock_code_agent_instance

                    # Execute the function under test
                    tool_calling_result, code_agent_result = agent_from_any_llm.multi_model_agent_execution(chosen_inference)

                    # Assertions
                    MockModel.assert_called_once()
                    MockToolCallingAgent.assert_called_once_with(
                        tools=[agent_from_any_llm.get_weather],
                        model=mock_model_instance,
                        verbosity_level=2
                    )
                    MockCodeAgent.assert_called_once_with(
                        tools=[agent_from_any_llm.get_weather],
                        model=mock_model_instance,
                        verbosity_level=2,
                        stream_outputs=True
                    )
                    mock_tool_calling_agent_instance.run.assert_called_once_with("What's the weather like in Paris?")
                    mock_code_agent_instance.run.assert_called_once_with("What's the weather like in Paris?")
                    self.assertEqual(tool_calling_result, expected_tool_response)
                    self.assertEqual(code_agent_result, expected_tool_response)

if __name__ == '__main__':
    unittest.main()
