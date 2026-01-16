from smolagents import (
    CodeAgent,
    InferenceClientModel,
    LiteLLMModel,
    OpenAIModel,
    ToolCallingAgent,
    TransformersModel,
    tool,
)


# Define available inference types
available_inferences = ["inference_client", "transformers", "ollama", "litellm", "openai"]

@tool
def get_weather(location: str, celsius: bool | None = False) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
        celsius: the temperature
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"


if __name__ == "__main__":
    # Choose which inference type to use!
    chosen_inference = "inference_client"
    multi_model_agent_execution(chosen_inference)
