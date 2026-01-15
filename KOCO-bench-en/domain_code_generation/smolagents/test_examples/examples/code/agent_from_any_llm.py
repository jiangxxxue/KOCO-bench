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

def multi_model_agent_execution(chosen_inference: str):
    """
    A function that demonstrates multi-model agent execution.
    """
    print(f"Chose model: '{chosen_inference}'")

    if chosen_inference == "inference_client":
        model = InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct", provider="nebius")

    elif chosen_inference == "transformers":
        model = TransformersModel(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", device_map="auto", max_new_tokens=1000)

    elif chosen_inference == "ollama":
        model = LiteLLMModel(
            model_id="ollama_chat/llama3.2",
            api_base="http://localhost:11434",
            api_key="your-api-key",
            num_ctx=8192,
        )

    elif chosen_inference == "litellm":
        model = LiteLLMModel(model_id="gpt-4o")

    elif chosen_inference == "openai":
        model = OpenAIModel(model_id="gpt-4o")
    
    else:
        raise ValueError(f"Unknown inference type: {chosen_inference}")

    tool_calling_agent = ToolCallingAgent(tools=[get_weather], model=model, verbosity_level=2)
    tool_calling_result = tool_calling_agent.run("What's the weather like in Paris?")
    print("ToolCallingAgent:", tool_calling_result)

    code_agent = CodeAgent(tools=[get_weather], model=model, verbosity_level=2, stream_outputs=True)
    code_agent_result = code_agent.run("What's the weather like in Paris?")
    print("CodeAgent:", code_agent_result)
    
    return tool_calling_result, code_agent_result

if __name__ == "__main__":
    # Choose which inference type to use!
    chosen_inference = "inference_client"
    multi_model_agent_execution(chosen_inference)
