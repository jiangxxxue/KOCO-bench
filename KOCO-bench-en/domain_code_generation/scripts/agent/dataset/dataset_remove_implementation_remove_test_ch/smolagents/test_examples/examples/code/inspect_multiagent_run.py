from smolagents import (
    CodeAgent,
    InferenceClientModel,
    ToolCallingAgent,
    VisitWebpageTool,
    WebSearchTool,
)


if __name__ == "__main__":
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from phoenix.otel import register

    register()
    SmolagentsInstrumentor().instrument(skip_dep_check=True)
    
    task = "If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?"
    multi_agent_coordination(task)
