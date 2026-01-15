from smolagents import (
    CodeAgent,
    InferenceClientModel,
    ToolCallingAgent,
    VisitWebpageTool,
    WebSearchTool,
)

def multi_agent_coordination(task: str):
    """
    Demonstrates multi-agent coordination.
    """
    model = InferenceClientModel(provider="nebius")

    search_agent = ToolCallingAgent(
        tools=[WebSearchTool(), VisitWebpageTool()],
        model=model,
        name="search_agent",
        description="This is an agent that can do web search.",
        return_full_result=True,
    )

    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[search_agent],
        return_full_result=True,
    )
    
    run_result = manager_agent.run(task)
    
    print("Here is the token usage for the manager agent", run_result.token_usage)
    print("Here are the timing informations for the manager agent:", run_result.timing)
    
    return run_result

if __name__ == "__main__":
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from phoenix.otel import register

    register()
    SmolagentsInstrumentor().instrument(skip_dep_check=True)
    
    task = "If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?"
    multi_agent_coordination(task)
