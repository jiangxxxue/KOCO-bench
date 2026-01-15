import os

from smolagents import CodeAgent, LiteLLMRouterModel, WebSearchTool


def get_model_list():
    """Returns a list of model configurations for the load balancer."""
    return [
        {
            "model_name": "model-group-1",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "model-group-1",
            "litellm_params": {
                "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "aws_region_name": os.getenv("AWS_REGION"),
            },
        },
    ]

def multi_llm_load_balancing(model_list, routing_strategy="simple-shuffle"):
    """
    Demonstrates multi-LLM load balancing and failover.
    """
    model = LiteLLMRouterModel(
        model_id="model-group-1",
        model_list=model_list,
        client_kwargs={"routing_strategy": routing_strategy},
    )
    agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True, return_full_result=True)
    
    full_result = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
    print(full_result)
    return full_result

if __name__ == "__main__":
    # Make sure to setup the necessary environment variables!
    llm_loadbalancer_model_list = get_model_list()
    multi_llm_load_balancing(llm_loadbalancer_model_list)
