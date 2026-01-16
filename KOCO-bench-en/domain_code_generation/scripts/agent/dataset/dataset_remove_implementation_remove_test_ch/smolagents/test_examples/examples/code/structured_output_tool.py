# How to run with uv:
#   uv run structured_output_tool.py
#
# Modify the smolagents dependency to point to the local smolagents repo or
# remove `@ file:///<path-to-smolagents>`
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "smolagents[mcp,litellm] @ file:///<path-to-smolagents>",
#   "pydantic",
# ]
# ///

from textwrap import dedent

from mcp import StdioServerParameters

from smolagents import CodeAgent, InferenceClientModel, LiteLLMModel, MCPClient  # noqa: F401


def weather_server_script() -> str:
    """Return an inline MCP server script that exposes a weather tool."""
    return dedent(
        '''
        from pydantic import BaseModel, Field
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Weather Service")

        class WeatherInfo(BaseModel):
            location: str = Field(description="The location name")
            temperature: float = Field(description="Temperature in Celsius")
            conditions: str = Field(description="Weather conditions")
            humidity: int = Field(description="Humidity percentage", ge=0, le=100)

        @mcp.tool(
            name="get_weather_info",
            description="Get weather information for a location as structured data.",
        )
        def get_weather_info(city: str) -> WeatherInfo:
            """Get weather information for a city."""
            return WeatherInfo(
                location=city,
                temperature=22.5,
                conditions="partly cloudy",
                humidity=65
            )

        mcp.run()
        '''
    )


