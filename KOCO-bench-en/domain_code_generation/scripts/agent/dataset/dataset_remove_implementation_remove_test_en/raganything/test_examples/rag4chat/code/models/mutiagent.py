from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.reasoning import ReasoningTools
from agno.tools.function import Function
from agno.team import Team
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Ollama(id="qwen3:8b"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Ollama(id="qwen3:8b"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True
)

team = Team(
    mode="coordinate",
    members=[web_agent, finance_agent],
    model=Ollama(id="qwen3:8b"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True
)

team.print_response("Summarize NVDA analyst recommendations and latest news", stream=True)