import os
from mcp.server.fastmcp import FastMCP
from ddgs import DDGS  
# Create an MCP server
mcp = FastMCP(
    name="Web Search Tool",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8050,       # only used for SSE transport
)

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search using ddgs and return summarized results.

    Args:
        query (str): The search query.
        max_results (int, optional): Number of results to fetch. Defaults to 5.

    Returns:
        str: A formatted string with search results.
    """
    try:
        results_text = f"🔎 Search results for: {query}\n\n"
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)

        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            link = r.get("href", "No link")
            snippet = r.get("body", "")
            results_text += f"{i}. {title}\n{link}\n{snippet}\n\n"

        return results_text if results_text.strip() else "No results found."

    except Exception as e:
        return f"Error: {str(e)}"

# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
import os
from mcp.server.fastmcp import FastMCP
from ddgs import DDGS  
# Create an MCP server
mcp = FastMCP(
    name="Web Search Tool",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8050,       # only used for SSE transport
)

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search using ddgs and return summarized results.

    Args:
        query (str): The search query.
        max_results (int, optional): Number of results to fetch. Defaults to 5.

    Returns:
        str: A formatted string with search results.
    """
    try:
        results_text = f"🔎 Search results for: {query}\n\n"
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)

        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            link = r.get("href", "No link")
            snippet = r.get("body", "")
            results_text += f"{i}. {title}\n{link}\n{snippet}\n\n"

        return results_text if results_text.strip() else "No results found."

    except Exception as e:
        return f"Error: {str(e)}"

# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
