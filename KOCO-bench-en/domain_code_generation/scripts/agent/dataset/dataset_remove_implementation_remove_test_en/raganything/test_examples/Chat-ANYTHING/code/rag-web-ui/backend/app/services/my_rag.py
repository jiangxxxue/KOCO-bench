import asyncio
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc


class MCPGeminiRAGClient:
    """Client combining RAGAnything (local documents) + Gemini LLM + MCP tools."""

    # -------------------- Load document --------------------
    # -------------------- MCP Connection --------------------
    async def connect_to_mcp(self, server_script_path: str = "app/services/server.py"):
        """Connect to MCP server and list available tools."""
        server_params = StdioServerParameters(command="python", args=[server_script_path])

        # نستخدم async with بشكل آمن
        self.stdio_cm = stdio_client(server_params)

        # نفتح الاتصال ونحتفظ بالمراجع
        self._stdio_context = await self.stdio_cm.__aenter__()
        self.stdio, self.write = self._stdio_context

        self.session_cm = ClientSession(self.stdio, self.write)
        self._session_context = await self.session_cm.__aenter__()
        self.session = self._session_context

        await self.session.initialize()

        tools_result = await self.session.list_tools()
        print("\n✅ Connected to MCP server with tools:")
        for tool in tools_result.tools:
            print(f"  - {tool.name}: {tool.description}")



    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Fetch all available tools from the MCP server."""
        tools_result = await self.session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
                "handler": self._make_tool_handler(tool.name),
            }
            for tool in tools_result.tools
        ]

    def _make_tool_handler(self, tool_name: str):
        """Create async handler for each MCP tool."""
        async def handler(args: Dict[str, Any]):
            result = await self.session.call_tool(tool_name, arguments=args)
            parts = []
            for c in result.content:
                if hasattr(c, "text") and c.text:
                    parts.append(c.text)
                elif hasattr(c, "data") and c.data:
                    parts.append(str(c.data))
            return "\n".join(parts) if parts else "No output from tool."
        return handler

    # -------------------- Main query logic --------------------

    # -------------------- Cleanup --------------------
    async def cleanup(self):
        """Gracefully close all sessions."""
        try:
            if hasattr(self, "_session_context"):
                await self.session_cm.__aexit__(None, None, None)
            if hasattr(self, "_stdio_context"):
                await self.stdio_cm.__aexit__(None, None, None)
            print("🧹 MCP session cleaned up successfully.")
        except Exception as e:
            print(f"⚠ Cleanup error: {e}")


