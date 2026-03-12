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

    def __init__(self, gemini_api_key: str):
        self.session: Optional[ClientSession] = None
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None
        self.history: List[Dict[str, str]] = []  # Conversation history

        # Initialize Gemini client
        self.gemini = genai.Client(api_key=gemini_api_key)

        # Configure RAGAnything
        config = RAGAnythingConfig(
            working_dir="./rag_storage",
            parser="mineru",
            parse_method="auto",
            enable_image_processing=False,
            enable_table_processing=True,
            enable_equation_processing=True,
            context_window=5,
            max_context_tokens=4000,
            context_mode="chunk",
            include_headers=True,
            include_captions=True,
            context_filter_content_types=["text"],
        )

        # -------------------- Embedding function --------------------
        async def generate_embedding(model: str, texts: list[str]):
            """Generate embeddings using Gemini."""
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini.models.embed_content(
                    model=model,
                    contents=[types.Content(parts=[types.Part(text=t)]) for t in texts],
                ),
            )
            return [embedding.values for embedding in response.embeddings]

        embedding_func = EmbeddingFunc(
            embedding_dim=3072,   # Gemini embeddings are 3072-dimensional
            max_token_size=8192,
            func=lambda texts: generate_embedding("gemini-embedding-001", texts),
        )

        # -------------------- LLM model function --------------------
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], tools=[], **kwargs):
            """LLM logic for Gemini with optional tool usage."""
            if not system_prompt:
                system_prompt = (
                    "You are connected to external tools such as web_search. "
                    "If the user asks about the current date, time, weather, news, sports, or anything that requires real-time information, "
                    "you MUST call the appropriate tool instead of answering directly. "
                    "When a tool returns results, you must extract the relevant facts "
                    "(dates, opponents, scores, etc.) and provide a clear, concise final answer. "
                    "Do not ignore tool outputs. "
                    "Always respond in the same language as the user's query."
                )

            # Prepare chat history for Gemini
            contents = []
            for msg in history_messages:
                role = msg.get("role", "user")
                if role not in ["user", "model"]:
                    role = "user"
                contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

            contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

            # Define MCP tools for Gemini
            tool_defs = []
            if tools:
                for tool in tools:
                    tool_defs.append(
                        types.Tool(
                            function_declarations=[
                                types.FunctionDeclaration(
                                    name=tool["name"],
                                    description=tool["description"],
                                    parameters=tool["parameters"],
                                )
                            ]
                        )
                    )

            # Ask Gemini
            response = self.gemini.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=tool_defs if tool_defs else None,
                    system_instruction=types.Content(parts=[types.Part(text=system_prompt)]),
                ),
            )

            candidate = response.candidates[0]

            # If Gemini calls a tool
            if candidate.content.parts and candidate.content.parts[0].function_call:
                fn_call = candidate.content.parts[0].function_call
                tool_name = fn_call.name
                args = fn_call.args if fn_call.args else {}

                for tool in tools:
                    if tool["name"] == tool_name:
                        tool_result = await tool["handler"](args)

                        # Ask Gemini again, including tool result
                        follow_up = self.gemini.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part(
                                        text=f"{prompt}\n\nThe tool '{tool_name}' returned:\n\n{tool_result}\n\n"
                                             f"Please summarize and give a direct answer."
                                    )],
                                ),
                            ],
                            config=types.GenerateContentConfig(
                                system_instruction=types.Content(parts=[types.Part(text=system_prompt)]),
                            ),
                        )

                        # Extract text parts safely
                        text_parts = []
                        if follow_up.candidates and follow_up.candidates[0].content.parts:
                            for part in follow_up.candidates[0].content.parts:
                                if hasattr(part, "text") and part.text:
                                    text_parts.append(part.text)
                        return "\n".join(text_parts).strip() or "No textual response from model."

            # Default: just return Gemini text
            text_parts = []
            if candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
            return "\n".join(text_parts).strip() or "No textual response from model."

        # Initialize RAGAnything
        self.rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
        )

    # -------------------- Load document --------------------
    async def load_documents(self):
        """Load and index your document into the vector database (LightRAG)."""
        await self.rag.process_document_complete(
            file_path=r"/mnt/c/Users/Zakaria Eisa/RAG-Anything/Agil_PDF.pdf",
            output_dir="./output",
            parse_method="auto",
        )
        print("✅ Document successfully loaded into RAG.")

    # -------------------- MCP Connection --------------------
    async def connect_to_mcp(self, server_script_path: str = "server.py"):
        """Connect to MCP server and list available tools."""
        server_params = StdioServerParameters(command="python", args=[server_script_path])
        self.stdio_cm = stdio_client(server_params)
        self.stdio, self.write = await self.stdio_cm.__aenter__()

        self.session_cm = ClientSession(self.stdio, self.write)
        self.session = await self.session_cm.__aenter__()
        await self.session.initialize()

        tools_result = await self.session.list_tools()
        print("\nConnected to MCP server with tools:")
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
    async def process_query(self, query: str) -> str:
        """Process user query with RAG + Gemini + MCP tools."""
        tools = await self.get_mcp_tools()
        self.history.append({"role": "user", "content": query})

        print("🔍 Searching in vector database...")
        rag_result = await self.rag.aquery(query, mode="hybrid")
        print(f"📝 RAG returned: {rag_result}")

        evaluation_prompt = f"""
        Here is the answer from the RAG knowledge base:
        "{rag_result}"

        Does this response directly answer the question "{query}" with factual detail?
        Reply with only one word: YES or NO.
        """

        judge = self.gemini.models.generate_content(
            model="gemini-2.0-flash",
            contents=[types.Content(parts=[types.Part(text=evaluation_prompt)])],
        )

        if "no" in judge.candidates[0].content.parts[0].text.lower():
            print("⚙️ Gemini judged RAG answer as unhelpful. Using external tools...")
            response = await self.rag.chat(query, tools=tools, multimodal=True, history_messages=self.history)
        else:
            print("✅ Gemini confirmed RAG answer is relevant.")
            self.history.append({"role": "model", "content": rag_result})
            return rag_result

        self.history.append({"role": "model", "content": response})
        return response

    # -------------------- Cleanup --------------------
    async def cleanup(self):
        """Gracefully close all sessions."""
        if hasattr(self, "session_cm"):
            await self.session_cm.__aexit__(None, None, None)
        if hasattr(self, "stdio_cm"):
            await self.stdio_cm.__aexit__(None, None, None)


# -------------------- Safe Input Utility --------------------
def safe_input(prompt="👉 Question: "):
    """Read user input safely, removing illegal Unicode surrogates."""
    try:
        query = input(prompt)
        # Remove invalid UTF-8 surrogates and control characters
        query = query.encode("utf-8", "ignore").decode("utf-8", "ignore").strip()
        return query
    except Exception as e:
        print(f"⚠️ Input error: {e}")
        return ""


# -------------------- Main entry point --------------------
async def main():
    client = MCPGeminiRAGClient(gemini_api_key="AIzaSyBFLZ8o7tOCppO60pLBKX9SsGev0FEsavs")
    await client.connect_to_mcp("server.py")

    await client.load_documents()

    print("📌 Type your question (or 'exit' to quit):")
    while True:
        query = safe_input()
        if query.lower() in ["exit", "quit"]:
            print("👋 Program ended.")
            break

        try:
            response = await client.process_query(query)
            print(f"✅ Answer: {response}\n")

            print("🕘 Conversation so far:")
            for msg in client.history:
                print(f"[{msg['role']}] {msg['content']}")
            print("-" * 40)
        except Exception as e:
            print(f"❌ Error: {e}")

    await client.cleanup()


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
