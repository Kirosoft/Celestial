import os
import sys
import json
import asyncio
from typing import Optional
from contextlib import AsyncExitStack

import openai
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import Scope, Receive, Send
from starlette.middleware import Middleware

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------
# MCPClient: Similar to the local OpenAI MCP client from before
# but with no interactive loop. We'll just expose "process_query"
# to be called by the Starlette route.
# -------------------------------------------------------------------
class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # Configure OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai.api_key:
            raise ValueError("No OPENAI_API_KEY found in environment (or .env).")

        # If you have a custom base URL for the OpenAI API
        custom_base = os.getenv("OPENAI_API_BASE")
        if custom_base:
            openai.api_base = custom_base

    async def connect_to_server(self, server_script_path: str):
        """
        Connect to an MCP server via stdio. Spawns the server locally.
        """
        if not server_script_path.endswith(".py"):
            raise ValueError("Please provide a .py server script.")

        python_executable = sys.executable
        server_params = StdioServerParameters(
            command=python_executable,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        # Initialize and list available tools
        await self.session.initialize()
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """
        Use OpenAI's chat API to interpret the query, then optionally call MCP tools.
        Returns the final text or tool call results as a string.
        """

        # 1) List available tools
        response = await self.session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in response.tools
        ]

        # 2) Construct a system prompt
        system_prompt = f"""
You are a helpful assistant with the ability to call tools.
The user has asked: {query}

Here are the available tools (name, description, input_schema):
{json.dumps(available_tools, indent=2)}

Respond with valid JSON of the form:
{{
  "content": [
    {{
      "type": "text",
      "text": "any reply text"
    }},
    {{
      "type": "tool_use",
      "name": "some_tool_name",
      "input": {{ ... }}
    }}
  ]
}}

No extra keys. Only valid JSON. The "input" object must match the tool's input schema if you call a tool.
If you do not need a tool, just return one item of type "text".
After you have used the tools return the response is a nice human readable form. For example if there are multiple results return a table.
        """.strip()

        # 3) OpenAI chat request (openai>=1.0.0)
        chat_response = openai.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )

        # 4) Extract the model's text
        assistant_message = chat_response.choices[0].message.content.strip()

        # 5) Parse the model's output as JSON
        try:
            parsed = json.loads(assistant_message)
        except json.JSONDecodeError:
            return "Model did not return valid JSON:\n" + assistant_message

        # 6) Process each content item
        final_text_chunks = []
        for item in parsed.get("content", []):
            if item.get("type") == "text":
                final_text_chunks.append(item.get("text", ""))

            elif item.get("type") == "tool_use":
                tool_name = item.get("name")
                tool_input = item.get("input", {})
                print(f"\nExecuting tool '{tool_name}' with input {tool_input}...")
                try:
                    tool_result = await self.session.call_tool(tool_name, tool_input)
                    final_text_chunks.append(
                        f"[Tool '{tool_name}' executed with result: {tool_result}]"
                    )
                except Exception as e:
                    final_text_chunks.append(
                        f"[Error executing tool '{tool_name}': {str(e)}]"
                    )
            else:
                final_text_chunks.append(f"Unrecognized item: {item}")

        return "\n".join(final_text_chunks)

    async def shutdown(self):
        """
        Called when the server is shutting down, to close resources.
        """
        await self.exit_stack.aclose()

    async def prettify_output(self, query: str) -> str:
        """
        Use OpenAI's chat API to interpret the query, then optionally call MCP tools.
        Returns the final text or tool call results as a string.
        """

        # 2) Construct a system prompt
        system_prompt = f"""
You are a helpful assistant that will receive JSON and format the output into a nice human readable string or table based on the json received.
The user has asked: {query}
Return the output as a nice string or table result.
        """.strip()

        # 3) OpenAI chat request (openai>=1.0.0)
        chat_response = openai.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )

        # 4) Extract the model's text
        assistant_message = chat_response.choices[0].message.content.strip()

        return assistant_message
    
# -------------------------------------------------------------------
# 2) Starlette HTTP app
# We'll define a route "/query" that accepts a 'q' query param,
# calls the MCP client's process_query, and returns the result as JSON.
# -------------------------------------------------------------------
mcp_client = MCPClient()

async def startup():
    """
    Runs on Starlette startup. Connect to your MCP server here.
    """
    # Adjust path to your server script
    await mcp_client.connect_to_server("celestial_engine.py")

async def shutdown():
    """
    Runs on Starlette shutdown. Close the MCP session.
    """
    await mcp_client.shutdown()

async def handle_query(request):
    """
    HTTP GET /query?q=some+question
    or POST /query with JSON body { "q": "some question" }
    """
    if request.method == "GET":
        q = request.query_params.get("q", "")
    else:
        body = await request.json()
        q = body.get("q", "")

    # Call the MCP-based LLM logic
    response_text = await mcp_client.process_query(q)

    nice_output = await mcp_client.prettify_output(response_text)

    return JSONResponse({"result": nice_output})

# Starlette routes
routes = [
    Route("/query", handle_query, methods=["GET", "POST"]),
]

app = Starlette(
    debug=True,
    routes=routes,
    on_startup=[startup],
    on_shutdown=[shutdown],
)

# If you prefer, you can run "uvicorn http_mcp_client:app" from the command line:
# uvicorn http_mcp_client:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)