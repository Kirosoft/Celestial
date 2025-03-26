import os
import sys
import json
import asyncio
from typing import Optional
from contextlib import AsyncExitStack

import openai
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

class MCPClient:
    def __init__(self):
        # Store references for session and resource management
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # Configure OpenAI credentials
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai.api_key:
            raise ValueError("No OPENAI_API_KEY found in environment (or .env).")

        # If you have a custom base URL for the OpenAI API (e.g., Azure, proxy):
        custom_base = os.getenv("OPENAI_API_BASE")
        if custom_base:
            openai.api_base = custom_base

    async def connect_to_server(self, server_script_path: str):
        """
        Connect to an MCP server given its script path (e.g. server.py).
        Spawns the server locally via stdio transport.
        """
        if not server_script_path.endswith(".py"):
            raise ValueError("Please provide a .py server script.")

        # Use the same Python interpreter for the server
        python_executable = sys.executable

        server_params = StdioServerParameters(
            command=python_executable,
            args=[server_script_path],
            env=None  # or pass custom environment if needed
        )

        # Create the stdio transport and session
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        # Initialize the session and list available tools
        await self.session.initialize()
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """
        Process a query by sending it to OpenAI's chat API with a system prompt
        that instructs the model to produce JSON describing text or tool calls.
        Then, if tool calls are requested, execute them via MCP and return combined output.
        """

        # 1) Retrieve available tools from the server
        response = await self.session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in response.tools
        ]

        # 2) Construct a system prompt that instructs the model to output JSON
        # describing either text or tool calls
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
        """.strip()

        # 3) Use the new chat interface in openai>=1.0.0
        chat_response = openai.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
        )

        # 4) Extract the model's response (the assistant's message)
        assistant_message = chat_response.choices[0].message.content.strip()

        # 5) Parse the model's output as JSON
        try:
            parsed = json.loads(assistant_message)
        except json.JSONDecodeError:
            # If the model didn't return valid JSON, treat the entire response as text
            return "Model did not return valid JSON:\n" + assistant_message

        # 6) Process each content item
        final_text_chunks = []
        content_list = parsed.get("content", [])

        for item in content_list:
            if item.get("type") == "text":
                # Normal text response
                text = item.get("text", "")
                final_text_chunks.append(text)

            elif item.get("type") == "tool_use":
                # The model wants to call a tool
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
                # Unrecognized item type
                final_text_chunks.append(f"Unrecognized item: {item}")

        return "\n".join(final_text_chunks)

    async def chat_loop(self):
        """Run an interactive chat loop for the user to send queries."""
        print("\nMCP Client Started! Type your query or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print("\nResponse:\n" + response)
            except Exception as e:
                print(f"\nError processing query: {e}")

    async def cleanup(self):
        """Clean up resources and close the session."""
        await self.exit_stack.aclose()

async def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python local_openai_mcp_client.py <path_to_server_script>")
    #     sys.exit(1)

    server_script_path = "celestial_engine.py"
    client = MCPClient()
    try:
        await client.connect_to_server(server_script_path)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

