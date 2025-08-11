from dotenv import load_dotenv
load_dotenv()

import os
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

server_url = f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}"

# On Windows prefer 'npx.cmd' to ensure the command resolves
command = "npx.cmd" if os.name == "nt" else "npx"

server_params = StdioServerParameters(
    command=command,
    args=["-y", "mcp-remote", server_url],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent("openai:gpt-4.1", tools)
            agent_response = await agent.ainvoke({"messages": "what is the temperature in bochum"})
            print(agent_response["messages"])

            print(agent_response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())