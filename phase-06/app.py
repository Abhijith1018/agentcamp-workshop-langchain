import os
from datetime import date

import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from tools import TOOLS

# Load environment variables
load_dotenv()

# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""You are a helpful AI assistant named Aria.

You have access to multiple tools:
- Local tools: get_weather for real-time weather
- MCP tools: LangChain documentation search and other services

Guidelines:
- For weather-related queries, use the get_weather tool
- For LangChain-related questions, use MCP documentation tools
- Do NOT make up real-time data
- Be clear and helpful

Current date: {date.today().strftime("%B %d, %Y")}
"""

# -------------------------------------------------
# LLM CONFIG
# -------------------------------------------------
def get_llm():
    return ChatOpenAI(
        model="openai/gpt-4.1-nano",
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.github.ai/inference",
        temperature=0.7,
    )

# -------------------------------------------------
# MCP TOOLS FETCHER
# -------------------------------------------------
async def get_mcp_tools():
    """
    Fetch tools from MCP servers.
    """
    mcp_client = MultiServerMCPClient(
        {
            "langchain_docs": {
                "transport": "http",
                "url": "https://docs.langchain.com/mcp",
            }
        }
    )
    return await mcp_client.get_tools()

# -------------------------------------------------
# AGENT FACTORY
# -------------------------------------------------
async def create_assistant_agent():
    llm = get_llm()

    # Fetch MCP tools
    mcp_tools = await get_mcp_tools()

    # Combine local + MCP tools
    agent = create_agent(
        model=llm,
        tools=[*TOOLS, *mcp_tools],
        system_prompt=SYSTEM_PROMPT,
    )

    return agent

# -------------------------------------------------
# CHAINLIT EVENTS
# -------------------------------------------------
@cl.on_chat_start
async def start():
    agent = await create_assistant_agent()

    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])

    await cl.Message(
        content="👋 Hi! I'm Aria. I can check the weather and search LangChain docs. Try asking!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    chat_history = cl.user_session.get("chat_history")

    # Add user message
    chat_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    full_response = ""
    steps = {}

    # Stream messages + tool updates
    async for stream_mode, data in agent.astream(
        {"messages": chat_history},
        stream_mode=["messages", "updates"],
    ):
        # Handle tool calls and results
        if stream_mode == "updates":
            for source, update in data.items():
                if source in ("model", "tools"):
                    last_msg = update["messages"][-1]

                    # Tool invocation
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            step = cl.Step(
                                f"🔧 {tool_call['name']}",
                                type="tool",
                            )
                            step.input = tool_call["args"]
                            await step.send()
                            steps[tool_call["id"]] = step

                    # Tool result
                    if isinstance(last_msg, ToolMessage):
                        step = steps.get(last_msg.tool_call_id)
                        if step:
                            step.output = last_msg.content
                            await step.update()

        # Stream assistant text
        if stream_mode == "messages":
            token, _ = data
            if isinstance(token, AIMessageChunk):
                full_response += token.content
                await msg.stream_token(token.content)

    await msg.send()

    # Save assistant response
    chat_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("chat_history", chat_history)
import os
from datetime import date

import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from tools import TOOLS

# Load environment variables
load_dotenv()

# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""You are a helpful AI assistant named Aria.

You have access to multiple tools:
- Local tools: get_weather for real-time weather
- MCP tools: LangChain documentation search and other services

Guidelines:
- For weather-related queries, use the get_weather tool
- For LangChain-related questions, use MCP documentation tools
- Do NOT make up real-time data
- Be clear and helpful

Current date: {date.today().strftime("%B %d, %Y")}
"""

# -------------------------------------------------
# LLM CONFIG
# -------------------------------------------------
def get_llm():
    return ChatOpenAI(
        model="openai/gpt-4.1-nano",
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.github.ai/inference",
        temperature=0.7,
    )

# -------------------------------------------------
# MCP TOOLS FETCHER
# -------------------------------------------------
async def get_mcp_tools():
    """
    Fetch tools from MCP servers.
    """
    mcp_client = MultiServerMCPClient(
        {
            "langchain_docs": {
                "transport": "http",
                "url": "https://docs.langchain.com/mcp",
            }
        }
    )
    return await mcp_client.get_tools()

# -------------------------------------------------
# AGENT FACTORY
# -------------------------------------------------
async def create_assistant_agent():
    llm = get_llm()

    # Fetch MCP tools
    mcp_tools = await get_mcp_tools()

    # Combine local + MCP tools
    agent = create_agent(
        model=llm,
        tools=[*TOOLS, *mcp_tools],
        system_prompt=SYSTEM_PROMPT,
    )

    return agent

# -------------------------------------------------
# CHAINLIT EVENTS
# -------------------------------------------------
@cl.on_chat_start
async def start():
    agent = await create_assistant_agent()

    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])

    await cl.Message(
        content="👋 Hi! I'm Aria. I can check the weather and search LangChain docs. Try asking!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    chat_history = cl.user_session.get("chat_history")

    # Add user message
    chat_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    full_response = ""
    steps = {}

    # Stream messages + tool updates
    async for stream_mode, data in agent.astream(
        {"messages": chat_history},
        stream_mode=["messages", "updates"],
    ):
        # Handle tool calls and results
        if stream_mode == "updates":
            for source, update in data.items():
                if source in ("model", "tools"):
                    last_msg = update["messages"][-1]

                    # Tool invocation
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            step = cl.Step(
                                f"🔧 {tool_call['name']}",
                                type="tool",
                            )
                            step.input = tool_call["args"]
                            await step.send()
                            steps[tool_call["id"]] = step

                    # Tool result
                    if isinstance(last_msg, ToolMessage):
                        step = steps.get(last_msg.tool_call_id)
                        if step:
                            step.output = last_msg.content
                            await step.update()

        # Stream assistant text
        if stream_mode == "messages":
            token, _ = data
            if isinstance(token, AIMessageChunk):
                full_response += token.content
                await msg.stream_token(token.content)

    await msg.send()

    # Save assistant response
    chat_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("chat_history", chat_history)
