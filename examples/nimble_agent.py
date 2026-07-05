"""Runnable example: a LlamaIndex agent using the Nimble web-search tool.

Requires:
    pip install llama-index-tools-nimble llama-index-llms-openai
    export NIMBLE_API_KEY=...      # Nimble web search
    export OPENAI_API_KEY=...      # the agent's LLM

Run:
    python examples/nimble_agent.py
"""

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from llama_index.tools.nimble import NimbleToolSpec


async def main() -> None:
    agent = FunctionAgent(
        tools=NimbleToolSpec().to_tool_list(),
        llm=OpenAI(model="gpt-4o-mini"),
        system_prompt=(
            "You are a research assistant. Use the web search tool to ground "
            "your answers in current sources, and cite the URLs you used."
        ),
    )
    response = await agent.run(
        "What is Nimble (nimbleway.com) and what does its Web API offer? "
        "Use web search and cite sources."
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
