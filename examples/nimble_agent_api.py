"""Runnable example: a LlamaIndex agent using the Nimble Agent API tool.

The Nimble tool executes deep-research runs on a preconfigured Web Search
Agent; the LlamaIndex agent decides when to call it and relays the
citation-backed answer. Agent runs are long-running (tens of seconds at low
effort, minutes at higher effort) — this is a research tool, not a quick
lookup.

Requires:
    pip install llama-index-tools-nimble llama-index-llms-openai
    export NIMBLE_API_KEY=...      # Nimble API key
    export NIMBLE_AGENT_ID=...     # a provisioned wsa_... agent instance
    export OPENAI_API_KEY=...      # the agent's LLM

Run:
    python examples/nimble_agent_api.py
"""

import asyncio
import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from llama_index.tools.nimble import NimbleAgentToolSpec


async def main() -> None:
    tool_spec = NimbleAgentToolSpec(agent_id=os.environ["NIMBLE_AGENT_ID"])
    agent = FunctionAgent(
        tools=tool_spec.to_tool_list(),
        llm=OpenAI(model="gpt-4o-mini"),
        system_prompt=(
            "You are a research assistant. For questions that need a "
            "researched, synthesized answer, call the Nimble research tool "
            "(it is slow but thorough) and relay its answer together with "
            "the source URLs it cites."
        ),
    )
    response = await agent.run(
        "Research: what do photography reviewers consider the best "
        "mirrorless cameras for travel, and why? Cite your sources."
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
