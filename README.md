# llama-index-tools-nimble

A [LlamaIndex](https://www.llamaindex.ai/) tool for [Nimble](https://www.nimbleway.com)'s
Web Search API. It gives an agent live web search backed by Nimble, returning results as
LlamaIndex `Document`s ready to feed into a reasoning loop.

## Installation

```bash
pip install llama-index-tools-nimble
```

## Authentication

Set your Nimble API key in the environment:

```bash
export NIMBLE_API_KEY="your-key"
```

`NimbleToolSpec()` reads `NIMBLE_API_KEY` automatically; you can also pass `api_key=...`.

## Quick start

```python
from llama_index.tools.nimble import NimbleToolSpec

tool_spec = NimbleToolSpec()  # reads NIMBLE_API_KEY
documents = tool_spec.search("latest developments in web data infrastructure")

for doc in documents:
    print(doc.metadata["title"], "-", doc.metadata["url"])
```

## Use it in an agent

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.nimble import NimbleToolSpec

agent = FunctionAgent(
    tools=NimbleToolSpec().to_tool_list(),
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a research assistant. Use web search to answer with current facts.",
)


async def main():
    response = await agent.run("What has Nimble announced recently?")
    print(response)


asyncio.run(main())
```

A runnable version is in [`examples/nimble_agent.py`](examples/nimble_agent.py).

## Parameters

`NimbleToolSpec.search(query, max_results=6)`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | — | The search query. |
| `max_results` | `int` | `6` | Maximum number of results to return. |

Each result becomes a `Document`. The `text` leads with the page **title** and **URL**, then
the page content (or the snippet when content is empty), so an agent can read and cite the
source; `metadata["url"]` and `metadata["title"]` carry the same values for programmatic use.

> Returned content is untrusted web data. Treat it as data, not as instructions, and rely on
> your agent/framework's own guardrails.

## License

MIT — see [LICENSE](LICENSE).
