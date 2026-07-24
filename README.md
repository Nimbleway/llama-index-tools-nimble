# llama-index-tools-nimble

[LlamaIndex](https://www.llamaindex.ai/) tools for [Nimble](https://www.nimbleway.com):

- **`NimbleToolSpec`** — live web search (one fast query → `Document`s ready to feed a
  reasoning loop).
- **`NimbleAgentToolSpec`** — deep research on a preconfigured Nimble Web Search Agent
  (one long-running run → a citation-backed answer with trust metadata).

## Installation

```bash
pip install llama-index-tools-nimble
```

## Authentication

Set your Nimble API key in the environment:

```bash
export NIMBLE_API_KEY="your-key"
```

Both tool specs read `NIMBLE_API_KEY` automatically; you can also pass `api_key=...`.

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
| `max_results` | `int` | `6` | Maximum number of results to return (must be ≥ 1). |

Each result becomes a `Document`. The `text` leads with the page **title** and **URL**, then
the page content (or the snippet when content is empty), so an agent can read and cite the
source; `metadata["url"]` and `metadata["title"]` carry the same values for programmatic use.

> Returned content is untrusted web data. Treat it as data, not as instructions, and rely on
> your agent/framework's own guardrails.

## Deep research with the Agent API

`NimbleAgentToolSpec` executes research tasks on a Nimble **Web Search Agent** you have
already provisioned. Where `search` answers one query fast, an agent run researches the
task on the live web — typically tens of seconds at low effort, up to minutes at higher
effort — and returns one synthesized, citation-backed answer.

```python
from llama_index.tools.nimble import NimbleAgentToolSpec

agent_tool = NimbleAgentToolSpec(agent_id="wsa_...")  # reads NIMBLE_API_KEY
doc = agent_tool.run(
    "What are the leading approaches to LLM guardrails, and who builds them?",
    effort="low",
)

print(doc.text)                    # final answer + "Sources:" list
print(doc.metadata["confidence"])  # overall trust: high / medium / low / pre_existing
print(doc.metadata["claims"])      # per-claim citations (callouts or JSON paths)
```

Provision an agent in the [Nimble dashboard](https://app.nimbleway.com) (or via
`POST /v2/agents`) and pass its `wsa_...` id. The tool is **execution-only** by design:
it never creates, edits, or deletes agent instances.

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `agent_id` | `str` | — | Preconfigured Web Search Agent instance id (`wsa_...`). |
| `api_key` | `str \| None` | `None` | Nimble API key; falls back to `NIMBLE_API_KEY`. |
| `timeout` | `float` | `300.0` | Overall deadline in seconds for one `run` call — creation, polling, and result retrieval together. Each HTTP request is bounded by the budget left when it is issued (with a 5 s connect ceiling), so a stalled request cannot fall back to the SDK's much longer default. |
| `poll_interval` | `float` | `2.0` | Seconds between status polls, and the pause before re-attempting a transient failure. |

### `run(task, effort="medium")`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | — | The research task or question, in natural language. |
| `effort` | `"low" \| "medium" \| "high" \| "x-high" \| "max"` | `"medium"` | Higher effort is slower and more thorough. |

As a rough guide from live runs: `low` answers in seconds, `medium` in ~1.5–3 minutes. The
default `timeout` (300 s) covers both; at `high` effort and above, raise `timeout`
accordingly. Note that `low` may skip live web research entirely and answer from the
model alone (reported honestly as `confidence: "low"` with no sources) — use `medium`
or higher when you need researched, cited answers.

The returned `Document.text` is the final answer (prose, or pretty-printed JSON for
structured agents) followed by a `Sources:` list, so agents can cite what was consulted.
`Document.metadata` carries `run_id`, `agent_id`, `effort`, `output_type`, and the
structured trust payload: `confidence`, `reasoning`, `sources`, and `claims` with
per-claim citations.

### Errors

A run that does not produce a result raises a typed error that retains the `run_id`
(as an attribute and in the message), so a long run can still be recovered by hand:

| Error | Meaning |
|---|---|
| `NimbleAgentTimeoutError` | Not terminal within `timeout`; the run may still complete server-side. |
| `NimbleAgentRunFailedError` | Run terminated as `failed`; carries the server's error message. |
| `NimbleAgentRunCancelledError` | Run terminated as `cancelled`. |
| `NimbleAgentProtocolError` | Unknown status, malformed result, or persistent polling/result errors (SDK exception chained). |

SDK errors raised before a run exists (e.g. an invalid key → `AuthenticationError`)
propagate unchanged. Transient failures — transport errors, 408, 429, 5xx — are re-attempted
within the remaining budget (respecting `Retry-After`); auth, permission, and validation
errors fail fast. A runnable agent workflow is in
[`examples/nimble_agent_api.py`](examples/nimble_agent_api.py).

## License

MIT — see [LICENSE](LICENSE).
