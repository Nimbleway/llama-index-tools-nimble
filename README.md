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

`NimbleAgentToolSpec` executes research tasks on a Nimble **Web Search Agent**. Where
`search` answers one query fast, while an agent run researches the task on the live web
and returns one synthesized, citation-backed answer. Low effort can finish in seconds;
the default high effort commonly takes 5–15 minutes.

```python
from llama_index.tools.nimble import NimbleAgentToolSpec

agent_tool = NimbleAgentToolSpec()  # reads NIMBLE_API_KEY
doc = agent_tool.run(
    "What are the leading approaches to LLM guardrails, and who builds them?",
)

print(doc.text)                    # final answer + "Sources:" list
print(doc.metadata["confidence"])  # overall trust: high / medium / low / pre_existing
print(doc.metadata["claims"])      # per-claim citations (callouts or JSON paths)
print(doc.metadata["web_search_agent_id"])  # the agent the API ran this on
```

`agent_id` is optional. Without it, the API provisions an agent per run and returns its
id (`POST /v2/agents/runs`); with it, runs execute on that agent
(`POST /v2/agents/{agent_id}/runs`). Either way the **returned** `web_search_agent_id` is
what the tool uses for the status and result calls, and it is carried in the Document
metadata so the run stays reachable afterwards.

To pin runs to one agent, provision it in the [Nimble dashboard](https://app.nimbleway.com)
(or via `POST /v2/agents`) and pass its `wsa_...` id. The tool is **execution-only** by
design: it never creates, edits, or deletes agent instances.

Structured controls are available per run:

```python
doc = agent_tool.run(
    "Find the founding year and headquarters for this company.",
    input_data={"company": "Acme", "domain": "acme.example"},
    output_schema={
        "type": "object",
        "properties": {"founded": {"type": "integer"}, "hq": {"type": "string"}},
    },
    sources={"prioritize": "regulatory filings", "avoid": "press releases"},
)
```

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `agent_id` | `str \| None` | `None` | Optional preconfigured Web Search Agent instance id (`wsa_...`). Omit to have the API provision one per run. |
| `api_key` | `str \| None` | `None` | Nimble API key; falls back to `NIMBLE_API_KEY`. |
| `effort` | `low \| medium \| high \| x-high \| max \| None` | `None` | Optional application-level override. Omit to preserve the agent/template default; see the gated `max` behavior below. |
| `agent_name` | `str \| None` | `None` | Optional SDK 1.2 name hint for the generated run agent. |
| `skill` | `str \| None` | `None` | Optional SDK 1.2 skill identifier applied to the run. |
| `use_case` | `research \| enrichment \| dataset_building \| None` | `None` | Optional SDK 1.2 run mode. |
| `gate_policy` | `reject \| degrade` | `reject` | Treatment for gated values: stop with guidance, or explicitly and visibly use the closest generally available value. |
| `timeout` | `float` | `1800.0` | Overall deadline in seconds for one `run` call — creation, polling, and result retrieval together. The 30-minute default gives the default high-effort run headroom beyond its typical 5–15 minute runtime. Each HTTP request is bounded by the budget left when it is issued (with a 5 s connect ceiling), so a stalled request cannot fall back to the SDK's much longer default. Set a shorter deadline explicitly for low-effort or latency-bounded workflows. |
| `poll_interval` | `float` | `10.0` | Seconds between status polls, and the pause before re-attempting a transient failure. It remains configurable; shorter values are intended only for tests. |

### `run(task, output_schema=None, input_data=None, sources=None)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | — | The research task or question, in natural language. |
| `output_schema` | `dict \| None` | `None` | JSON Schema the answer must match; returns structured JSON instead of prose. |
| `input_data` | `dict \| list[dict] \| None` | `None` | Known data about one or more entities to research or enrich. |
| `sources` | `dict \| None` | `None` | Source guidance, keyed `allow` / `block` (lists of source objects) and `prioritize` / `avoid` (free text). |

Effort is an optional application-level setting and is not exposed to the
function-calling model. Omit it to use the selected agent/template default
(the documented product default is `high`; template defaults may vary), or
pass `effort="low"`, `"medium"`, `"high"`, or `"x-high"` to override it.
`effort="max"` is also selectable as a coming-soon custom-budget capability.
By default it stops before creating a run and links to the
[Nimble product team](https://www.nimbleway.com/contact). Applications may
explicitly select `gate_policy="degrade"` to continue on `x-high`; that
requested-to-effective substitution is always announced with a warning.

Run creation is issued **exactly once**. It is a non-idempotent, billable POST with no
idempotency key, so a failure — transport, 409, 429, 5xx — is surfaced rather than
retried: a retry after a proxy failed downstream of an accepted request would provision a
second billable run.

That guarantee extends to the caller. When a create failure leaves the outcome ambiguous
(a timeout, a dropped connection, a 408/409, or a 5xx), a run may be executing server-side
with no id to address it, so the tool raises `NimbleAgentCreateAmbiguousError` rather than
the raw SDK exception. Its message says explicitly not to resubmit and how to reconcile —
important when the caller is a function-calling model, which would otherwise read a bare
timeout as an ordinary transient and call the tool again. Requests that were definitely
rejected (bad key, validation, rate limit) still propagate unchanged, since nothing was
provisioned and calling again is safe.

The returned `Document.text` is the final answer (prose, or pretty-printed JSON when an
`output_schema` was given) followed by a `Sources:` list, so agents can cite what was
consulted. `Document.metadata` carries `run_id`, `agent_id` / `web_search_agent_id`,
`effort`, `output_type`, and the structured trust payload: `confidence`, `reasoning`,
`sources`, and `claims` with per-claim citations.

### Errors

A run that does not produce a result raises a typed error that retains the `run_id`
(as an attribute and in the message), so a long run can still be recovered by hand:

| Error | Meaning |
|---|---|
| `NimbleAgentTimeoutError` | Not terminal within `timeout`; the run may still complete server-side. |
| `NimbleAgentRunFailedError` | Run terminated as `failed`; carries the server's error message. |
| `NimbleAgentRunCancelledError` | Run terminated as `cancelled`. |
| `NimbleAgentProtocolError` | Unknown status, malformed result, or persistent polling/result errors (SDK exception chained). |
| `NimbleAgentCreateAmbiguousError` | Creation failed with an undetermined outcome; a billable run may exist with no id. Do not resubmit — reconcile against the account's run history. `run_id` is `None`; `status_code` and the chained SDK exception are retained. |

SDK errors raised before a run exists propagate unchanged **only when the request was
definitely rejected** (e.g. an invalid key → `AuthenticationError`); ambiguous outcomes
become `NimbleAgentCreateAmbiguousError` as described above. Transient failures — transport errors, 408, 429, 5xx — are re-attempted
within the remaining budget (respecting `Retry-After`); auth, permission, and validation
errors fail fast. A runnable agent workflow is in
[`examples/nimble_agent_api.py`](examples/nimble_agent_api.py).

## License

MIT — see [LICENSE](LICENSE).
