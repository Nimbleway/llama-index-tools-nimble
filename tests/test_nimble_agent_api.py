"""Unit tests for NimbleAgentToolSpec. No API key or network required.

The SDK's ``agents.runs`` resource methods are mocked, but their return values
are built through the *real* generated response models (``model_validate`` for
wire-shaped payloads, ``model_construct`` to smuggle in contract-violating
states), so the mapping is exercised against the SDK's actual model layer.
"""

from unittest.mock import MagicMock

import httpx
import pytest
from llama_index.core.schema import Document
from nimble_python import (
    AuthenticationError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from nimble_python.types.agents.run_create_response import RunCreateResponse
from nimble_python.types.agents.run_get_response import RunGetResponse
from nimble_python.types.agents.run_result_response import (
    TaskRunFailedResultPublicV2,
    TaskRunResultPublicV2,
)

from llama_index.tools.nimble import (
    NimbleAgentProtocolError,
    NimbleAgentRunCancelledError,
    NimbleAgentRunFailedError,
    NimbleAgentTimeoutError,
    NimbleAgentToolSpec,
    NimbleToolSpec,
)

AGENT_ID = "wsa_00000000-0000-0000-0000-0000000000aa"
RUN_ID = "task_run_00000000-0000-0000-0000-0000000000bb"
CANARY_KEY = "canary-key-XYZ-do-not-leak"


# ---------------------------------------------------------------- builders


def _run_payload(status="queued", **overrides):
    payload = {
        "id": RUN_ID,
        "interaction_id": "int_1",
        "status": status,
        "is_active": status in ("queued", "running"),
        "effort": "medium",
        "created_at": "2026-07-22T00:00:00Z",
        "web_search_agent_id": AGENT_ID,
    }
    payload.update(overrides)
    return payload


def _created(status="queued", **overrides):
    return RunCreateResponse.model_validate(_run_payload(status, **overrides))


def _run(status, **overrides):
    return RunGetResponse.model_validate(_run_payload(status, **overrides))


def _trust_payload(sources=None, claims=None, confidence="high", reasoning="ok"):
    if sources is None:
        sources = [
            {"url": "https://example.org/a", "type": "primary", "title": "Source A"}
        ]
    if claims is None:
        claims = [
            {
                "callout": 1,
                "citations": [
                    {"url": "https://example.org/a", "excerpts": ["a quote"]}
                ],
                "confidence": "high",
                "reasoning": "verified",
            }
        ]
    return {
        "sources": sources,
        "confidence": confidence,
        "reasoning": reasoning,
        "claims": claims,
    }


def _text_result(content="The final answer.", **trust_kwargs):
    return TaskRunResultPublicV2.model_validate(
        {
            "run": _run_payload("completed"),
            "output": {
                "type": "text",
                "content": content,
                "trust": _trust_payload(**trust_kwargs),
            },
        }
    )


def _json_result(content=None):
    if content is None:
        content = {"company": "Acme", "founded": 1999}
    return TaskRunResultPublicV2.model_validate(
        {
            "run": _run_payload("completed"),
            "output": {
                "type": "json",
                "content": content,
                "trust": _trust_payload(
                    claims=[
                        {
                            "path": "$.founded",
                            "citations": [{"url": "https://example.org/a"}],
                            "confidence": "medium",
                            "reasoning": "single source",
                        }
                    ]
                ),
            },
        }
    )


def _failed_result():
    return TaskRunFailedResultPublicV2.model_validate(
        {
            "run": _run_payload("failed"),
            "error": {"message": "upstream failure", "ref_id": RUN_ID},
        }
    )


def _api_error(cls, status_code):
    request = httpx.Request("GET", "https://api.test/v2")
    response = httpx.Response(status_code, request=request)
    return cls("boom", response=response, body=None)


def _spec(monkeypatch, api_key="test-key", timeout=300.0, poll_interval=2.0):
    """Build a NimbleAgentToolSpec whose SDK client is a MagicMock."""
    import nimble_python

    client = MagicMock()
    captured_kwargs = {}

    def _fake_nimble(**kwargs):
        captured_kwargs.update(kwargs)
        return client

    monkeypatch.setattr(nimble_python, "Nimble", _fake_nimble)
    spec = NimbleAgentToolSpec(
        agent_id=AGENT_ID,
        api_key=api_key,
        timeout=timeout,
        poll_interval=poll_interval,
    )
    return spec, client, captured_kwargs


class _FakeClock:
    """Deterministic stand-in for time.monotonic/time.sleep in the poller."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    from llama_index.tools.nimble import agent as agent_module

    clock = _FakeClock()
    monkeypatch.setattr(agent_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(agent_module.time, "sleep", clock.sleep)
    return clock


# ---------------------------------------------------------------- lifecycle


def test_full_lifecycle_queued_running_completed(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.side_effect = [_run("running"), _run("completed")]
    client.agents.runs.result.return_value = _text_result()

    doc = spec.run("research task")

    client.agents.runs.create.assert_called_once_with(
        AGENT_ID, input="research task", effort="medium"
    )
    client.agents.runs.get.assert_called_with(RUN_ID, agent_id=AGENT_ID)
    assert client.agents.runs.get.call_count == 2
    client.agents.runs.result.assert_called_once_with(RUN_ID, agent_id=AGENT_ID)
    assert isinstance(doc, Document)
    assert doc.text.startswith("The final answer.")


def test_immediate_completion_skips_polling(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result()

    doc = spec.run("task")

    client.agents.runs.get.assert_not_called()
    assert isinstance(doc, Document)


def test_effort_passes_through_to_create(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result()

    spec.run("task", effort="x-high")

    client.agents.runs.create.assert_called_once_with(
        AGENT_ID, input="task", effort="x-high"
    )


def test_timeout_retains_run_id_and_skips_result(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch, timeout=10.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.return_value = _run("running")

    with pytest.raises(NimbleAgentTimeoutError) as excinfo:
        spec.run("task")

    err = excinfo.value
    assert err.run_id == RUN_ID
    assert err.agent_id == AGENT_ID
    assert err.status == "running"
    assert RUN_ID in str(err)
    # the poller respected the configured interval and the overall deadline
    assert fake_clock.sleeps == [2.0] * 5
    client.agents.runs.result.assert_not_called()


def test_failed_run_raises_with_server_message(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.return_value = _run(
        "failed", error={"message": "quota exhausted", "ref_id": RUN_ID}
    )

    with pytest.raises(NimbleAgentRunFailedError) as excinfo:
        spec.run("task")

    err = excinfo.value
    assert err.run_id == RUN_ID
    assert err.status == "failed"
    assert "quota exhausted" in str(err)
    assert RUN_ID in str(err)
    client.agents.runs.result.assert_not_called()


def test_failed_run_without_error_detail(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("failed")

    with pytest.raises(NimbleAgentRunFailedError) as excinfo:
        spec.run("task")

    assert "no error detail" in str(excinfo.value)


def test_cancelled_run_raises_typed_error(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.return_value = _run("cancelled")

    with pytest.raises(NimbleAgentRunCancelledError) as excinfo:
        spec.run("task")

    assert excinfo.value.run_id == RUN_ID
    client.agents.runs.result.assert_not_called()


def test_cancelled_run_surfaces_server_detail_when_present(monkeypatch, fake_clock):
    # `error` is not type-coupled to `failed`; a cancellation reason the
    # server attaches must not be silently dropped.
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created(
        "cancelled", error={"message": "budget exceeded", "ref_id": RUN_ID}
    )

    with pytest.raises(NimbleAgentRunCancelledError) as excinfo:
        spec.run("task")

    assert "budget exceeded" in str(excinfo.value)


def test_unknown_status_is_protocol_error(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("queued")
    # model_construct bypasses validation, mimicking a lenient SDK passing
    # through a status this adapter does not know.
    client.agents.runs.get.return_value = RunGetResponse.model_construct(
        **_run_payload("paused")
    )

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert excinfo.value.run_id == RUN_ID
    assert excinfo.value.status == "paused"


def test_create_returning_unknown_status_is_protocol_error(monkeypatch, fake_clock):
    # The unknown-status guard must fire on the very first observed state
    # (the creation response), not only on later polls.
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = RunCreateResponse.model_construct(
        **_run_payload("paused")
    )

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert excinfo.value.run_id == RUN_ID
    client.agents.runs.get.assert_not_called()


def test_completed_run_with_failed_result_payload(monkeypatch, fake_clock):
    # /result answered with the failed-variant payload for a run that polled
    # as completed — a contract violation, not a mappable answer.
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _failed_result()

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert excinfo.value.run_id == RUN_ID
    assert "no output payload" in str(excinfo.value)


def test_conflict_on_result_is_protocol_error_with_cause(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.side_effect = _api_error(ConflictError, 409)

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert excinfo.value.run_id == RUN_ID
    assert isinstance(excinfo.value.__cause__, ConflictError)


@pytest.mark.parametrize(
    ("error_cls", "status_code"),
    [
        (PermissionDeniedError, 403),
        (NotFoundError, 404),
        (UnprocessableEntityError, 422),
        (RateLimitError, 429),
        (InternalServerError, 500),
    ],
)
def test_api_error_while_polling_is_wrapped(
    monkeypatch, fake_clock, error_cls, status_code
):
    # The contract's full HTTP-status list: whatever escapes the SDK's own
    # retry layer during polling wraps to a protocol error retaining run_id.
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.side_effect = _api_error(error_cls, status_code)

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert excinfo.value.run_id == RUN_ID
    assert isinstance(excinfo.value.__cause__, error_cls)


def test_auth_error_on_create_propagates_unwrapped(monkeypatch, fake_clock):
    # Before a run exists there is no run_id to retain; the SDK error is the
    # most informative thing to surface.
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.side_effect = _api_error(AuthenticationError, 401)

    with pytest.raises(AuthenticationError):
        spec.run("task")


# ---------------------------------------------------------------- mapping


def test_text_result_document_mapping(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result()

    doc = spec.run("task")

    # citations must survive stringification: answer + sources in the text
    assert doc.text.startswith("The final answer.")
    assert "Sources:" in doc.text
    assert "- Source A — https://example.org/a (primary)" in doc.text

    meta = doc.metadata
    assert meta["run_id"] == RUN_ID
    assert meta["agent_id"] == AGENT_ID
    assert meta["effort"] == "medium"
    assert meta["output_type"] == "text"
    assert meta["confidence"] == "high"
    assert meta["reasoning"] == "ok"
    assert meta["sources"][0]["url"] == "https://example.org/a"
    assert meta["claims"][0]["callout"] == 1
    assert meta["claims"][0]["citations"][0]["excerpts"] == ["a quote"]


def test_json_result_document_mapping(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _json_result()

    doc = spec.run("task")

    # structured content rendered as readable JSON, still citing sources
    assert '"company": "Acme"' in doc.text
    assert "Sources:" in doc.text
    assert doc.metadata["output_type"] == "json"
    # JSON-output claims are keyed by JSON path, not callout
    assert doc.metadata["claims"][0]["path"] == "$.founded"


def test_result_without_sources_has_no_sources_section(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result(sources=[], claims=[])

    doc = spec.run("task")

    assert "Sources:" not in doc.text
    assert doc.metadata["sources"] == []


def test_source_without_title_renders_bare_url(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result(
        sources=[{"url": "https://example.org/no-title", "type": "secondary"}]
    )

    doc = spec.run("task")

    assert "- https://example.org/no-title (secondary)" in doc.text


def test_output_maps_correctly_when_type_tag_is_absent(monkeypatch, fake_clock):
    # `output.type` is Optional in the response model; discrimination must
    # work purely from the content shape when the tag is missing.
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")

    text_payload = {
        "run": _run_payload("completed"),
        "output": {"content": "untagged answer", "trust": _trust_payload()},
    }
    client.agents.runs.result.return_value = TaskRunResultPublicV2.model_validate(
        text_payload
    )
    doc = spec.run("task")
    assert doc.metadata["output_type"] == "text"
    assert doc.text.startswith("untagged answer")

    json_payload = {
        "run": _run_payload("completed"),
        "output": {"content": {"k": "v"}, "trust": _trust_payload(claims=[])},
    }
    client.agents.runs.result.return_value = TaskRunResultPublicV2.model_validate(
        json_payload
    )
    doc = spec.run("task")
    assert doc.metadata["output_type"] == "json"
    assert '"k": "v"' in doc.text


# ---------------------------------------------------------------- surface


def test_to_tool_list_exposes_run_with_description(monkeypatch):
    spec, _, _ = _spec(monkeypatch)
    tools = spec.to_tool_list()
    tool = next(t for t in tools if t.metadata.name == "run")
    # the docstring is the contract the LLM reads to decide how to call it
    assert tool.metadata.description


def test_param_descriptions_reach_the_tool_schema(monkeypatch):
    # LlamaIndex's docstring parser requires the `name (type): desc` form;
    # without it the per-parameter descriptions silently never reach the
    # JSON schema a function-calling model reads.
    spec, _, _ = _spec(monkeypatch)
    tool = next(t for t in spec.to_tool_list() if t.metadata.name == "run")
    fields = tool.metadata.fn_schema.model_fields
    assert fields["task"].description
    assert fields["effort"].description
    assert "research" in fields["task"].description.lower()


@pytest.mark.parametrize("bad_task", ["", "   "])
def test_task_must_be_non_empty(monkeypatch, bad_task):
    spec, client, _ = _spec(monkeypatch)
    with pytest.raises(ValueError):
        spec.run(bad_task)
    client.agents.runs.create.assert_not_called()


def test_effort_must_be_a_known_level(monkeypatch):
    spec, client, _ = _spec(monkeypatch)
    with pytest.raises(ValueError):
        spec.run("task", effort="turbo")  # type: ignore[arg-type]
    client.agents.runs.create.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"agent_id": ""},
        {"agent_id": "   "},
        {"timeout": 0},
        {"timeout": -1},
        {"poll_interval": 0},
        {"poll_interval": -1},
    ],
)
def test_constructor_rejects_bad_arguments(monkeypatch, kwargs):
    import nimble_python

    monkeypatch.setattr(nimble_python, "Nimble", lambda **kw: MagicMock())
    full_kwargs = {"agent_id": AGENT_ID, **kwargs}
    with pytest.raises(ValueError):
        NimbleAgentToolSpec(**full_kwargs)


def test_client_source_header_matches_search_tool(monkeypatch):
    """Both tool specs must send the same X-Client-Source attribution.

    Constructs BOTH specs and compares the headers they actually pass, so a
    drift on either side fails this test (not just a drift in agent.py).
    """
    import nimble_python

    seen = []

    def _fake(**kwargs):
        seen.append(kwargs.get("default_headers"))
        return MagicMock()

    monkeypatch.setattr(nimble_python, "Nimble", _fake)
    NimbleAgentToolSpec(agent_id=AGENT_ID, api_key="k")
    NimbleToolSpec(api_key="k")

    assert seen[0] == seen[1] == {"X-Client-Source": "llama-index-tools-nimble"}


def test_init_without_api_key_does_not_raise(monkeypatch):
    import nimble_python

    monkeypatch.setattr(nimble_python, "Nimble", lambda **kw: MagicMock())
    spec = NimbleAgentToolSpec(agent_id=AGENT_ID)  # SDK reads NIMBLE_API_KEY
    assert spec.client is not None


# ---------------------------------------------------------------- redaction


def test_api_key_is_not_stored_on_the_spec(monkeypatch):
    """The spec itself must never hold the key.

    (The real SDK client's own repr is out of scope here — `client` is a
    mock; the design guarantee under test is that the adapter hands the key
    to the SDK and keeps no copy.)
    """
    spec, _, _ = _spec(monkeypatch, api_key=CANARY_KEY)
    assert CANARY_KEY not in repr(vars(spec))


def test_api_key_absent_from_run_errors(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch, api_key=CANARY_KEY, timeout=4.0)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.return_value = _run("running")

    with pytest.raises(NimbleAgentTimeoutError) as excinfo:
        spec.run("task")
    assert CANARY_KEY not in str(excinfo.value)
    assert CANARY_KEY not in repr(excinfo.value)

    client.agents.runs.get.return_value = _run(
        "failed", error={"message": "server side detail", "ref_id": RUN_ID}
    )
    with pytest.raises(NimbleAgentRunFailedError) as excinfo2:
        spec.run("task")
    assert CANARY_KEY not in str(excinfo2.value)


def test_api_key_absent_from_document_and_tool_schema(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch, api_key=CANARY_KEY)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result()

    doc = spec.run("task")
    assert CANARY_KEY not in doc.text
    assert CANARY_KEY not in repr(doc.metadata)

    tool = next(t for t in spec.to_tool_list() if t.metadata.name == "run")
    assert CANARY_KEY not in tool.metadata.description


def test_api_key_absent_from_chained_cause(monkeypatch, fake_clock):
    # Walk the __cause__ chain of a wrapped SDK error: neither our message
    # nor the chained SDK exception's str/repr may carry the key.
    spec, client, _ = _spec(monkeypatch, api_key=CANARY_KEY)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.side_effect = _api_error(ConflictError, 409)

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    exc: BaseException | None = excinfo.value
    seen_links = 0
    while exc is not None:
        assert CANARY_KEY not in str(exc)
        assert CANARY_KEY not in repr(exc)
        exc = exc.__cause__
        seen_links += 1
    assert seen_links >= 2  # our error + the chained SDK error


# ------------------------------------------------------- wire-level binding


def test_real_client_binds_real_sdk_signatures(monkeypatch, fake_clock):
    """Drive run() through the REAL Nimble client over a mocked transport.

    A bare MagicMock accepts any call shape, so the rest of the suite cannot
    catch a kwarg rename in the SDK's create/get/result methods. Here the
    real resource methods bind their actual signatures before hitting an
    httpx.MockTransport — still credential-free and network-free — and the
    attribution header is asserted on the wire.
    """
    import nimble_python
    from nimble_python import Nimble as RealNimble

    seen: list[tuple[str, str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (request.method, request.url.path, request.headers.get("X-Client-Source"))
        )
        if request.method == "POST":
            return httpx.Response(202, json=_run_payload("queued"))
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "run": _run_payload("completed"),
                    "output": {
                        "type": "text",
                        "content": "wire answer",
                        "trust": _trust_payload(),
                    },
                },
            )
        return httpx.Response(200, json=_run_payload("completed"))

    def _fake(**kwargs):
        return RealNimble(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    monkeypatch.setattr(nimble_python, "Nimble", _fake)
    spec = NimbleAgentToolSpec(agent_id=AGENT_ID, api_key="test-key")

    doc = spec.run("wire task")

    assert doc.text.startswith("wire answer")
    assert [(method, path) for method, path, _ in seen] == [
        ("POST", f"/v2/agents/{AGENT_ID}/runs"),
        ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}"),
        ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}/result"),
    ]
    assert all(header == "llama-index-tools-nimble" for _, _, header in seen)
