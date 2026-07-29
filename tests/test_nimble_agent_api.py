"""Unit tests for NimbleAgentToolSpec. No API key or network required.

The SDK's ``agents.runs`` resource methods are mocked, but their return values
are built through the *real* generated response models (``model_validate`` for
wire-shaped payloads, ``model_construct`` to smuggle in contract-violating
states), so the mapping is exercised against the SDK's actual model layer.
"""

import inspect
import json
from types import SimpleNamespace
from typing import get_args
from unittest.mock import ANY, MagicMock

import httpx
import pytest
from llama_index.core.schema import Document
from nimble_python import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
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
# The id the API generates when a run is created without a configured agent.
GENERATED_AGENT_ID = "wsa_11111111-1111-1111-1111-1111111111cc"
RUN_ID = "task_run_00000000-0000-0000-0000-0000000000bb"
CANARY_KEY = "canary-key-XYZ-do-not-leak"

# Fields the released 1.2 run-create API accepts.
ALLOWED_CREATE_KEYS = {
    "agent_name",
    "input",
    "effort",
    "enable_events",
    "input_data",
    "output_schema",
    "previous_interaction_id",
    "skill",
    "sources",
    "timeout",
    "use_case",
}


# ---------------------------------------------------------------- builders


def _run_payload(status="queued", **overrides):
    payload = {
        "id": RUN_ID,
        "interaction_id": "int_1",
        "status": status,
        "is_active": status in ("queued", "running"),
        "effort": "low",
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


def _text_result(content="The final answer.", run_overrides=None, **trust_kwargs):
    return TaskRunResultPublicV2.model_validate(
        {
            "run": _run_payload("completed", **(run_overrides or {})),
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


def _spec(
    monkeypatch,
    api_key="test-key",
    timeout=300.0,
    poll_interval=10.0,
    agent_id=AGENT_ID,
    effort=None,
    agent_name=None,
    skill=None,
    use_case=None,
    gate_policy="reject",
):
    """Build a NimbleAgentToolSpec whose SDK client is a MagicMock.

    ``agent_id=None`` exercises the generated-agent route, where the run is
    created with ``client.agents.run(...)`` instead of
    ``client.agents.runs.create(agent_id, ...)``.

    Individual timing tests may pass a shorter interval as a test-only
    override so their deterministic fake clocks can exercise deadline and
    retry boundaries compactly.
    """
    import nimble_python

    client = MagicMock()
    # The spec calls .with_options(max_retries=0); the real SDK returns an
    # equivalent client, so the mock returns itself.
    client.with_options.return_value = client
    captured_kwargs = {}

    def _fake_nimble(**kwargs):
        captured_kwargs.update(kwargs)
        return client

    monkeypatch.setattr(nimble_python, "Nimble", _fake_nimble)
    spec = NimbleAgentToolSpec(
        agent_id=agent_id,
        api_key=api_key,
        effort=effort,
        agent_name=agent_name,
        skill=skill,
        use_case=use_case,
        gate_policy=gate_policy,
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
    """Deterministic time *and* jitter, so sleep schedules are exact."""
    from llama_index.tools.nimble import agent as agent_module

    clock = _FakeClock()
    monkeypatch.setattr(agent_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(agent_module.time, "sleep", clock.sleep)
    # Back-off jitter is 1 - 0.25 * random(); pin random() to 0 so the
    # jittered delay is exactly the computed back-off.
    monkeypatch.setattr(agent_module, "random", lambda: 0.0)
    return clock


# ---------------------------------------------------------------- lifecycle


def test_full_lifecycle_queued_running_completed(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.side_effect = [_run("running"), _run("completed")]
    client.agents.runs.result.return_value = _text_result()

    doc = spec.run("research task")

    client.agents.runs.create.assert_called_once_with(
        AGENT_ID, input="research task", timeout=ANY
    )
    client.agents.runs.get.assert_called_with(RUN_ID, agent_id=AGENT_ID, timeout=ANY)
    assert client.agents.runs.get.call_count == 2
    client.agents.runs.result.assert_called_once_with(
        RUN_ID, agent_id=AGENT_ID, timeout=ANY
    )
    assert isinstance(doc, Document)
    assert doc.text.startswith("The final answer.")


def test_immediate_completion_skips_polling(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result()

    doc = spec.run("task")

    client.agents.runs.get.assert_not_called()
    assert isinstance(doc, Document)


# ------------------------------------------------- C01/C02: optional identity


def test_absent_agent_id_uses_the_generic_run_route(monkeypatch, fake_clock):
    """C01 — no configured agent: exactly one POST /v2/agents/runs."""
    spec, client, _ = _spec(monkeypatch, agent_id=None)
    client.agents.run.return_value = _created(
        "completed", web_search_agent_id=GENERATED_AGENT_ID
    )
    client.agents.runs.result.return_value = _text_result(
        run_overrides={"web_search_agent_id": GENERATED_AGENT_ID}
    )

    doc = spec.run("task")

    client.agents.run.assert_called_once_with(input="task", timeout=ANY)
    client.agents.runs.create.assert_not_called()
    assert doc.metadata["agent_id"] == GENERATED_AGENT_ID


def test_present_agent_id_uses_the_persistent_run_route(monkeypatch, fake_clock):
    """C02 — configured agent: exactly one POST /v2/agents/{agent_id}/runs."""
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result()

    spec.run("task")

    client.agents.runs.create.assert_called_once_with(
        AGENT_ID, input="task", timeout=ANY
    )
    client.agents.run.assert_not_called()


def test_agent_id_is_optional_at_construction(monkeypatch):
    import nimble_python

    monkeypatch.setattr(nimble_python, "Nimble", lambda **kw: MagicMock())
    assert NimbleAgentToolSpec().agent_id is None


# ------------------------------------------------------ C04: optional effort


@pytest.mark.parametrize("agent_id", [AGENT_ID, None])
def test_every_create_body_omits_unspecified_effort(monkeypatch, fake_clock, agent_id):
    """C04 — both routes preserve the server-side default when unspecified."""
    spec, client, _ = _spec(monkeypatch, agent_id=agent_id)
    created = _created("completed", web_search_agent_id=agent_id or GENERATED_AGENT_ID)
    client.agents.runs.create.return_value = created
    client.agents.run.return_value = created
    client.agents.runs.result.return_value = _text_result(
        run_overrides={"web_search_agent_id": agent_id or GENERATED_AGENT_ID}
    )

    spec.run("task")

    create = client.agents.runs.create if agent_id else client.agents.run
    assert "effort" not in create.call_args.kwargs


@pytest.mark.parametrize("tier", ["low", "medium", "high", "x-high"])
def test_constructor_effort_override_is_forwarded(monkeypatch, fake_clock, tier):
    """C04 — documented, generally available effort overrides are preserved."""
    spec, client, _ = _spec(monkeypatch, effort=tier)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result()

    spec.run("task")

    assert client.agents.runs.create.call_args.kwargs["effort"] == tier


def test_effort_is_application_controlled_not_model_controlled(monkeypatch):
    """C04 — applications choose effort; the function-calling model cannot.

    ``max`` remains selectable at application level but is resolved before
    dispatch under the gated-feature policy.
    """
    from llama_index.tools.nimble.agent import _EFFORT_LEVELS, EffortLevel

    expected = ("low", "medium", "high", "x-high", "max")
    assert _EFFORT_LEVELS == expected
    assert get_args(EffortLevel) == expected

    spec, client, _ = _spec(monkeypatch)
    tool = next(t for t in spec.to_tool_list() if t.metadata.name == "run")
    assert "effort" not in tool.metadata.fn_schema.model_fields
    assert "effort" not in inspect.signature(spec.run).parameters

    with pytest.raises(TypeError):
        spec.run("task", effort="high")
    client.agents.runs.create.assert_not_called()
    client.agents.run.assert_not_called()


def test_max_rejects_before_client_creation_with_positive_guidance(monkeypatch):
    with pytest.raises(ValueError) as captured:
        _spec(monkeypatch, effort="max")
    message = str(captured.value)
    assert "coming-soon" in message
    assert "nothing was sent and no run was created" in message
    assert "https://www.nimbleway.com/contact" in message


def test_explicit_max_degradation_is_announced_and_sends_x_high(
    monkeypatch, fake_clock
):
    with pytest.warns(UserWarning, match="effective effort='x-high'"):
        spec, client, _ = _spec(monkeypatch, effort="max", gate_policy="degrade")
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result()

    spec.run("task")

    assert spec.requested_effort == "max"
    assert client.agents.runs.create.call_args.kwargs["effort"] == "x-high"


@pytest.mark.parametrize("agent_id", [AGENT_ID, None])
def test_gated_effort_is_re_checked_at_dispatch(monkeypatch, fake_clock, agent_id):
    """A gated tier assigned after construction still never reaches the wire.

    The gate resolves in __init__, but `effort` is a plain public attribute:
    without a check at the point that actually spends money, `spec.effort =
    "max"` would send a gated tier silently — the one outcome the policy
    exists to prevent.
    """
    spec, client, _ = _spec(monkeypatch, agent_id=agent_id)
    spec.effort = "max"

    with pytest.raises(ValueError) as captured:
        spec.run("task")

    assert "coming-soon" in str(captured.value)
    assert "nothing was sent and no run was created" in str(captured.value)
    client.agents.runs.create.assert_not_called()
    client.agents.run.assert_not_called()


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
    ],
)
def test_non_retryable_error_while_polling_is_wrapped(
    monkeypatch, fake_clock, error_cls, status_code
):
    # Auth/permission/validation failures are terminal: wrap immediately as a
    # protocol error retaining run_id, never spend the budget re-attempting.
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.side_effect = _api_error(error_cls, status_code)

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert excinfo.value.run_id == RUN_ID
    assert isinstance(excinfo.value.__cause__, error_cls)
    assert client.agents.runs.get.call_count == 1


@pytest.mark.parametrize(
    ("error_cls", "status_code"),
    [
        (RateLimitError, 429),
        (InternalServerError, 500),
    ],
)
def test_transient_error_while_polling_is_absorbed_then_bounded(
    monkeypatch, fake_clock, error_cls, status_code
):
    # Transients are re-attempted (the SDK's own retry loop is disabled, so
    # this tool owns them) but only inside the deadline — never past it.
    spec, client, _ = _spec(monkeypatch, timeout=10.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.side_effect = _api_error(error_cls, status_code)

    with pytest.raises(NimbleAgentTimeoutError) as excinfo:
        spec.run("task")

    assert excinfo.value.run_id == RUN_ID
    assert isinstance(excinfo.value.__cause__, error_cls)
    assert client.agents.runs.get.call_count > 1  # genuinely re-attempted
    assert fake_clock.now <= 10.0  # and still inside the configured bound


def test_auth_error_on_create_propagates_unwrapped(monkeypatch, fake_clock):
    # Before a run exists there is no run_id to retain; the SDK error is the
    # most informative thing to surface.
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.side_effect = _api_error(AuthenticationError, 401)

    with pytest.raises(AuthenticationError):
        spec.run("task")


# ------------------------------------------------------- deadline bounding


def _stalling(fake_clock):
    """A call that hangs for exactly the timeout it was given, then fails.

    This is what an HTTP client does with a stalled request: block for the
    per-request timeout, then raise. Driving the fake clock by the *passed*
    timeout is what makes these tests discriminating — if the tool stopped
    passing the remaining budget, the stall would fall back to the SDK's own
    per-request timeout and overrun the bound.
    """

    def _call(*args, timeout, **kwargs):
        # httpx blocks for the read phase of the budget it was handed.
        fake_clock.now += timeout.read
        raise APITimeoutError(request=httpx.Request("GET", "https://api.test/v2"))

    return _call


def test_stalled_create_cannot_outlive_the_deadline(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch, timeout=10.0, poll_interval=2.0)
    client.agents.runs.create.side_effect = _stalling(fake_clock)

    # No run exists yet, so the SDK error surfaces unwrapped (existing
    # contract) — the point here is that it happens *within* the bound.
    with pytest.raises(APITimeoutError):
        spec.run("task")

    assert fake_clock.now <= 10.0


def test_stalled_poll_cannot_outlive_the_deadline(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch, timeout=10.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.side_effect = _stalling(fake_clock)

    with pytest.raises(NimbleAgentTimeoutError) as excinfo:
        spec.run("task")

    assert excinfo.value.run_id == RUN_ID
    assert isinstance(excinfo.value.__cause__, APITimeoutError)
    # The whole call — create, sleep, and the stalled poll — fits the bound.
    assert fake_clock.now <= 10.0
    client.agents.runs.result.assert_not_called()


def test_stalled_result_cannot_outlive_the_deadline(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch, timeout=10.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.side_effect = _stalling(fake_clock)

    with pytest.raises(NimbleAgentTimeoutError) as excinfo:
        spec.run("task")

    assert excinfo.value.run_id == RUN_ID
    assert excinfo.value.status == "completed"
    assert fake_clock.now <= 10.0


def test_every_request_carries_the_remaining_budget(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch, timeout=10.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.return_value = _run("completed")
    client.agents.runs.result.return_value = _text_result()

    spec.run("task")

    # create at t=0 → 10s left; get after one 2s sleep → 8s; result → 8s.
    assert client.agents.runs.create.call_args.kwargs["timeout"].read == pytest.approx(
        10.0
    )
    assert client.agents.runs.get.call_args.kwargs["timeout"].read == pytest.approx(8.0)
    assert client.agents.runs.result.call_args.kwargs["timeout"].read == pytest.approx(
        8.0
    )


def test_request_timeout_never_goes_non_positive(monkeypatch, fake_clock):
    # A budget consumed to the millisecond must still yield a timeout the
    # HTTP client accepts (it then fails fast) rather than 0 or negative.
    spec, client, _ = _spec(monkeypatch, timeout=4.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.side_effect = _stalling(fake_clock)

    with pytest.raises(NimbleAgentTimeoutError):
        spec.run("task")

    for call in client.agents.runs.get.call_args_list:
        assert call.kwargs["timeout"].read > 0
        assert call.kwargs["timeout"].connect > 0


def test_retry_after_is_honored_within_the_budget(monkeypatch, fake_clock):
    # The SDK honored Retry-After before its retry loop was disabled here;
    # a rate-limited caller must still back off as the server asks.
    spec, client, _ = _spec(monkeypatch, timeout=60.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("queued")
    request = httpx.Request("GET", "https://api.test/v2")
    throttled = httpx.Response(429, request=request, headers={"retry-after": "9"})
    client.agents.runs.get.side_effect = [
        RateLimitError("slow down", response=throttled, body=None),
        _run("completed"),
    ]
    client.agents.runs.result.return_value = _text_result()

    spec.run("task")

    # 2s poll interval, then a 9s back-off as instructed (not another 2s).
    assert fake_clock.sleeps == [2.0, 9.0]


def test_retry_after_is_clamped_to_the_remaining_budget(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch, timeout=6.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("queued")
    request = httpx.Request("GET", "https://api.test/v2")
    throttled = httpx.Response(429, request=request, headers={"retry-after": "30"})
    client.agents.runs.get.side_effect = RateLimitError(
        "slow down", response=throttled, body=None
    )

    with pytest.raises(NimbleAgentTimeoutError):
        spec.run("task")

    # 2s poll interval, then the 30s the server asked for, clamped to the 4s
    # actually left — a fixed-interval retry would have slept [2, 2, 2].
    assert fake_clock.sleeps == [2.0, 4.0]
    assert fake_clock.now == 6.0


def test_implausible_retry_after_falls_back_to_backoff(monkeypatch, fake_clock):
    # The SDK ignored a Retry-After above 60s rather than trusting it; a
    # server (or proxy) asking for 10 minutes must not park the caller.
    spec, client, _ = _spec(monkeypatch, timeout=60.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("queued")
    request = httpx.Request("GET", "https://api.test/v2")
    throttled = httpx.Response(429, request=request, headers={"retry-after": "600"})
    client.agents.runs.get.side_effect = [
        RateLimitError("slow down", response=throttled, body=None),
        _run("completed"),
    ]
    client.agents.runs.result.return_value = _text_result()

    spec.run("task")

    assert fake_clock.sleeps == [2.0, 2.0]  # back-off, not 600s


def test_malformed_retry_after_falls_back_to_poll_interval(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch, timeout=60.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("queued")
    request = httpx.Request("GET", "https://api.test/v2")
    # HTTP-date form: valid HTTP, not a float — must not crash the poller.
    throttled = httpx.Response(
        429, request=request, headers={"retry-after": "Wed, 23 Jul 2026 12:00:00 GMT"}
    )
    client.agents.runs.get.side_effect = [
        RateLimitError("slow down", response=throttled, body=None),
        _run("completed"),
    ]
    client.agents.runs.result.return_value = _text_result()

    spec.run("task")

    assert fake_clock.sleeps == [2.0, 2.0]


def test_sdk_internal_retries_are_disabled(monkeypatch):
    # The SDK's own retry loop would issue several requests per call, each
    # with its own timeout, and could outlive the overall deadline; this tool
    # owns retries instead. It is also the load-bearing half of C03: without
    # max_retries=0 the SDK would issue up to three POSTs *inside* a single
    # create call, which no assertion on this adapter's call count could see.
    _, client, _ = _spec(monkeypatch)
    client.with_options.assert_called_once_with(max_retries=0)


@pytest.mark.parametrize("agent_id", [AGENT_ID, None])
@pytest.mark.parametrize(
    ("error_cls", "status_code"),
    [
        (APITimeoutError, None),  # transport failure
        (APIConnectionError, None),  # transport failure
        (APIStatusError, 408),
        (ConflictError, 409),
        (RateLimitError, 429),
        (InternalServerError, 500),
        (InternalServerError, 503),
    ],
)
def test_create_is_never_retried(
    monkeypatch, fake_clock, error_cls, status_code, agent_id
):
    """C03 — one POST per run(), whatever the failure, on either route.

    Creating a run is a non-idempotent, billable POST with no idempotency
    key. A failure response is not evidence the backend declined the work: a
    proxy that fails *after* acceptance would leave an orphaned billable run
    that this caller cannot even see, because a failed create surfaces no run
    id. So the failure propagates on the first attempt, every time.
    """
    spec, client, _ = _spec(
        monkeypatch, timeout=300.0, poll_interval=2.0, agent_id=agent_id
    )
    request = httpx.Request("POST", "https://api.test/v2")
    if status_code is None:
        failure = (
            APITimeoutError(request=request)
            if error_cls is APITimeoutError
            else APIConnectionError(request=request)
        )
    else:
        failure = _api_error(error_cls, status_code)
    create = client.agents.runs.create if agent_id else client.agents.run
    create.side_effect = failure

    with pytest.raises(type(failure)):
        spec.run("task")

    assert create.call_count == 1
    assert fake_clock.sleeps == []  # not even a back-off before giving up


def test_poll_attempts_are_capped(monkeypatch, fake_clock):
    # A persistently failing endpoint must not be re-hit for the whole budget.
    spec, client, _ = _spec(monkeypatch, timeout=300.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.side_effect = _api_error(InternalServerError, 500)

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert client.agents.runs.get.call_count == 5
    assert excinfo.value.run_id == RUN_ID
    assert fake_clock.now < 300.0


def test_backoff_is_exponential_and_capped(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch, timeout=300.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.side_effect = _api_error(InternalServerError, 500)

    with pytest.raises(NimbleAgentProtocolError):
        spec.run("task")

    # first the 2s poll interval, then 2, 4, 8, 8 — doubling, capped at 8.
    assert fake_clock.sleeps == [2.0, 2.0, 4.0, 8.0, 8.0]


def test_server_can_veto_a_retry_with_x_should_retry(monkeypatch, fake_clock):
    # A 500 is normally retryable; the server's explicit "false" wins.
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("queued")
    request = httpx.Request("GET", "https://api.test/v2")
    response = httpx.Response(500, request=request, headers={"x-should-retry": "false"})
    client.agents.runs.get.side_effect = InternalServerError(
        "no retry", response=response, body=None
    )

    with pytest.raises(NimbleAgentProtocolError):
        spec.run("task")

    assert client.agents.runs.get.call_count == 1


def test_conflict_on_result_is_retried_then_reported(monkeypatch, fake_clock):
    """A 409 in the read-after-write window is worth one more look.

    Discarding the output of a multi-minute billable run because its result
    was not readable on the first attempt is the expensive mistake here.
    """
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.side_effect = [
        _api_error(ConflictError, 409),
        _text_result(),
    ]

    doc = spec.run("task")

    assert client.agents.runs.result.call_count == 2
    assert doc.text.startswith("The final answer.")


def test_retry_after_ms_is_honored(monkeypatch, fake_clock):
    spec, client, _ = _spec(monkeypatch, timeout=60.0, poll_interval=2.0)
    client.agents.runs.create.return_value = _created("queued")
    request = httpx.Request("GET", "https://api.test/v2")
    throttled = httpx.Response(429, request=request, headers={"retry-after-ms": "4500"})
    client.agents.runs.get.side_effect = [
        RateLimitError("slow down", response=throttled, body=None),
        _run("completed"),
    ]
    client.agents.runs.result.return_value = _text_result()

    spec.run("task")

    assert fake_clock.sleeps == [2.0, 4.5]


def test_request_timeout_keeps_a_tight_connect_ceiling(monkeypatch, fake_clock):
    # A bare float would give every phase the whole budget; a black-holed
    # connection must fail fast instead.
    spec, client, _ = _spec(monkeypatch, timeout=300.0)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result()

    spec.run("task")

    sent = client.agents.runs.create.call_args.kwargs["timeout"]
    assert sent.read == pytest.approx(300.0)
    assert sent.connect == pytest.approx(5.0)


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
    assert meta["web_search_agent_id"] == AGENT_ID
    assert meta["effort"] == "low"
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


def test_completed_run_without_trust_is_protocol_error(monkeypatch, fake_clock):
    # A completed result whose output carries no trust block is a contract
    # violation: it must surface as a typed protocol error retaining run
    # context, never a bare AttributeError from dereferencing trust.
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    trustless = TaskRunResultPublicV2.model_construct(
        run=RunGetResponse.model_validate(_run_payload("completed")),
        output=SimpleNamespace(content="answer", trust=None),
    )
    client.agents.runs.result.return_value = trustless

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert excinfo.value.run_id == RUN_ID
    assert "no output payload" in str(excinfo.value)


def test_empty_content_without_sources_still_yields_non_empty_text(
    monkeypatch, fake_clock
):
    # A run can complete with blank content and no sources; the Document must
    # never have empty text (which breaks downstream nodes), and the fallback
    # keeps the run id so the result stays traceable.
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result(
        content="   ", sources=[], claims=[]
    )

    doc = spec.run("task")

    assert doc.text.strip()
    assert RUN_ID in doc.text


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
    # LlamaIndex's docstring parser requires the `name (type): desc` form AND
    # captures only the first physical line — so each param's first line must
    # be a complete, self-contained description or the schema text a
    # function-calling model reads ends up truncated mid-sentence.
    spec, _, _ = _spec(monkeypatch)
    tool = next(t for t in spec.to_tool_list() if t.metadata.name == "run")
    fields = tool.metadata.fn_schema.model_fields

    task_desc = fields["task"].description
    assert task_desc and "research" in task_desc.lower()
    assert task_desc.rstrip().endswith(".")

    for name in ("output_schema", "input_data", "sources"):
        desc = fields[name].description
        assert desc and desc.rstrip().endswith(".")
        assert fields[name].default is None  # optional for the model


@pytest.mark.parametrize("bad_task", ["", "   "])
def test_task_must_be_non_empty(monkeypatch, bad_task):
    spec, client, _ = _spec(monkeypatch)
    with pytest.raises(ValueError):
        spec.run(bad_task)
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


# ------------------------------------------------------- C05/C06: identity


def test_generated_identity_is_used_for_status_and_result(monkeypatch, fake_clock):
    """C05 — the returned agent id, not the configured one, drives the run.

    A generated agent's id exists only in the create response, so dropping it
    strands the run: nothing else can address the poll or result call.
    """
    spec, client, _ = _spec(monkeypatch, agent_id=None)
    client.agents.run.return_value = _created(
        "queued", web_search_agent_id=GENERATED_AGENT_ID
    )
    client.agents.runs.get.return_value = _run(
        "completed", web_search_agent_id=GENERATED_AGENT_ID
    )
    client.agents.runs.result.return_value = _text_result(
        run_overrides={"web_search_agent_id": GENERATED_AGENT_ID}
    )

    doc = spec.run("task")

    client.agents.runs.get.assert_called_with(
        RUN_ID, agent_id=GENERATED_AGENT_ID, timeout=ANY
    )
    client.agents.runs.result.assert_called_once_with(
        RUN_ID, agent_id=GENERATED_AGENT_ID, timeout=ANY
    )
    # …and it survives into the Document, so the run stays reachable later.
    assert doc.metadata["run_id"] == RUN_ID
    assert doc.metadata["web_search_agent_id"] == GENERATED_AGENT_ID


def test_generated_identity_is_retained_in_errors(monkeypatch, fake_clock):
    """C05 — a failed generated run is still traceable by its returned pair."""
    spec, client, _ = _spec(monkeypatch, agent_id=None)
    client.agents.run.return_value = _created(
        "failed", web_search_agent_id=GENERATED_AGENT_ID
    )

    with pytest.raises(NimbleAgentRunFailedError) as excinfo:
        spec.run("task")

    assert excinfo.value.agent_id == GENERATED_AGENT_ID
    assert excinfo.value.run_id == RUN_ID


def test_create_without_agent_identity_is_a_protocol_error(monkeypatch, fake_clock):
    """C05 — a create response with no agent id cannot be polled at all."""
    spec, client, _ = _spec(monkeypatch, agent_id=None)
    client.agents.run.return_value = RunCreateResponse.model_construct(
        **{**_run_payload("queued"), "web_search_agent_id": None}
    )

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert "web_search_agent_id" in str(excinfo.value)
    client.agents.runs.get.assert_not_called()


def test_create_on_a_different_agent_is_rejected(monkeypatch, fake_clock):
    """C06 — never silently substitute the configured id for the returned one.

    Papering over the mismatch would poll an agent the run does not belong
    to: a 404 at best, another run's answer at worst.
    """
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created(
        "queued", web_search_agent_id=GENERATED_AGENT_ID
    )

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert GENERATED_AGENT_ID in str(excinfo.value)
    assert excinfo.value.agent_id == AGENT_ID
    client.agents.runs.get.assert_not_called()


def test_poll_response_for_another_owner_is_rejected(monkeypatch, fake_clock):
    """C06 — a status response reporting a different pair is not this run."""
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.return_value = _run(
        "completed", web_search_agent_id=GENERATED_AGENT_ID
    )

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert "identity mismatch" in str(excinfo.value)
    client.agents.runs.result.assert_not_called()


@pytest.mark.parametrize("missing", ["id", "web_search_agent_id"])
def test_poll_response_without_identity_is_rejected(monkeypatch, fake_clock, missing):
    """C05/C06 — an unverifiable poll envelope is not "no contradiction".

    Absence is the case this guard exists for: a response that names neither
    the run nor the agent cannot be shown to be this run's, so accepting it
    would let the exact routing failure being guarded against pass silently.
    """
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("queued")
    client.agents.runs.get.return_value = RunGetResponse.model_construct(
        **{**_run_payload("completed"), missing: None}
    )

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert "identity mismatch" in str(excinfo.value)
    assert excinfo.value.run_id == RUN_ID
    client.agents.runs.result.assert_not_called()


@pytest.mark.parametrize("missing", ["id", "web_search_agent_id"])
def test_result_without_identity_is_rejected(monkeypatch, fake_clock, missing):
    """C05/C06 — same positive check on the result envelope."""
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    result = _text_result()
    setattr(result.run, missing, None)
    client.agents.runs.result.return_value = result

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert "identity mismatch" in str(excinfo.value)


def test_check_owner_requires_both_ids_present(monkeypatch):
    """C05/C06 — the guard itself, asserted directly on the absence case."""
    spec, _, _ = _spec(monkeypatch)

    with pytest.raises(NimbleAgentProtocolError):
        spec._check_owner(SimpleNamespace(), RUN_ID, AGENT_ID)

    # the fully-identified case still passes
    spec._check_owner(
        SimpleNamespace(id=RUN_ID, web_search_agent_id=AGENT_ID), RUN_ID, AGENT_ID
    )


@pytest.mark.parametrize(
    "run_overrides",
    [
        {"id": "task_run_someone-else"},
        {"web_search_agent_id": GENERATED_AGENT_ID},
    ],
)
def test_result_for_another_owner_is_rejected(monkeypatch, fake_clock, run_overrides):
    """C06 — a result envelope for a different pair must not be mapped.

    Mapping it would attribute another run's answer, and its citations, to
    this call — the one failure mode a trust-carrying Document must not have.
    """
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result(run_overrides=run_overrides)

    with pytest.raises(NimbleAgentProtocolError) as excinfo:
        spec.run("task")

    assert "identity mismatch" in str(excinfo.value)


# ------------------------------------------- C07/C08/C09/C10: run controls


_SCHEMA = {"type": "object", "properties": {"founded": {"type": "integer"}}}
_ROWS = [{"company": "Acme", "domain": "acme.test"}]
_SOURCES = {
    "allow": [{"url": "https://example.org"}],
    "block": [{"url": "https://spam.test"}],
    "prioritize": "regulatory filings",
    "avoid": "press releases",
}


@pytest.mark.parametrize("agent_id", [AGENT_ID, None])
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # C07 research controls, C08 enrichment controls, C09 dataset controls
        ({"output_schema": _SCHEMA, "sources": _SOURCES}, ["output_schema", "sources"]),
        (
            {"input_data": _ROWS, "output_schema": _SCHEMA, "sources": _SOURCES},
            ["input_data", "output_schema", "sources"],
        ),
        ({"output_schema": _SCHEMA}, ["output_schema"]),
        ({"input_data": _ROWS[0]}, ["input_data"]),  # single object, not a list
    ],
)
def test_structured_controls_pass_through_unchanged(
    monkeypatch, fake_clock, kwargs, expected, agent_id
):
    """Controls reach the create body byte-identical, on both routes.

    Rewriting a caller's JSON Schema or source guidance would silently change
    what the run is asked to produce, so they are validated and forwarded, not
    normalized.
    """
    spec, client, _ = _spec(monkeypatch, agent_id=agent_id)
    created = _created("completed", web_search_agent_id=agent_id or GENERATED_AGENT_ID)
    client.agents.runs.create.return_value = created
    client.agents.run.return_value = created
    client.agents.runs.result.return_value = _text_result(
        run_overrides={"web_search_agent_id": agent_id or GENERATED_AGENT_ID}
    )

    spec.run("task", **kwargs)

    create = client.agents.runs.create if agent_id else client.agents.run
    sent = create.call_args.kwargs
    for key in expected:
        assert sent[key] == kwargs[key]
    # unset controls are omitted entirely, not sent as null
    assert set(sent) - {"input", "effort", "timeout"} == set(expected)


@pytest.mark.parametrize("agent_id", [AGENT_ID, None])
def test_sdk_12_typed_run_fields_reach_both_create_routes(
    monkeypatch, fake_clock, agent_id
):
    """Released SDK 1.2 fields are typed, forwarded, and never use extra_body."""
    spec, client, _ = _spec(
        monkeypatch,
        agent_id=agent_id,
        agent_name="market-research",
        skill="competitive-intelligence",
        use_case="research",
    )
    created = _created("completed", web_search_agent_id=agent_id or GENERATED_AGENT_ID)
    client.agents.runs.create.return_value = created
    client.agents.run.return_value = created
    client.agents.runs.result.return_value = _text_result(
        run_overrides={"web_search_agent_id": agent_id or GENERATED_AGENT_ID}
    )

    spec.run("task", output_schema=_SCHEMA, input_data=_ROWS, sources=_SOURCES)

    create = client.agents.runs.create if agent_id else client.agents.run
    sent = create.call_args.kwargs
    assert sent["agent_name"] == "market-research"
    assert sent["skill"] == "competitive-intelligence"
    assert sent["use_case"] == "research"
    assert "extra_body" not in sent
    assert set(sent) <= ALLOWED_CREATE_KEYS


def test_unsupported_source_keys_are_rejected(monkeypatch):
    """C10 — the closed source-key set blocks smuggling an unpublished field."""
    spec, client, _ = _spec(monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        spec.run("task", sources={"allow": [], "skill": "deep_research"})

    assert "skill" in str(excinfo.value)
    client.agents.runs.create.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_schema": "not-a-dict"},
        {"output_schema": [{"type": "object"}]},
        {"input_data": "not-a-row"},
        {"input_data": []},
        {"input_data": ["not-a-row"]},
        {"sources": ["https://example.org"]},
    ],
)
def test_malformed_controls_are_rejected_before_any_billable_call(monkeypatch, kwargs):
    spec, client, _ = _spec(monkeypatch)

    with pytest.raises(ValueError):
        spec.run("task", **kwargs)

    client.agents.runs.create.assert_not_called()
    client.agents.run.assert_not_called()


def test_empty_sources_object_is_omitted(monkeypatch, fake_clock):
    # An LLM filling an optional object often emits `{}`; sending it adds a
    # meaningless key to a billable request rather than expressing anything.
    spec, client, _ = _spec(monkeypatch)
    client.agents.runs.create.return_value = _created("completed")
    client.agents.runs.result.return_value = _text_result()

    spec.run("task", sources={})

    assert "sources" not in client.agents.runs.create.call_args.kwargs


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


def test_generic_route_binds_real_sdk_signatures(monkeypatch, fake_clock):
    """C01/C04/C05/C10 asserted on the wire, not on a mock's call log.

    A MagicMock accepts any call shape, so only the real client proves that
    ``agents.run`` exists with these kwargs, that the generic path is
    ``POST /v2/agents/runs``, and that the generated agent id is what the
    follow-up URLs are built from.
    """
    import nimble_python
    from nimble_python import Nimble as RealNimble

    seen: list[tuple[str, str]] = []
    posted: dict = {}
    generated = _run_payload("queued", web_search_agent_id=GENERATED_AGENT_ID)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.method == "POST":
            posted.update(json.loads(request.content))
            return httpx.Response(202, json=generated)
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "run": _run_payload(
                        "completed", web_search_agent_id=GENERATED_AGENT_ID
                    ),
                    "output": {
                        "type": "json",
                        "content": {"founded": 1999},
                        "trust": _trust_payload(claims=[]),
                    },
                },
            )
        return httpx.Response(
            200, json=_run_payload("completed", web_search_agent_id=GENERATED_AGENT_ID)
        )

    monkeypatch.setattr(
        nimble_python,
        "Nimble",
        lambda **kw: RealNimble(
            **kw, http_client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )
    spec = NimbleAgentToolSpec(api_key="test-key")

    doc = spec.run("wire task", output_schema=_SCHEMA, sources=_SOURCES)

    assert seen == [
        ("POST", "/v2/agents/runs"),
        ("GET", f"/v2/agents/{GENERATED_AGENT_ID}/runs/{RUN_ID}"),
        ("GET", f"/v2/agents/{GENERATED_AGENT_ID}/runs/{RUN_ID}/result"),
    ]
    assert "effort" not in posted
    assert posted["output_schema"] == _SCHEMA
    assert posted["sources"] == _SOURCES
    assert set(posted) <= ALLOWED_CREATE_KEYS - {"timeout"}
    assert doc.metadata["web_search_agent_id"] == GENERATED_AGENT_ID


def test_remaining_budget_reaches_the_wire(monkeypatch, fake_clock):
    """The deadline is enforced by the HTTP client, not just by our arithmetic.

    Drives the real Nimble client over a mocked transport and reads the
    timeout httpx actually resolved for each outgoing request, proving the
    remaining budget is what bounds every call — the guarantee the tool's
    `timeout` argument advertises.
    """
    import nimble_python
    from nimble_python import Nimble as RealNimble

    read_timeouts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        read_timeouts.append(request.extensions["timeout"]["read"])
        if request.method == "POST":
            return httpx.Response(202, json=_run_payload("queued"))
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "run": _run_payload("completed"),
                    "output": {
                        "type": "text",
                        "content": "bounded",
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
    spec = NimbleAgentToolSpec(
        agent_id=AGENT_ID, api_key="test-key", timeout=30.0, poll_interval=2.0
    )

    spec.run("bounded task")

    # create at t=0 → 30s left; one 2s poll interval elapses before the poll
    # and the result fetch, so both are bounded by the 28s that remain.
    assert read_timeouts == pytest.approx([30.0, 28.0, 28.0])
    # …and the client actually used is the retry-disabled one, not just a
    # discarded copy (asserting on the real object, not the mock's call log).
    assert spec.client.max_retries == 0
