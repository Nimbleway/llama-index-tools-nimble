"""Typed errors raised by :class:`~llama_index.tools.nimble.NimbleAgentToolSpec`.

Every error carries the identifiers needed to recover a run by hand
(``agent_id``, ``run_id``) both as attributes and inside ``str(exc)``: agent
frameworks usually surface only the message, while programmatic callers want
the attributes. Messages are built from safe parts only — never from the API
key or raw request state. The underlying SDK exception, when there is one, is
chained via ``__cause__``.
"""


class NimbleAgentRunError(Exception):
    """Base error for a Nimble Agent run that did not produce a result.

    Attributes:
        run_id: The ``task_run_...`` identifier of the run. ``None`` only
            when creation itself failed without returning one — see
            :class:`NimbleAgentCreateAmbiguousError`.
        agent_id: The ``wsa_...`` agent instance the run belongs to — the id
            the API returned for the run. ``None`` only when the run was
            created without a configured agent and the response carried no
            agent id, which is itself the failure being reported.
        status: The last run status observed, when one was observed.
    """

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None,
        agent_id: str | None,
        status: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.agent_id = agent_id
        self.status = status
        detail = f"agent_id={agent_id}, run_id={run_id}"
        if status is not None:
            detail += f", status={status}"
        super().__init__(f"{message} ({detail})")


class NimbleAgentCreateAmbiguousError(NimbleAgentRunError):
    """Run creation failed without saying whether a run actually started.

    A transport failure, a timeout, a 408/409, or a 5xx can all arrive
    *after* the backend accepted the request, so a billable run may be
    executing server-side even though this call raised. No id came back, so
    nothing here can address it.

    This is deliberately a distinct type rather than the raw SDK exception.
    The immediate caller is frequently a function-calling model, and a bare
    ``APITimeoutError`` reads to a model like any other transient — it calls
    the tool again and pays for a second run. The message says, in words a
    model will act on, not to resubmit automatically, and how to reconcile
    instead.

    ``run_id`` is always ``None``: its absence is the problem being reported.
    The originating SDK exception is chained via ``__cause__``, and
    ``status_code`` carries the HTTP status when there was one.
    """

    def __init__(
        self,
        message: str,
        *,
        agent_id: str | None,
        status_code: int | None = None,
    ) -> None:
        self.status_code = status_code
        super().__init__(message, run_id=None, agent_id=agent_id)


class NimbleAgentTimeoutError(NimbleAgentRunError):
    """The run did not reach a terminal status within the configured timeout.

    The run may still complete server-side; ``run_id`` can be used to poll or
    fetch it later.
    """


class NimbleAgentRunFailedError(NimbleAgentRunError):
    """The run reached the terminal status ``failed``."""


class NimbleAgentRunCancelledError(NimbleAgentRunError):
    """The run reached the terminal status ``cancelled``."""


class NimbleAgentProtocolError(NimbleAgentRunError):
    """The API behaved outside the documented run contract.

    Raised for an unknown run status, a completed run whose result payload is
    missing or malformed, or an SDK/HTTP error while polling or fetching the
    result (chained via ``__cause__``).
    """
