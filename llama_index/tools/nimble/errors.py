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
        run_id: The ``task_run_...`` identifier of the run.
        agent_id: The ``wsa_...`` agent instance the run belongs to.
        status: The last run status observed, when one was observed.
    """

    def __init__(
        self,
        message: str,
        *,
        run_id: str,
        agent_id: str,
        status: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.agent_id = agent_id
        self.status = status
        detail = f"agent_id={agent_id}, run_id={run_id}"
        if status is not None:
            detail += f", status={status}"
        super().__init__(f"{message} ({detail})")


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
