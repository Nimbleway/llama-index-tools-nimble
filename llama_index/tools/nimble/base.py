"""Nimble Web Search tool spec for LlamaIndex."""

from typing import Any

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class NimbleToolSpec(BaseToolSpec):
    """Nimble Web Search tool spec.

    Wraps Nimble's Web Search API (``POST /v1/search``) and exposes it to a
    LlamaIndex agent as a ``search`` tool. Construct with an API key, or set the
    ``NIMBLE_API_KEY`` environment variable.
    """

    spec_functions = ["search"]

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the tool spec.

        Args:
            api_key: Nimble API key. If omitted, the SDK reads ``NIMBLE_API_KEY``
                from the environment.
        """
        from nimble_python import Nimble

        self.client = Nimble(
            api_key=api_key,
            default_headers={"X-Client-Source": "llama-index-tools-nimble"},
        )

    def search(self, query: str, max_results: int = 6) -> list[Document]:
        """Search the web with Nimble and return the results as Documents.

        Args:
            query: The search query.
            max_results: The maximum number of results to return.

        Returns:
            A list of Documents. Each Document's text leads with the result's
            title and source URL followed by the page content (or the snippet
            when content is empty); the url and title are also in its metadata.
        """
        # SDK errors (AuthenticationError / PermissionDeniedError / other APIError)
        # propagate to the caller: the agent framework surfaces them to the LLM, and a
        # direct caller gets the real, informative exception rather than a silent [].
        #
        # focus="general" + search_depth="lite" + output_format="markdown" are the
        # non-enterprise defaults. Passing focus explicitly keeps results on the broad
        # web/research mode; omitting it lets per-query auto-selection make relevance
        # erratic. The agent does not choose focus — v1 fixes it to general.
        response = self.client.search(
            query=query,
            max_results=max_results,
            focus="general",
            search_depth="lite",
            output_format="markdown",
        )
        return [self._to_document(result) for result in response.results]

    @staticmethod
    def _to_document(result: Any) -> Document:
        """Render one Nimble result as a Document.

        The title and URL are embedded in the text (not only metadata): an agent
        only sees a tool's stringified output, where Document metadata is dropped,
        so the source must live in the text for the model to read and cite it.
        'lite' depth can return empty content, so fall back to the snippet.
        """
        title = result.title or ""
        url = result.url or ""
        body = result.content or result.description or ""
        text = f"{title}\nURL: {url}\n\n{body}".strip()
        return Document(text=text, extra_info={"url": url, "title": title})
