"""Unit tests for NimbleToolSpec. No API key or network required — the SDK is mocked."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from llama_index.core.schema import Document

from llama_index.tools.nimble import NimbleToolSpec


def _result(content="", description="", title="title", url="https://example.com"):
    """A stand-in for nimble_python's Result model (only the fields the tool reads)."""
    return SimpleNamespace(
        content=content, description=description, title=title, url=url
    )


def _spec_with_results(results, monkeypatch):
    """Build a NimbleToolSpec whose underlying Nimble client returns `results`."""
    import nimble_python

    client = MagicMock()
    client.search.return_value = SimpleNamespace(
        results=results, total_results=len(results)
    )
    monkeypatch.setattr(nimble_python, "Nimble", lambda **kwargs: client)
    return NimbleToolSpec(api_key="test-key"), client


def test_maps_results_to_documents(monkeypatch):
    spec, _ = _spec_with_results(
        [_result(content="body text", title="T1", url="https://a.com")], monkeypatch
    )
    docs = spec.search("q")
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    # text embeds title + url (so the agent can cite) then the body
    assert docs[0].text == "T1\nURL: https://a.com\n\nbody text"
    assert "https://a.com" in docs[0].text
    assert docs[0].metadata["url"] == "https://a.com"
    assert docs[0].metadata["title"] == "T1"


def test_falls_back_to_description_when_content_empty(monkeypatch):
    spec, _ = _spec_with_results(
        [_result(content="", description="snippet")], monkeypatch
    )
    docs = spec.search("q")
    # content empty -> body falls back to the description snippet
    assert docs[0].text.endswith("snippet")


def test_passes_query_max_results_and_default_params(monkeypatch):
    spec, client = _spec_with_results([], monkeypatch)
    spec.search("hello", max_results=3)
    # The non-enterprise defaults are sent on every call: fixing focus="general"
    # keeps results in broad web/research mode (omitting it makes relevance erratic).
    client.search.assert_called_once_with(
        query="hello",
        max_results=3,
        focus="general",
        search_depth="lite",
        output_format="markdown",
    )


def test_empty_results_returns_empty_list(monkeypatch):
    spec, _ = _spec_with_results([], monkeypatch)
    assert spec.search("q") == []


def test_to_tool_list_exposes_search(monkeypatch):
    spec, _ = _spec_with_results([], monkeypatch)
    tools = spec.to_tool_list()
    names = {t.metadata.name for t in tools}
    assert "search" in names


def test_to_tool_list_search_has_description(monkeypatch):
    spec, _ = _spec_with_results([], monkeypatch)
    tool = next(t for t in spec.to_tool_list() if t.metadata.name == "search")
    # the docstring is the description the LLM reads to decide how to call the tool
    assert tool.metadata.description


def test_empty_body_keeps_title_and_url(monkeypatch):
    spec, _ = _spec_with_results(
        [_result(content="", description="", title="T", url="https://x.com")],
        monkeypatch,
    )
    docs = spec.search("q")
    assert docs[0].text == "T\nURL: https://x.com"


def test_all_empty_fields_still_produce_non_empty_text(monkeypatch):
    spec, _ = _spec_with_results(
        [_result(content="", description="", title="", url="")], monkeypatch
    )
    docs = spec.search("q")
    # never empty: an empty Document.text breaks downstream nodes
    assert "URL:" in docs[0].text


def test_hard_caps_results_at_max_results(monkeypatch):
    # Nimble's max_results is a soft cap; the tool must enforce it client-side.
    results = [_result(content=f"body {i}", url=f"https://x/{i}") for i in range(10)]
    spec, _ = _spec_with_results(results, monkeypatch)
    assert len(spec.search("q", max_results=3)) == 3


@pytest.mark.parametrize("bad", [0, -1])
def test_max_results_must_be_positive(monkeypatch, bad):
    spec, _ = _spec_with_results([], monkeypatch)
    with pytest.raises(ValueError):
        spec.search("q", max_results=bad)


def test_init_without_api_key_does_not_raise(monkeypatch):
    import nimble_python

    monkeypatch.setattr(nimble_python, "Nimble", lambda **kwargs: MagicMock())
    spec = NimbleToolSpec()  # api_key=None -> SDK reads NIMBLE_API_KEY
    assert spec.client is not None
