"""Test LLMClient fallback rotation across multi-endpoint chains.

Verifies CC-design fix: range = len(base_urls) * (1 + _MAX_RETRIES),
direct endpoint indexing (no NameError-prone if/else fallback),
correct 401/403 break, and successful primary-endpoint path.
"""

from __future__ import annotations

import httpx
import pytest
from unittest.mock import AsyncMock, Mock, patch

from core.llm_client import LLMClient, _MAX_RETRIES


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_BASE_URLS = [
    "https://api.deepseek.com",
    "https://api.openai.com",
    "https://api.moonshot.cn",
    "https://openrouter.ai/api",
]
_ENDPOINT_COUNT = len(_BASE_URLS)
_RETRIES_PER = 1 + _MAX_RETRIES  # 3


def _make_mock_async_client_factory(call_sequence: list[str]):
    """Return a side-effect callable that replaces ``httpx.AsyncClient(...)``.

    Every constructed instance exposes:
    - ``.base_url`` → the rstrip'd endpoint
    - ``.aclose`` → no-op AsyncMock
    - ``.post`` → AsyncMock chained to the shared ``call_sequence`` list

    The *behaviour* of ``.post`` is set on each instance by the test (via
    ``instance.post.side_effect = ...``) because each test needs different
    success / failure sequences.
    """

    def _factory(*args, **kwargs):
        base_url = kwargs.get("base_url", "")
        inst = Mock()
        inst.base_url = base_url
        inst.aclose = AsyncMock()

        async def _post_tracker(path, json=None):
            call_sequence.append(base_url)

        # default: a no-op tracker; tests override side_effect
        inst.post = AsyncMock(side_effect=_post_tracker)
        return inst

    return _factory


def _ok_response(content: str = "test content") -> Mock:
    """Return a Mock httpx.Response that passes raise_for_status + json."""
    resp = Mock()
    resp.status_code = 200
    resp.raise_for_status = Mock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _http_error(status_code: int, text: str = "error") -> httpx.HTTPStatusError:
    """Build a minimal httpx.HTTPStatusError for exception paths."""
    req = Mock()
    resp = Mock()
    resp.status_code = status_code
    resp.text = text
    return httpx.HTTPStatusError(f"HTTP {status_code}", request=req, response=resp)


# ---------------------------------------------------------------------------
# Test 1 — full 10-call fallback sequence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_fallback_sequence():
    """First 9 post calls fail (timeout / 500), 10th succeeds on openrouter.

    Expected call round-robin:
    - deepseek  ×3  (attempts 0-2)
    - openai    ×3  (attempts 3-5)
    - moonshot  ×3  (attempts 6-8)
    - openrouter ×1 → success (attempt 9)
    ─────────────────
    Total: 10 post calls
    """
    call_sequence: list[str] = []

    factory = _make_mock_async_client_factory(call_sequence)

    with patch("core.llm_client.httpx.AsyncClient", side_effect=factory):
        client = LLMClient(api_key="dummy", base_url=_BASE_URLS[0])

        # Replace *the first* instance's post with our scenario.
        # (Endpoint-switch creates new instances — we'll intercept them below.)

        call_count = [0]

        async def _scenario_post(path, json=None):
            call_count[0] += 1
            call_sequence.append(client._client.base_url)
            if call_count[0] <= 9:
                # alternate timeout and HTTP 500
                if call_count[0] % 2 == 0:
                    raise httpx.TimeoutException("timeout")
                else:
                    raise _http_error(500, "internal server error")
            # 10th call → success
            return _ok_response("fallback success")

        # Patch the first client
        client._client.post = AsyncMock(side_effect=_scenario_post)

        # Watch for new client creation → attach same scenario to new instances
    
        def _intercept_new_client(*args, **kwargs):
            """When chat() creates a new httpx.AsyncClient, inject our scenario."""
            inst = Mock()
            inst.base_url = kwargs.get("base_url", "")
            inst.aclose = AsyncMock()
            inst.post = AsyncMock(side_effect=_scenario_post)
            return inst

        with patch(
            "core.llm_client.httpx.AsyncClient", side_effect=_intercept_new_client
        ):
            result = await client.chat(
                [{"role": "user", "content": "hello"}]
            )

    # Did we get a result?
    assert result == "fallback success"

    # Deduplicate: call_sequence is append-only and mutual; each post appends
    # once via _scenario_post.  The factory's _post_tracker is overridden.
    # We should see exactly 10 entries.
    assert len(call_sequence) == 10, f"Expected 10 calls, got {len(call_sequence)}"

    # Verify round-robin: 3 per endpoint, and openrouter reached.
    assert call_sequence[0:3] == [_BASE_URLS[0]] * 3  # deepseek ×3
    assert call_sequence[3:6] == [_BASE_URLS[1]] * 3  # openai   ×3
    assert call_sequence[6:9] == [_BASE_URLS[2]] * 3  # moonshot ×3
    assert call_sequence[9:10] == [_BASE_URLS[3]] * 1  # openrouter ×1


# ---------------------------------------------------------------------------
# Test 2 — 401/403 breaks immediately (no fallback)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_401_breaks_no_retry():
    """Primary endpoint returns 401 → break, exactly 1 post call."""
    call_sequence: list[str] = []
    factory = _make_mock_async_client_factory(call_sequence)

    with patch("core.llm_client.httpx.AsyncClient", side_effect=factory):
        client = LLMClient(api_key="dummy", base_url=_BASE_URLS[0])

    async def _post_401(path, json=None):
        call_sequence.append(client._client.base_url)
        raise _http_error(401, "Unauthorized")

    client._client.post = AsyncMock(side_effect=_post_401)

    result = await client.chat([{"role": "user", "content": "hello"}])

    assert result is None
    assert len(call_sequence) == 1
    assert call_sequence[0] == _BASE_URLS[0]


@pytest.mark.asyncio
async def test_403_breaks_no_retry():
    """Primary endpoint returns 403 → break, exactly 1 post call."""
    call_sequence: list[str] = []
    factory = _make_mock_async_client_factory(call_sequence)

    with patch("core.llm_client.httpx.AsyncClient", side_effect=factory):
        client = LLMClient(api_key="dummy", base_url=_BASE_URLS[0])

    async def _post_403(path, json=None):
        call_sequence.append(client._client.base_url)
        raise _http_error(403, "Forbidden")

    client._client.post = AsyncMock(side_effect=_post_403)

    result = await client.chat([{"role": "user", "content": "hello"}])

    assert result is None
    assert len(call_sequence) == 1
    assert call_sequence[0] == _BASE_URLS[0]


# ---------------------------------------------------------------------------
# Test 3 — primary endpoint succeeds on first try
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_primary_endpoint_succeeds_immediately():
    """Primary endpoint returns content → exactly 1 post call, returns content."""
    call_sequence: list[str] = []
    factory = _make_mock_async_client_factory(call_sequence)

    with patch("core.llm_client.httpx.AsyncClient", side_effect=factory):
        client = LLMClient(api_key="dummy", base_url=_BASE_URLS[0])

    async def _post_ok(path, json=None):
        call_sequence.append(client._client.base_url)
        return _ok_response("hello world")

    client._client.post = AsyncMock(side_effect=_post_ok)

    result = await client.chat([{"role": "user", "content": "hi"}])

    assert result == "hello world"
    assert len(call_sequence) == 1
    assert call_sequence[0] == _BASE_URLS[0]
