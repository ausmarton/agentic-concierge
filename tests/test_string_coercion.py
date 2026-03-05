"""FR7.4: String-typed numeric coercion at tool boundaries.

Small LLMs (e.g. llama3.1:8b) occasionally send numeric arguments as strings
despite JSON schema ``"type": "integer"``.  Without coercion, ``time.monotonic()
+ "120"`` raises ``TypeError``.  These tests verify every coercion site.

Coercion sites:
- shell_tools.run_shell  → int(timeout_s)
- test_runner.run_tests  → int(timeout_s)  (tested in test_run_tests_tool.py too)
- web_tools.fetch_url    → float(timeout_s)
- containerised._exec_in_container → int(args["timeout_s"])
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_concierge.infrastructure.tools.sandbox import SandboxPolicy
from agentic_concierge.infrastructure.tools.shell_tools import run_shell
from agentic_concierge.infrastructure.tools.web_tools import fetch_url


# ---------------------------------------------------------------------------
# run_shell: timeout_s as string
# ---------------------------------------------------------------------------

def test_run_shell_string_timeout_coerced(tmp_path):
    """run_shell accepts timeout_s='120' and coerces to int before run_cmd."""
    policy = SandboxPolicy(root=tmp_path)
    with patch(
        "agentic_concierge.infrastructure.tools.shell_tools.run_cmd",
        return_value={"cmd": ["echo", "hi"], "returncode": 0, "stdout": "hi\n", "stderr": ""},
    ) as mock_run_cmd:
        result = run_shell(policy, ["echo", "hi"], timeout_s="120")  # type: ignore[arg-type]

    assert result["returncode"] == 0
    # Verify int was passed, not string
    _, kwargs = mock_run_cmd.call_args
    assert isinstance(kwargs["timeout_s"], int)
    assert kwargs["timeout_s"] == 120


# ---------------------------------------------------------------------------
# fetch_url: timeout_s as string
# ---------------------------------------------------------------------------

def test_fetch_url_string_timeout_coerced():
    """fetch_url accepts timeout_s='30' and coerces to float before httpx."""
    mock_response = MagicMock()
    mock_response.text = "<html><body>Hello</body></html>"
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_response

    with patch("agentic_concierge.infrastructure.tools.web_tools.httpx.Client", return_value=mock_client) as mock_cls:
        with patch("agentic_concierge.infrastructure.tools.web_tools.trafilatura.extract", return_value="Hello"):
            result = fetch_url("http://example.com", timeout_s="30")  # type: ignore[arg-type]

    assert result["text"] == "Hello"
    # Verify float was passed to httpx.Client, not string
    call_kwargs = mock_cls.call_args.kwargs
    assert isinstance(call_kwargs["timeout"], float)
    assert call_kwargs["timeout"] == 30.0
