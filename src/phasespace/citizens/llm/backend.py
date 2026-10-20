"""One LLM call: the headless Claude Code CLI, or any object with the same interface.

`ClaudeCLI` runs ``claude -p`` once per call, with

- the prompt on stdin and a fixed, short system prompt replacing Claude Code's own;
- the working directory set to a fresh, empty temporary directory (removed afterwards),
  so no project CLAUDE.md/AGENTS.md, settings or memory is discovered;
- the model and the effort level pinned (``--model``, ``--effort``; protocol Appendix A);
- every tool disabled (``--tools ""``), no settings sources, no MCP servers, no slash
  commands or skills, safe mode (no user customisations), one turn, no session saved;
- an environment built from an allowlist (`config.B4_ENV_ALLOWLIST`: PATH, HOME, USER,
  LANG, LC_ALL, TMPDIR, SHELL, TERM; protocol-validation A28), so no variable of the
  caller (an enclosing Claude Code session, API keys, tokens) reaches the call; the
  names (never values) of the variables passed are recorded per run. The CLI
  authenticates through the macOS keychain with these alone (checked 2026-10-20).

Each response's reported model usage (``modelUsage``) must name exactly the requested
model; anything else is an infrastructure error (``model_mismatch``, A26), as are
timeouts, non-zero exits, CLI/API errors and a missing result (A27): `infra_error`.

The exact argv is `ClaudeCLI.argv()` and is recorded in every run, with the CLI version
(``claude --version``).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Final, Protocol

from phasespace.citizens import config as cfg
from phasespace.citizens.llm.prompts import SYSTEM_PROMPT

DEFAULT_MODEL: Final = cfg.B4_MODEL
DEFAULT_EFFORT: Final = cfg.B4_EFFORT
TIMEOUT_S: Final = float(cfg.B4_TIMEOUT_S)
ENV_ALLOWLIST: Final = cfg.B4_ENV_ALLOWLIST


@dataclass(frozen=True, slots=True)
class CallResult:
    """One attempt: the raw CLI output and what was extracted from it."""

    raw_stdout: str
    raw_stderr: str
    exit_code: int | None  # None on timeout
    latency_s: float
    timed_out: bool
    result_text: str | None  # the JSON response's "result" field, if any
    is_error: bool
    cost_usd: float | None
    models: list[str] = field(default_factory=list[str])  # keys of "modelUsage"
    usage: dict[str, Any] | None = None
    session_id: str | None = None
    duration_ms: float | None = None
    infra_error: str | None = None  # A27: set when no model answer could be obtained

    def to_json(self) -> dict[str, Any]:
        return {
            "raw_stdout": self.raw_stdout,
            "raw_stderr": self.raw_stderr,
            "exit_code": self.exit_code,
            "latency_s": self.latency_s,
            "timed_out": self.timed_out,
            "result_text": self.result_text,
            "is_error": self.is_error,
            "cost_usd": self.cost_usd,
            "models": self.models,
            "usage": self.usage,
            "session_id": self.session_id,
            "duration_ms": self.duration_ms,
            "infra_error": self.infra_error,
        }


class Backend(Protocol):
    """Anything that answers one prompt (the real CLI, or a fake in tests)."""

    async def complete(self, prompt: str) -> CallResult: ...

    def describe(self) -> dict[str, Any]: ...


def infrastructure_error(result: CallResult, expected_model: str | None) -> str | None:
    """A27 / A26: why `result` carries no usable model answer, or None.

    Timeouts, non-zero exits, CLI or API errors (rate limits included; the CLI reports
    them as ``is_error``), a missing result text and a reported model usage that is not
    exactly ``[expected_model]`` are infrastructure errors. A result text that is not a
    valid answer is not: it is the model's (invalid) answer.
    """
    if result.timed_out:
        return "timeout"
    if result.exit_code not in (0, None):
        return "nonzero_exit"
    if result.is_error:
        return "api_error"
    if result.result_text is None:
        return "no_result"
    if expected_model is not None and result.models != [expected_model]:
        return "model_mismatch"
    return None


def interpret(
    stdout: str,
    stderr: str,
    exit_code: int | None,
    latency: float,
    expected_model: str | None = None,
) -> CallResult:
    """Extract result text, error flag, cost and model ids from ``--output-format json``;
    `infra_error` is set by `infrastructure_error` (model checked if `expected_model`)."""
    payload: Any = None
    try:
        payload = json.loads(stdout) if stdout.strip() else None
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        kind = "nonzero_exit" if exit_code not in (0, None) else "unreadable_output"
        return CallResult(
            stdout, stderr, exit_code, latency, False, None, True, None, infra_error=kind
        )
    d: dict[str, Any] = payload  # pyright: ignore[reportUnknownVariableType]
    result = d.get("result")
    cost = d.get("total_cost_usd")
    usage = d.get("usage")
    model_usage = d.get("modelUsage")
    duration = d.get("duration_ms")
    session = d.get("session_id")
    out = CallResult(
        raw_stdout=stdout,
        raw_stderr=stderr,
        exit_code=exit_code,
        latency_s=latency,
        timed_out=False,
        result_text=result if isinstance(result, str) else None,
        is_error=bool(d.get("is_error", False)) or exit_code != 0,
        cost_usd=float(cost) if isinstance(cost, int | float) else None,
        models=sorted(str(k) for k in model_usage) if isinstance(model_usage, dict) else [],  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        usage=usage if isinstance(usage, dict) else None,  # pyright: ignore[reportUnknownArgumentType]
        session_id=session if isinstance(session, str) else None,
        duration_ms=float(duration) if isinstance(duration, int | float) else None,
    )
    return replace(out, infra_error=infrastructure_error(out, expected_model))


def call_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """The subprocess environment: only the allowlisted variables present (A28)."""
    env = os.environ if source is None else source
    return {k: env[k] for k in ENV_ALLOWLIST if k in env}


def passed_environment_names(source: Mapping[str, str] | None = None) -> list[str]:
    """Names (never values) of the variables passed to each call."""
    return sorted(call_environment(source))


@dataclass(frozen=True, slots=True)
class ClaudeCLI:
    """The headless Claude Code CLI with minimal context (see the module docstring)."""

    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    system_prompt: str = SYSTEM_PROMPT
    timeout_s: float = TIMEOUT_S
    executable: str = "claude"

    def argv(self) -> list[str]:
        return [
            self.executable,
            "-p",
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--output-format",
            "json",
            "--system-prompt",
            self.system_prompt,
            "--tools",
            "",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--safe-mode",
            "--max-turns",
            "1",
        ]

    def describe(self) -> dict[str, Any]:
        """The call's identity (argv, cwd rule, timeout, model, CLI version).

        The names of the environment variables passed to the call depend on the invoking
        shell, not on the call, so they are reported separately
        (`passed_environment_names`) and are not part of a run's resumable configuration.
        """
        return {
            "kind": "claude_cli",
            "argv": self.argv(),
            "prompt_via": "stdin",
            "cwd": "a fresh empty directory from tempfile.mkdtemp() per call, removed after",
            "timeout_s": self.timeout_s,
            "model_requested": self.model,
            "effort_requested": self.effort,
            "cli_version": cli_version(self.executable),
        }

    async def complete(self, prompt: str) -> CallResult:
        env = call_environment()
        workdir = tempfile.mkdtemp(prefix="cp-")
        start = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.argv(),
                cwd=workdir,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(prompt.encode("utf-8")), timeout=self.timeout_s
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = time.perf_counter() - start
                return CallResult(
                    "", "", None, elapsed, True, None, True, None, infra_error="timeout"
                )
            latency = time.perf_counter() - start
            return interpret(
                out.decode("utf-8", "replace"),
                err.decode("utf-8", "replace"),
                proc.returncode,
                latency,
                expected_model=self.model,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def cli_version_number(text: object) -> str | None:
    """'2.1.304 (Claude Code)' -> '2.1.304' (None if absent)."""
    if not isinstance(text, str) or not text.strip():
        return None
    return text.split()[0]


def cli_version(executable: str = "claude") -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None
