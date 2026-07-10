"""Pure tool-call loop guardrail primitives.

The controller in this module is intentionally side-effect free: it tracks
per-turn tool-call observations and returns decisions. Runtime code owns whether
those decisions become warning guidance, synthetic tool results, or controlled
turn halts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlsplit

from utils import safe_json_loads
from agent.tool_result_classification import file_mutation_result_landed


# Web-fetch-family tools whose single argument (or argument list) names a URL
# the model chose itself. These are the tools where a stuck model evades the
# exact-signature guards above by mutating the URL slug on every retry
# (verywellfamily.com/oci-card-renewal -> /oci-renewal-guide -> ...) while
# hammering the same unreachable host. See ``domain_failure_budget`` below.
WEB_FETCH_TOOL_NAMES = frozenset({"fetch_resilient", "web_extract", "browser_navigate"})

IDEMPOTENT_TOOL_NAMES = frozenset(
    {
        "read_file",
        "search_files",
        "web_search",
        "web_extract",
        "session_search",
        "browser_snapshot",
        "browser_console",
        "browser_get_images",
        "mcp_filesystem_read_file",
        "mcp_filesystem_read_text_file",
        "mcp_filesystem_read_multiple_files",
        "mcp_filesystem_list_directory",
        "mcp_filesystem_list_directory_with_sizes",
        "mcp_filesystem_directory_tree",
        "mcp_filesystem_get_file_info",
        "mcp_filesystem_search_files",
    }
)

MUTATING_TOOL_NAMES = frozenset(
    {
        "terminal",
        "execute_code",
        "write_file",
        "patch",
        "todo",
        "memory",
        "skill_manage",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_scroll",
        "browser_navigate",
        "send_message",
        "cronjob",
        "delegate_task",
        "process",
    }
)


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop detection.

    Warnings are enabled by default and never prevent tool execution. Hard stops
    are explicit opt-in so interactive CLI/TUI sessions get a gentle nudge unless
    the user enables circuit-breaker behavior in config.yaml.
    """

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    repeated_result_warn_after: int = 3
    repeated_result_halt_after: int = 5
    repeated_result_min_chars: int = 200
    assistant_repeat_warn_after: int = 2
    assistant_repeat_halt_after: int = 3
    assistant_repeat_min_chars: int = 40
    domain_failure_budget: int = 6
    idempotent_tools: frozenset[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)
    mutating_tools: frozenset[str] = field(default_factory=lambda: MUTATING_TOOL_NAMES)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ToolCallGuardrailConfig":
        """Build config from the `tool_loop_guardrails` config.yaml section."""
        if not isinstance(data, Mapping):
            return cls()

        warn_after = data.get("warn_after")
        if not isinstance(warn_after, Mapping):
            warn_after = {}
        hard_stop_after = data.get("hard_stop_after")
        if not isinstance(hard_stop_after, Mapping):
            hard_stop_after = {}

        defaults = cls()
        return cls(
            warnings_enabled=_as_bool(data.get("warnings_enabled"), defaults.warnings_enabled),
            hard_stop_enabled=_as_bool(data.get("hard_stop_enabled"), defaults.hard_stop_enabled),
            exact_failure_warn_after=_positive_int(
                warn_after.get("exact_failure", data.get("exact_failure_warn_after")),
                defaults.exact_failure_warn_after,
            ),
            same_tool_failure_warn_after=_positive_int(
                warn_after.get("same_tool_failure", data.get("same_tool_failure_warn_after")),
                defaults.same_tool_failure_warn_after,
            ),
            no_progress_warn_after=_positive_int(
                warn_after.get("idempotent_no_progress", data.get("no_progress_warn_after")),
                defaults.no_progress_warn_after,
            ),
            exact_failure_block_after=_positive_int(
                hard_stop_after.get("exact_failure", data.get("exact_failure_block_after")),
                defaults.exact_failure_block_after,
            ),
            same_tool_failure_halt_after=_positive_int(
                hard_stop_after.get("same_tool_failure", data.get("same_tool_failure_halt_after")),
                defaults.same_tool_failure_halt_after,
            ),
            no_progress_block_after=_positive_int(
                hard_stop_after.get("idempotent_no_progress", data.get("no_progress_block_after")),
                defaults.no_progress_block_after,
            ),
            repeated_result_warn_after=_positive_int(
                warn_after.get("repeated_result", data.get("repeated_result_warn_after")),
                defaults.repeated_result_warn_after,
            ),
            repeated_result_halt_after=_positive_int(
                hard_stop_after.get("repeated_result", data.get("repeated_result_halt_after")),
                defaults.repeated_result_halt_after,
            ),
            repeated_result_min_chars=_positive_int(
                data.get("repeated_result_min_chars"),
                defaults.repeated_result_min_chars,
            ),
            assistant_repeat_warn_after=_positive_int(
                warn_after.get("assistant_repeat", data.get("assistant_repeat_warn_after")),
                defaults.assistant_repeat_warn_after,
            ),
            assistant_repeat_halt_after=_positive_int(
                hard_stop_after.get("assistant_repeat", data.get("assistant_repeat_halt_after")),
                defaults.assistant_repeat_halt_after,
            ),
            assistant_repeat_min_chars=_positive_int(
                data.get("assistant_repeat_min_chars"),
                defaults.assistant_repeat_min_chars,
            ),
            domain_failure_budget=_positive_int(
                hard_stop_after.get("domain_failure", data.get("domain_failure_budget")),
                defaults.domain_failure_budget,
            ),
        )


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable, non-reversible identity for a tool name plus canonical args."""

    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> "ToolCallSignature":
        canonical = canonical_tool_args(args or {})
        return cls(tool_name=tool_name, args_hash=_sha256(canonical))

    def to_metadata(self) -> dict[str, str]:
        """Return public metadata without raw argument values."""
        return {"tool_name": self.tool_name, "args_hash": self.args_hash}


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """Decision returned by the tool-call guardrail controller."""

    action: str = "allow"  # allow | warn | block | halt
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0
    signature: ToolCallSignature | None = None

    @property
    def allows_execution(self) -> bool:
        return self.action in {"allow", "warn"}

    @property
    def should_halt(self) -> bool:
        return self.action in {"block", "halt"}

    def to_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action": self.action,
            "code": self.code,
            "message": self.message,
            "tool_name": self.tool_name,
            "count": self.count,
        }
        if self.signature is not None:
            data["signature"] = self.signature.to_metadata()
        return data


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """Return sorted compact JSON for parsed tool arguments."""
    if not isinstance(args, Mapping):
        raise TypeError(f"tool args must be a mapping, got {type(args).__name__}")
    return json.dumps(
        args,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def classify_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Safety-fallback classifier used only when callers don't pass ``failed``.

    Mirrors ``agent.display._detect_tool_failure`` exactly so the guardrail
    never disagrees with the CLI's user-visible ``[error]`` tag. Production
    callers in ``run_agent.py`` always pass an explicit ``failed=`` derived
    from ``_detect_tool_failure``; this function exists so standalone callers
    (tests, tooling) still get consistent behavior.
    """
    if result is None:
        return False, ""
    if file_mutation_result_landed(tool_name, result):
        return False, ""

    if tool_name == "terminal":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            exit_code = data.get("exit_code")
            if exit_code is not None and exit_code != 0:
                return True, f" [exit {exit_code}]"
        return False, ""

    if tool_name == "memory":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            if data.get("success") is False and "exceed the limit" in data.get("error", ""):
                return True, " [full]"

    lower = result[:500].lower()
    if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
        return True, " [error]"

    return False, ""


class ToolCallGuardrailController:
    """Per-turn controller for repeated failed/non-progressing tool calls."""

    def __init__(self, config: ToolCallGuardrailConfig | None = None):
        self.config = config or ToolCallGuardrailConfig()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._result_repeat_counts: dict[str, int] = {}
        self._assistant_msg_counts: dict[str, int] = {}
        self._domain_failure_counts: dict[str, int] = {}
        self._halt_decision: ToolGuardrailDecision | None = None

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if not self.config.hard_stop_enabled:
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        if tool_name in WEB_FETCH_TOOL_NAMES:
            for host in _hosts_from_args(tool_name, args):
                failure_count = self._domain_failure_counts.get(host, 0)
                if failure_count >= self.config.domain_failure_budget:
                    decision = ToolGuardrailDecision(
                        action="block",
                        code="domain_failure_budget_block",
                        message=(
                            f"Blocked {tool_name}: {host} has failed {failure_count} times "
                            "this turn (HTTP error status and/or blocked responses across "
                            "different URLs on this host). Guessing another URL on this "
                            "domain will not help. Use web_search if available, try a "
                            "different source/domain, or ask the user how to proceed."
                        ),
                        tool_name=tool_name,
                        count=failure_count,
                        signature=signature,
                    )
                    self._halt_decision = decision
                    return decision

        exact_count = self._exact_failure_counts.get(signature, 0)
        if exact_count >= self.config.exact_failure_block_after:
            decision = ToolGuardrailDecision(
                action="block",
                code="repeated_exact_failure_block",
                message=(
                    f"Blocked {tool_name}: the same tool call failed {exact_count} "
                    "times with identical arguments. Stop retrying it unchanged; "
                    "change strategy or explain the blocker."
                ),
                tool_name=tool_name,
                count=exact_count,
                signature=signature,
            )
            self._halt_decision = decision
            return decision

        if self._is_idempotent(tool_name):
            record = self._no_progress.get(signature)
            if record is not None:
                _result_hash, repeat_count = record
                if repeat_count >= self.config.no_progress_block_after:
                    decision = ToolGuardrailDecision(
                        action="block",
                        code="idempotent_no_progress_block",
                        message=(
                            f"Blocked {tool_name}: this read-only call returned the same "
                            f"result {repeat_count} times. Stop repeating it unchanged; "
                            "use the result already provided or try a different query."
                        ),
                        tool_name=tool_name,
                        count=repeat_count,
                        signature=signature,
                    )
                    self._halt_decision = decision
                    return decision

        return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

    def after_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str | None,
        *,
        failed: bool | None = None,
    ) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if failed is None:
            failed, _ = classify_tool_failure(tool_name, result)

        # Per-domain failure budget. Independent of the exact-signature and
        # result-repetition trackers below: those key on the tool call's
        # arguments or result content, both of which the model varies on
        # every retry (a new URL slug guessed each time), so they never
        # accumulate. This tracks failures by registrable host instead, so
        # verywellfamily.com/oci-card-renewal -> /oci-renewal-guide ->
        # /oci-renewal-process all count against the same bucket. Never
        # raises: unparseable URLs/results simply contribute no observation.
        try:
            self._track_domain_failures(tool_name, args, result, failed)
        except Exception:
            pass

        # Result-repetition loop detection. This is deliberately independent of
        # the tool name, its argument signature, and whether the call "failed":
        # the worst observed loops are *successful* calls with *varying* args
        # (e.g. execute_code wrapping web fetches) that keep returning the same
        # blocked/404 body. The failure- and signature-keyed counters below all
        # miss that shape, so we key purely on the result content here.
        repeat_decision = self._track_result_repetition(tool_name, result, signature)
        if repeat_decision is not None and repeat_decision.action == "halt":
            self._halt_decision = repeat_decision
            return repeat_decision

        if failed:
            exact_count = self._exact_failure_counts.get(signature, 0) + 1
            self._exact_failure_counts[signature] = exact_count
            self._no_progress.pop(signature, None)

            same_count = self._same_tool_failure_counts.get(tool_name, 0) + 1
            self._same_tool_failure_counts[tool_name] = same_count

            if self.config.hard_stop_enabled and same_count >= self.config.same_tool_failure_halt_after:
                decision = ToolGuardrailDecision(
                    action="halt",
                    code="same_tool_failure_halt",
                    message=(
                        f"Stopped {tool_name}: it failed {same_count} times this turn. "
                        "Stop retrying the same failing tool path and choose a different approach."
                    ),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )
                self._halt_decision = decision
                return decision

            if self.config.warnings_enabled and exact_count >= self.config.exact_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="repeated_exact_failure_warning",
                    message=(
                        f"{tool_name} has failed {exact_count} times with identical arguments. "
                        "This looks like a loop; inspect the error and change strategy "
                        "instead of retrying it unchanged."
                    ),
                    tool_name=tool_name,
                    count=exact_count,
                    signature=signature,
                )

            if self.config.warnings_enabled and same_count >= self.config.same_tool_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="same_tool_failure_warning",
                    message=_tool_failure_recovery_hint(tool_name, same_count),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )

            return repeat_decision or ToolGuardrailDecision(
                tool_name=tool_name, count=exact_count, signature=signature
            )

        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return repeat_decision or ToolGuardrailDecision(
                tool_name=tool_name, signature=signature
            )

        result_hash = _result_hash(result)
        previous = self._no_progress.get(signature)
        repeat_count = 1
        if previous is not None and previous[0] == result_hash:
            repeat_count = previous[1] + 1
        self._no_progress[signature] = (result_hash, repeat_count)

        if self.config.warnings_enabled and repeat_count >= self.config.no_progress_warn_after:
            return ToolGuardrailDecision(
                action="warn",
                code="idempotent_no_progress_warning",
                message=(
                    f"{tool_name} returned the same result {repeat_count} times. "
                    "Use the result already provided or change the query instead of "
                    "repeating it unchanged."
                ),
                tool_name=tool_name,
                count=repeat_count,
                signature=signature,
            )

        return repeat_decision or ToolGuardrailDecision(
            tool_name=tool_name, count=repeat_count, signature=signature
        )

    def observe_assistant_message(self, content: str | None) -> ToolGuardrailDecision | None:
        """Detect the model narrating the same thing turn after turn.

        Complementary to result-repetition: the worst loops sometimes change
        their tool *and* their tool result yet the model keeps emitting the
        identical sentence ("I'm hitting the same wall with X. Let me try
        alternative sources.") because it is genuinely stuck, not because any
        one tool repeats. We key on the normalized assistant text so this fires
        even when the underlying tool calls and results all differ.

        Returns a warn decision at ``assistant_repeat_warn_after`` and a halt at
        ``assistant_repeat_halt_after``; ``None`` otherwise. Short or empty
        messages are ignored so brief acknowledgements never trip it.
        """
        normalized = _normalize_assistant_text(content)
        if len(normalized) < self.config.assistant_repeat_min_chars:
            return None

        key = _sha256(normalized)
        count = self._assistant_msg_counts.get(key, 0) + 1
        self._assistant_msg_counts[key] = count

        if self.config.hard_stop_enabled and count >= self.config.assistant_repeat_halt_after:
            decision = ToolGuardrailDecision(
                action="halt",
                code="assistant_repeat_halt",
                message=(
                    f"Stopped: you have repeated essentially the same message {count} times "
                    "while making no progress. Repeating the plan is not advancing it. Stop, "
                    "summarize what you have actually obtained so far, and either report the "
                    "blocker to the user or ask how they want to proceed."
                ),
                count=count,
            )
            self._halt_decision = decision
            return decision

        if self.config.warnings_enabled and count >= self.config.assistant_repeat_warn_after:
            return ToolGuardrailDecision(
                action="warn",
                code="assistant_repeat_warning",
                message=(
                    f"You have now said essentially the same thing {count} times "
                    "('try alternative sources'/'hitting the same wall'). Restating the plan "
                    "is not progress. Either take a concretely different action or stop and "
                    "report what you have and the blocker — do not repeat this narration again."
                ),
                count=count,
            )

        return None

    def _track_result_repetition(
        self,
        tool_name: str,
        result: str | None,
        signature: ToolCallSignature,
    ) -> ToolGuardrailDecision | None:
        """Detect repeated identical tool results regardless of args/outcome.

        Returns a halt decision once the same substantial result has recurred
        ``repeated_result_halt_after`` times this turn, a warn decision at the
        warn threshold, or ``None`` otherwise. Short results are ignored so
        trivial outputs (``"[]"``, ``"OK"``, empty strings) never trip it.
        """
        if not result:
            return None
        text = _repetition_text(result)
        if len(text) < self.config.repeated_result_min_chars:
            return None

        result_hash = _result_hash(text)
        count = self._result_repeat_counts.get(result_hash, 0) + 1
        self._result_repeat_counts[result_hash] = count

        if self.config.hard_stop_enabled and count >= self.config.repeated_result_halt_after:
            return ToolGuardrailDecision(
                action="halt",
                code="repeated_result_halt",
                message=(
                    f"Stopped: the last {count} tool calls returned the same result even "
                    "though the arguments changed. You are looping without making progress "
                    "(the data source is unavailable/blocking). Stop retrying variations; "
                    "report what you already have and the blocker, or ask how to proceed."
                ),
                tool_name=tool_name,
                count=count,
                signature=signature,
            )

        if self.config.warnings_enabled and count >= self.config.repeated_result_warn_after:
            return ToolGuardrailDecision(
                action="warn",
                code="repeated_result_warning",
                message=(
                    f"The last {count} tool calls returned an identical result despite "
                    "changed arguments. This is a no-progress loop — the target likely "
                    "isn't reachable. Use what you have or report the blocker instead of "
                    "trying more variations of the same approach."
                ),
                tool_name=tool_name,
                count=count,
                signature=signature,
            )

        return None

    def _is_idempotent(self, tool_name: str) -> bool:
        if tool_name in self.config.mutating_tools:
            return False
        return tool_name in self.config.idempotent_tools

    def _track_domain_failures(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        result: str | None,
        failed: bool,
    ) -> None:
        """Update per-host failure counters for web-fetch-family tools.

        A 2xx/3xx success for a host resets its counter; anything classified
        as a failure (tool-level error, ``blocked: true``, or a parsed HTTP
        ``status`` >= 400 — including the ``ok: true, status: 404`` shape
        that other classifiers miss) increments it. The accumulated count is
        read back in ``before_call`` to block further calls to that host.
        """
        if tool_name not in WEB_FETCH_TOOL_NAMES:
            return
        for host, is_failure in _web_fetch_domain_outcomes(tool_name, args, result, failed):
            if not host:
                continue
            if is_failure:
                self._domain_failure_counts[host] = self._domain_failure_counts.get(host, 0) + 1
            else:
                self._domain_failure_counts.pop(host, None)


def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """Build a synthetic role=tool content string for a blocked tool call."""
    return json.dumps(
        {
            "error": decision.message,
            "guardrail": decision.to_metadata(),
        },
        ensure_ascii=False,
    )


def append_toolguard_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """Append runtime guidance to the current tool result content."""
    if decision.action not in {"warn", "halt"} or not decision.message:
        return result
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    suffix = (
        f"\n\n[{label}: "
        f"{decision.code}; count={decision.count}; {decision.message}]"
    )
    return (result or "") + suffix


def _tool_failure_recovery_hint(tool_name: str, count: int) -> str:
    """Action-oriented guidance for recovering from repeated tool failures."""
    common = (
        f"{tool_name} has failed {count} times this turn. This looks like a loop. "
        "Do not switch to text-only replies; keep using tools, but diagnose before retrying. "
        "First inspect the latest error/output and verify your assumptions. "
    )
    if tool_name == "terminal":
        return common + (
            "For terminal failures, run a small diagnostic such as `pwd && ls -la` "
            "in the same tool, then try an absolute path, a simpler command, a different "
            "working directory, or a different tool such as read_file/write_file/patch."
        )
    return common + (
        "Try different arguments, a narrower query/path, an absolute path when relevant, "
        "or a different tool that can make progress. If the blocker is external, report "
        "the blocker after one diagnostic attempt instead of repeating the same failing path."
    )


def _coerce_args(args: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return args if isinstance(args, Mapping) else {}


def _registrable_host(url: Any) -> str | None:
    """Lowercased netloc with the ``www.`` prefix stripped, or ``None``.

    Deliberately simple (no public-suffix-list dependency): good enough to
    bucket ``verywellfamily.com/oci-card-renewal`` and
    ``verywellfamily.com/oci-renewal-guide-5215361`` together, which is all
    this guard needs. Never raises — any malformed input yields ``None``.
    """
    if not url or not isinstance(url, str):
        return None
    try:
        netloc = urlsplit(url).netloc
        if not netloc:
            return None
        if "@" in netloc:
            netloc = netloc.rsplit("@", 1)[-1]
        host = netloc.split(":")[0].strip().lower()
        if not host:
            return None
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


def _urls_from_args(tool_name: str, args: Mapping[str, Any]) -> list[str]:
    """Pull the URL(s) a web-fetch-family tool call was about to request."""
    if tool_name == "web_extract":
        raw = args.get("urls")
        return [u for u in raw if isinstance(u, str)] if isinstance(raw, list) else []
    single = args.get("url")
    return [single] if isinstance(single, str) and single else []


def _hosts_from_args(tool_name: str, args: Mapping[str, Any]) -> list[str]:
    """Distinct registrable hosts a call would touch, in first-seen order."""
    hosts: list[str] = []
    for url in _urls_from_args(tool_name, args):
        host = _registrable_host(url)
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def _web_fetch_domain_outcomes(
    tool_name: str,
    args: Mapping[str, Any],
    result: str | None,
    failed: bool,
) -> list[tuple[str, bool]]:
    """Return ``(host, is_failure)`` pairs observed in a web-fetch tool result.

    ``web_extract`` batches multiple URLs per call and reports a per-URL
    ``error`` in its ``results`` list, so each entry is resolved against the
    result payload (falling back to the request URLs on a parse failure —
    those are then treated as failures only via the caller-supplied
    ``failed`` flag, never fabricated). Single-URL tools (``fetch_resilient``,
    ``browser_navigate``) resolve one host from the result's own ``url``/
    ``final_url`` field, falling back to the request argument, and are a
    failure if the caller already classified the call as failed, the payload
    says ``blocked: true``, or a parsed ``status`` is >= 400 — the last of
    these is the case a plain error/blocked check alone misses (``ok: true``
    with ``status: 404``).
    """
    parsed = safe_json_loads(result or "")

    if tool_name == "web_extract":
        entries: list[tuple[str, bool]] = []
        if isinstance(parsed, Mapping) and isinstance(parsed.get("results"), list):
            for item in parsed["results"]:
                if not isinstance(item, Mapping):
                    continue
                host = _registrable_host(item.get("url"))
                if host:
                    entries.append((host, bool(item.get("error"))))
        if entries:
            return entries
        # Result didn't parse into the expected shape (e.g. a top-level
        # error) — fall back to the requested hosts, tagged with whatever
        # the caller already determined about the call as a whole.
        return [(host, failed) for host in _hosts_from_args(tool_name, args)]

    url = None
    if isinstance(parsed, Mapping):
        url = parsed.get("url") or parsed.get("final_url")
    if not url:
        candidates = _urls_from_args(tool_name, args)
        url = candidates[0] if candidates else None
    host = _registrable_host(url)
    if not host:
        return []

    is_failure = bool(failed)
    if not is_failure and isinstance(parsed, Mapping):
        if parsed.get("blocked"):
            is_failure = True
        else:
            status = parsed.get("status")
            if status is not None:
                try:
                    is_failure = int(status) >= 400
                except (TypeError, ValueError):
                    pass
    return [(host, is_failure)]


def _normalize_assistant_text(content: str | None) -> str:
    """Lowercase, collapse whitespace, and drop reasoning tags for stable
    narration comparison. Keeps detection robust to trivial reformatting while
    still treating genuinely different sentences as different."""
    if not content:
        return ""
    text = content if isinstance(content, str) else str(content)
    import re

    text = re.sub(r"</?(?:REASONING_SCRATCHPAD|think|reasoning)>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _repetition_text(result: Any) -> str:
    """Normalize a tool result for repetition tracking.

    Multimodal results (``{"_multimodal": True, "content": [...]}``, e.g.
    vision_analyze) embed base64 image payloads, so ``str(result)`` is unique
    per image and identical *text* parts can never be seen as repetition —
    the guard was blind to the repeated "Image loaded into your context"
    placeholder (observed 6x in session 20260701_121806 with zero guard
    activity). Represent them as their text parts plus a short digest of each
    non-text payload: re-loading the *same* image repeatedly still counts as
    repetition, while distinct images (legitimate page-scroll screenshots)
    still count as progress.
    """
    if isinstance(result, Mapping) and result.get("_multimodal"):
        parts: list[str] = []
        for part in result.get("content") or []:
            if not isinstance(part, Mapping):
                parts.append(str(part))
                continue
            if part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            else:
                payload = part.get("image_url")
                if isinstance(payload, Mapping):
                    payload = payload.get("url")
                parts.append(f"[media:{_sha256(str(payload or ''))[:16]}]")
        return "\n".join(parts)
    return result if isinstance(result, str) else str(result)


def _result_hash(result: str | None) -> str:
    parsed = safe_json_loads(result or "")
    if parsed is not None:
        try:
            canonical = json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except TypeError:
            canonical = str(parsed)
    else:
        canonical = result or ""
    return _sha256(canonical)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
