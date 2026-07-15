"""Pure tool-call guardrail primitive tests."""

import json

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    canonical_tool_args,
    classify_tool_failure,
)


def test_tool_call_signature_hashes_canonical_nested_unicode_args_without_exposing_raw_args():
    args_a = {
        "z": [{"β": "☤", "a": 1}],
        "a": {"y": 2, "x": "secret-token-value"},
    }
    args_b = {
        "a": {"x": "secret-token-value", "y": 2},
        "z": [{"a": 1, "β": "☤"}],
    }

    assert canonical_tool_args(args_a) == canonical_tool_args(args_b)
    sig_a = ToolCallSignature.from_call("web_search", args_a)
    sig_b = ToolCallSignature.from_call("web_search", args_b)

    assert sig_a == sig_b
    assert len(sig_a.args_hash) == 64
    metadata = sig_a.to_metadata()
    assert metadata == {"tool_name": "web_search", "args_hash": sig_a.args_hash}
    assert "secret-token-value" not in json.dumps(metadata)
    assert "☤" not in json.dumps(metadata)


def test_default_config_is_soft_warning_only_with_hard_stop_disabled():
    cfg = ToolCallGuardrailConfig()

    assert cfg.warnings_enabled is True
    assert cfg.hard_stop_enabled is False
    assert cfg.exact_failure_warn_after == 2
    assert cfg.same_tool_failure_warn_after == 3
    assert cfg.no_progress_warn_after == 2
    assert cfg.exact_failure_block_after == 5
    assert cfg.same_tool_failure_halt_after == 8
    assert cfg.no_progress_block_after == 5


def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
            },
            "hard_stop_after": {
                "exact_failure": 6,
                "same_tool_failure": 7,
                "idempotent_no_progress": 8,
            },
        }
    )

    assert cfg.warnings_enabled is False
    assert cfg.hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 3
    assert cfg.same_tool_failure_warn_after == 4
    assert cfg.no_progress_warn_after == 5
    assert cfg.exact_failure_block_after == 6
    assert cfg.same_tool_failure_halt_after == 7
    assert cfg.no_progress_block_after == 8


def test_default_repeated_identical_failed_call_warns_without_blocking():
    controller = ToolCallGuardrailController()
    args = {"query": "same"}

    decisions = []
    for _ in range(5):
        assert controller.before_call("web_search", args).action == "allow"
        decisions.append(
            controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
        )

    assert decisions[0].action == "allow"
    assert [d.action for d in decisions[1:]] == ["warn", "warn", "warn", "warn"]
    assert {d.code for d in decisions[1:]} == {"repeated_exact_failure_warning"}
    assert controller.before_call("web_search", args).action == "allow"
    assert controller.halt_decision is None


def test_hard_stop_enabled_blocks_repeated_exact_failure_before_next_execution():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_warn_after=2,
            exact_failure_block_after=2,
            same_tool_failure_halt_after=99,
        )
    )
    args = {"query": "same"}

    assert controller.before_call("web_search", args).action == "allow"
    first = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert first.action == "allow"

    assert controller.before_call("web_search", args).action == "allow"
    second = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert second.action == "warn"
    assert second.code == "repeated_exact_failure_warning"

    blocked = controller.before_call("web_search", args)
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"
    assert blocked.count == 2


def test_success_resets_exact_signature_failure_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, exact_failure_block_after=2, same_tool_failure_halt_after=99)
    )
    args = {"query": "same"}

    controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    controller.after_call("web_search", args, '{"ok":true}', failed=False)

    assert controller.before_call("web_search", args).action == "allow"
    controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert controller.before_call("web_search", args).action == "allow"


def test_file_mutation_lint_error_result_is_not_a_tool_failure():
    write_result = json.dumps({
        "bytes_written": 12,
        "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
    })
    patch_result = json.dumps({
        "success": True,
        "diff": "--- a/tmp.py\n+++ b/tmp.py\n",
        "lsp_diagnostics": "<diagnostics>ERROR [1:1] type mismatch</diagnostics>",
    })

    assert classify_tool_failure("write_file", write_result) == (False, "")
    assert classify_tool_failure("patch", patch_result) == (False, "")


def test_same_tool_varying_args_warns_by_default_without_halting():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(same_tool_failure_warn_after=2, same_tool_failure_halt_after=3)
    )

    first = controller.after_call("terminal", {"command": "cmd-1"}, '{"exit_code":1}', failed=True)
    second = controller.after_call("terminal", {"command": "cmd-2"}, '{"exit_code":1}', failed=True)
    third = controller.after_call("terminal", {"command": "cmd-3"}, '{"exit_code":1}', failed=True)
    fourth = controller.after_call("terminal", {"command": "cmd-4"}, '{"exit_code":1}', failed=True)

    assert first.action == "allow"
    assert [second.action, third.action, fourth.action] == ["warn", "warn", "warn"]
    assert {second.code, third.code, fourth.code} == {"same_tool_failure_warning"}
    assert "Do not switch to text-only replies" in second.message
    assert "keep using tools" in second.message
    assert "diagnose before retrying" in second.message
    assert "different tool" in second.message
    assert controller.halt_decision is None


def test_hard_stop_enabled_halts_same_tool_varying_args_failure_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_block_after=99,
            same_tool_failure_warn_after=2,
            same_tool_failure_halt_after=3,
        )
    )

    first = controller.after_call("terminal", {"command": "cmd-1"}, '{"exit_code":1}', failed=True)
    assert first.action == "allow"
    second = controller.after_call("terminal", {"command": "cmd-2"}, '{"exit_code":1}', failed=True)
    assert second.action == "warn"
    assert second.code == "same_tool_failure_warning"
    third = controller.after_call("terminal", {"command": "cmd-3"}, '{"exit_code":1}', failed=True)
    assert third.action == "halt"
    assert third.code == "same_tool_failure_halt"
    assert third.count == 3


def test_idempotent_no_progress_repeated_result_warns_without_blocking_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )
    args = {"path": "/tmp/same.txt"}
    result = "same file contents"

    for _ in range(4):
        assert controller.before_call("read_file", args).action == "allow"
        decision = controller.after_call("read_file", args, result, failed=False)

    assert decision.action == "warn"
    assert decision.code == "idempotent_no_progress_warning"
    assert controller.before_call("read_file", args).action == "allow"
    assert controller.halt_decision is None


def test_hard_stop_enabled_blocks_idempotent_no_progress_future_repeat():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=2,
        )
    )
    args = {"path": "/tmp/same.txt"}
    result = "same file contents"

    assert controller.before_call("read_file", args).action == "allow"
    assert controller.after_call("read_file", args, result, failed=False).action == "allow"
    assert controller.before_call("read_file", args).action == "allow"
    warn = controller.after_call("read_file", args, result, failed=False)
    assert warn.action == "warn"
    assert warn.code == "idempotent_no_progress_warning"

    blocked = controller.before_call("read_file", args)
    assert blocked.action == "block"
    assert blocked.code == "idempotent_no_progress_block"


def test_successful_mutating_call_resets_no_progress_counts():
    """Interleaved page interactions must not accumulate snapshot repeats.

    Regression for the browser false-positive: a session alternating
    browser_click and browser_snapshot on a slow SPA reached the no-progress
    block even though every snapshot legitimately followed a state-changing
    action. A successful mutating call resets the no-progress ledger.
    """
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=2,
        )
    )
    snapshot_result = "same accessibility tree"

    for _ in range(5):
        assert controller.before_call("browser_snapshot", {}).action == "allow"
        controller.after_call("browser_snapshot", {}, snapshot_result, failed=False)
        controller.after_call("browser_click", {"ref": "@e5"}, '{"success": true}', failed=False)

    assert controller.before_call("browser_snapshot", {}).action == "allow"
    assert controller.halt_decision is None


def test_failed_mutating_call_does_not_reset_no_progress_counts():
    """A mutating call that failed changed nothing — repeats keep counting."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=2,
        )
    )
    snapshot_result = "same accessibility tree"

    for _ in range(2):
        assert controller.before_call("browser_snapshot", {}).action == "allow"
        controller.after_call("browser_snapshot", {}, snapshot_result, failed=False)
        controller.after_call(
            "browser_click", {"ref": "@e5"}, '{"success": false, "error": "no such ref"}',
            failed=True,
        )

    blocked = controller.before_call("browser_snapshot", {})
    assert blocked.action == "block"
    assert blocked.code == "idempotent_no_progress_block"


def test_browser_wait_success_resets_no_progress_counts():
    """browser_wait counts as progress: waiting is the sanctioned way to poll."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=2,
        )
    )
    snapshot_result = "loading spinner"

    for _ in range(3):
        assert controller.before_call("browser_snapshot", {}).action == "allow"
        controller.after_call("browser_snapshot", {}, snapshot_result, failed=False)
        controller.after_call(
            "browser_wait", {"seconds": 3}, '{"success": true, "waited_seconds": 3.0}',
            failed=False,
        )

    assert controller.before_call("browser_snapshot", {}).action == "allow"
    assert controller.halt_decision is None


def test_mutating_or_unknown_tools_are_not_blocked_for_repeated_identical_success_output_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )

    for _ in range(3):
        assert controller.before_call("write_file", {"path": "/tmp/x", "content": "x"}).action == "allow"
        assert controller.after_call("write_file", {"path": "/tmp/x", "content": "x"}, "ok", failed=False).action == "allow"
        assert controller.before_call("custom_tool", {"x": 1}).action == "allow"
        assert controller.after_call("custom_tool", {"x": 1}, "ok", failed=False).action == "allow"


def test_reset_for_turn_clears_bounded_guardrail_state():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, exact_failure_block_after=2, no_progress_block_after=2)
    )
    controller.after_call("web_search", {"query": "same"}, '{"error":"boom"}', failed=True)
    controller.after_call("web_search", {"query": "same"}, '{"error":"boom"}', failed=True)
    controller.after_call("read_file", {"path": "/tmp/x"}, "same", failed=False)
    controller.after_call("read_file", {"path": "/tmp/x"}, "same", failed=False)

    assert controller.before_call("web_search", {"query": "same"}).action == "block"
    assert controller.before_call("read_file", {"path": "/tmp/x"}).action == "block"

    controller.reset_for_turn()

    assert controller.before_call("web_search", {"query": "same"}).action == "allow"
    assert controller.before_call("read_file", {"path": "/tmp/x"}).action == "allow"


def test_repeated_identical_result_halts_successful_varying_arg_loop():
    """The real-world loop: execute_code 'succeeds' with different args every
    call but returns the same blocked/404 body. Failure- and signature-keyed
    counters miss it; result-repetition detection must catch it."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            repeated_result_warn_after=3,
            repeated_result_halt_after=5,
        )
    )
    blocked_body = (
        '{"status": "success", "output": "=== LEVELS.FYI COMPENSATION API ==='
        + "<!DOCTYPE html>" + "x" * 300 + '"}'
    )

    decisions = []
    for i in range(5):
        # Different arguments every call, never classified as a failure.
        decisions.append(
            controller.after_call(
                "execute_code", {"code": f"fetch_attempt_{i}()"}, blocked_body, failed=False
            )
        )

    assert decisions[2].action == "warn"
    assert decisions[2].code == "repeated_result_warning"
    assert decisions[4].action == "halt"
    assert decisions[4].code == "repeated_result_halt"
    assert controller.halt_decision is decisions[4]


def test_repeated_result_ignores_short_and_distinct_outputs():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, repeated_result_halt_after=3)
    )
    # Short identical results must not trip the guard.
    for _ in range(5):
        d = controller.after_call("execute_code", {"code": "x"}, '{"ok":true}', failed=False)
        assert d.action == "allow"
    # Long but genuinely distinct results = real progress, never halts.
    for i in range(5):
        body = '{"status":"success","output":"' + f"unique-{i}-" + "y" * 300 + '"}'
        d = controller.after_call("execute_code", {"code": f"c{i}"}, body, failed=False)
        assert d.action == "allow"
    assert controller.halt_decision is None


def test_observe_assistant_message_warns_then_halts_on_repeated_narration():
    """The model restating the same sentence every turn (different tools/results)
    must warn, then halt, even though no single tool call repeats."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            assistant_repeat_warn_after=2,
            assistant_repeat_halt_after=3,
        )
    )
    line = "I'm hitting the same wall with levels.fyi. Let me try alternative salary sources."

    first = controller.observe_assistant_message(line)
    second = controller.observe_assistant_message("  I'M HITTING the same WALL with levels.fyi.   Let me try alternative salary sources.  ")
    third = controller.observe_assistant_message(line)

    assert first is None
    assert second.action == "warn" and second.code == "assistant_repeat_warning"
    assert third.action == "halt" and third.code == "assistant_repeat_halt"
    assert controller.halt_decision is third


def test_observe_assistant_message_ignores_short_and_distinct_messages():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, assistant_repeat_halt_after=2)
    )
    # Short messages never count.
    assert controller.observe_assistant_message("ok") is None
    assert controller.observe_assistant_message("on it") is None
    # Distinct substantial messages are genuine progress.
    assert controller.observe_assistant_message("Fetching the compensation table from the first source now.") is None
    assert controller.observe_assistant_message("That failed, so I am parsing the cached JSON snapshot instead.") is None
    assert controller.halt_decision is None


def _multimodal_vision_result(image_b64: str, question: str = "Describe everything visible.") -> dict:
    """Shape returned by vision_analyze's native path (see tools/vision_tools.py)."""
    return {
        "_multimodal": True,
        "content": [
            {
                "type": "text",
                "text": (
                    "Image loaded into your context — you can see it natively now. "
                    "Use your built-in vision to answer the user."
                    f"\n\nQuestion: {question}" + " " * 120
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ],
        "text_summary": "Image attached natively for the main model.",
    }


def test_repeated_multimodal_result_same_image_trips_guard():
    """Session 20260701_121806: identical multimodal placeholders repeated 6x with
    zero guard activity because str(result) embedded unique base64 payloads.
    Re-loading the *same* image must count as repetition."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            repeated_result_warn_after=3,
            repeated_result_halt_after=5,
        )
    )
    same = _multimodal_vision_result("A" * 4000)
    decisions = [
        controller.after_call("vision_analyze", {"image_url": f"/tmp/shot_{i}.png"}, same, failed=False)
        for i in range(5)
    ]
    assert decisions[2].action == "warn"
    assert decisions[2].code == "repeated_result_warning"
    assert decisions[4].action == "halt"
    assert decisions[4].code == "repeated_result_halt"


def test_repeated_multimodal_result_distinct_images_is_progress():
    """Distinct screenshots with an identical text part (scrolling through a long
    page) are legitimate progress and must NOT halt, even past the threshold."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, repeated_result_halt_after=5)
    )
    for i in range(8):
        result = _multimodal_vision_result(f"IMG{i}" * 1000)
        d = controller.after_call(
            "vision_analyze", {"image_url": f"/tmp/shot_{i}.png"}, result, failed=False
        )
        assert d.action == "allow"
    assert controller.halt_decision is None


# ── Per-domain failure budget ────────────────────────────────────────────
#
# Real session evidence: with no web_search available, the model fabricates
# URLs on a host that doesn't have the page, and evades the exact-signature
# and result-repetition guards above by mutating the slug on every retry
# (verywellfamily.com/oci-card-renewal -> /oci-renewal-guide ->
# /oci-renewal-process -> /oci-card-renewal-5215361 -> ...). One session made
# 95 fetch_resilient calls this way: 34 HTTP 404, 16 blocked — all with
# "ok": true in the payload, so a guard that only looks at tool-level errors
# never sees them. These tests key on registrable host instead of args/result
# content.


def _fetch_result(url: str, *, status: int = 200, ok: bool = True, blocked: bool = False) -> str:
    """Shape returned by tools.resilient_fetch_tool.resilient_fetch."""
    return json.dumps({"ok": ok, "url": url, "status": status, "blocked": blocked, "text": "x" * 50})


def test_domain_failure_budget_blocks_after_six_slug_mutated_404s_but_not_other_host():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, domain_failure_budget=6)
    )
    host = "verywellfamily.com"
    slugs = [
        "oci-card-renewal",
        "oci-renewal-guide",
        "oci-renewal-process",
        "oci-card-renewal-5215361",
        "oci-renewal-checklist",
        "oci-renewal-faq",
    ]

    for slug in slugs:
        url = f"https://{host}/{slug}"
        assert controller.before_call("fetch_resilient", {"url": url}).action == "allow"
        # fetch_resilient reports ok=true even for a 404 (only bot-blocks set
        # blocked=true), and the caller-computed `failed` flag mirrors that —
        # this is exactly the shape a naive error-only guard would miss.
        decision = controller.after_call(
            "fetch_resilient",
            {"url": url},
            _fetch_result(url, status=404, ok=True, blocked=False),
            failed=False,
        )
        assert decision.action == "allow"

    blocked = controller.before_call(
        "fetch_resilient", {"url": f"https://{host}/oci-renewal-final-answer"}
    )
    assert blocked.action == "block"
    assert blocked.code == "domain_failure_budget_block"
    assert host in blocked.message
    assert blocked.count == 6
    assert controller.halt_decision is blocked

    # A different host must be entirely unaffected.
    other = controller.before_call(
        "fetch_resilient", {"url": "https://irs.gov/oci-renewal-info"}
    )
    assert other.action == "allow"


def test_domain_failure_budget_success_resets_the_host_counter():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, domain_failure_budget=6)
    )
    host = "example.com"

    for i in range(5):
        url = f"https://{host}/path-{i}"
        controller.after_call(
            "fetch_resilient", {"url": url}, _fetch_result(url, status=404), failed=False
        )
    assert controller.before_call("fetch_resilient", {"url": f"https://{host}/x"}).action == "allow"

    # A 200 resets the streak.
    ok_url = f"https://{host}/works"
    controller.after_call(
        "fetch_resilient", {"url": ok_url}, _fetch_result(ok_url, status=200), failed=False
    )

    # 5 more failures after the reset must NOT reach the budget of 6.
    for i in range(5):
        url = f"https://{host}/retry-{i}"
        controller.after_call(
            "fetch_resilient", {"url": url}, _fetch_result(url, status=404), failed=False
        )
    assert controller.before_call("fetch_resilient", {"url": f"https://{host}/one-more"}).action == "allow"

    # The 6th failure since the reset crosses the budget again.
    final_url = f"https://{host}/retry-final"
    controller.after_call(
        "fetch_resilient", {"url": final_url}, _fetch_result(final_url, status=404), failed=False
    )
    blocked = controller.before_call("fetch_resilient", {"url": f"https://{host}/blocked-now"})
    assert blocked.action == "block"
    assert blocked.count == 6


def test_domain_failure_budget_counts_blocked_and_error_and_http_404_shapes_together():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, domain_failure_budget=3)
    )
    host = "paywalled-news.example"

    # Shape 1: bot-detection block (blocked: true, ok: false).
    controller.after_call(
        "fetch_resilient",
        {"url": f"https://{host}/a"},
        json.dumps({"ok": False, "url": f"https://{host}/a", "status": None, "blocked": True}),
        failed=False,
    )
    # Shape 2: tool-level error, classified failed=True by the caller.
    controller.after_call(
        "fetch_resilient", {"url": f"https://{host}/b"}, '{"error":"timeout"}', failed=True
    )
    # Shape 3: ok: true + status 404 — the guard's core target.
    controller.after_call(
        "fetch_resilient",
        {"url": f"https://{host}/c"},
        _fetch_result(f"https://{host}/c", status=404, ok=True, blocked=False),
        failed=False,
    )

    blocked = controller.before_call("fetch_resilient", {"url": f"https://{host}/d"})
    assert blocked.action == "block"
    assert blocked.count == 3


def test_domain_failure_budget_config_override_via_hard_stop_after_mapping():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {"hard_stop_enabled": True, "hard_stop_after": {"domain_failure": 2}}
    )
    assert cfg.domain_failure_budget == 2

    controller = ToolCallGuardrailController(cfg)
    host = "example.org"
    for i in range(2):
        url = f"https://{host}/x{i}"
        controller.after_call(
            "fetch_resilient", {"url": url}, _fetch_result(url, status=500), failed=False
        )

    blocked = controller.before_call("fetch_resilient", {"url": f"https://{host}/x2"})
    assert blocked.action == "block"
    assert blocked.count == 2


def test_domain_failure_budget_covers_web_extract_batch_per_url_results():
    """web_extract batches up to 5 URLs per call; each result entry carries
    its own `error`, so a single call can touch/fail multiple hosts."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, domain_failure_budget=2)
    )
    host = "blocked-docs.example"
    other_host = "fine-docs.example"

    payload = json.dumps(
        {
            "results": [
                {"url": f"https://{host}/one", "error": "404 Not Found"},
                {"url": f"https://{other_host}/one", "error": None, "content": "ok"},
            ]
        }
    )
    controller.after_call(
        "web_extract",
        {"urls": [f"https://{host}/one", f"https://{other_host}/one"]},
        payload,
        failed=False,
    )
    payload2 = json.dumps({"results": [{"url": f"https://{host}/two", "error": "404 Not Found"}]})
    controller.after_call("web_extract", {"urls": [f"https://{host}/two"]}, payload2, failed=False)

    blocked = controller.before_call("web_extract", {"urls": [f"https://{host}/three"]})
    assert blocked.action == "block"
    assert blocked.tool_name == "web_extract"

    assert controller.before_call(
        "web_extract", {"urls": [f"https://{other_host}/two"]}
    ).action == "allow"


def test_domain_failure_budget_covers_browser_navigate_without_url_in_result():
    """browser_navigate's failure payload has no `url` field at all
    (`{"success": false, "error": ...}`) — the host must fall back to args."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, domain_failure_budget=2)
    )
    host = "flaky-site.example"
    fail_payload = json.dumps({"success": False, "error": "Navigation failed"})

    controller.after_call("browser_navigate", {"url": f"https://{host}/a"}, fail_payload, failed=True)
    controller.after_call("browser_navigate", {"url": f"https://{host}/b"}, fail_payload, failed=True)

    blocked = controller.before_call("browser_navigate", {"url": f"https://{host}/c"})
    assert blocked.action == "block"


def test_domain_failure_budget_ignores_non_web_fetch_tools():
    # Other thresholds pinned high so only the domain-failure budget itself
    # (deliberately set to 1) is under test — terminal isn't a web-fetch tool
    # so it must still be allowed after 5 identical failures.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            domain_failure_budget=1,
            exact_failure_block_after=99,
            same_tool_failure_halt_after=99,
        )
    )
    for _ in range(5):
        controller.after_call(
            "terminal", {"command": "curl https://example.com/x"}, '{"exit_code":1}', failed=True
        )
    assert controller.before_call(
        "terminal", {"command": "curl https://example.com/x"}
    ).action == "allow"


def test_domain_failure_budget_fails_open_on_malformed_urls_and_results():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, domain_failure_budget=1)
    )
    # None URL, non-JSON result, missing args entirely — must never raise.
    controller.after_call("fetch_resilient", {"url": None}, "not json at all", failed=True)
    assert controller.before_call("fetch_resilient", {"url": "::not-a-url::"}).action == "allow"
    controller.after_call("fetch_resilient", {}, None, failed=False)
    assert controller.before_call("fetch_resilient", {}).action == "allow"
    controller.after_call("web_extract", {"urls": "not-a-list"}, "{}", failed=False)
    assert controller.before_call("web_extract", {"urls": "not-a-list"}).action == "allow"
